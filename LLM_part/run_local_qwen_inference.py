import json
import os
import re
import sys
import gc
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from LLM_part.student_schema import (
    COMPACT_FIELD_NAMES,
    build_student_system_prompt,
    extract_basic_info_from_context,
    normalize_compact_result,
)


BASE_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.5-2B-Base")
ADAPTER_PATH = os.getenv("QWEN_OUTPUT_DIR", "LLM_part/SFT_Qwen1.7b/qwen35_2b_finetuned")
CONTEXT_FILE = os.getenv("LOCAL_INFERENCE_CONTEXT_FILE", "LLM_part/user_contexts_big_50_unified.jsonl")
OUTPUT_FILE = os.getenv("LOCAL_INFERENCE_OUTPUT_FILE", "LLM_part/user_inferences_big_50_local_compact.jsonl")
RAW_OUTPUT_FILE = os.getenv("LOCAL_INFERENCE_RAW_OUTPUT_FILE", "LLM_part/user_inferences_big_50_local_compact_raw.jsonl")
MAX_USERS = int(os.getenv("LOCAL_INFERENCE_MAX_USERS", "50"))
WHOLE_JSON_MAX_NEW_TOKENS = int(os.getenv("LOCAL_INFERENCE_MAX_NEW_TOKENS", "1280"))
FIELD_MAX_NEW_TOKENS = int(os.getenv("LOCAL_INFERENCE_FIELD_MAX_NEW_TOKENS", "160"))
HF_TOKEN = os.getenv("HF_TOKEN")
FIELDWISE_SYSTEM_PROMPT = """
你是推荐系统中的用户兴趣字段生成器。

你只负责生成一个指定字段的最终中文文本值。
不要输出 JSON。
不要输出字段名。
不要输出 Markdown。
不要输出引号、代码块、解释、前言、补充说明。
不要重复题目。

输出必须是适合直接写入 JSON 字段值的一段中文，长度尽量控制在 1-2 句。
""".strip()
JSON_FIELD_MARKERS = ["user_basic_information", *COMPACT_FIELD_NAMES, "generated_time"]


def parse_json_content(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        stripped = content.lstrip()
        try:
            parsed, _ = decoder.raw_decode(stripped)
            return parsed
        except json.JSONDecodeError:
            pass
        left = content.find("{")
        right = content.rfind("}")
        if left != -1 and right != -1 and right > left:
            sliced = content[left:right + 1]
            try:
                return json.loads(sliced)
            except json.JSONDecodeError:
                parsed, _ = decoder.raw_decode(sliced)
                return parsed
        raise


def repair_truncated_json(raw_text: str) -> dict | None:
    """Attempt to repair common JSON truncation / formatting issues."""
    text = raw_text.strip()
    if not text.startswith("{"):
        left = text.find("{")
        if left == -1:
            return None
        text = text[left:]

    # Try 0: Remove invisible/control characters that break JSON
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    if cleaned != text:
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Try 1: Remove trailing comma before closing brace
    repaired = re.sub(r",\s*}$", "}", text)
    for attempt in [repaired, text]:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass

    # Try 2: Fix unescaped quotes within string values
    fixed = re.sub(r'(?<=[^\\])"(?=[^,:}\]])', r'\\"', text)
    if fixed != text:
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # Try 3: Close unclosed braces
    for suffix in ["}", '"}', '"]}', "}]}", '}]}"']:
        candidate = text.rstrip() + suffix
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Try 4: Close unclosed string + brace
    for suffix in ['"}', "'}"]:
        candidate = text.rstrip().rstrip(",") + suffix
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Try 5: Try to extract each field individually (regex-based)
    fields = {}
    for field_name in ["long_term_intent", "life_stage", "psychological_demand",
                        "retrieval_suggestions", "interest_growth_points",
                        "user_id", "user_active_degree", "gender", "generated_time"]:
        patterns = [
            rf'"{field_name}"\s*:\s*"((?:[^"\\]|\\.)*)"',
            rf'"{field_name}"\s*:\s*"([^"]*)"',
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.DOTALL)
            if m:
                val = m.group(1).replace('\\"', '"').replace("\\n", " ").strip()
                fields[field_name] = val
                break
    if len(fields) >= 5:
        # Build a minimal valid JSON from extracted fields
        result = {"user_basic_information": {}}
        for f in ["user_id", "user_active_degree", "gender"]:
            if f in fields:
                result["user_basic_information"][f] = fields.pop(f)
        for f in ["long_term_intent", "life_stage", "psychological_demand",
                   "retrieval_suggestions", "interest_growth_points"]:
            result[f] = fields.get(f, "")
        result["generated_time"] = fields.get("generated_time", "")
        return result

    return None


def apply_chat_template(tokenizer, messages, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def decode_generated_text(tokenizer, generated_ids, prompt_length: int) -> str:
    output_ids = generated_ids[0][prompt_length:]
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def clean_plain_text(text: str) -> str:
    text = text.strip()
    text = text.replace('\\"', '"').replace("\\'", "'")
    text = text.replace("\\n", " ").replace("\\r", " ")
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_field_from_json_like_text(text: str, field_name: str) -> str | None:
    patterns = [
        rf'"{re.escape(field_name)}"\s*:\s*"((?:[^"\\]|\\.)*)"',
        rf"'{re.escape(field_name)}'\s*:\s*'((?:[^'\\]|\\.)*)'",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            value = match.group(1)
            value = value.replace('\\"', '"').replace("\\n", " ").replace("\\r", " ")
            return clean_plain_text(value)
    return None


def strip_json_shell(text: str, field_name: str) -> str | None:
    start_match = re.search(rf'["\']{re.escape(field_name)}["\']\s*:\s*', text, flags=re.DOTALL)
    if not start_match:
        return None

    payload = text[start_match.end():].lstrip()
    if payload.startswith('"') or payload.startswith("'"):
        payload = payload[1:]

    other_markers = [marker for marker in JSON_FIELD_MARKERS if marker != field_name]
    next_pattern = rf'["\'](?:{"|".join(re.escape(marker) for marker in other_markers)})["\']\s*:'
    next_match = re.search(next_pattern, payload, flags=re.DOTALL)
    if next_match:
        payload = payload[:next_match.start()]

    payload = payload.rstrip().rstrip(",")
    payload = payload.rstrip("}]) \n\r\t")
    payload = payload.rstrip('"\''" ")
    return clean_plain_text(payload)


def clean_field_text(text: str, field_name: str | None = None) -> str:
    text = text.strip()
    if field_name:
        extracted = extract_field_from_json_like_text(text, field_name)
        if extracted:
            return extracted
        if any(marker in text for marker in JSON_FIELD_MARKERS):
            shell_stripped = strip_json_shell(text, field_name)
            if shell_stripped:
                return shell_stripped
    return clean_plain_text(text)


def sanitize_result(result: dict, fallback_user_id: str, basic_info: dict) -> dict:
    for field_name in COMPACT_FIELD_NAMES:
        result[field_name] = clean_field_text(str(result.get(field_name, "")), field_name)

    basic = result.setdefault("user_basic_information", {})
    user_id_value = str(basic.get("user_id", "")).strip()
    if user_id_value in {"", "用户ID", "字符串", "user_id"} or not user_id_value.isdigit():
        basic["user_id"] = str(fallback_user_id)

    active_value = str(basic.get("user_active_degree", "")).strip()
    if active_value in {"字符串", "活跃度", "活跃程度", "user_active_degree"}:
        basic["user_active_degree"] = basic_info.get("user_active_degree", "")

    gender_value = str(basic.get("gender", "")).strip()
    if gender_value in {"字符串", "性别", "gender"}:
        basic["gender"] = basic_info.get("gender", "")

    return result


print("加载 tokenizer...")
hf_kwargs = {"trust_remote_code": True}
if HF_TOKEN:
    hf_kwargs["token"] = HF_TOKEN
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **hf_kwargs)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("加载 4bit 基座模型...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    **hf_kwargs
)

print("加载 LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
system_prompt = build_student_system_prompt()


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def generate_whole_json(context: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context},
    ]
    prompt_text = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=WHOLE_JSON_MAX_NEW_TOKENS,
            do_sample=False,
        )
    return decode_generated_text(tokenizer, generated_ids, len(model_inputs.input_ids[0]))


def generate_single_field(context: str, field_name: str) -> str:
    field_prompt = (
        context
        + f"\n\n任务：只生成字段 `{field_name}` 的最终中文文本值。"
        + "\n要求："
        + "\n1. 只输出该字段值本身。"
        + "\n2. 不要输出 JSON。"
        + "\n3. 不要输出字段名。"
        + "\n4. 不要输出 user_basic_information。"
        + "\n5. 不要输出其它字段内容。"
        + "\n6. 只保留 1-2 句中文。"
    )
    messages = [
        {"role": "system", "content": FIELDWISE_SYSTEM_PROMPT},
        {"role": "user", "content": field_prompt},
    ]
    prompt_text = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=FIELD_MAX_NEW_TOKENS,
            do_sample=False,
        )
    raw_text = decode_generated_text(tokenizer, generated_ids, len(model_inputs.input_ids[0]))
    return clean_field_text(raw_text, field_name)


contexts = []
with open(CONTEXT_FILE, "r", encoding="utf-8-sig") as file:
    for line in file:
        if line.strip():
            contexts.append(json.loads(line))

contexts = contexts[:MAX_USERS]
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
Path(RAW_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

completed_user_ids: set = set()
output_exists = Path(OUTPUT_FILE).exists()
if output_exists and Path(OUTPUT_FILE).stat().st_size > 0:
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                uid = obj.get("user_id") or (
                    obj.get("user_basic_information", {}).get("user_id")
                )
                if uid:
                    completed_user_ids.add(str(uid))
            except json.JSONDecodeError:
                continue
    print(f"[resume] 检测到 {len(completed_user_ids)} 个已完成用户，将跳过")

pending_contexts = [row for row in contexts if str(row["user_id"]) not in completed_user_ids]
total_count = len(pending_contexts)
if total_count == 0:
    print(f"[OK] 所有 {len(contexts)} 个用户已完成推理，退出")
    sys.exit(0)
print(f"待处理: {total_count} 个用户 (总计 {len(contexts)}，已完成 {len(completed_user_ids)})")

file_mode = "a" if output_exists and completed_user_ids else "w"
with open(OUTPUT_FILE, file_mode, encoding="utf-8") as file:
    with open(RAW_OUTPUT_FILE, file_mode, encoding="utf-8") as raw_file:
        for idx, row in enumerate(pending_contexts, 1):
            user_id = row["user_id"]
            sample_id = row.get("sample_id", "")
            anchor_date = row.get("anchor_date", "")
            basic_info = extract_basic_info_from_context(row["context"], user_id)

            try:
                whole_json_raw = generate_whole_json(row["context"])
                raw_result = None
                parse_mode = "whole_json"
                try:
                    raw_result = parse_json_content(whole_json_raw)
                except Exception:
                    repaired = repair_truncated_json(whole_json_raw)
                    if repaired is not None:
                        raw_result = repaired
                        parse_mode = "whole_json_repaired"
                    else:
                        raise
                result = normalize_compact_result(
                    raw_result,
                    fallback_user_id=user_id,
                    fallback_active_degree=basic_info["user_active_degree"],
                    fallback_gender=basic_info["gender"],
                    generated_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                result = sanitize_result(result, str(user_id), basic_info)
                result["generated_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if sample_id:
                    result["sample_id"] = sample_id
                if anchor_date:
                    result["anchor_date"] = anchor_date
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()
                raw_file.write(
                    json.dumps(
                        {
                            "user_id": user_id,
                            "mode": parse_mode,
                            "status": "ok",
                            "raw_output": whole_json_raw,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raw_file.flush()
                print(f"[{idx}/{total_count}] 用户 {user_id} 推理完成 ({parse_mode})")
                cleanup_cuda()
                continue
            except Exception as error:
                raw_file.write(
                    json.dumps(
                        {
                            "user_id": user_id,
                            "mode": "whole_json",
                            "status": "retry_fieldwise",
                            "error": str(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raw_file.flush()
                cleanup_cuda()

            try:
                result = {"user_basic_information": basic_info}
                raw_fields = {}
                for field_name in COMPACT_FIELD_NAMES:
                    value = generate_single_field(row["context"], field_name)
                    result[field_name] = value
                    raw_fields[field_name] = value
                result["generated_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result = normalize_compact_result(
                    result,
                    fallback_user_id=user_id,
                    fallback_active_degree=basic_info["user_active_degree"],
                    fallback_gender=basic_info["gender"],
                    generated_time=result["generated_time"],
                )
                result = sanitize_result(result, str(user_id), basic_info)
                result["generated_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if sample_id:
                    result["sample_id"] = sample_id
                if anchor_date:
                    result["anchor_date"] = anchor_date
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()
                raw_file.write(
                    json.dumps(
                        {
                            "user_id": user_id,
                            "mode": "fieldwise",
                            "status": "ok",
                            "raw_output": raw_fields,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raw_file.flush()
                print(f"[{idx}/{total_count}] 用户 {user_id} 推理完成 (fieldwise)")
                cleanup_cuda()
            except Exception as error:
                raw_file.write(
                    json.dumps(
                        {
                            "user_id": user_id,
                            "mode": "fieldwise",
                            "status": "error",
                            "error": str(error),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raw_file.flush()
                print(f"[{idx}/{total_count}] 用户 {user_id} 推理失败，已跳过: {error}")
                cleanup_cuda()

print(f"[OK] 本地推理结果已保存到: {OUTPUT_FILE}")
