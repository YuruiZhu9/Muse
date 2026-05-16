import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LLM_part.student_schema import build_student_system_prompt, normalize_compact_result


DEFAULT_CONTEXT_FILE = Path(os.getenv("SFT_CONTEXT_FILE", "LLM_part/user_contexts_big_50_unified.jsonl"))
DEFAULT_INFERENCE_FILE = Path(os.getenv("SFT_INFERENCE_FILE", "LLM_part/user_inferences_big_50_unified.jsonl"))
DEFAULT_OUTPUT_FILE = Path(os.getenv("SFT_OUTPUT_FILE", "LLM_part/SFT_Qwen1.7b/trandata_qwen35_compact_50.jsonl"))
MAX_SAMPLES = int(os.getenv("SFT_MAX_SAMPLES", "0"))


def load_jsonl_map(path: Path, key_field: str) -> dict:
    result = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            obj = json.loads(line)
            if key_field == "user_id":
                if "user_id" in obj:
                    key = str(obj["user_id"])
                else:
                    key = str(obj.get("user_basic_information", {}).get("user_id"))
            else:
                key = str(obj.get(key_field))
            if key and key != "None":
                result[key] = obj
    return result


def detect_join_key(contexts: dict, inferences: dict) -> str:
    if contexts and inferences:
        sample_key_in_context = next(iter(contexts.keys()))
        sample_key_in_inference = next(iter(inferences.keys()))
        if sample_key_in_context and sample_key_in_inference:
            return "sample_id"
    return "user_id"


def choose_join_key(path: Path) -> str:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "sample_id" in obj:
                return "sample_id"
            return "user_id"
    return "user_id"


def main():
    system_prompt = build_student_system_prompt()
    context_join_key = choose_join_key(DEFAULT_CONTEXT_FILE)
    inference_join_key = choose_join_key(DEFAULT_INFERENCE_FILE)

    if context_join_key != inference_join_key:
        raise ValueError(
            f"Context join key ({context_join_key}) and inference join key ({inference_join_key}) do not match."
        )

    join_key = context_join_key
    contexts = load_jsonl_map(DEFAULT_CONTEXT_FILE, join_key)
    inferences = load_jsonl_map(DEFAULT_INFERENCE_FILE, join_key)

    shared_keys = sorted(set(contexts.keys()) & set(inferences.keys()))
    if MAX_SAMPLES > 0:
        shared_keys = shared_keys[:MAX_SAMPLES]
    DEFAULT_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    skipped = 0
    with DEFAULT_OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for key in shared_keys:
            context_row = contexts[key]
            inference_row = inferences[key]
            user_id = str(
                context_row.get("user_id")
                or inference_row.get("user_id")
                or inference_row.get("user_basic_information", {}).get("user_id")
                or key
            )
            context = context_row["context"]
            if isinstance(inference_row, dict) and inference_row.get("error"):
                skipped += 1
                print(f"[SKIP] {key}: inference row contains error -> {inference_row.get('error')}")
                continue

            try:
                assistant_obj = normalize_compact_result(
                    inference_row,
                    fallback_user_id=user_id,
                    generated_time=""
                )
            except Exception as error:
                skipped += 1
                print(f"[SKIP] {key}: normalize failed -> {error}")
                continue

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
                {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False)},
            ]

            file.write(
                json.dumps(
                    {
                        "user_id": user_id,
                        join_key: key,
                        "messages": messages,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rows += 1

    print(f"[OK] SFT dataset saved to: {DEFAULT_OUTPUT_FILE}")
    print(f"[OK] Samples: {rows}")
    print(f"[OK] Skipped: {skipped}")


if __name__ == "__main__":
    main()
