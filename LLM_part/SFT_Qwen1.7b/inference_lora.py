import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.5-2B-Base")
ADAPTER_PATH = os.getenv("QWEN_OUTPUT_DIR", str((SCRIPT_DIR / "qwen35_2b_finetuned").resolve()))
HF_TOKEN = os.getenv("HF_TOKEN")

hf_kwargs = {"trust_remote_code": True}
if HF_TOKEN:
    hf_kwargs["token"] = HF_TOKEN

print("加载基座模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    **hf_kwargs,
)

print("加载微调权重...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, **hf_kwargs)

messages = [
    {"role": "user", "content": "请简要说明 SFT 微调的作用。"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("生成中...")
with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,
    )

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("-" * 20)
print(f"回答: {response}")
