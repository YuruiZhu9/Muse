import os
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.5-2B-Base")
DEFAULT_DATA_PATH = SCRIPT_DIR / "trandata_qwen35_compact_50.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "qwen35_2b_finetuned"
DATA_PATH = Path(os.getenv("QWEN_SFT_DATA_PATH", str(DEFAULT_DATA_PATH))).resolve()
OUTPUT_DIR = Path(os.getenv("QWEN_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))).resolve()
NUM_EPOCHS = float(os.getenv("QWEN_NUM_EPOCHS", "1"))
BATCH_SIZE = int(os.getenv("QWEN_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.getenv("QWEN_GRAD_ACCUM", "8"))
LEARNING_RATE = float(os.getenv("QWEN_LEARNING_RATE", "2e-4"))
HF_TOKEN = os.getenv("HF_TOKEN")


if MODEL_ID.startswith("Qwen/Qwen3.5"):
    current_ver = transformers.__version__
    try:
        ver_parts = tuple(int(x) for x in current_ver.split(".")[:3])
    except Exception:
        ver_parts = (0, 0, 0)
    if ver_parts < (5, 2, 0):
        raise RuntimeError(
            f"当前 transformers={current_ver}，不支持 qwen3_5 架构。"
            "请先升级到 >=5.2.0，或安装 transformers main 分支。"
        )


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("正在加载 Tokenizer...")
hf_kwargs = {"trust_remote_code": True}
if HF_TOKEN:
    hf_kwargs["token"] = HF_TOKEN

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **hf_kwargs)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    **hf_kwargs,
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=10,
    fp16=False,
    bf16=True,
    save_strategy="epoch",
    optim="paged_adamw_32bit",
    report_to="none",
    gradient_checkpointing=True,
)


def format_prompts(example):
    messages = example["messages"]
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
    formatting_func=format_prompts,
)

print("开始训练...")
trainer.train()

print("保存模型中...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
trainer.model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"微调完成，模型已保存至 {OUTPUT_DIR}")
