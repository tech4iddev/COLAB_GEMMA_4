import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# --- KONFIGURASI ---
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "datasets/clean_theory_dataset.jsonl")

# Path Output ke Google Drive
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/Structural_AI_Project/model_llama"
# Fallback jika Drive tidak mounted (untuk testing)
OUTPUT_DIR = DRIVE_OUTPUT_DIR if os.path.exists("/content/drive") else "./model_llama_local"

print(f"🚀 Memuat Model {MODEL_NAME} untuk Training Teori...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    load_in_4bit=True,
)

# Tambahkan LoRA Adapter
model = FastLanguageModel.get_peft_model(
    model,
    r=16, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# --- PREPARASI DATASET (FORMAT LLAMA 3.2) ---
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    
    # Llama 3.2 standard chat template
    system_prompt = "Anda adalah Structural AI Engineer Expert. Jawablah dengan runtut, detail, dan sertakan referensi SNI."
    
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Menyederhanakan prompt ke format chat Llama 3
        text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        text += f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}\n{input}<|eot_id|>"
        text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True,)

# --- TRAINING ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=30,
        max_steps=400,
        learning_rate=2e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_llama",
        report_to="none",
    ),
)

print("⚡ Memulai Training Llama 3.2...")
trainer.train()

# --- SIMPAN MODEL ---
print("💾 Menyimpan Model Hasil Training Llama...")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")

print(f"✅ Selesai! Model Llama tersimpan di {OUTPUT_DIR}")
