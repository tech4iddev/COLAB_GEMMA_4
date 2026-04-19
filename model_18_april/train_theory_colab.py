# --- LOG UPDATE ---
# Tanggal: 2026-04-19 12:45
# Update: Menggunakan dataset teori bersih (3.8k data), penyesuaian LR 2e-5, dan sinkronisasi path Colab.
# ------------------
import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# --- KONFIGURASI ---
MODEL_NAME = "unsloth/gemma-2-9b-it-bnb-4bit"
# Mencari dataset relatif terhadap lokasi script ini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "datasets/clean_theory_dataset.jsonl")
OUTPUT_DIR = "/content/gemma2-9b-structural-theory-only"

print("🚀 Memuat Model Gemma 2 9B untuk Training Teori...")

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

# --- PREPARASI DATASET ---
# Kita tambahkan instruksi gaya bahasa langsung ke dalam prompt training
style_instruction = "Jawablah dengan runtut, detail, dan sebutkan referensi pasal SNI yang relevan jika tersedia."

prompt_style = """<start_of_turn>user
""" + style_instruction + """
{}
{} <end_of_turn>
<start_of_turn>model
{}<end_of_turn>"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = prompt_style.format(instruction, input, output)
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
        max_steps=400, # Fokus pada kualitas, bukan kuantitas
        learning_rate=2e-5, # Lebih lambat agar lebih stabil
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none", # Matikan W&B agar tidak minta input login
    ),
)

print("⚡ Memulai Training Teori Bersih...")
trainer.train()

# --- SIMPAN MODEL ---
print("💾 Menyimpan Model Hasil Training Teori...")
model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")

print(f"✅ Selesai! Model tersimpan di {OUTPUT_DIR}")
