# --- LOG UPDATE ---
# Tanggal: 2026-04-19 15:52
# Update: Script training Qwen 2.5 7B menggunakan dataset Hybrid Expert dan format ChatML untuk akurasi teknis maksimal.
# ------------------
import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# --- KONFIGURASI ---
# Qwen 2.5 sangat kuat untuk rumus dan teknis
MODEL_NAME = "unsloth/Qwen2.5-7B-it-bnb-4bit" 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "datasets/final_expert_hybrid.jsonl")

# Path Output ke Google Drive
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/Structural_AI_Project/model_qwen"
OUTPUT_DIR = DRIVE_OUTPUT_DIR if os.path.exists("/content/drive") else "./model_qwen_local"

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

# --- PREPARASI DATASET (FORMAT CHATML UNTUK QWEN) ---
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    
    # Qwen standard ChatML template
    system_prompt = "Anda adalah Structural AI Engineer Expert. Jawablah dengan runtut, detail, dan sertakan referensi SNI."
    
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        text += f"<|im_start|>user\n{instruction}\n{input}<|im_end|>\n"
        text += f"<|im_start|>assistant\n{output}<|im_end|>"
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
        learning_rate=1e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_qwen",
        report_to="none",
    ),
)

print("⚡ Memulai Training Qwen 2.5...")
trainer.train()

# --- SIMPAN MODEL ---
print("💾 Menyimpan Model Hasil Training Qwen...")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")

# --- BLOK TESTING OTOMATIS ---
print("\n" + "="*50)
print("🧪 TESTING OTOMATIS QWEN 2.5...")
print("="*50)

FastLanguageModel.for_inference(model)

def run_test(prompt):
    text = f"<|im_start|>system\nAnda adalah Structural AI Engineer Expert. Jawablah dengan runtut, detail, dan sertakan referensi SNI.<|im_end|>\n"
    text += f"<|im_start|>user\n{prompt}<|im_end|>\n"
    text += f"<|im_start|>assistant\n"
    
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=512, 
        repetition_penalty=1.2, 
        temperature=0.3,
        do_sample=True,
        use_cache=True
    )
    result = tokenizer.batch_decode(outputs)[0]
    final_output = result.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    return final_output

test_questions = [
    "Jelaskan apa itu beban gempa nominal sesuai SNI 1726:2019?",
    "Bagaimana prosedur menentukan kategori risiko struktur gedung?",
    "Sebutkan kombinasi pembebanan untuk metode LRFD."
]

for i, q in enumerate(test_questions):
    print(f"\n[HASIL TEST {i+1}]: {q}")
    print("-" * 30)
    print(run_test(q))
    print("\n" + "="*50)

print(f"✅ Selesai! Model Qwen tersimpan di {OUTPUT_DIR}")
