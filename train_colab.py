from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os

# 1. Konfigurasi Model
model_name = "unsloth/gemma-4-4b-it-bnb-4bit"
max_seq_length = 2048
dtype = None # None untuk auto-detection
load_in_4bit = True # Gunakan 4bit quantization

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 2. Tambahkan LoRA Adapter
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 3. Prompt Template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

# 4. Load Dataset
# Pastikan file final ini sudah dibuat lewat merge_datasets.py
dataset_file = "dataset_final.jsonl"
if not os.path.exists(dataset_file):
    print(f"❌ File {dataset_file} tidak ditemukan. Jalankan merge_datasets.py dahulu.")
else:
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # 5. Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Ganti sesuai kebutuhan
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    # 6. Jalankan Training
    print("🚀 Memulai Training di Google Colab...")
    trainer_stats = trainer.train()
    
    # 7. Simpan Adapter LoRA (Versi Ringan)
    model.save_pretrained("gemma4_lora_model_colab")
    tokenizer.save_pretrained("gemma4_lora_model_colab")
    print("✅ LoRA Adapter disimpan!")

    # 8. EXPORT KE GGUF (PENTING untuk Mac Mini M4)
    # Ini akan menggabungkan model dan adapter, lalu mengubahnya ke format GGUF 4-bit (Ringan untuk 16GB RAM)
    print("🚀 Mengonversi model ke format GGUF untuk Mac Mini M4...")
    model.save_pretrained_gguf("gemma4_final_gguf", tokenizer, quantization_method = "q4_k_m")
    print("✨ BERHASIL! File GGUF siap di folder 'gemma4_final_gguf'.")
    print("Materi ini bisa langsung Anda download ke Mac Mini dan dijalankan!")
