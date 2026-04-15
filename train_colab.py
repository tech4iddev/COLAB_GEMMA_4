from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import os
import shutil

# 1. Konfigurasi Model
model_name = "unsloth/gemma-2-9b-it-bnb-4bit"
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
# Karena dataset QA kita sering memiliki "input" yang kosong, 
# kita buat 2 jenis prompt: yang pakai input tambahan dan yang tidak.
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

alpaca_prompt_no_input = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Jika kolom input kosong, jangan cetak tag "### Input:" agar Gemma tidak keliru
        if input_text.strip() == "":
            text = alpaca_prompt_no_input.format(instruction, output) + tokenizer.eos_token
        else:
            text = alpaca_prompt.format(instruction, input_text, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

# 4. Load Dataset
# Mengarahkan langsung ke hasil dari script generate_qa_dataset.py
dataset_file = "/content/COLAB_GEMMA_4/training_data/dataset_sni_qa_cot.jsonl"
if not os.path.exists(dataset_file):
    print(f"❌ File {dataset_file} tidak ditemukan. Jalankan extract_colab.py dan generate_qa_dataset.py dahulu.")
else:
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # 5. Deteksi Google Drive untuk Auto-Save Checkpoints
    use_gdrive = os.path.exists("/content/drive/MyDrive")
    gdrive_base = "/content/drive/MyDrive/COLAB_GEMMA_4/trained_models"
    checkpoint_dir = f"{gdrive_base}/checkpoints" if use_gdrive else "outputs"
    
    if use_gdrive:
        os.makedirs(gdrive_base, exist_ok=True)
        print(f"📁 Auto-Save Google Drive Aktif! Checkpoints akan lari ke: {checkpoint_dir}")

    # 6. Trainer
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
            output_dir = checkpoint_dir,
            save_strategy = "steps",
            save_steps = 20, # Menyimpan status (checkpoint) tiap 20 langkah
        ),
    )

    # 7. Jalankan Training
    print("🚀 Memulai Training di Google Colab...")
    trainer_stats = trainer.train()
    
    # 8. Simpan Adapter LoRA (Versi Ringan)
    lora_path = "gemma4_lora_model_colab"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print("✅ LoRA Adapter disimpan secara lokal!")
    
    if use_gdrive:
        shutil.copytree(lora_path, f"{gdrive_base}/{lora_path}", dirs_exist_ok=True)
        print("   -> 💾 LoRA Adapter berhasil disinkronisasi ke Google Drive.")

    # 9. EXPORT KE GGUF (PENTING untuk Mac Mini M4)
    # Ini akan menggabungkan model dan adapter, lalu mengubahnya ke format GGUF 4-bit (Ringan untuk 16GB RAM)
    print("🚀 Mengonversi model ke format GGUF untuk Mac Mini M4...")
    gguf_path = "gemma4_final_gguf"
    model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method = "q4_k_m")
    print(f"✨ BERHASIL! File GGUF siap di folder lokal '{gguf_path}'.")
    
    if use_gdrive:
        try:
            # Mengamankan seluruh folder gguf_path jika berupa folder
            if os.path.isdir(gguf_path):
                shutil.copytree(gguf_path, f"{gdrive_base}/{gguf_path}", dirs_exist_ok=True)
            else:
                # Berjaga-jaga jika unsloth save output sebagai single file
                for f in glob.glob(f"{gguf_path}*gguf"):
                    shutil.copy(f, gdrive_base)
            print("   -> 💾 File GGUF Utama berhasil disinkronisasi ke Google Drive.")
        except Exception as e:
            print(f"   -> ⚠️ Gagal mengamankan GGUF ke Drive: {e}")
            
    print("Materi ini bisa langsung Anda download ke Mac Mini dan dijalankan melalui LM Studio!")
