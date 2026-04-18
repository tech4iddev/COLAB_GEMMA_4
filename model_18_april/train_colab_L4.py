"""
Script Training Gemma 2 9B - Optimized for Google Colab L4 GPU
Lokasi: model_18_april/train_colab_L4.py
"""
import os
print("\n[DEBUG] File: train_colab_L4.py | Update: 2026-04-18 23:37 (OOM Fix)")

try:
    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset
except ImportError:
    print("📦 Menginstall dependencies (Unsloth)...")
    import os
    os.system('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    os.system('pip install unsloth_zoo')
    os.system('pip install --no-deps xformers trl peft accelerate bitsandbytes')
    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

def train_on_colab():
    # 1. Konfigurasi Model
    max_seq_length = 4096  # L4 GPU memiliki VRAM cukup untuk context panjang
    dtype = None           # Auto detect (akan menggunakan bfloat16 di L4)
    load_in_4bit = True    # 4-bit quantization tetap direkomendasikan untuk efisiensi

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/gemma-2-9b-it-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 2. Konfigurasi LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 3. Format Prompt (Gemma Style)
    prompt_style = """<start_of_turn>user
{}
{}<end_of_turn>
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

    # 4. Load Dataset
    dataset_path = "datasets/dataset_final_18_april.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ File {dataset_path} tidak ditemukan. Pastikan sudah upload ke Colab.")
        return

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # 5. Konfigurasi Training (Optimized for L4 + GDrive Sync)
    # Tentukan Path di Google Drive
    drive_base_path = "/content/drive/MyDrive/Structural_AI_Project"
    output_dir = os.path.join(drive_base_path, "outputs")
    final_model_path = os.path.join(drive_base_path, "gemma2-9b-structural-18april")

    if not os.path.exists(drive_base_path):
        try:
            os.makedirs(drive_base_path)
            print(f"📁 Created base directory on Drive: {drive_base_path}")
        except:
            print("⚠️ Gagal membuat folder di Drive. Pastikan Drive sudah di-mount!")

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = True, 
        args = TrainingArguments(
            per_device_train_batch_size = 2, # Diturunkan dari 4 ke 2 untuk menghemat VRAM
            gradient_accumulation_steps = 8, # Dinaikkan dari 4 ke 8 (Total global batch tetap 16)
            warmup_steps = 10,
            max_steps = 120, 
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(), 
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = output_dir, # Langsung ke Drive
            save_total_limit = 2,    # Simpan 2 checkpoint terakhir saja di Drive
            save_steps = 30,          # Simpan checkpoint setiap 30 step
        ),
    )

    # 6. Jalankan Training
    print(f"🚀 Memulai Training di L4 GPU... (Syncing to Drive: {output_dir})")
    trainer.train()

    # 7. Simpan Model Akhir (Penting: Simpan Merged 16bit Terlebih Dahulu)
    # Tentukan Path di Google Drive
    gguf_drive_path = os.path.join(drive_base_path, "GGUF_MODELS")
    if not os.path.exists(gguf_drive_path):
        os.makedirs(gguf_drive_path)

    print(f"📦 Menyimpan model merged 16-bit ke: {final_model_path}")
    model.save_pretrained_merged(final_model_path, tokenizer, save_method = "merged_16bit",)

    # 8. Konversi Otomatis ke GGUF (Optimized for Mac M4)
    print("\n🚀 Memulai Konversi Otomatis ke GGUF (Q4_K_M)...")
    gguf_filename = "gemma2-9b-structural-18april-Q4_K_M.gguf"
    
    model.save_pretrained_gguf(
        "structural_ai_model", 
        tokenizer, 
        quantization_method = "q4_k_m"
    )

    # 9. Pindahkan file GGUF ke Drive
    import shutil
    for file in os.listdir("."):
        if file.endswith(".gguf"):
            shutil.move(file, os.path.join(gguf_drive_path, gguf_filename))
            print(f"✨ ALL DONE! Model GGUF Anda siap di Drive: {gguf_drive_path}/{gguf_filename}")
            break

if __name__ == "__main__":
    import os
    # Optimasi untuk mencegah fragmentasi memori (Mengatasi CUDA OOM)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    train_on_colab()
