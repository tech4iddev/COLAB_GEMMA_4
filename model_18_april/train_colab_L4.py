"""
Script Training Gemma 2 9B - Optimized for Google Colab L4 GPU
Lokasi: model_18_april/train_colab_L4.py
"""
import os
print("\n[DEBUG] File: train_colab_L4.py | Update: 2026-04-19 00:23 (Target 0.8)")

try:
    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer
    from transformers import TrainingArguments, TrainerCallback
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
    from transformers import TrainingArguments, TrainerCallback
    from datasets import load_dataset

# 0. Custom Callback untuk Dynamic Stopping
class SmartStoppingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            current_loss = logs.get("loss")
            current_epoch = state.epoch
            
            # KRITERIA STOP: Sudah 1 Epoch DAN Loss di bawah 0.8
            if current_epoch >= 1.0 and current_loss is not None and current_loss < 0.8:
                print(f"\n🎯 SMART STOP: Target tercapai! (Epoch: {current_epoch:.2f}, Loss: {current_loss:.4f})")
                print("Lanjut ke proses Final Saving & GGUF...")
                control.should_training_stop = True

def train_on_colab():
    # 1. Konfigurasi Dasar
    max_seq_length = 2048  # Safe context for L4
    load_in_4bit = True

    # 2. Inisialisasi Model & Tokenizer
    print("📦 Loading model Gemma 2 9B (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/gemma-2-9b-it-bnb-4bit",
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
    )

    # 3. Konfigurasi LoRA (Optimized for Structural Engineering)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 4. Format Prompt (Gemma Style)
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

    # 5. Load & Format Dataset
    print("📂 Loading dataset...")
    dataset_path = "datasets/dataset_final_18_april.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset tidak ditemukan di: {dataset_path}")
        return

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # 6. Konfigurasi Training (SAFE MODE - Mencegah CUDA OOM)
    drive_base_path = "/content/drive/MyDrive/Structural_AI_Project"
    output_dir = os.path.join(drive_base_path, "outputs")
    final_model_path = os.path.join(drive_base_path, "gemma2-9b-structural-18april")

    if not os.path.exists(drive_base_path):
        os.makedirs(drive_base_path, exist_ok=True)

    print("🚀 Menyiapkan Trainer (Safe Mode)...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = True, 
        callbacks = [SmartStoppingCallback()], # Aktifkan Smart Stop
        args = TrainingArguments(
            per_device_train_batch_size = 1,  # Safe batch size
            gradient_accumulation_steps = 16, # Global Batch = 16
            warmup_steps = 10,
            max_steps = 1500, # Dinaikkan untuk mencapai +- 2 Epoch (11800/16 * 2)
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(), 
            logging_steps = 1,
            optim = "paged_adamw_8bit",      # Memory efficient optimizer
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = output_dir, 
            save_total_limit = 2,
            save_steps = 50, # Simpan setiap 50 step
        ),
    )

    # 7. Jalankan Training (Dengan Fitur Auto-Resume)
    print(f"🔥 Memulai Training (Syncing to: {output_dir})...")
    
    # Cek apakah ada checkpoint untuk di-resume
    resume_checkpoint = None
    if os.path.exists(output_dir):
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if "checkpoint-" in d]
        if checkpoints:
            resume_checkpoint = True # Akan mengambil yang terbaru secara otomatis dari output_dir
            print(f"🔄 Checkpoint ditemukan! Melanjutkan training dari: {output_dir}")

    trainer.train(resume_from_checkpoint = resume_checkpoint)

    # 8. Simpan Model Akhir Merged
    print(f"📦 Menyimpan model merged ke Drive: {final_model_path}")
    model.save_pretrained_merged(final_model_path, tokenizer, save_method = "merged_16bit",)

    # 9. Konversi Otomatis ke GGUF Q4_K_M (Untuk Mac M4)
    print("\n🛠️ Memulai Konversi GGUF (Q4_K_M)...")
    gguf_drive_path = os.path.join(drive_base_path, "GGUF_MODELS")
    if not os.path.exists(gguf_drive_path):
        os.makedirs(gguf_drive_path, exist_ok=True)

    model.save_pretrained_gguf(
        "structural_ai_model", 
        tokenizer, 
        quantization_method = "q4_k_m"
    )

    # Pindahkan GGUF ke Drive
    import shutil
    for file in os.listdir("."):
        if file.endswith(".gguf"):
            shutil.move(file, os.path.join(gguf_drive_path, "gemma2-9b-structural-18april-Q4_K_M.gguf"))
            print(f"✅ SEMUA SELESAI! GGUF siap di: {gguf_drive_path}")
            break

if __name__ == "__main__":
    # Cegah fragmentasi memori
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    train_on_colab()
