"""
Script Training Gemma 2 9B - Optimized for Google Colab L4 GPU
Lokasi: model_18_april/train_colab_L4.py
"""
import os
print("\n[DEBUG] File: train_colab_L4.py | Update: 2026-04-19 09:28 (Dry Run 10 Steps)")

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

def run_post_train_test(model_path):
    """Fungsi untuk mengetes model secara instan setelah merging (Versi Sinkron 2026)"""
    try:
        from unsloth import FastLanguageModel
        import torch
        print(f"\n🧪 [AUTO-TEST] Menjalankan tes instan pada: {model_path}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            max_seq_length = 2048,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(model)

        question = "Hitung kapasitas tarik profil baja jika fy=250 MPa dan A=3000 mm2."
        messages = [{"role": "user", "content": question}]
        
        # LOGIKA 2-TAHAP (Anti EOS Melompong)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        print("🤖 Model sedang berpikir...")
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 512, 
            temperature = 0.1,
            pad_token_id = tokenizer.eos_token_id
        )
        response = tokenizer.batch_decode(outputs)[0]
        
        answer = response.split("<start_of_turn>model\n")[-1].replace("<end_of_turn>", "").strip()
        print("\n" + "="*50)
        print("📢 HASIL TEST INSTAN SETELAH TRAINING:")
        print("-" * 50)
        print(answer)
        print("="*50 + "\n")
        
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"⚠️ Gagal menjalankan auto-test: {e}")

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

    print("\n[DEBUG] File: train_colab_L4.py | Update: 2026-04-19 09:44 (Indo-Focus Mode)")

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Anchor Indonesian: Mencegah bahasa Spanyol/Drift
            instruction_with_anchor = f"[Gunakan Bahasa Indonesia]\n{instruction}"
            text = prompt_style.format(instruction_with_anchor, input, output)
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

    # 6. Konfigurasi Training (FULL MODE - 600 Steps)
    drive_base_path = "/content/drive/MyDrive/Structural_AI_Project"
    output_dir = os.path.join(drive_base_path, "outputs")
    final_model_path = os.path.join(drive_base_path, "gemma2-9b-structural-18april")

    if not os.path.exists(drive_base_path):
        os.makedirs(drive_base_path, exist_ok=True)

    training_args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 16,
        warmup_steps = 50, # Kembali ke setting stabil
        max_steps = 600,   # Training SUNGGUHAN
        learning_rate = 5e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(), 
        logging_steps = 10,
        optim = "paged_adamw_8bit",
        weight_decay = 0.1, # Dinaikkan untuk stabilitas bahasa
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir, 
        save_total_limit = 2,
        save_steps = 200,
        report_to = "none",
    )

    # HOTFIX: Paksa variabel agar ada (Kompatibilitas Transformers v4.46+)
    if not hasattr(training_args, "push_to_hub_token"):
        setattr(training_args, "push_to_hub_token", None)

    print("🚀 Menyiapkan Trainer (Safe Mode)...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = True, 
        callbacks = [SmartStoppingCallback()],
        args = training_args,
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

    # 8. Simpan Merged 16bit Model (Local First for Speed & Safety)
    local_final_path = "/content/gemma2-9b-structural-final"
    print(f"🚀 Menyimpan Merged 16bit Model secara LOKAL ke: {local_final_path}")
    model.save_pretrained_merged(local_final_path, tokenizer, save_method = "merged_16bit")
    
    # --- STEP: AUTO-TEST (Lari dari local supaya cepat) ---
    run_post_train_test(local_final_path)
    
    # 9. Sync ke Google Drive (Safe Transfer)
    print(f"📦 Menyinkronkan model ke Google Drive: {final_model_path}...")
    import shutil
    
    if os.path.exists(final_model_path):
        print("⚠️ Folder lama di Drive ditemukan, menghapus untuk mencegah konflik...")
        shutil.rmtree(final_model_path)
    
    # Copying from local to drive
    shutil.copytree(local_final_path, final_model_path)
    print(f"✅ SINKRONISASI SELESAI! Model aman di Drive: {final_model_path}")
    
    print("\n💡 NOTE: Silakan jalankan 'model_18_april/export_to_gguf.py' jika ingin konversi ke GGUF.")

if __name__ == "__main__":
    # Cegah fragmentasi memori
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    train_on_colab()
