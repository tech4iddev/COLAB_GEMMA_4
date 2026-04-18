"""
Script Training Gemma 2 9B - Optimized for Google Colab L4 GPU
Lokasi: model_18_april/train_colab_L4.py
"""

try:
    from unsloth import FastLanguageModel
    import torch
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset
except ImportError:
    print("📦 Menginstall dependencies (Unsloth)...")
    import os
    os.system('pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    os.system("pip install --no-deps xformers trl peft accelerate bitsandbytes")
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

    # 5. Konfigurasi Training (Optimized for L4)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = True, # Mempercepat training untuk data SNI yang bervariasi panjangnya
        args = TrainingArguments(
            per_device_train_batch_size = 4, # L4 bisa handle batch size lebih besar
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 120, # Dengan 12k data, 120-200 steps adalah titik mulai yang baik
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(), # L4 mendukung bfloat16 (lebih stabil)
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs_structural_ai",
        ),
    )

    # 6. Jalankan Training
    print("🚀 Memulai Training di L4 GPU...")
    trainer.train()

    # 7. Simpan Model Akhir
    # Simpan dalam format GGUF jika ingin digunakan langsung di Local Mac/Ollama dilepas
    # Atau simpan Merged 16bit untuk akurasi terbaik
    model.save_pretrained_merged("gemma2-9b-structural-18april", tokenizer, save_method = "merged_16bit",)
    print("✨ Selesai! Model disimpan di folder: gemma2-9b-structural-18april")

if __name__ == "__main__":
    import os
    train_on_colab()
