"""
Script Training Gemma 2 9B menggunakan Unsloth.
Lokasi: model_18_april
"""

from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

def train_model(train_file="dataset_final_18_april.jsonl"):
    max_seq_length = 2048 # Sesuai kebutuhan dokumen teknis
    dtype = None # None untuk auto detection
    load_in_4bit = True # Gunakan 4bit quantization untuk hemat VRAM

    # 1. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/gemma-2-9b-it-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 2. Tambahkan LoRA Adapters
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
    )

    # 3. Format Prompt
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
    dataset = load_dataset("json", data_files=train_file, split="train")
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # 5. Konfigurasi Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Bisa set True untuk mempercepat training short sequences
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Atur sesuai keinginan (misal 60-120 steps)
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

    # 6. Mulai Training
    trainer.train()

    # 7. Simpan Model
    model.save_pretrained_merged("gemma4_final_18_april", tokenizer, save_method = "merged_16bit",)
    print("✨ Model berhasil dilatih dan disimpan di folder gemma4_final_18_april!")

if __name__ == "__main__":
    train_model()
