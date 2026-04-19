"""
Script Export Model ke GGUF - Optimized for Mac M4 (Ollama/LM Studio)
Lokasi: model_18_april/export_to_gguf.py
"""
import os
import shutil

# Try to import Unsloth, install if missing (for Colab)
try:
    from unsloth import FastLanguageModel
    import torch
except ImportError:
    print("📦 Menginstall dependencies (Unsloth)...")
    os.system('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    os.system('pip install unsloth_zoo')
    os.system('pip install --no-deps xformers trl peft accelerate bitsandbytes')
    from unsloth import FastLanguageModel
    import torch

def export_now():
    # 1. Konfigurasi Path
    drive_base_path = "/content/drive/MyDrive/Structural_AI_Project"
    model_path = os.path.join(drive_base_path, "gemma2-9b-structural-18april")
    output_gguf_name = "gemma2-9b-structural-18april-Q4_K_M"
    gguf_drive_path = os.path.join(drive_base_path, "GGUF_MODELS")

    # 2. Cek Mount Drive
    if not os.path.exists("/content/drive"):
        print("⚠️ Google Drive belum di-mount. Mencoba mounting...")
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except ImportError:
            print("❌ Bukan di environment Colab? Pastikan path model benar.")

    if not os.path.exists(model_path):
        print(f"❌ Model merged tidak ditemukan di: {model_path}")
        print("Pastikan training di 'train_colab_L4.py' sudah selesai.")
        return

    print(f"📦 Loading model untuk konversi GGUF dari: {model_path}...")
    
    # 3. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    # 4. Export ke GGUF
    print(f"🚀 Memulai konversi ke GGUF ({output_gguf_name})...")
    
    model.save_pretrained_gguf(
        "structural_ai_temp", # Nama sementara
        tokenizer, 
        quantization_method = "q4_k_m"
    )

    # 5. Pindahkan hasil GGUF ke Google Drive
    if not os.path.exists(gguf_drive_path):
        os.makedirs(gguf_drive_path, exist_ok=True)
    
    found = False
    for file in os.listdir("."):
        if file.endswith(".gguf"):
            final_filename = f"{output_gguf_name}.gguf"
            shutil.move(file, os.path.join(gguf_drive_path, final_filename))
            print(f"✅ BERHASIL! File GGUF siap di Drive: {os.path.join(gguf_drive_path, final_filename)}")
            found = True
            break
    
    if not found:
        print("⚠️ Konversi selesai tapi file .gguf tidak ditemukan di folder root.")

if __name__ == "__main__":
    export_now()
