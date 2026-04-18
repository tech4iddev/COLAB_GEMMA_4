"""
Script Export Model ke GGUF - Optimized for Mac M4 (Ollama/LM Studio)
Lokasi: model_18_april/export_to_gguf.py
"""

from unsloth import FastLanguageModel
import os

def export_now():
    # 1. Tentukan path model (Hasil training sebelumnya di Drive)
    # Sesuaikan jika folder di Drive Anda berbeda
    model_path = "/content/drive/MyDrive/Structural_AI_Project/gemma2-9b-structural-18april"
    output_gguf_name = "gemma2-9b-structural-18april-Q4_K_M"

    if not os.path.exists(model_path):
        print(f"❌ Model tidak ditemukan di: {model_path}")
        print("Pastikan training sudah selesai dan Drive sudah di-mount.")
        return

    print(f"📦 Loading model untuk konversi GGUF dari: {model_path}...")
    
    # 2. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    # 3. Export ke GGUF (Metode Q4_K_M direkomendasikan untuk Mac M4 16GB)
    print("🚀 Memulai proses konversi ke GGUF (Q4_K_M)... Ini akan memakan waktu beberapa menit.")
    
    # Catatan: Unsloth akan mengunduh llama.cpp secara otomatis untuk proses ini
    model.save_pretrained_gguf(
        output_gguf_name, 
        tokenizer, 
        quantization_method = "q4_k_m"
    )

    # 4. Pindahkan hasil GGUF ke Google Drive
    final_drive_path = "/content/drive/MyDrive/Structural_AI_Project/GGUF_MODELS"
    if not os.path.exists(final_drive_path):
        os.makedirs(final_drive_path)
    
    # Cari file yang berakhir dengan .gguf
    import shutil
    for file in os.listdir("."):
        if file.endswith(".gguf"):
            shutil.move(file, os.path.join(final_drive_path, file))
            print(f"✨ BERHASIL! File GGUF siap di-download dari Drive: {final_drive_path}/{file}")

if __name__ == "__main__":
    export_now()
