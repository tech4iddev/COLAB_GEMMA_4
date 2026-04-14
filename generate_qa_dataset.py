import os
import glob
import json
import time
from tqdm import tqdm
from openai import OpenAI

# ==========================================
# KONFIGURASI API OPENAI & PATH
# ==========================================
# Gunakan API Key OpenAI Anda
API_KEY = "MASUKKAN_API_KEY_OPENAI_ANDA_DISINI"

# Jalur baca hasil ekstraksi PDF Anda
INPUT_DIR = "/content/COLAB_GEMMA_4/sni_markdown"
# Jalur output file JSONL kualitas tinggi
OUTPUT_FILE = "/content/COLAB_GEMMA_4/training_data/dataset_sni_qa_cot.jsonl"

client = OpenAI(api_key=API_KEY)
MODEL_NAME = 'gpt-4o-mini'

def generate_qa_pairs_from_text(text_chunk, filename):
    prompt = f"""
    Anda adalah seorang Ahli Teknik Sipil Struktur dan Perancang Jembatan Senior.
    Tugas Anda adalah merangkum dokumen SNI (Standar Nasional Indonesia) menjadi dataset untuk melatih AI lain.
    
    Berikut adalah teks SNI dari file {filename}:
    \"\"\"{text_chunk}\"\"\"
    
    Buatkan dataset instruksi menjadi:
    1. Tiga (3) Pertanyaan teoritis mendasar dan mendalam (QA) beserta jawaban komprehensif berdasarkan regulasi pada teks di atas.
    2. Satu (1) "Skenario Analisa Studi Kasus" (jika memungkinkan dari teks). Yakni ada angka permisalan (bentang jembatan, beban, dsb) lalu berikan simulasi hitungan step-by-step / Chain-of-Thought secara matematis. Jika teks tidak mendukung hitungan, ganti dengan skenario keputusan/SOP teknik.
    
    PENTING! Otomatiskan balasan Anda HANYA berupa JSON Array murni, tanpa backticks Markdown (```json). Gunakan format ini:
    [
      {{
        "instruction": "Pertanyaan atau deskripsi masalah skenario",
        "input": "",
        "output": "Jawaban mendetail atau perhitungan matematika logika langkah demi langkah (Step 1, Step 2, dst)."
      }},
      ...
    ]
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful structural engineering AI assistant that generates valid JSON outputs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        response_text = response.choices[0].message.content.strip()
        
        # Pembersihan backticks Markdown jika AI masih membandel me-return markdown block
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:-3].strip()
            
        data = json.loads(response_text)
        return data
    except Exception as e:
        print(f"\n⚠️ Terjadi kesalahan parsing JSON atau Request OpenAI API: {e}")
        return []

def main():
    # Pastikan mencari di dalam /content/COLAB_GEMMA_4 agar valid untuk struktur path Colab Anda.
    # Jika dijalankan secara lokal di VS Code, hapus /content/COLAB_GEMMA_4/ pada variabel INPUT_DIR
    search_path = os.path.join(INPUT_DIR, "**", "*.md")
    # Jika foldernya dicari lokal karena script dieksekusi dari dalam folder repot ini:
    if not os.path.exists(INPUT_DIR):
         search_path = os.path.join("sni_markdown", "**", "*.md")

    md_files = glob.glob(search_path, recursive=True)
    
    if not md_files:
        print(f"❌ Tidak ditemukan file .md di path {INPUT_DIR}.")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE.replace("/content/COLAB_GEMMA_4/", "")), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"🚀 Memulai Ekstraksi Dataset Cerdas lewat OpenAI API pada {len(md_files)} file SNI...")
    
    # Buka output sebagai a (append) untuk berjaga-jaga jika terputus
    out_path = OUTPUT_FILE if os.path.exists("/content") else OUTPUT_FILE.replace("/content/COLAB_GEMMA_4/", "")
    
    # Pengamanan Google Drive secara Simultan per-baris
    gdrive_out_path = "/content/drive/MyDrive/COLAB_GEMMA_4/training_data/dataset_sni_qa_cot.jsonl"
    use_gdrive = os.path.exists("/content/drive/MyDrive")
    if use_gdrive:
        os.makedirs(os.path.dirname(gdrive_out_path), exist_ok=True)
        print("📁 Sinkronisasi ke Google Drive telah AKTIF!")
        
    with open(out_path, "a", encoding="utf-8") as f_out:
        f_gdrive = open(gdrive_out_path, "a", encoding="utf-8") if use_gdrive else None
        try:
            for md_path in tqdm(md_files, desc="Memproses Dokumen"):
                filename = os.path.basename(md_path)
                
                with open(md_path, "r", encoding="utf-8") as f_in:
                    content = f_in.read()
                    
                # Pemotongan setiap ~4000 karakter (~700 kata) agar prompt API optimal dan muat 
                chunk_size = 4000
                chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                
                for index, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 300:
                        continue # Abaikan sisa teks yang terlalu pendek agar tidak jadi error no-context
                    
                    # Memanggil OpenAI API
                    qa_data = generate_qa_pairs_from_text(chunk, filename)
                    
                    if qa_data and isinstance(qa_data, list):
                        for qa in qa_data:
                            # Tulis langsung baris per baris ke lokal dan Google Drive
                            line_str = json.dumps(qa, ensure_ascii=False) + "\n"
                            f_out.write(line_str)
                            f_out.flush() # Force write ke disk
                            
                            if f_gdrive:
                                f_gdrive.write(line_str)
                                f_gdrive.flush() # Tercatat aman di GDrive saat itu juga
                    
                    # Jeda minimal 1 detik untuk menghindari Rate Limit ringan (sesuaikan dengan tipe tier OpenAI Anda)
                    time.sleep(1)
        finally:
            if f_gdrive:
                f_gdrive.close()

    print(f"\n✨ Dataset QA Teoritis & Analisa Perhitungan CoT Selesai Dibuat!")

if __name__ == "__main__":
    main()
