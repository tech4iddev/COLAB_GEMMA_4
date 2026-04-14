import subprocess
import os
import glob
from tqdm import tqdm
# Konfigurasi Path Khusus Google Colab
PROJECT_DIR = "/content/COLAB_GEMMA_4"
DATASETS_DIR = os.path.join(PROJECT_DIR, "SNI Struktur")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "sni_markdown")
# Di Google Colab, perintah marker_single langsung tersedia setelah pip install marker-pdf
MARKER_BIN = "marker_single"

def extract_all_pdfs():
    # Pastikan folder output ada
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    pdf_files = glob.glob(os.path.join(DATASETS_DIR, "**", "*.pdf"), recursive=True)
    
    if not pdf_files:
        print(f"❌ Tidak ditemukan file PDF di {DATASETS_DIR}. Pastikan Anda sudah clone repo dengan benar.")
        return

    print(f"🚀 Memulai ekstraksi massal di Google Colab...")

    for pdf_path in tqdm(pdf_files, desc="Progres Ekstraksi", unit="file"):
        # Tentukan folder output berdasarkan nama folder material
        rel_path = os.path.relpath(pdf_path, DATASETS_DIR)
        material_folder = os.path.dirname(rel_path)
        
        current_output_dir = os.path.join(OUTPUT_DIR, material_folder)
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)

        filename = os.path.basename(pdf_path)
        tqdm.write(f"\n📂 Memproses: {filename}")
        
        try:
            # Menggunakan subprocess.run dengan shell=True agar path cli lebih mudah ditemukan
            cmd = f"{MARKER_BIN} \"{pdf_path}\" --output_dir \"{current_output_dir}\""
            process = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if process.returncode == 0:
                tqdm.write(f"✅ Selesai mengekstrak: {filename}")
            else:
                tqdm.write(f"❌ Gagal mengekstrak: {filename}\nDetail eror: {process.stderr.strip()[:200]}")

        except Exception as e:
            tqdm.write(f"❌ Terjadi kesalahan pada {filename}: {e}")

    print("\n✨ Seluruh proses ekstraksi SNI di Colab selesai!")

if __name__ == "__main__":
    extract_all_pdfs()
