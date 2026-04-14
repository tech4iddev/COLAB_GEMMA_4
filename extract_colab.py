import subprocess
import os
import glob

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

    for pdf_path in pdf_files:
        # Tentukan folder output berdasarkan nama folder material
        rel_path = os.path.relpath(pdf_path, DATASETS_DIR)
        material_folder = os.path.dirname(rel_path)
        
        current_output_dir = os.path.join(OUTPUT_DIR, material_folder)
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)

        filename = os.path.basename(pdf_path)
        print(f"\n📂 Memproses Material: {material_folder}")
        print(f"📄 File: {filename}")
        
        try:
            # Menggunakan Popen untuk streaming output real-time di notebook Colab
            process = subprocess.Popen([
                MARKER_BIN, 
                pdf_path, 
                "--output_dir", current_output_dir
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            # Menampilkan log progress secara real-time
            for line in process.stdout:
                line_text = line.strip()
                if line_text:
                    print(f"  > {line_text}")

            process.wait()

            if process.returncode == 0:
                print(f"✅ Selesai mengekstrak: {filename}")
            else:
                print(f"❌ Gagal mengekstrak: {filename}")

        except Exception as e:
            print(f"❌ Terjadi kesalahan pada {filename}: {e}")

    print("\n✨ Seluruh proses ekstraksi SNI di Colab selesai!")

if __name__ == "__main__":
    extract_all_pdfs()
