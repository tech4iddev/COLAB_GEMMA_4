import subprocess
import os
import glob

# Konfigurasi Path
PROJECT_DIR = "/Users/caturimamsaputro/Documents/GEMMA_4_unsloth"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "sni_markdown")
MARKER_BIN = os.path.join(PROJECT_DIR, "venv", "bin", "marker_single")

def extract_all_pdfs():
    # Mencari semua file PDF di direktori datasets_sni dan subdirektorinya
    DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets_sni")
    pdf_files = glob.glob(os.path.join(DATASETS_DIR, "**", "*.pdf"), recursive=True)
    
    if not pdf_files:
        print(f"❌ Tidak ditemukan file PDF di {DATASETS_DIR}")
        return

    print(f"🚀 Ditemukan {len(pdf_files)} file PDF. Memulai ekstraksi massal...")

    for pdf_path in pdf_files:
        # Tentukan folder output berdasarkan nama folder materialnya
        rel_path = os.path.relpath(pdf_path, DATASETS_DIR)
        material_folder = os.path.dirname(rel_path)
        
        # Buat folder output yang sejajar dengan folder material
        current_output_dir = os.path.join(OUTPUT_DIR, material_folder)
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)

        filename = os.path.basename(pdf_path)
        print(f"\n📄 Memproses: {material_folder}/{filename}")
        
        try:
            # Menggunakan Popen untuk streaming output real-time
            process = subprocess.Popen([
                MARKER_BIN, 
                pdf_path, 
                "--output_dir", current_output_dir
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            # Baca output per baris secara real-time
            for line in process.stdout:
                print(f"  > {line.strip()}")

            process.wait()

            if process.returncode == 0:
                print(f"✅ Berhasil mengekstrak {filename}")
            else:
                print(f"❌ Gagal mengekstrak {filename}")

        except Exception as e:
            print(f"❌ Error saat memproses {filename}: {e}")

    print("\n✨ Semua proses ekstraksi selesai!")

if __name__ == "__main__":
    extract_all_pdfs()
