import subprocess
import os
import glob
import shutil
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
        base_filename = os.path.splitext(filename)[0]
        
        # Mengecek apakah file sudah pernah diekstrak (marker versi lama atau baru)
        md_path_1 = os.path.join(current_output_dir, base_filename, f"{base_filename}.md")
        md_path_2 = os.path.join(current_output_dir, f"{base_filename}.md")
        
        if os.path.exists(md_path_1) or os.path.exists(md_path_2):
            tqdm.write(f"\n⏭️ Melewati: {filename} (Sudah ter-ekstrak)")
            continue
            
        tqdm.write(f"\n📂 Memproses: {filename}")
        
        try:
            # Sisipkan Environment Variable untuk membungkam spam log TensorFlow & CUDA
            env_vars = os.environ.copy()
            env_vars["TF_CPP_MIN_LOG_LEVEL"] = "3" 
            env_vars["PYTHONWARNINGS"] = "ignore"

            # Menggunakan subprocess.Popen agar log tiap file terlihat/live streaming
            cmd = f"{MARKER_BIN} \"{pdf_path}\" --output_dir \"{current_output_dir}\""
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env_vars)
            
            # Print output di tiap proses file
            for line in process.stdout:
                line_text = line.strip()
                if not line_text:
                    continue
                # FILTERING: Sembunyikan library warnings yang tidak penting
                spam_keywords = ["tensorflow", "cuda", "computation_placer", "W0000", "E0000", "oneDNN", "AVX2"]
                if any(spam in line_text for spam in spam_keywords):
                    continue
                    
                print(f"   > {line_text}")

            process.wait()

            if process.returncode == 0:
                tqdm.write(f"✅ Selesai mengekstrak: {filename}")
                
                # Otomatis backup ke GDrive secara bertahap setelah tiap file
                try:
                    if os.path.exists("/content/drive/MyDrive"):
                        gdrive_dest = os.path.join("/content/drive/MyDrive/COLAB_GEMMA_4/sni_markdown", material_folder)
                        os.makedirs(gdrive_dest, exist_ok=True)
                        # Gunakan dirs_exist_ok=True untuk merge dan overwrite yang sudah ada
                        shutil.copytree(current_output_dir, gdrive_dest, dirs_exist_ok=True)
                        tqdm.write("   -> 💾 Tersimpan otomatis ke Google Drive.")
                except Exception as ex_drive:
                    tqdm.write(f"   -> ⚠️ Gagal simpan ke GDrive: {ex_drive}")

            else:
                tqdm.write(f"❌ Gagal mengekstrak: {filename} (Detail dapat dilihat di log atas)")

        except Exception as e:
            tqdm.write(f"❌ Terjadi kesalahan pada {filename}: {e}")

    print("\n✨ Seluruh proses ekstraksi SNI di Colab selesai!")

if __name__ == "__main__":
    extract_all_pdfs()
