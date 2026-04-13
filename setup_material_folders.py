import os
import shutil

# Konfigurasi
BASE_DIR = "/Users/caturimamsaputro/Documents/GEMMA_4_unsloth"
DATA_DIR = os.path.join(BASE_DIR, "datasets_sni")

# Struktur folder berbasis material
folders = {
    "material_beton": ["2847", "Beton"],
    "material_baja": ["1729", "Baja"],
    "material_geoteknik": ["8460", "Geoteknik"],
    "material_kayu": ["7973", "Kayu"]
}

def setup():
    print("🚀 Memulai penataan folder dataset SNI...")
    
    # 1. Buat folder induk jika belum ada
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"✅ Folder induk {DATA_DIR} berhasil dibuat.")

    # 2. Buat sub-folder material
    for folder_name in folders.keys():
        path = os.path.join(DATA_DIR, folder_name)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"  📁 Sub-folder {folder_name} siap.")

    # 3. Identifikasi dan pindahkan file PDF yang ada di root
    files_in_root = [f for f in os.listdir(BASE_DIR) if f.lower().endswith(".pdf")]
    
    count = 0
    for file in files_in_root:
        moved = False
        file_lower = file.lower()
        for folder_name, keywords in folders.items():
            for kw in keywords:
                if kw.lower() in file_lower:
                    src = os.path.join(BASE_DIR, file)
                    dst = os.path.join(DATA_DIR, folder_name, file)
                    try:
                        shutil.move(src, dst)
                        print(f"🚚 Memindahkan: {file} ➡️ {folder_name}")
                        count += 1
                        moved = True
                    except Exception as e:
                        print(f"❌ Gagal memindahkan {file}: {e}")
                    break
            if moved: break
            
    print(f"\n✨ Selesai! {count} file berhasil dirapikan ke dalam 'datasets_sni'.")

if __name__ == "__main__":
    setup()
