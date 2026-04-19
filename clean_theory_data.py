# --- LOG UPDATE ---
# Tanggal: 2026-04-19 12:45
# Update: Script pembersih dataset untuk membuang tabel rusak dan noise OCR (Berhasil membuang 3k data kotor).
# ------------------
import json
import re

input_file = "model_18_april/datasets/dataset_theory.jsonl"
output_file = "model_18_april/datasets/clean_theory_dataset.jsonl"

print(f"🧹 Memulai pembersihan dataset: {input_file}")

clean_count = 0
skipped_count = 0

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        try:
            data = json.loads(line)
            output_text = data.get("output", "")
            
            # KRITERIA PEMBERSIHAN:
            # 1. Jika mengandung karakter '|' lebih dari 3 kali dalam satu baris (ciri tabel rusak)
            # 2. Jika mengandung kata-kata sampah OCR seperti "Picture_" atau "_page_"
            # 3. Jika teksnya terlalu pendek atau hanya berisi angka/simbol
            
            is_dirty = False
            
            # Cek tabel rusak
            if output_text.count("|") > 4:
                is_dirty = True
            
            # Cek noise OCR
            if "_page_" in output_text or "Picture_" in output_text:
                is_dirty = True
                
            # Cek daftar isi pecah (contoh: Daft | ar isi)
            if re.search(r"\w+\s*\|\s*\w+", output_text):
                is_dirty = True

            if not is_dirty:
                f_out.write(json.dumps(data) + "\n")
                clean_count += 1
            else:
                skipped_count += 1
                
        except Exception as e:
            skipped_count += 1

print(f"✅ Pembersihan Selesai!")
print(f"📊 Total Data Bersih: {clean_count}")
print(f"🗑️ Data Dibuang: {skipped_count}")
print(f"📂 File disimpan di: {output_file}")
