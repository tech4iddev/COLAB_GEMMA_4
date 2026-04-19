import json
import re

input_file = "model_18_april/datasets/clean_theory_dataset.jsonl"
output_file = "model_18_april/datasets/super_clean_theory.jsonl"

def clean_text(text):
    # 1. Hapus tag HTML seperti <span id=...>
    text = re.sub(r'<[^>]+>', '', text)
    # 2. Hapus referensi Gambar/Halaman Markdown
    text = re.sub(r'!\[\]\(.*?\)', '', text)
    text = re.sub(r'#+\s*.*id=.*', '', text)
    # 3. Hapus angka-angka halaman yang berceceran (misal: 13 | 4.1)
    text = re.sub(r'\d+\s*\|\s*\d+', '', text)
    # 4. Hapus karakter aneh yang berulang
    text = re.sub(r'[-_]{3,}', '', text)
    # 5. Gabungkan kata yang terpisah baris (Daft | ar -> Daftar)
    text = text.replace(" | ", "")
    # 6. Bersihkan spasi ganda
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print(f"🧼 Memulai Pencucian Total Dataset: {input_file}")

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        try:
            data = json.loads(line)
            data["instruction"] = clean_text(data["instruction"])
            data["input"] = clean_text(data["input"])
            data["output"] = clean_text(data["output"])
            
            # Hanya simpan yang isinya masih cukup panjang (berarti ada ilmunya)
            if len(data["output"]) > 50:
                f_out.write(json.dumps(data) + "\n")
        except:
            continue

print(f"✅ Pencucian Selesai! File disimpan di: {output_file}")
