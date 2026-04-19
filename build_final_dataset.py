# --- LOG UPDATE ---
# Tanggal: 2026-04-19 15:47
# Update: Menggabungkan Expert Anchor (pembobotan tinggi) dengan Clean Theory untuk menghentikan halusinasi rumus.
# ------------------
import json
import random

anchor_file = "model_llama/datasets/expert_sni_anchor.jsonl"
theory_file = "model_18_april/datasets/super_clean_theory.jsonl"
output_file = "model_llama/datasets/final_expert_hybrid.jsonl"

print("🏗️ Menyusun Dataset Hybrid (Expert + Theory)...")

final_data = []

# 1. Masukkan Data Ahli dengan Bobot Tinggi (Duplikasi 50x)
with open(anchor_file, 'r', encoding='utf-8') as f:
    anchors = [json.loads(line) for line in f]
    for _ in range(50):
        final_data.extend(anchors)
print(f"💎 Menambahkan {len(anchors) * 50} entri Expert Anchor (Duplikasi 50x)")

# 2. Masukkan Data Teori Bersih dengan Bobot Normal
with open(theory_file, 'r', encoding='utf-8') as f:
    theories = [json.loads(line) for line in f]
    final_data.extend(theories)
print(f"📖 Menambahkan {len(theories)} entri Teori Bersih")

# 3. Shuffle (Acak) agar model belajar secara merata
random.shuffle(final_data)

with open(output_file, 'w', encoding='utf-8') as f:
    for entry in final_data:
        f.write(json.dumps(entry) + "\n")

print(f"✅ Dataset Hybrid Selesai! Total: {len(final_data)} entri.")
print(f"📂 Lokasi: {output_file}")
