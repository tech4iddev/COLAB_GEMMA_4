import json
import os

def generate_steel_dataset(input_json, output_jsonl):
    if not os.path.exists(input_json):
        print(f"❌ File {input_json} tidak ditemukan.")
        return

    with open(input_json, "r") as f:
        data = json.load(f)

    dataset = []
    profiles = data.get("profiles", {})

    for category, items in profiles.items():
        for item in items:
            size = item["size"]
            weight = item["weight"]
            
            # 1. Pertanyaan Berat
            dataset.append({
                "instruction": f"Berapa berat per meter untuk profil baja {category} {size}?",
                "input": "",
                "output": f"Berdasarkan standar profil baja Indonesia, profil {category} {size} memiliki berat satuan sebesar **{weight} kg/m**."
            })

            # 2. Pertanyaan Spesifikasi Teknis (Khusus IWF/H-Beam)
            if category in ["IWF", "H_BEAM"]:
                h, b, t1, t2 = item["H"], item["B"], item["t1"], item["t2"]
                dataset.append({
                    "instruction": f"Sebutkan spesifikasi dimensi profil {category} {size}.",
                    "input": "",
                    "output": f"Profil {category} {size} memiliki dimensi teknis sebagai berikut:\n"
                              f"- Tinggi (H): {h} mm\n"
                              f"- Lebar (B): {b} mm\n"
                              f"- Tebal Badan (t1): {t1} mm\n"
                              f"- Tebal Sayap (t2): {t2} mm\n"
                              f"- Berat Satuan: {weight} kg/m."
                })

            # 3. Pertanyaan Cek Tebal
            if "t1" in item:
                dataset.append({
                    "instruction": f"Berapa tebal badan (web thickness) dari baja {category} {size}?",
                    "input": "",
                    "output": f"Tebal badan (t1) untuk profil {category} {size} adalah {item['t1']} mm."
                })

    # Simpan ke JSONL
    with open(output_jsonl, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"✅ Berhasil membuat {len(dataset)} data tanya-jawab baja di {output_jsonl}")

if __name__ == "__main__":
    BASE_DIR = "/Users/caturimamsaputro/Documents/GEMMA_4_unsloth"
    input_file = os.path.join(BASE_DIR, "datasets_sni/material_baja/database_baja_indonesia.json")
    output_file = os.path.join(BASE_DIR, "dataset_steel_qa.jsonl")
    generate_steel_dataset(input_file, output_file)
