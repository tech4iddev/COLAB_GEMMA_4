import json
import os

def generate_ahs_dataset(input_json, output_jsonl):
    if not os.path.exists(input_json):
        return

    with open(input_json, "r") as f:
        data = json.load(f)

    dataset = []
    
    # 1. QA - Beton
    beton_data = data.get("ahsp_beton", {})
    for mutu, bahan in beton_data.items():
        dataset.append({
            "instruction": f"Berapa kebutuhan bahan (Semen, Pasir, Kerikil) untuk membuat 1 m3 beton mutu {mutu} sesuai AHSP PUPR 2022?",
            "input": "",
            "output": f"Berdasarkan AHSP PUPR 2022, untuk membuat 1 m3 beton mutu {mutu} diperlukan bahan sebagai berikut:\n"
                      f"- Semen: {bahan['semen']['qty']} {bahan['semen']['unit']}\n"
                      f"- Pasir Beton: {bahan['pasir_beton']['qty']} {bahan['pasir_beton']['unit']}\n"
                      f"- Kerikil/Split: {bahan['kerikil']['qty']} {bahan['kerikil']['unit']}\n"
                      f"- Air: {bahan['air']['qty']} {bahan['air']['unit']}."
        })

    # 2. QA - Pembesian
    besi_data = data.get("ahsp_pembesian", {})
    for item, bahan in besi_data.items():
        dataset.append({
            "instruction": f"Berapa koefisien kebutuhan bahan untuk pekerjaan perakitan {item}?",
            "input": "",
            "output": f"Sesuai standar AHSP, untuk perakitan {item} diperlukan:\n"
                      f"- Besi Beton: {bahan['besi_beton']['qty']} {bahan['besi_beton']['unit']} (termasuk waste)\n"
                      f"- Kawat Beton (Bendrat): {bahan['kawat_beton']['qty']} {bahan['kawat_beton']['unit']}."
        })

    # Simpan ke JSONL
    with open(output_jsonl, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"✅ Berhasil membuat {len(dataset)} data tanya-jawab AHS di {output_jsonl}")

if __name__ == "__main__":
    BASE_DIR = "/Users/caturimamsaputro/Documents/GEMMA_4_unsloth"
    input_file = os.path.join(BASE_DIR, "database_ahs_pupr.json")
    output_file = os.path.join(BASE_DIR, "dataset_ahs_qa.jsonl")
    generate_ahs_dataset(input_file, output_file)
