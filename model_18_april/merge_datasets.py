import json
import os

def merge_jsonl(files, output_file):
    final_data = []
    print("📢 Memulai proses penggabungan dataset...")
    
    for file in files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                print(f"   ➕ Menambahkan {len(lines)} data dari: {file}")
                for line in lines:
                    final_data.append(json.loads(line))
        else:
            print(f"   ⚠️ File tidak ditemukan: {file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in final_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"\n✨ SELESAI! Total {len(final_data)} data siap di {output_file}")

if __name__ == "__main__":
    datasets = [
        "dataset_theory.jsonl",
        "dataset_analysis.jsonl",
        "datasets/ultimate_dataset.jsonl"
    ]
    merge_jsonl(datasets, "dataset_final_18_april.jsonl")
