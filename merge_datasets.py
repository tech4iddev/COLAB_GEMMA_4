import json
import glob
import os

def merge_all_jsonl():
    output_file = "dataset_final.jsonl"
    # Cari semua file .jsonl di folder training_data
    jsonl_files = glob.glob("training_data/dataset_*.jsonl")
    
    if not jsonl_files:
        print("❌ Tidak ditemukan file dataset (.jsonl) untuk digabung.")
        return

    print(f"🔄 Menggabungkan {len(jsonl_files)} dataset...")
    
    all_data = []
    for file in jsonl_files:
        print(f"  + Membaca {file}")
        with open(file, "r") as f:
            for line in f:
                all_data.append(json.loads(line))
    
    # Simpan ke satu file final
    with open(output_file, "w") as f:
        for entry in all_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"\n✨ BERHASIL! {len(all_data)} data gabungan siap di: {output_file}")

if __name__ == "__main__":
    merge_all_jsonl()
