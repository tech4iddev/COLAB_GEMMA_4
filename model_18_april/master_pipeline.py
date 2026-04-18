import subprocess
import json
import os

def run_script(script_name):
    print(f"🚀 Menjalankan {script_name}...")
    try:
        result = subprocess.run(["python3", script_name], capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Gagal menjalankan {script_name}:")
        print(e.stderr)

def merge_all(output_file="datasets/dataset_final_18_april.jsonl"):
    files_to_merge = [
        "datasets/dataset_theory.jsonl",
        "datasets/ultimate_dataset.jsonl"
    ]
    
    final_data = []
    print("📂 Memulai proses penggabungan seluruh dataset di folder 'datasets'...")
    
    for file in files_to_merge:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                print(f"   ➕ Menambahkan {len(lines)} data dari: {file}")
                for line in lines:
                    try:
                        final_data.append(json.loads(line))
                    except:
                        continue
        else:
            print(f"   ⚠️ File tidak ditemukan: {file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in final_data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"\n✨ SELESAI! Total {len(final_data)} dataset siap digunakan di: {output_file}")

if __name__ == "__main__":
    # Create datasets folder if not exists
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
        
    # 1. Jalankan semua generator & konverter
    run_script("convert_md_to_jsonl.py")
    run_script("generate_dataset_all_sni.py")
    
    # 2. Gabungkan hasilnya
    merge_all()
