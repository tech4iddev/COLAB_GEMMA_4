import json
import glob
import os

def convert_md_to_jsonl():
    input_dir = "sni_markdown"
    output_file = "training_data/dataset_sni_theory.jsonl"
    
    # Mencari semua file markdown di folder sni_markdown
    md_files = glob.glob(os.path.join(input_dir, "**", "*.md"), recursive=True)
    
    if not md_files:
        print(f"❌ Tidak ditemukan file .md di {input_dir}. Pastikan ekstraksi sudah selesai.")
        return

    print(f"📄 Menemukan {len(md_files)} file materi SNI. Memulai konversi...")
    
    dataset = []
    
    for md_path in md_files:
        filename = os.path.basename(md_path)
        # Ambil topik dari folder materialnya
        topic = os.path.basename(os.path.dirname(md_path)).replace("material_", "").capitalize()
        
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Memecah dokumen berdasarkan paragraf agar tidak terlalu panjang (max 1000 karakter per chunk)
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            
            for chunk in chunks:
                if len(chunk.strip()) > 50: # Abaikan potongan teks yang terlalu pendek/kosong
                    dataset.append({
                        "instruction": f"Informasi teknis mengenai SNI {topic} ({filename}).",
                        "input": "Berikan penjelasan mengenai materi yang terkandung dalam dokumen ini.",
                        "output": chunk.strip()
                    })

    # Pastikan folder training_data ada
    if not os.path.exists("training_data"):
        os.makedirs("training_data")

    # Simpan sebagai JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✨ BERHASIL! {len(dataset)} materi teori SNI telah siap di {output_file}")

if __name__ == "__main__":
    convert_md_to_jsonl()
