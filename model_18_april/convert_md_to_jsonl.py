import json
import glob
import os

print("\n[DEBUG] File: convert_md_to_jsonl.py | Update: 2026-04-18 23:35")

def convert_md_to_jsonl(input_dir="./sni_markdown", output_file="datasets/dataset_theory.jsonl"):
    """
    Mengonversi file Markdown hasil ekstraksi PDF menjadi format JSONL untuk training.
    """
    # Mencari semua file markdown di folder input
    md_files = glob.glob(os.path.join(input_dir, "**", "*.md"), recursive=True)
    
    if not md_files:
        print(f"❌ Tidak ditemukan file .md di {input_dir}.")
        return

    # Pastikan folder output ada
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created directory: {output_dir}")

    print(f"📄 Menemukan {len(md_files)} file materi. Memulai konversi...")
    
    dataset = []
    
    for md_path in md_files:
        filename = os.path.basename(md_path)
        # Identifikasi topik dari nama folder
        topic_path = os.path.dirname(md_path)
        topic = os.path.basename(topic_path).replace("material_", "").capitalize()
        
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Memecah dokumen berdasarkan paragraf (max 1200 karakter)
            # Dibuat sedikit bertumpuk (overlap) agar konteks tidak terputus
            chunk_size = 1200
            overlap = 200
            
            chunks = []
            for i in range(0, len(content), chunk_size - overlap):
                chunks.append(content[i:i + chunk_size])
            
            for chunk in chunks:
                if len(chunk.strip()) > 100:
                    dataset.append({
                        "instruction": f"Jelaskan informasi teknis mengenai {topic} berdasarkan dokumen {filename}.",
                        "input": "Berikan kutipan atau penjelasan dari standar regulasi terkait.",
                        "output": chunk.strip()
                    })

    # Simpan sebagai JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✨ BERHASIL! {len(dataset)} data teori tersimpan di {output_file}")

if __name__ == "__main__":
    # Default folder adalah folder sni_markdown di root proyek
    convert_md_to_jsonl()
