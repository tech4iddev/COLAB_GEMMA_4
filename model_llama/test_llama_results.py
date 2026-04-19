# --- LOG UPDATE ---
# Tanggal: 2026-04-19 12:51
# Update: Script testing hasil training Llama 3.2 3B untuk verifikasi kualitas jawaban dan SNI.
# ------------------
import torch
from unsloth import FastLanguageModel

# --- KONFIGURASI ---
# Ganti dengan path folder hasil training Anda di GDrive
MODEL_PATH = "/content/drive/MyDrive/Structural_AI_Project/model_llama"

print(f"🚀 Memuat Model untuk Testing: {MODEL_PATH}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

def test_model(prompt):
    system_prompt = "Anda adalah Structural AI Engineer Expert. Jawablah dengan runtut, detail, dan sertakan referensi SNI."
    
    # Format Llama 3.2
    text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
    text += f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
    text += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
    result = tokenizer.batch_decode(outputs)[0]
    
    # Ambil hanya bagian jawaban assistant
    final_output = result.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[-1].replace("<|eot_id|>", "").replace("<bos>", "").strip()
    return final_output

# --- DAFTAR TEST CASE ---
test_cases = [
    "Jelaskan apa itu beban gempa nominal sesuai SNI 1726:2019?",
    "Bagaimana prosedur menentukan kategori risiko struktur gedung?",
    "Sebutkan kombinasi pembebanan untuk metode LRFD."
]

print("\n" + "="*50)
print("🧪 MEMULAI TESTING MODEL LLAMA...")
print("="*50)

for i, q in enumerate(test_cases):
    print(f"\n[Test Case {i+1}]: {q}")
    print("-" * 30)
    response = test_model(q)
    print(response)
    print("\n" + "="*50)
