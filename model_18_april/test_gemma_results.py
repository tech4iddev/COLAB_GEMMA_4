# --- LOG UPDATE ---
# Tanggal: 2026-04-19 12:51
# Update: Script testing hasil training Gemma 2 9B untuk verifikasi perbaikan halusinasi tabel SNI.
# ------------------
import torch
from unsloth import FastLanguageModel

# --- KONFIGURASI ---
# Ganti dengan path folder hasil training Anda
MODEL_PATH = "/content/gemma2-9b-structural-theory-only"

print(f"🚀 Memuat Model untuk Testing: {MODEL_PATH}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

def test_model(prompt):
    style_instruction = "Jawablah dengan runtut, detail, dan sebutkan referensi pasal SNI yang relevan jika tersedia."
    
    # Format Gemma 2
    text = f"<start_of_turn>user\n{style_instruction}\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024, 
        repetition_penalty=1.2, # Penalti agar tidak looping
        temperature=0.3, 
        do_sample=True,
        use_cache=True
    )
    result = tokenizer.batch_decode(outputs)[0]
    
    # Ambil hanya bagian jawaban model
    final_output = result.split("<start_of_turn>model\n")[-1].replace("<end_of_turn>", "").replace("<bos>", "").strip()
    return final_output

# --- DAFTAR TEST CASE ---
test_cases = [
    "Apa yang dimaksud dengan faktor reduksi kekuatan (phi) dalam SNI 2847?",
    "Jelaskan persyaratan spasi tulangan sengkang pada daerah tumpuan balok.",
    "Bagaimana cara menghitung beban angin untuk bangunan tertutup?"
]

print("\n" + "="*50)
print("🧪 MEMULAI TESTING MODEL GEMMA...")
print("="*50)

for i, q in enumerate(test_cases):
    print(f"\n[Test Case {i+1}]: {q}")
    print("-" * 30)
    response = test_model(q)
    print(response)
    print("\n" + "="*50)
