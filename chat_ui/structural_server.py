import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
from fastapi.responses import FileResponse, StreamingResponse
import json

app = FastAPI(title="Structural AI Assistant v2")

# Aktifkan CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- KONFIGURASI MODEL TERBARU ---
# Pastikan file .gguf sudah Anda download dan taruh di folder yang sama dengan script ini
# Atau sesuaikan path-nya jika ditaruh di tempat lain
MODEL_PATH = "../gemma2-9b-structural-18april.Q4_K_M.gguf"

llm = None

def load_model():
    global llm
    if os.path.exists(MODEL_PATH):
        print(f"🚀 Memuat Model Structural AI Terbaru: {MODEL_PATH}")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,     # Context window sesuai kemampuan Gemma 2
            n_threads=8,    # Optimal untuk Mac M4
            n_gpu_layers=-1 # Full GPU Acceleration (Metal)
        )
    else:
        # Cek juga di folder root project
        alt_path = "gemma2-9b-structural-18april.Q4_K_M.gguf"
        if os.path.exists(alt_path):
            print(f"🚀 Memuat Model (Local Path): {alt_path}")
            llm = Llama(model_path=alt_path, n_ctx=4096, n_threads=8, n_gpu_layers=-1)
        else:
            print(f"⚠️ Model {MODEL_PATH} tidak ditemukan.")
            print("Silakan pindahkan hasil download GGUF ke folder root project.")

class ChatRequest(BaseModel):
    messages: list

@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.post("/chat")
async def chat(request: ChatRequest):
    global llm
    if llm is None: load_model()
    if llm is None: return {"response": "Model belum siap. Harap cek path file GGUF."}

    def generate():
        system_instruction = (
            "Anda adalah Structural AI Engineer Expert. "
            "JANGAN PERNAH langsung memberikan 'Hasil Analisa'. "
            "Anda WAJIB memberikan penjabaran mendetail sebelum memberikan hasil akhir dengan urutan tetap:\n"
            "1. Data & Parameter: Identifikasi variabel yang diketahui.\n"
            "2. Referensi SNI: Sebutkan nomor pasal atau tabel standar yang digunakan.\n"
            "3. Langkah Analisis: Tampilkan rumus (LaTeX) dan proses logika secara bertahap.\n"
            "4. Kesimpulan: Berikan hasil akhir dan saran teknis.\n"
            "Jawablah dengan Bahasa Indonesia profesional."
        )
        
        # Biarkan library menangani <bos> secara otomatis
        full_prompt = ""
        for i, msg in enumerate(request.messages):
            role = "user" if msg['role'] == 'user' else "model"
            content = msg['content']
            if i == 0 and role == "user":
                full_prompt += f"<start_of_turn>user\n{system_instruction}\n\n{content}<end_of_turn>\n"
            else:
                full_prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
        
        full_prompt += "<start_of_turn>model\n"

        try:
            stream = llm(
                full_prompt,
                max_tokens=2048,
                temperature=0.3, # Naikkkan dikit agar tidak macet di rumus panjang
                top_p=0.9,
                repeat_penalty=1.2,
                stop=["<end_of_turn>", "<eos>", "<start_of_turn>"],
                stream=True
            )
            for chunk in stream:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        yield text
        except Exception as e:
            yield f"\n[Server Error: {str(e)}]"

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=3000)
