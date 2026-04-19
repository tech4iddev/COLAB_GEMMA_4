import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import uvicorn
import os

app = FastAPI(title="Structural Engineering AI Assistant")

# Aktifkan CORS agar frontend bisa berkomunikasi dengan backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse

# Konfigurasi Path Model
MODEL_PATH = "gemma-2-9b-it.Q4_K_M.gguf"

@app.get("/")
async def read_root():
    # Sajikan file index.html saat mengakses http://localhost:8000
    return FileResponse("index.html")

# Inisialisasi Model (Akan dimuat saat server dijalankan)
llm = None

def load_model():
    global llm
    if os.path.exists(MODEL_PATH):
        print(f"🚀 Memuat model: {MODEL_PATH} pada Mac M4...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=8, # Sesuaikan dengan core Mac M4
            n_gpu_layers=-1 # Menggunakan GPU Apple Metal sepenuhnya
        )
    else:
        print(f"⚠️ Model {MODEL_PATH} belum ditemukan. Silakan download dari Colab.")

class ChatRequest(BaseModel):
    messages: list # Menerima list pesan untuk memori

from fastapi.responses import FileResponse, StreamingResponse
import json

@app.post("/chat")
async def chat(request: ChatRequest):
    global llm
    if llm is None:
        load_model()
    
    if llm is None:
        return {"response": "Model belum siap."}

    def generate():
        # Membangun prompt dari riwayat pesan (Gemma format)
        full_prompt = ""
        for msg in request.messages:
            role = "user" if msg['role'] == 'user' else "model"
            full_prompt += f"<start_of_turn>{role}\n{msg['content']}<end_of_turn>\n"
        full_prompt += "<start_of_turn>model\n"

        stream = llm(
            full_prompt,
            max_tokens=2048,
            stop=["<end_of_turn>"],
            stream=True
        )
        for chunk in stream:
            text = chunk["choices"][0]["text"]
            if text:
                yield text

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    load_model()
    uvicorn.run(app, host="0.0.0.0", port=3000)
