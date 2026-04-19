from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
import uvicorn
import os
import json
import asyncio

# Konfigurasi Path
MODEL_PATH = "mlx-community/gemma-4-e4b-4bit"
ADAPTER_PATH = "/Users/caturimamsaputro/Documents/GEMMA_4_unsloth/gemma4_final_structural"

print("🚀 Sedang memuat Gemma 4 + Adapter Structural di M4 GPU...")
try:
    model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
    print("✅ Model berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    exit(1)

app = FastAPI()

# Pengaturan CORS agar Browser bisa mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get_ui():
    return FileResponse("chat_ui.html")

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(request: ChatRequest):
    async def event_generator():
        try:
            # Identity Grounding
            system_msg = "Kamu adalah Gemma 4 AI spesialis Structural Engineering (SNI 1729:2020 & OpenSeesPy). Jawablah dengan akurat."
            formatted_prompt = f"<start_of_turn>user\n{system_msg}\n\n{request.prompt}<end_of_turn>\n<start_of_turn>model\n"
            
            sampler = make_sampler(temp=0.3)
            
            # Mode Streaming - Kirim token demi token
            for response in stream_generate(model, tokenizer, prompt=formatted_prompt, max_tokens=1024, sampler=sampler):
                yield response.text
                await asyncio.sleep(0.005) # Delay sangat kecil agar terasa smooth
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(event_generator(), media_type="text/plain")

if __name__ == "__main__":
    print("-" * 30)
    print("WEBSITE CHAT GEMMA 4 AKTIF (STREAMING MODE)")
    print("Akses: http://localhost:8000")
    print("-" * 30)
    uvicorn.run(app, host="0.0.0.0", port=8000)
