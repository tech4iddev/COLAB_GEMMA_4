#!/bin/bash

# Script: launch_structural_ai.sh
# Tujuan: Menjalankan Structural AI Chat Service pada Mac M4

echo "--------------------------------------------------"
echo "🚀 Memulai Structural AI Chat Assistant v2"
echo "--------------------------------------------------"

# 1. Navigasi ke root project (asumsi script ditaruh di root)
ROOT_DIR=$(pwd)
CHAT_DIR="$ROOT_DIR/chat_ui"
MODEL_FILE="gemma2-9b-structural-18april.Q4_K_M.gguf"

# 2. Cek apakah model ada
if [ ! -f "$ROOT_DIR/$MODEL_FILE" ]; then
    echo "⚠️  Model GGUF tidak ditemukan di root project!"
    echo "Pastikan file $MODEL_FILE sudah ditaruh di: $ROOT_DIR"
    exit 1
fi

# 3. Cek environment python
if [ -d "venv" ]; then
    echo "📦 Mengaktifkan Virtual Environment..."
    source venv/bin/activate
fi

# 4. Masuk ke folder chat_ui dan jalankan server
cd "$CHAT_DIR"
echo "🌐 Server akan berjalan di: http://localhost:3000"
echo "--------------------------------------------------"

python3 structural_server.py
