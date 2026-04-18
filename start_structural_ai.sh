#!/bin/bash

# --- KONFIGURASI ---
PORT=3000
DOMAIN="macmini.cemapp.com"
PROJECT_DIR="/Users/caturimamsaputro/Documents/GEMMA_4_unsloth"

echo "----------------------------------------------------"
echo "🚀 MEMULAI SYSTEM STRUCTURAL AI ASSISTANT (PUBLIC)"
echo "----------------------------------------------------"

# 1. Masuk ke folder proyek
cd $PROJECT_DIR

# 2. Membersihkan proses lama jika ada
echo "🧹 Membersihkan port $PORT..."
lsof -ti:$PORT | xargs kill -9 2>/dev/null

# 3. Menjalankan Backend AI
echo "🧠 Menjalankan Backend Gemma 2 (M4 Stack)..."
rm -f server.log

# Mengaktifkan Virtual Environment
if [ -d "venv" ]; then
    source venv/bin/activate
    python mac_server.py > server.log 2>&1 &
else
    python3 mac_server.py > server.log 2>&1 &
fi
SERVER_PID=$!

echo "⏳ Menunggu server inisialisasi model (bisa memakan waktu 10-30 detik)..."
MAX_RETRIES=60
COUNT=0
while ! lsof -i:$PORT > /dev/null; do
    sleep 1
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Server gagal dijalankan dalam waktu $MAX_RETRIES detik."
        cat server.log
        kill $SERVER_PID
        exit 1
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Proses server terhenti secara mendadak."
        cat server.log
        exit 1
    fi
    printf "."
done

echo ""
echo "✅ Server backend telah aktif di port $PORT!"
echo "----------------------------------------------------"
echo "🌐 Menghubungkan ke Public Domain: $DOMAIN..."
echo "----------------------------------------------------"
echo "✅ AI Anda aktif di: https://$DOMAIN"
echo "----------------------------------------------------"

# Menjalankan Named Tunnel 'macmini' yang sudah terdaftar
/opt/homebrew/bin/cloudflared tunnel run macmini

# Tutup server jika script dihentikan
trap "kill $SERVER_PID" EXIT
