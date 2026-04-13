import subprocess
import os

def start_training():
    print("🚀 Memulai Fine-tuning Gemma 4 E4B dengan QLoRA di Mac Mini M4...")
    
    # Memastikan folder adapters ada
    if not os.path.exists("adapters"):
        os.makedirs("adapters")

    # Perintah MLX-LM untuk melakukan training
    # Ini akan secara otomatis melakukan 4-bit quantization dan LoRA
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--config", "config.yaml"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Terjadi kesalahan saat training: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Training dihentikan oleh pengguna.")

if __name__ == "__main__":
    # Memastikan menggunakan venv
    start_training()
