# Persiapan Training di Google Colab (Folder: COLAB_GEMMA_4)

Berikut adalah urutan proses run/script yang perlu Anda jalankan di cell Google Colab secara berurutan sebelum memulai proses training atau generate dataset.

## 1. Mount Google Drive (Opsional tapi disarankan)
Langkah ini sangat berguna agar dataset, hasil generate, dan model hasil training otomatis tersimpan ke Drive Anda dan tidak hilang saat runtime Colab terputus atau direstart.

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Clone atau Pull Repository Git (COLAB_GEMMA_4)
Jika Anda bekerja langsung di storage sementara Colab, jalankan `git clone`. Jika bekerja di dalam Google Drive, Anda bisa masuk ke direktori tersebut dan jalankan `git pull` untuk mendapatkan pembaruan terbaru.

**Jika Clone Pertama Kali:**
```bash
!git clone <URL_REPOSITORY_ANDA> /content/COLAB_GEMMA_4
%cd /content/COLAB_GEMMA_4
```

**Jika Folder Sudah Ada (Update Code dengan Git Pull):**
```bash
%cd /content/COLAB_GEMMA_4
!git pull origin main
```
*(Sesuaikan path folder jika Anda menaruhnya di dalam Google Drive, misal: `%cd /content/drive/MyDrive/COLAB_GEMMA_4`)*

## 3. Install Dependencies (Requirements)
Install semua library Python yang dibutuhkan. Pastikan di dalam folder tersebut terdapat file `requirements.txt`.

```bash
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" peft accelerate bitsandbytes openai tqdm
!pip install --upgrade trl
```
> **Catatan Tambahan:** Unsloth adalah framework akselerasi khusus yang wajib di-install dengan metode repositori Git di atas agar langsung cocok dengan dukungan model Gemma di mesin Colab/GPU L4.

## 4. Login Hugging Face Hub
Login ke Hugging Face sangat wajib dilakukan jika Anda ingin menggunakan model base dari Hugging Face (seperti Gemma) karena model tersebut memerlukan akses (Gated Model), serta untuk mengunggah model hasil fine-tuning nantinya.

Pastikan Anda sudah mendapatkan **Write Access Token** dari menu *Settings > Access Tokens* di akun Hugging Face Anda.

**Menggunakan CLI Baru (Langsung memasukkan token):**
```bash
!hf auth login --token "MASUKKAN_TOKEN_HF_ANDA_DISINI"
```

**Atau Menggunakan Widget Login (Interaktif):**
```python
from huggingface_hub import notebook_login
notebook_login()
```

## 5. Login Weights & Biases (W&B) - *Opsional*
Jika Anda menggunakan Weights & Biases (`wandb`) untuk tracking metrik training (loss, learning rate, dll), lakukan login.

```bash
!pip install wandb
!wandb login "MASUKKAN_TOKEN_WANDB_ANDA_DISINI"
```

## 6. Verifikasi Penggunaan GPU
Sebelum memulai training, sangat penting untuk memeriksa apakah sesi Colab Anda sudah mendapatkan alokasi GPU dari Google.

```bash
!nvidia-smi
```

---

## 7. Eksekusi Script Utama
Setelah persiapan selesai, Anda bisa melanjutkan ke proses utama yaitu generate dataset dan training.

**Contoh Urutan:**
```bash
# 1. Menjalankan script ekstraksi PDF (jika belum)
!python extract_colab.py

# 2. Menjalankan script QA Generator cerdas (OpenAI API) untuk bikin Dataset Pelatihan
!python generate_qa_dataset.py

# 3. Menjalankan script untuk training model menggunakan Unsloth (setelah dataset siap)
!python train_colab.py
```
