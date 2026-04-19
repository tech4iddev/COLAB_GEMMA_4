# Panduan Training Structural AI (Gemma 2 9B) di Google Colab

Dokumen ini berisi langkah-langkah lengkap untuk menjalankan training menggunakan spesifikasi **L4 GPU** di Google Colab.

---

## 1. Persiapan Environment
Pilih Runtime **L4 GPU** di menu `Runtime > Change runtime type`. Kemudian jalankan blok kode berikut untuk mount Drive dan install dependencies.

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone atau Update Repository
Jika Anda baru pertama kali, gunakan **Clone**. Jika sudah ada folder lama, gunakan **Update (Pull)** agar mendapatkan script terbaru.

```python
import os

# 1. Clone jika belum ada
if not os.path.exists("/content/structural_ai"):
    !git clone https://github.com/tech4iddev/COLAB_GEMMA_4.git /content/structural_ai
    print("✅ Repository berhasil di-clone.")
else:
    # 2. Pull jika sudah ada untuk mendapatkan update terbaru
    %cd /content/structural_ai
    !git pull origin main
    print("✅ Repository berhasil di-update (Pull).")

%cd /content/structural_ai/model_18_april
```

# 3. Install Dependencies (PENTING: Versi Terkunci)
Jalankan ini secara spesifik untuk menghindari bug "Bisu" pada Gemma 2.

```python
# 1. Uninstall versi default yang mungkin bermasalah
!pip uninstall -y unsloth transformers trl peft accelerate xformers bitsandbytes

# 2. Install Versi Stabil (Golden Version 2026)
!pip install --no-deps "xformers<0.0.35" "trl<0.13.0" peft accelerate bitsandbytes
!pip install --upgrade "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --upgrade "transformers==4.51.3" # <--- JANGAN DIGANTI
```
**WAJIB**: Setelah instalasi selesai, klik menu **Runtime ➡️ Restart Session**.
```

---

## 2. Sinkronisasi Data & Dataset
Jalankan master pipeline untuk memastikan seluruh data SNI terbaru (Markdown) dan data sintetis (Hitungan) sudah tergabung menjadi satu.

```python
# Menjalankan pipeline pembuatan dataset (Theory + Analysis)
!python master_pipeline.py
```

Setelah dijalankan, file final akan berada di `datasets/dataset_final_18_april.jsonl`.

---

## 3. Proses Training & Export GGUF Otomatis
Jalankan script training berikut. Script ini sudah saya rancang untuk melakukan **seluruh proses** secara berurutan:
1.  **Training**: Fins-tuning Gemma 2 9B.
2.  **Saving**: Menyimpan model ke Google Drive.
3.  **GGUF Export**: Mengonversi model ke format GGUF (Q4_K_M) untuk Mac M4.

```python
# Jalankan All-in-One Training & Export Pipeline
!python train_colab_L4.py
```

---

## 4. Lokasi Hasil di Google Drive
Setelah script selesai (All Done), Anda dapat mengambil hasilnya di:
*   **GGUF (Untuk Mac)**: `My Drive / Structural_AI_Project / GGUF_MODELS /`
*   **Raw Model (16-bit)**: `My Drive / Structural_AI_Project / gemma2-9b-structural-18april`

---
## 5. Tips Anti-Gagal (Golden Rules)
1.  **Stop "Bisu" (`<eos>`)**: Gunakan **Logika Testing 2-Tahap** (Teks ➡️ Tokenizer) agar `attention_mask` terkirim. Kode aman:
    ```python
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    ```
2.  **Stop "Ngawur" (`myself`)**: Jika model mulai mengulang kata aneh, berarti **Overfitting**. Gunakan Learning Rate rendah (**5e-5**) dan batasi di **Step 600**.
3.  **Bursa Transfer Versi**: Jika Python 3.12 Colab update, pastikan tetap mengunci `transformers==4.51.3`.
4.  **Hardware**: Prioritaskan **L4 GPU**. T4 GPU adalah alternatif paling stabil.

---
*Dibuat untuk Milestone Model 18 April - Structural Engineering AI (Sync: 09:30)*



---

## Tips Training:
*   **Waktu Training**: Dengan 12k data dan batch size 4 di L4, training 120 steps biasanya memakan waktu sekitar 15-25 menit.
*   **Loss Monitoring**: Perhatikan nilai `Loss`. Jika loss masih turun tajam di akhir step 120, Anda bisa mengedit `train_colab_L4.py` dan menaikkan `max_steps` ke 200.
*   **VRAM**: Jika mendapatkan error `Out of Memory`, turunkan `per_device_train_batch_size` menjadi 2 di file `train_colab_L4.py`.

---
*Dibuat untuk Milestone Model 18 April - Structural Engineering AI*
