# Panduan Training Structural AI (Gemma 2 9B) di Google Colab

Dokumen ini berisi langkah-langkah lengkap untuk menjalankan training menggunakan spesifikasi **L4 GPU** di Google Colab.

---

## 1. Persiapan Environment
Pilih Runtime **L4 GPU** di menu `Runtime > Change runtime type`. Kemudian jalankan blok kode berikut untuk mount Drive dan install dependencies.

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone Repository
# Ganti URL_REPO dengan URL repository GitHub Anda
!git clone https://github.com/tech4iddev/COLAB_GEMMA_4.git /content/structural_ai
%cd /content/structural_ai/model_18_april

# 3. Install Dependencies
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install unsloth_zoo
!pip install --no-deps xformers trl peft accelerate bitsandbytes
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

## 3. Proses Training (Sinkronisasi Real-time ke Drive)
Jalankan script training. Perhatikan bahwa script ini secara otomatis akan menyimpan checkpoint setiap 30 step langsung ke Google Drive Anda untuk mencegah kehilangan data jika koneksi terputus.

```python
# Memulai proses training (Simpan otomatis ke MyDrive/Structural_AI_Project)
!python train_colab_L4.py
```

---

## 4. Lokasi Hasil Training
Setelah selesai, hasil model dan log training dapat Anda temukan di Google Drive pada folder:
`My Drive / Structural_AI_Project / gemma2-9b-structural-18april`

---
## Tips Penting:
*   **WAJIB MOUNT DRIVE**: Jika Drive tidak di-mount, training tidak akan dimulai karena script tidak bisa menemukan lokasi penyimpanan permanen.
*   **Waktu Training**: Estimasi 20-30 menit untuk 120 steps pada GPU L4.


---

## Tips Training:
*   **Waktu Training**: Dengan 12k data dan batch size 4 di L4, training 120 steps biasanya memakan waktu sekitar 15-25 menit.
*   **Loss Monitoring**: Perhatikan nilai `Loss`. Jika loss masih turun tajam di akhir step 120, Anda bisa mengedit `train_colab_L4.py` dan menaikkan `max_steps` ke 200.
*   **VRAM**: Jika mendapatkan error `Out of Memory`, turunkan `per_device_train_batch_size` menjadi 2 di file `train_colab_L4.py`.

---
*Dibuat untuk Milestone Model 18 April - Structural Engineering AI*
