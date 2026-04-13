# Roadmap Proyek: AI Structural Engineering Assistant (Gemma 4)

Dokumen ini berisi urutan proses dari awal hingga Anda memiliki asisten AI yang ahli dalam Teori, Hitungan, dan Analisa Konstruksi (SNI).

---

## 🏗️ FASE 1: Persiapan Data (SELESAI)
Pada fase ini kita menyiapkan "Bahan Baku" untuk otak AI.
1.  **Struktur Folder:** Mengelompokkan PDF ke material (Beton, Baja, Geoteknik, Kayu).
2.  **Database Material:** Membuat JSON profil baja Indonesia dan AHSP PUPR 2022.
3.  **Data Sintetis:** Membuat 500+ pasang QA perhitungan beton untuk logika matematika AI.
4.  **Koneksi:** Sinkronisasi seluruh file ke GitHub & Google Colab.

## 📄 FASE 2: Ekstraksi & Digitalisasi (SEDANG BERJALAN)
Mengubah dokumen PDF yang sulit dibaca menjadi teks Markdown yang bersih.
1.  **Running `extract_colab.py`:** Berjalan di Google Colab.
2.  **Pembersihan Data:** (Opsional) Menghapus teks yang tidak terbaca dari hasil scan.
3.  **Merge Datasets:** Menyatukan data SNI (Teori) + Data Hitungan + Data Material menjadi satu file: `dataset_final.jsonl`.

## 🧠 FASE 3: Fine-Tuning / Training (LANGKAH BERIKUTNYA)
Mengajarkan dataset ke Model Gemma 4 menggunakan mesin Unsloth di Colab.
1.  **Load Model:** Mengambil Base Model Gemma 2 9B (via Unsloth).
2.  **Training Process:** Menjalankan `train_colab.py` (Lama proses: 15-45 menit tergantung jumlah data).
3.  **Save LoRA Adapters:** Menyimpan "kecerdasan baru" AI ke folder `gemma4_lora_model`.

## 🎯 FASE 4: Hasil Akhir & Implementasi
Menguji apakah AI sudah benar-benar pintar.
1.  **Chat UI:** Menjalankan `app.py` atau mencoba notebook testing.
2.  **Uji Coba 3 Pilar:**
    *   **Teori:** Tanya "Apa syarat minimum tulangan menurut SNI 2847?"
    *   **Hitungan:** Berikan parameter dimensi balok dan minta hitung kapasitasnya.
    *   **Analisa:** Berikan kasus kerusakan struktur dan minta rekomendasi teknis.
3.  **Deployment:** Model siap digunakan di laptop Anda atau Cloud.

---

### Perintah Penting yang Harus Diingat di Colab:
```bash
# Gabungkan semua data sebelum training
!python merge_datasets.py

# Mulai Mengajar AI
!python train_colab.py
```

---
*Dibuat oleh Antigravity untuk Proyek Structural Engineering AI*
