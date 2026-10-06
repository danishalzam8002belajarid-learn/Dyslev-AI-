# Dyslexia Analysis AI

Aplikasi web berbasis Streamlit untuk menganalisis potensi disleksia dari gambar (khususnya tulisan tangan) menggunakan model AI **EfficientNet-B0** (TensorFlow & TFLite Float32) dengan visualisasi **Grad-CAM**.

## 1. Persiapan Dataset

Sebelum melatih model, pastikan Anda telah menyiapkan dataset gambar tulisan tangan dengan struktur folder berikut:

```
data/
  └── train/
        ├── Disleksia/  (Berisi gambar tulisan tangan dengan indikasi disleksia)
        └── Normal/     (Berisi gambar tulisan tangan normal)
```

**Penting:** Nama folder kelas (`Disleksia` dan `Normal`) akan otomatis dikenali dan diurutkan secara alfabetis. `Disleksia` (Indeks 0) dan `Normal` (Indeks 1). Script `app.py` sudah disesuaikan dengan urutan alfabetis ini.

## 2. Instalasi Dependensi

Jalankan perintah berikut untuk menginstal semua library yang dibutuhkan:
```bash
pip install -r requirements.txt
```

## 3. Melatih Model

Jalankan script berikut untuk memulai proses training dari awal dan mengonversi model ke format `.tflite` (Float32):
```bash
python train_and_convert.py
```
Setelah selesai, dua file model akan terbuat:
1. `dyslexia_model.h5` (Digunakan untuk ekstraksi fitur Grad-CAM)
2. `dyslexia_model.tflite` (Digunakan untuk inferensi cepat/prediksi probabilitas)

## 4. Menjalankan Aplikasi Web Secara Lokal

Setelah model berhasil dilatih dan file `.tflite` serta `.h5` telah dibuat, Anda bisa menjalankan aplikasi Streamlit:
```bash
streamlit run app.py
```

## 5. Deployment Gratis dengan Hugging Face Spaces (Direkomendasikan)

Hugging Face Spaces sangat direkomendasikan karena memberikan kapasitas RAM gratis sebesar ~16GB, yang cukup untuk menampung Keras Model `.h5` saat menjalankan fungsi Grad-CAM.

### Langkah-langkah deploy ke Hugging Face Spaces:
1. Buat akun di [Hugging Face](https://huggingface.co/).
2. Buat "Space" baru, pilih **Streamlit** sebagai Space SDK.
3. Clone repositori space yang baru dibuat ke komputer Anda.
4. Salin file-file berikut ke folder repositori tersebut:
   - `app.py`
   - `grad_cam.py`
   - `requirements.txt`
   - `dyslexia_model.tflite`
   - `dyslexia_model.h5`
5. Push (commit) file-file tersebut kembali ke Hugging Face.
6. Aplikasi akan otomatis ter-build dan berjalan di internet secara gratis!
