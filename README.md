# Dyslexia Analysis AI

Aplikasi web berbasis Streamlit dengan dua alur analisis yang berbeda:

- **Pengaturan analisis** tersedia melalui tombol popover yang tetap menempel di pojok kanan atas saat halaman digulir. Pengaturan menyediakan EfficientNet saja (klasifikasi gambar utuh), OpenCV saja (deteksi/crop komponen lalu EfficientNet mengklasifikasikan tiap crop), atau keduanya, serta threshold dan luas minimum komponen. Pada mode keduanya, ringkasan perbandingan dan unduhan gabungan ditampilkan lebih dahulu, diikuti bagian hasil EfficientNet dan OpenCV secara terpisah.
- **EfficientNet-B0** (TensorFlow & TFLite Float32) mengklasifikasikan keseluruhan gambar sebagai `Disleksia` atau `Normal`, dan dapat menampilkan visualisasi **Grad-CAM**. Hasilnya adalah skrining berbasis gambar, bukan diagnosis klinis.
- **Eksperimen crop OpenCV + EfficientNet** menggunakan komponen tinta OpenCV sebagai crop, lalu mengklasifikasikan setiap crop dengan model EfficientNet-B0 yang sama. Output utamanya satu ringkasan per gambar: persentase crop berlabel Disleksia dan Normal dari seluruh crop yang berhasil dianalisis. Persentase adalah proporsi komponen yang diberi label oleh model, bukan peluang diagnosis. Rincian prediksi per crop tersedia terpisah. Model EfficientNet saat ini dilatih untuk gambar tulisan utuh, bukan crop per huruf; jadi prediksi crop bersifat eksploratif dan belum tervalidasi. Crop komponen tinta belum tentu sama dengan satu huruf.
- **Skala Pengukuran Huruf** menggunakan OpenCV untuk mengelompokkan komponen tinta secara perkiraan menurut garis dasar vertikal dan mengukur ukurannya dalam piksel. Tabel OpenCV terpisah dari tabel prediksi EfficientNet dan menampilkan CV tinggi, CV lebar, serta indeks variasi ukuran per baris. Rinciannya mencantumkan nomor komponen, nomor baris, posisi, lebar, tinggi, luas tinta, dan rasio tinggi/lebar.

Gambar komponen terdeteksi selalu ditampilkan pada hasil OpenCV. Kontrol **Skala Komponen & Variasi Huruf** menyembunyikan atau menampilkan tabel skala komponen, hasil pengukuran gambar, dan variasi per baris secara bersamaan; kontrol ini tertutup secara default. Tabel rincian prediksi crop tetap ditampilkan.

Alur **Analisis Tulisan Tangan** menerima satu atau beberapa gambar sekaligus melalui **Upload file**, atau satu foto melalui **Kamera langsung**. Foto kamera diproses dengan alur analisis yang sama seperti file yang diunggah; jika akses kamera browser tidak tersedia, foto dapat dipilih dari perangkat. Grid panel gabungan muncul sebelum tabel perbandingan. Tabel perbandingan menggabungkan semua gambar; setelahnya hasil ditampilkan per gambar dengan urutan **Model EfficientNet-B0** lalu **OpenCV Detection**. Tabel prediksi EfficientNet tidak memakai subjudul tambahan. OpenCV menampilkan ringkasan crop, gambar komponen, rincian prediksi per huruf, lalu kontrol **Skala Komponen & Variasi Huruf**. Saat dibuka, kontrol itu memuat tabel skala komponen, hasil pengukuran gambar, lalu variasi per baris. Kontrol opsional tertutup secara default. Unduhan CSV dan gambar dipisahkan per gambar; mode perbandingan menyediakan unduhan gabungan CSV dan arsip gambar kedua alur.

Pengukuran dan klasifikasi crop hanya eksperimen teknis awal. Persentase agregat crop dihitung dari jumlah crop dengan label Disleksia atau Normal dibagi jumlah seluruh crop yang berhasil diklasifikasikan. Itu adalah proporsi label model dalam gambar, bukan peluang dyslexia, skor risiko, atau diagnosis. EfficientNet saat ini dilatih untuk gambar tulisan utuh, sehingga perilakunya pada crop belum tervalidasi. Untuk setiap baris, `CV tinggi (%) = simpangan baku tinggi ÷ rata-rata tinggi × 100` dan `CV lebar (%) = simpangan baku lebar ÷ rata-rata lebar × 100`, menggunakan simpangan baku populasi. **Indeks variasi ukuran per baris** adalah rata-rata CV tinggi dan CV lebar. Baris dengan kurang dari dua komponen tidak memiliki nilai variasi. Angka CV adalah variasi ukuran, bukan kemungkinan atau deteksi dysgraphia. Ukuran piksel tidak sama dengan ukuran fisik tanpa kalibrasi skala. Pengelompokan baris memakai kedekatan posisi baseline komponen dan dapat keliru untuk titik huruf, ascender/descender, tulisan sambung, atau tata letak yang tidak rata. Ukuran komponen juga dipengaruhi skala/resolusi gambar, karakter huruf, pemisahan tinta, dan kualitas foto. Penggunaan sebagai alat skrining memerlukan segmentasi huruf yang tervalidasi, data berlabel dan representatif, serta validasi klinis dan ambang keputusan yang ditetapkan sebelumnya. Komponen kecil di bawah batas luas minimum dapat diabaikan.

## Dataset untuk Pengukuran Huruf

Dataset Kaggle **Dyslexia Handwriting Image Dataset** yang saat ini diekstrak ke folder `Dyslexia_Dataset` berisi gambar potongan huruf kecil di kelas `Normal`, `Reversal`, dan `Corrected`. Gambar tersebut dapat dipakai untuk eksperimen pengukuran crop huruf, tetapi bukan untuk melokalisasi huruf pada satu halaman tulisan. Kategori itu juga bukan label diagnosis dyslexia.

Untuk pengembangan eksperimen berikutnya:

- [NIST Special Database 19 (SD19)](https://www.nist.gov/srd/nist-special-database-19) menyediakan banyak crop karakter tulisan tangan berlabel dan formulir tulisan. Crop cocok untuk menguji pengukuran per-glyph, tetapi halaman formulirnya tidak menyediakan anotasi kotak/mask per huruf yang siap dipakai untuk melatih detektor. Halaman NIST tidak mencantumkan lisensi penggunaan spesifik; periksa ketentuan sebelum penggunaan ulang.
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) lebih menyerupai tulisan alami pada halaman, dengan anotasi teks/baris/kata dan informasi komponen. Informasi komponen tidak selalu sepadan dengan karakter, terutama pada tulisan sambung. Akses memerlukan registrasi dan penggunaannya dibatasi untuk riset non-komersial.
- Dataset di [Roboflow Universe](https://universe.roboflow.com/search?q=handwriting%20character) harus diperiksa satu per satu. Pastikan gambar merupakan halaman tulisan (bukan hanya crop huruf), ekspor benar-benar memiliki box/mask pada tiap huruf, dan lisensinya mengizinkan penggunaan.

Belum ada sumber terverifikasi di atas yang menyediakan dataset siap latih dengan kotak/mask per karakter pada halaman tulisan alami. Untuk tujuan itu, tahap berikutnya adalah memilih halaman tulisan yang lisensinya sesuai lalu memberi anotasi huruf secara manual, setidaknya pada subset untuk pelatihan dan evaluasi. Dataset-dataset tersebut tidak memberikan label diagnosis dyslexia/dysgraphia.

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
1. `dyslexia_model.h5` (Digunakan untuk Grad-CAM)
2. `dyslexia_model_v2.tflite` (Digunakan aplikasi untuk inferensi EfficientNet)

## 4. Menjalankan Aplikasi Web Secara Lokal

Setelah model berhasil dilatih dan file `.tflite` serta `.h5` telah dibuat, Anda bisa menjalankan aplikasi Streamlit:
```bash
streamlit run app.py
```

## 5. Deployment di Streamlit Community Cloud

Repository aplikasi menyertakan `app.py`, `requirements.txt`, file model TFLite dan Keras, logo, serta modul pengukuran yang diperlukan saat runtime.

1. Push perubahan ke repository GitHub.
2. Masuk ke [Streamlit Community Cloud](https://share.streamlit.io/) dengan akun GitHub yang memiliki akses ke repository.
3. Pilih **Create app**, lalu pilih repository, branch, dan `app.py` sebagai file utama.
4. Klik **Deploy** dan pantau log build jika instalasi dependensi atau startup gagal.

Untuk repository public, aplikasi biasanya public secara default. Repository private juga dapat digunakan jika akun GitHub terhubung dan memiliki akses; akses aplikasi dapat diatur dari pengaturan sharing Streamlit. Periksa [dokumentasi deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) dan [pengaturan akses aplikasi](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app). Batas sumber daya Community Cloud lebih rendah daripada beberapa platform lain, sehingga keberhasilan deploy bergantung pada ukuran dependensi/model dan batas runtime yang berlaku.
