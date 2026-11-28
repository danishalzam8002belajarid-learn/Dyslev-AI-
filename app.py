import base64
import hashlib
import io
import math
import os
import textwrap
import time
import zipfile
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from handwriting_measurement import (
    crop_connected_regions,
    measure_connected_regions,
    summarize_line_variability,
)

ENGLISH_TRANSLATIONS = {
    "DISCLAIMER PENELITIAN • ALAT BANTU SKRINING": "RESEARCH DISCLAIMER • SCREENING ASSISTANT",
    "Alat Bantu Deteksi Dini Disgrafia & Disleksia": "Early Screening Tool for Dysgraphia & Dyslexia",
    "Khusus Anak Usia 4 – 7 Tahun": "Designed for Children Aged 4 – 7 Years",
    "Berbasis Indikator Tulisan Tangan (EfficientNet-B0 & OpenCV)": "Based on Handwriting Indicators (EfficientNet-B0 & OpenCV)",
    "Aplikasi ini hanya berfungsi sebagai alat bantu skrining awal (prototipe penelitian) berdasarkan pola visual tulisan tangan, bukan alat diagnosis medis atau klinis final.": "This application serves strictly as an early screening assistant (research prototype) based on handwriting visual patterns, not a final medical or clinical diagnosis.",
    "Model dioptimalkan dan dievaluasi untuk karakteristik tulisan tangan anak pada rentang usia 4–7 tahun.": "The model is optimized and evaluated for handwriting characteristics of children aged 4–7 years.",
    "Hasil prediksi dan tingkat keyakinan (confidence) tetap perlu dikonsultasikan dengan psikolog, terapis okupasi, atau tenaga profesional terkait.": "Prediction results and confidence levels should always be consulted with a psychologist, occupational therapist, or relevant specialist.",
    "Saya Mengerti & Lanjutkan": "I Understand & Continue",
    "Sasaran Usia 4–7 Tahun • Alat Bantu Skrining Indikator Tulisan Tangan (Bukan Diagnosis Klinis)": "Target Age 4–7 Years • Handwriting Indicator Screening Assistant (Not a Clinical Diagnosis)",
    "Kesimpulan Akhir & Akumulasi Skor Hybrid (EfficientNet-B0 + OpenCV)": "Final Hybrid Conclusion & Accumulated Score (EfficientNet-B0 + OpenCV)",
    "Akumulasi Skor Hybrid menggabungkan kedua model secara tertimbang (70% Skor EfficientNet-B0 + 30% Skor OpenCV Crop) menjadi satu keputusan akhir dengan kalibrasi confidence.": "The Hybrid Score combines both models with weighted fusion (70% EfficientNet-B0 Score + 30% OpenCV Crop Score) into one final calibrated conclusion.",
    "Kesimpulan Hybrid (Gabungan 2 Model)": "Hybrid Conclusion (2-Model Fusion)",
    "Skor EfficientNet-B0 (%)": "EfficientNet-B0 Score (%)",
    "Skor OpenCV Crop (%)": "OpenCV Crop Score (%)",
    "Akumulasi Skor Hybrid (%)": "Hybrid Accumulated Score (%)",
    "Kesepakatan Model": "Model Agreement",
    "Kesimpulan Akhir Hybrid": "Final Hybrid Conclusion",
    "Confidence Hybrid": "Hybrid Confidence",
    "Sepakat (Kedua Model)": "Agreed (Both Models)",
    "Berbeda (Perlu Tinjauan)": "Divergent (Needs Review)",
    "Hanya EfficientNet-B0": "EfficientNet-B0 Only",
    "Hanya OpenCV Crop": "OpenCV Crop Only",
    "Lihat tabel kesimpulan hybrid seluruh sampel ({count} sampel)": "View full hybrid conclusion table ({count} samples)",
    "Unduh CSV": "Download CSV",
    "Unduh Gambar (PNG)": "Download Image (PNG)",
    "ZIP (Gambar PNG per tabel)": "ZIP (PNG Images per Table)",
    "Unduh ZIP (Gambar PNG)": "Download ZIP (PNG Images)",
    "Unduh ZIP (CSV)": "Download ZIP (CSV)",
    "Alat Deteksi Dysgraphia": "Dysgraphia Detection Tool",
    "Alat Deteksi Dysgraphia & Dyseleksia": "Dysgraphia & Dyslexia Detection Tool",
    "Model EfficientNet-B0 & OpenCV": "EfficientNet-B0 & OpenCV Models",
    "EfficientNet-B0 (Klasifikasi)": "EfficientNet-B0 (Classification)",
    "Aktif": "Active",
    "Digunakan": "In Use",
    "Tidak Digunakan": "Not Used",
    "Visualisasi Grad-CAM": "Grad-CAM Visualization",
    "Format output": "Output format",
    "Model .h5 tidak tersedia, heatmap dilewati.": "The .h5 model is unavailable; heatmap skipped.",
    "{filename} — EfficientNet: {label} ({confidence:.1f}%)": (
        "{filename} — EfficientNet: {label} ({confidence:.1f}%)"
    ),
    "{filename} — {count} komponen tinta terukur": (
        "{filename} — {count} measured ink components"
    ),
    "Memulai analisis...": "Starting analysis...",
    "Gagal menyiapkan hasil gambar: {filename}": "Failed to prepare result image: {filename}",
    "Gagal menyiapkan grid heatmap untuk diunduh.": "Failed to prepare the heatmap grid for download.",
    "Heatmap gagal untuk {filename}: {error}": "Heatmap failed for {filename}: {error}",
    "TensorFlow belum terpasang ({error})": "TensorFlow is not installed ({error})",
    "File .tflite EfficientNet tidak ditemukan": "EfficientNet .tflite file not found",
    "Gambar Asli": "Original Image",
    "Bagian": "Section",
    "Metode Prediksi": "Scoring Method",
    "Gambar utuh": "Full image",
    "Rata-rata tertimbang skor per bagian": "Pixel-area-weighted average of section scores",
    "Skor Klasifikasi per Bagian Grad-CAM": "Classification Scores per Grad-CAM Section",
    "Setiap bagian diklasifikasikan secara mandiri. Prediksi gambar pada tabel di atas adalah rata-rata tertimbang skor Disleksia dari seluruh bagian.": (
        "Each section is classified independently. The image prediction in the "
        "table above is the pixel-area-weighted average of section Dyslexia scores."
    ),
    "Grid with Heatmap": "Grid with Heatmap",
    "Heatmap + Gambar": "Heatmap + Image",
    "Heatmap Murni": "Heatmap Only",
    "Prediksi EfficientNet (gambar utuh)": "EfficientNet Prediction (full image)",
    "Confidence EfficientNet": "EfficientNet Confidence",
    "Jenis Analisis Model": "Analysis Model Type",
    "Status": "Status",
    "Model": "Model",
    "Komponen diprediksi Disleksia (%)": "Components Predicted as Dyslexia (%)",
    "Jumlah crop dianalisis": "Crops Analyzed",
    "Jumlah Komponen": "Component Count",
    "Jumlah Baris": "Line Count",
    "Jumlah komponen": "Component Count",
    "Nama File": "File Name",
    "File Name": "File Name",
    "Prediksi": "Prediction",
    "Tingkat Kepercayaan": "Confidence",
    "Level Confidence": "Confidence Level",
    "Raw Score (Disleksia)": "Raw Score (Dyslexia)",
    "Ukuran Gambar": "Image Dimensions",
    "Waktu Analisis": "Analysis Time",
    "Komponen diprediksi Normal (%)": "Components Predicted as Normal (%)",
    "Komponen": "Component",
    "Baris": "Line",
    "Prediksi eksperimen": "Experimental Prediction",
    "Skor Disleksia (%)": "Dyslexia Score (%)",
    "Skor Normal (%)": "Normal Score (%)",
    "Skor kelas terpilih (%)": "Selected Class Score (%)",
    "Variasi tinggi (%)": "Height Variation (%)",
    "Variasi lebar (%)": "Width Variation (%)",
    "Indeks variasi ukuran (%)": "Size Variation Index (%)",
    "rasio tinggi/lebar": "Height/Width Ratio",
    "lebar (px)": "Width (px)",
    "tinggi (px)": "Height (px)",
    "luas tinta (px)": "Ink Area (px)",
    "Rata-rata indeks variasi ukuran (%)": "Average Size Variation Index (%)",
    "Rendah": "Low",
    "Tinggi": "High",
    "Data kurang": "Insufficient data",
    "Model tidak tersedia": "Model unavailable",
    "Grid panel hasil gabungan": "Combined Results Grid",
    "Grid heatmap dan panel hasil": "Heatmap and Results Panel Grid",
    "Gambar perbandingan": "Comparison images",
    "Panel gabungan hasil EfficientNet dan Grad-CAM": "Combined EfficientNet and Grad-CAM panels",
    "Unduh Tabel - {filename}": "Download Table - {filename}",
    "Unduh Tabel": "Download Table",
    "Unduh Gambar": "Download Image",
    "Unduh Grid Panel Gabungan": "Download Combined Results Grid",
    "Unduh ZIP": "Download ZIP",
    "Unduh Data Pilihan": "Download Selected Data",
    "Pilih tabel dan format": "Choose tables and format",
    "Cakupan data": "Data scope",
    "Semua gambar": "All images",
    "Cakupan ini berlaku untuk semua tabel yang dipilih.": (
        "This scope applies to all selected tables."
    ),
    "Grid panel gabungan selalu mencakup semua gambar.": (
        "The combined results grid always includes all images."
    ),
    "Format unduhan": "Download format",
    "Excel (beberapa sheet)": "Excel (multiple sheets)",
    "ZIP (CSV per tabel)": "ZIP (one CSV per table)",
    "Unduh Excel": "Download Excel",
    "Pilih minimal satu data atau grid panel untuk diunduh.": (
        "Select at least one dataset or results grid to download."
    ),
    "EfficientNet": "EfficientNet",
    "OpenCV": "OpenCV",
    "Perbandingan Model": "Model Comparison",
    "Keseluruhan": "Overall",
    "Disarankan: bagi otomatis semua gambar lebar untuk Grad-CAM": (
        "Recommended: automatically split all wide images for Grad-CAM"
    ),
    "Semua gambar lebar yang tercantum akan dibagi otomatis untuk Grad-CAM. Matikan centang ini jika ingin memprosesnya tanpa pembagian.": (
        "All listed wide images will be split automatically for Grad-CAM. "
        "Uncheck this to process them without splitting."
    ),
    "Gambar lebar yang tercantum direkomendasikan untuk dibagi otomatis dengan pengaturan di bawah.": (
        "The listed wide images are recommended for automatic splitting with "
        "the setting below."
    ),
    "Pembagian per gambar hanya tersedia pada Analisis Lengkap saat model Grad-CAM EfficientNet tersedia.": (
        "Per-image splitting is available only in Full Analysis when the "
        "EfficientNet Grad-CAM model is available."
    ),
    "Ringkasan EfficientNet": "EfficientNet Summary",
    "Skor per Bagian Grad-CAM": "Grad-CAM Section Scores",
    "Ringkasan Perbandingan": "Comparison Summary",
    "Pengukuran Gambar OpenCV": "OpenCV Image Measurements",
    "Ringkasan Crop OpenCV": "OpenCV Crop Summary",
    "Prediksi Komponen OpenCV": "OpenCV Component Predictions",
    "Skala Komponen": "Component Scale",
    "Variasi per Baris": "Variation by Line",
    "Komponen terdeteksi": "Detected Components",
    "Rincian prediksi per huruf": "Per-Component Prediction Details",
    "Skala Komponen & Variasi Huruf": "Component Scale & Letter Variation",
    "Hasil pengukuran gambar": "Image Measurement Results",
    "Variasi per baris": "Variation by Line",
    "Analisis Tulisan Tangan": "Handwriting Analysis",
    "Analisis pola gambar tulisan tangan tanpa ribet teknis. Cocok untuk guru, orang tua, hingga terapis. Dilengkapi berbagai fitur pendukung:": (
        "Analyze handwriting patterns without technical complexity. Suitable for "
        "teachers, parents, and therapists, with the following features:"
    ),
    "Ekspor Hasil CSV": "Export Results to CSV",
    "Eksperimen Crop OpenCV": "OpenCV Crop Experiment",
    "Eksperimen Crop OpenCV + EfficientNet": "OpenCV Crop Experiment + EfficientNet",
    "Eksperimen Crop OpenCV + EfficientNet.": "OpenCV Crop Experiment + EfficientNet.",
    "EfficientNet saja": "EfficientNet only",
    "OpenCV saja (crop + EfficientNet)": "OpenCV only (crop + EfficientNet)",
    "Keduanya": "Both",
    "Pengaturan": "Settings",
    "Atur model dan parameter analisis.": "Configure analysis models and parameters.",
    "Bahasa / Language": "Language / Bahasa",
    "Ganti bahasa / Change language": "Ganti bahasa / Change language",
    "Gambar berikut memiliki lebar lebih dari {threshold} piksel. Input model diubah menjadi 224 × 224 piksel, sehingga detail tulisan mungkin sulit terlihat:": (
        "The following images are wider than {threshold} pixels. Model input is "
        "resized to 224 × 224 pixels, so handwriting details may be difficult to see:"
    ),
    "Bagian {index}/{count} - {filename}": "Part {index}/{count} - {filename}",
    "Pengaturan Analisis": "Analysis Settings",
    "Pilih alur analisis": "Choose an analysis workflow",
    "OpenCV menggunakan EfficientNet untuk mengklasifikasikan setiap crop; mode ini tidak menjalankan klasifikasi gambar utuh.": (
        "OpenCV uses EfficientNet to classify each crop; this mode does not "
        "classify the full image."
    ),
    "OpenCV saja berarti deteksi/crop OpenCV lalu EfficientNet mengklasifikasikan crop; EfficientNet gambar utuh hanya berjalan pada mode yang menyertakannya.": (
        "OpenCV only runs component detection/cropping, then EfficientNet classifies "
        "the crops. Full-image EfficientNet classification runs only in workflows "
        "that include it."
    ),
    "Threshold Disleksia": "Dyslexia Threshold",
    "Gambar diprediksi Disleksia jika skor kelas Disleksia > {threshold:.2f}. Ambang ini bukan standar klinis.": (
        "An image is predicted as Dyslexia if its Dyslexia class score is > "
        "{threshold:.2f}. This threshold is not a clinical standard."
    ),
    "OpenCV: Pengukuran & Crop Eksperimental": "OpenCV: Experimental Measurement & Cropping",
    "Mengukur dan memotong komponen tinta; crop diproses dengan EfficientNet yang sama. Satu komponen belum tentu satu huruf.": (
        "Measures and crops ink components; crops are processed by the same "
        "EfficientNet model. A component may not represent a single letter."
    ),
    "Luas minimum komponen (piksel)": "Minimum component area (pixels)",
    "Komponen yang lebih kecil dari nilai ini akan diabaikan.": (
        "Components smaller than this value will be ignored."
    ),
    "Status Komponen": "Model Status",
    "Klasifikasi EfficientNet-B0": "EfficientNet-B0 Classification",
    "Mendukung Analisis Awal Disleksia Berbasis AI": "Supporting Preliminary AI-Based Dyslexia Analysis",
    "Diwujudkan melalui berbagai fitur unggulan untuk membantu analisis Anda. Hasil merupakan prototipe awal, bukan diagnosis.": (
        "Explore features designed to support your analysis. Results are from an "
        "early prototype and are not a diagnosis."
    ),
    "Menganalisis gambar tulisan tangan dan memprediksi kategori Disleksia atau Normal. Visualisasi Grad-CAM menunjukkan area gambar yang memengaruhi prediksi. Hasil merupakan prediksi prototipe, bukan bukti klinis atau diagnosis.": (
        "Analyzes handwriting images and predicts Dyslexia or Normal. Grad-CAM "
        "visualizes image regions that influenced the prediction. This prototype "
        "prediction is not clinical evidence or a diagnosis."
    ),
    "OpenCV memotong komponen tinta, lalu EfficientNet-B0 yang sama memberi skor kelas pada setiap crop. Crop belum tentu satu huruf dan hasilnya masih eksperimen, bukan diagnosis.": (
        "OpenCV crops ink components, then the same EfficientNet-B0 model scores "
        "each crop. A crop may not be a single letter; results are experimental, "
        "not a diagnosis."
    ),
    "Pemrosesan Banyak Gambar": "Batch Processing",
    "Mendukung pemrosesan puluhan gambar sekaligus dalam satu waktu, mempermudah pengujian data dan riset dalam skala besar secara instan.": (
        "Process dozens of images at once to streamline large-scale data testing "
        "and research."
    ),
    "Laporan & Ekspor Data": "Reports & Data Export",
    "Dapatkan hasil analisis lengkap beserta metrik raw score dan confidence level yang siap diekspor ke format CSV untuk kebutuhan riset lanjutan.": (
        "Get complete analysis results, including raw scores and confidence levels, "
        "ready to export as CSV for further research."
    ),
    "Eksperimen 1:** EfficientNet-B0 pada gambar utuh": "Experiment 1:** EfficientNet-B0 on the full image",
    "Eksperimen 2:** OpenCV crop → EfficientNet-B0 pada tiap komponen": "Experiment 2:** OpenCV crops → EfficientNet-B0 on each component",
    "**Eksperimen 1:** EfficientNet-B0 pada gambar utuh  \n**Eksperimen 2:** OpenCV crop → EfficientNet-B0 pada tiap komponen": (
        "**Experiment 1:** EfficientNet-B0 on the full image  \n"
        "**Experiment 2:** OpenCV crops → EfficientNet-B0 on each component"
    ),
    "Perjelas kontras tulisan": "Enhance handwriting contrast",
    "Memekatkan bagian gelap dan mencerahkan bagian terang dengan peregangan kontras, sambil mempertahankan tingkat abu-abu. Tidak membalik atau membinarisasi warna. File asli tidak diubah.": (
        "Darkens dark areas and brightens light areas using contrast stretching "
        "while preserving grayscale levels. Does not invert or binarize the image. "
        "The original file remains unchanged."
    ),
    "Sumber gambar": "Image source",
    "Upload file": "Upload files",
    "Kamera langsung": "Live camera",
    "Pilih satu atau beberapa gambar": "Select one or more images",
    "Pratinjau berbentuk potret. Jika kamera menampilkan lanskap, gambar akan tampak lebih kecil di tengah tanpa terpotong. Resolusi diminta hingga 1080p sesuai dukungan perangkat; aplikasi tidak mengubah ukuran atau mengompresi ulang foto. Streamlit menyimpan hasil kamera dalam JPEG kualitas maksimum.": (
        "The preview uses a portrait frame. Landscape camera feeds appear smaller "
        "in the center without cropping. Up to 1080p is requested, depending on "
        "device support. The app does not resize or recompress photos. Streamlit "
        "stores camera captures as maximum-quality JPEG."
    ),
    "Ambil Gambar": "Take Picture",
    "Foto kamera siap dianalisis: {count}": "Camera photos ready for analysis: {count}",
    "Lihat foto yang sudah diambil": "View captured photos",
    "Hapus semua foto kamera": "Remove all camera photos",
    "Atau tambahkan beberapa foto dari perangkat": "Or add multiple photos from your device",
    "Pilih satu atau beberapa foto jika kamera langsung tidak tersedia, atau untuk menambahkan foto lain.": (
        "Select one or more photos if the live camera is unavailable or to add "
        "more photos."
    ),
    "Analisis Lengkap": "Full Analysis",
    "Hanya Tabel Ringkasan": "Summary Table Only",
    "Mulai Analisis": "Start Analysis",
    "Hasil Perbandingan Model EfficientNet dan OpenCV": "EfficientNet and OpenCV Model Comparison",
    "Perbandingan ini menampilkan klasifikasi EfficientNet pada gambar utuh dan proporsi label Disleksia/Normal dari crop OpenCV yang diklasifikasikan EfficientNet. Persentase crop adalah proporsi label komponen, bukan kemungkinan diagnosis; segmentasi komponen belum tentu memotong huruf secara tepat.": (
        "This comparison shows EfficientNet classification of the full image and "
        "the proportion of Dyslexia/Normal labels among OpenCV crops classified "
        "by EfficientNet. Crop percentages are component-label proportions, not "
        "diagnostic probabilities; component segmentation may not isolate "
        "individual letters."
    ),
    "Perbandingan menampilkan prediksi EfficientNet per gambar dan proporsi label crop OpenCV. Untuk gambar lebar yang dibagi otomatis, prediksi EfficientNet adalah rata-rata tertimbang skor Disleksia tiap bagian. Persentase crop OpenCV adalah proporsi label, bukan kemungkinan diagnosis.": (
        "This comparison shows EfficientNet predictions per image and the "
        "proportion of OpenCV crop labels. For automatically split wide images, the "
        "EfficientNet prediction is the pixel-area-weighted average of section "
        "Dyslexia scores. OpenCV crop percentages are label proportions, not "
        "diagnostic probabilities."
    ),
    "Hasil gambar — {filename}": "Image results — {filename}",
    "Model EfficientNet-B0": "EfficientNet-B0 Model",
    "Deteksi OpenCV": "OpenCV Detection",
    "Model Crop Komponen + EfficientNet": "Component Crop Model + EfficientNet",
    "Persentase merupakan proporsi crop yang diprediksi Disleksia atau Normal pada gambar ini, bukan peluang diagnosis. Crop belum tentu satu huruf; model EfficientNet dilatih untuk gambar utuh sehingga hasil crop masih eksperimen.": (
        "Percentages are the proportions of crops predicted as Dyslexia or Normal "
        "in this image, not diagnostic probabilities. A crop may not be a single "
        "letter; EfficientNet was trained on full images, so crop results are "
        "experimental."
    ),
    "Jenis Analisis": "Analysis Model Type",
    "Inference Time (s)": "Inference Time (s)",
    "x (px)": "x (px)",
    "y (px)": "y (px)",
    "Disleksia": "Dyslexia",
    "EfficientNet tidak tersedia; crop tidak dapat diklasifikasikan.": (
        "EfficientNet is unavailable; crops cannot be classified."
    ),
    "Gagal memuat model": "Failed to load model",
    "Lanjut ke Analisis": "Continue to Analysis",
    "Informasi Analisis": "Analysis Information",
    "Aplikasi memiliki dua alur eksperimen: EfficientNet-B0 mengklasifikasikan gambar utuh secara default, atau bagian gambar lebar secara mandiri setelah gambar yang direkomendasikan dibagi otomatis. OpenCV memotong komponen tinta lalu mengirim setiap crop ke EfficientNet-B0 yang sama; OpenCV bukan model AI terpisah.": (
        "The app offers two experimental workflows: EfficientNet-B0 classifies "
        "the full image by default, or independently classifies sections of a "
        "wide image after automatically splitting the recommended images. OpenCV crops ink "
        "components and sends each crop to the same EfficientNet-B0 model; "
        "OpenCV is not a separate AI model."
    ),
    "- **EfficientNet saja:** klasifikasi gambar tulisan secara utuh. Untuk gambar lebar yang direkomendasikan untuk dibagi, tiap bagian diklasifikasikan mandiri dan skornya digabung dengan rata-rata tertimbang luas piksel.\n- **OpenCV saja:** deteksi komponen, crop, lalu klasifikasi crop dengan EfficientNet.\n- **Keduanya:** jalankan dua alur dan bandingkan hasilnya.\n- Skor pada crop bersifat eksperimen dan bukan peluang diagnosis; komponen tinta belum tentu satu huruf.": (
        "- **EfficientNet only:** classify the full handwriting image. For "
        "wide images recommended for splitting, classify each section "
        "independently and combine "
        "scores using a pixel-area-weighted average.\n"
        "- **OpenCV only:** detect components, crop them, then classify crops "
        "with EfficientNet.\n"
        "- **Both:** run and compare both workflows.\n"
        "- Crop scores are experimental, not diagnostic probabilities; an ink "
        "component may not be a single letter."
    ),
    "Muat Ulang Aplikasi": "Reload Application",
    "Pilih tindakan. Refresh data akan menghapus gambar yang diunggah, tetapi mempertahankan pengaturan analisis. Muat ulang akan mengembalikan aplikasi ke halaman awal.": (
        "Choose an action. Refreshing data removes uploaded and captured images "
        "but keeps analysis settings. Reloading returns the app to the home page."
    ),
    "Batal": "Cancel",
    "Refresh data aja": "Clear Images",
    "Muat Ulang": "Reload",
    "Kosongkan gambar yang diunggah atau mulai kembali dari awal.": (
        "Clear uploaded images or restart the app."
    ),
    "Selesai": "Complete",
    "Terindikasi": "Indicated",
    "Tidak tersedia": "Unavailable",
    "Memuat Aplikasi": "Loading Application",
    "Program Studi Statistika • Universitas Singaperbangsa Karawang": (
        "Statistics Study Program • Singaperbangsa Karawang University"
    ),
    "Anova Outlier 4 • Program Studi Statistika UNSIKA": (
        "Anova Outlier 4 • Statistics Study Program UNSIKA"
    ),
    "Dyslev AI — Dyslexia & Dysgraphia Detection AI": (
        "Dyslev AI — Dyslexia & Dysgraphia Detection AI"
    ),
    "EfficientNet-B0 (TFLite & Grad-CAM) | OpenCV Component Analysis | Fast & Efficient": (
        "EfficientNet-B0 (TFLite & Grad-CAM) | OpenCV Component Analysis | Fast & Efficient"
    ),
    "Disclaimer: Riset Mahasiswa yang menggunakan model EfficientNet-B0 (TFLite & Grad-CAM) dan OpenCV, digunakan untuk pengembangan dan eksperimen Program Studi Statistika UNSIKA.": (
        "Disclaimer: Student Research utilizing EfficientNet-B0 (TFLite & Grad-CAM) and OpenCV models, developed for academic development and experimentation in the Statistics Study Program, UNSIKA."
    ),
    "Diikutsertakan dalam kompetisi Malaysia Data Innovation Talent (MDIT) 2026 yang diselenggarakan oleh Universiti Teknologi MARA (UiTM) Malaysia.": (
        "Competed in the Malaysia Data Innovation Talent (MDIT) 2026 organized by Universiti Teknologi MARA (UiTM) Malaysia."
    ),
    "© 2026 Anova Outlier 4 — Program Studi Statistika, Universitas Singaperbangsa Karawang. All Rights Reserved.": (
        "© 2026 Anova Outlier 4 — Statistics Study Program, Singaperbangsa Karawang University. All Rights Reserved."
    ),
    "Advisor": "Advisor",
    "Team Members": "Team Members",
    "Dosen Pembimbing Riset": "Research Advisor",
    "Tim Peneliti Mahasiswa": "Student Research Team",
    "Program Studi Statistika, Universitas Singaperbangsa Karawang": (
        "Statistics Study Program, Singaperbangsa Karawang University"
    ),
    "Disclaimer Riset & Kompetisi": "Research & Competition Disclaimer",
    "Riset Mahasiswa ini menggunakan model Deep Learning EfficientNet-B0 (TensorFlow Lite & Keras Grad-CAM) untuk klasifikasi citra tulisan tangan serta algoritma Computer Vision OpenCV untuk pengukuran dan segmentasi komponen tinta. Sistem ini dikembangkan sebagai media pengembangan keilmuan dan eksperimen pada Program Studi Statistika, Universitas Singaperbangsa Karawang (UNSIKA), serta diikutsertakan dalam kompetisi Malaysia Data Innovation Talent (MDIT) 2026 yang diselenggarakan oleh Universiti Teknologi MARA (UiTM) Malaysia. Seluruh hasil analisis bersifat prototipe eksperimental dan bukan pengganti diagnosis klinis.": (
        "This Student Research project utilizes the EfficientNet-B0 Deep Learning model "
        "(TensorFlow Lite & Keras Grad-CAM) for handwriting image classification alongside "
        "OpenCV Computer Vision algorithms for ink component measurement and segmentation. "
        "This system is developed for academic development and experimentation within the "
        "Statistics Study Program, Singaperbangsa Karawang University (UNSIKA), and is "
        "competed in the Malaysia Data Innovation Talent (MDIT) 2026 organized by "
        "Universiti Teknologi MARA (UiTM) Malaysia. All analysis results are experimental "
        "prototypes and do not replace clinical diagnosis."
    ),
    "Pemindaian AI Aktif": "AI Scan Active",
    "Analisis Selesai": "Analysis Complete",
    "Gambar {current}/{total}": "Image {current}/{total}",
    "Sampel {current}/{total}": "Sample {current}/{total}",
    "Memproses sampel {current} dari {total} pada seluruh model aktif...": (
        "Processing sample {current} of {total} across all active models..."
    ),
    "Alur Aktif: EfficientNet-B0 & OpenCV": "Active Workflow: EfficientNet-B0 & OpenCV",
    "Alur Aktif: EfficientNet-B0 (Klasifikasi & Grad-CAM)": (
        "Active Workflow: EfficientNet-B0 (Classification & Grad-CAM)"
    ),
    "Alur Aktif: OpenCV Crop & EfficientNet-B0": (
        "Active Workflow: OpenCV Crop & EfficientNet-B0"
    ),
    "Daftar & Filter Sampel Hasil Analisis": "Analysis Sample List & Filter",
    "Menampilkan menu daftar sampel karena jumlah gambar lebih dari 10 ({count} sampel). Gunakan pencarian nama gambar dan filter kategori di bawah untuk melihat detail setiap sampel.": (
        "Showing the sample list menu because more than 10 images were analyzed ({count} samples). Use the filename search and category filter below to inspect each sample."
    ),
    "Total Sampel": "Total Samples",
    "Sampel Disleksia": "Dyslexia Samples",
    "Sampel Non-Disleksia": "Non-Dyslexia Samples",
    "Filter kategori sampel": "Filter sample category",
    "Semua Sampel": "All Samples",
    "Disleksia": "Dyslexia",
    "Non-Disleksia": "Non-Dyslexia",
    "Cari berdasarkan nama gambar": "Search by image filename",
    "Ketik nama gambar...": "Type image filename...",
    "Pilih sampel dari daftar ({count} hasil cocok)": (
        "Select a sample from the list ({count} matching results)"
    ),
    "Lihat tabel daftar sampel terfilter ({count} sampel)": (
        "View filtered sample list table ({count} samples)"
    ),
    "Lihat tabel ringkasan perbandingan seluruh sampel ({count} sampel)": (
        "View full comparison summary table ({count} samples)"
    ),
    "Unduh Tabel Terfilter": "Download Filtered Table",
    "Tidak ada sampel yang cocok dengan pencarian atau filter yang dipilih.": (
        "No samples match the selected search query or filter."
    ),
    "Kategori": "Category",
    "Ringkasan Skor": "Score Summary",
    "1. Prapemrosesan & Kontras": "1. Preprocessing & Contrast",
    "2. EfficientNet-B0 & Grad-CAM": "2. EfficientNet-B0 & Grad-CAM",
    "3. Deteksi & Crop OpenCV": "3. OpenCV Detection & Crops",
    "Menyiapkan citra dan menyelaraskan kontras tulisan...": (
        "Preparing image and standardizing handwriting contrast..."
    ),
    "Memuat model Grad-CAM untuk memetakan fokus jaringan saraf...": (
        "Loading Grad-CAM model to map neural network focus..."
    ),
    "Menjalankan inferensi EfficientNet-B0 & visualisasi Grad-CAM...": (
        "Running EfficientNet-B0 inference & Grad-CAM visualization..."
    ),
    "Memindai bagian {part}/{total_parts} untuk Grad-CAM resolusi tinggi...": (
        "Scanning section {part}/{total_parts} for high-resolution Grad-CAM..."
    ),
    "Mendeteksi komponen tinta & mengelompokkan baris tulisan...": (
        "Detecting ink components & grouping handwriting lines..."
    ),
    "Mengklasifikasikan komponen tinta {current}/{total} dengan EfficientNet-B0...": (
        "Classifying ink component {current}/{total} with EfficientNet-B0..."
    ),
    "Seluruh tahapan analisis gambar telah selesai diproses.": (
        "All image analysis stages have finished processing."
    ),
    "Info AI": "AI Insight",
    "Peregangan kontras (persentil 2–98%) mempertegas goresan pena tanpa menghilangkan gradasi abu-abu alami.": (
        "Contrast stretching (2nd–98th percentile) sharpens pen strokes while preserving natural grayscale gradients."
    ),
    "Grad-CAM memetakan aktivasi layer konvolusi terakhir untuk menampilkan area tulisan yang paling memengaruhi prediksi.": (
        "Grad-CAM maps the final convolutional layer activations to highlight handwriting regions that most influenced the prediction."
    ),
    "Gambar lebar (>448 px) dibagi menjadi beberapa segmen vertikal agar detail goresan tidak menyusut saat di-resize ke 224×224 px.": (
        "Wide images (>448 px) are split into vertical segments so stroke details are not lost when resized to 224×224 px."
    ),
    "OpenCV mengelompokkan komponen tinta per baris dan menghitung koefisien variasi (CV) tinggi serta lebar secara otomatis.": (
        "OpenCV groups ink components by baseline and automatically calculates height and width coefficients of variation (CV)."
    ),
}

st.session_state.setdefault("ui_language", "Indonesia")
UI_LANGUAGE = st.session_state["ui_language"]


def tr(text: str) -> str:
    if UI_LANGUAGE == "English":
        return ENGLISH_TRANSLATIONS.get(text, text)
    return text


def toggle_ui_language() -> None:
    st.session_state["ui_language"] = (
        "English" if st.session_state["ui_language"] == "Indonesia" else "Indonesia"
    )
    st.session_state["page_refresh_seq"] = (
        st.session_state.get("page_refresh_seq", 0) + 1
    )


def translate_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    if UI_LANGUAGE != "English":
        return frame
    translated = frame.rename(columns=lambda column: tr(str(column))).copy()
    return translated.apply(
        lambda column: column.map(
            lambda value: tr(value) if isinstance(value, str) else value
        )
    )


def localized_result_caption(output: dict[str, Any]) -> str:
    if "caption_data" not in output:
        return tr(output.get("caption", ""))
    if output["analysis"] == MODEL_EFFNET:
        return tr("{filename} — EfficientNet: {label} ({confidence:.1f}%)").format(
            filename=output["filename"],
            label=tr(output["caption_data"]["label"]),
            confidence=output["caption_data"]["confidence"] * 100,
        )
    return tr("{filename} — {count} komponen tinta terukur").format(
        filename=output["filename"],
        count=output["caption_data"]["component_count"],
    )


def _get_uploaded_file_meta(uploaded_file: Any) -> tuple[int, int, str, str]:
    cache: dict[Any, tuple[int, int, str, str]] = st.session_state.setdefault(
        "_uploaded_file_meta_cache", {}
    )
    raw_bytes = uploaded_file.getvalue()
    cache_key = (
        uploaded_file.name,
        len(raw_bytes),
        getattr(uploaded_file, "file_id", None),
        raw_bytes[:64],
        raw_bytes[-64:],
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    fingerprint = hashlib.sha256(
        f"{uploaded_file.name}:{content_hash}".encode("utf-8")
    ).hexdigest()[:16]
    with Image.open(io.BytesIO(raw_bytes)) as image:
        width, height = image.size
    meta = (width, height, content_hash, fingerprint)
    cache[cache_key] = meta
    return meta


def uploaded_image_fingerprint(uploaded_file: Any) -> str:
    return _get_uploaded_file_meta(uploaded_file)[3]


def images_over_width(
    files: list[Any],
    max_width: int,
) -> list[tuple[int, str, int, int, str]]:
    oversized = []
    for file_index, uploaded_file in enumerate(files):
        width, height, _, fingerprint = _get_uploaded_file_meta(uploaded_file)
        if width > max_width:
            oversized.append(
                (
                    file_index,
                    uploaded_file.name,
                    width,
                    height,
                    fingerprint,
                )
            )
    return oversized


def split_image_into_equal_width_parts(
    image: Image.Image,
    max_width: int,
) -> list[Image.Image]:
    width, height = image.size
    part_count = (width + max_width - 1) // max_width
    if part_count <= 1:
        return [image]
    base_width, remainder = divmod(width, part_count)
    parts = []
    for part_index in range(part_count):
        left = part_index * base_width + min(part_index, remainder)
        right = left + base_width + int(part_index < remainder)
        parts.append(image.crop((left, 0, right, height)))
    return parts


def calibrate_prediction_confidence(
    raw_prob: float,
    threshold: float,
    temperature: float = 2.8,
    max_cap: float = 0.92,
) -> float:
    """Calibrate sigmoid confidence via threshold-centered temperature scaling
    to prevent overconfidence on ambiguous or misclassified samples."""
    p = min(max(float(raw_prob), 1e-5), 1.0 - 1e-5)
    t_c = min(max(float(threshold), 1e-5), 1.0 - 1e-5)
    logit_p = math.log(p / (1.0 - p))
    logit_t = math.log(t_c / (1.0 - t_c))
    cal_prob = 1.0 / (1.0 + math.exp(-(logit_p - logit_t) / temperature))
    conf = cal_prob if raw_prob > threshold else (1.0 - cal_prob)
    return min(conf, max_cap)


def aggregate_efficientnet_predictions(
    predictions: list[dict[str, Any]],
    areas: list[int],
    threshold: float,
) -> dict[str, Any]:
    if not predictions or len(predictions) != len(areas):
        raise ValueError("Predictions and pixel areas must be non-empty and aligned.")
    total_area = sum(areas)
    if total_area <= 0 or any(area <= 0 for area in areas):
        raise ValueError("Every image section must have a positive pixel area.")

    dyslexia_score = sum(
        prediction["raw"] * area
        for prediction, area in zip(predictions, areas)
    ) / total_area
    label = "Disleksia" if dyslexia_score > threshold else "Normal"
    confidence = calibrate_prediction_confidence(dyslexia_score, threshold)
    return {
        "label": label,
        "confidence": confidence,
        "raw": dyslexia_score,
        "class_scores": {
            "Disleksia": dyslexia_score,
            "Normal": 1.0 - dyslexia_score,
        },
        "time": sum(prediction["time"] for prediction in predictions),
    }


# Setup Halaman
st.set_page_config(
    page_title="Dyslexia Analysis AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS untuk tampilan premium
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f8fafc;
    }
    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: #040d12 !important;
        background-image:
            radial-gradient(circle at 12% 8%, rgba(0, 223, 154, 0.09) 0%, transparent 42%),
            radial-gradient(circle at 88% 85%, rgba(0, 223, 154, 0.06) 0%, transparent 45%),
            linear-gradient(180deg, #040d12 0%, #06141a 50%, #040d12 100%) !important;
        color: #f8fafc !important;
    }
    [data-testid="stHeader"] {
        background: rgba(4, 13, 18, 0.78) !important;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(0, 223, 154, 0.12);
    }
    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stHeadingWithActionElements"] h1,
    [data-testid="stHeadingWithActionElements"] h2,
    [data-testid="stHeadingWithActionElements"] h3,
    [data-testid="stHeadingWithActionElements"] h4 {
        color: #f8fafc !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: -0.015em;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] {
        color: rgba(248, 250, 252, 0.9) !important;
    }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: rgba(248, 250, 252, 0.68) !important;
    }
    hr {
        border-color: rgba(0, 223, 154, 0.18) !important;
    }
    /* Hero Container senada dengan AI Scanner Card */
    .hero-container {
        position: relative;
        background: linear-gradient(135deg, #06141a 0%, #0b222c 60%, #081b23 100%);
        background-image: 
            radial-gradient(circle at 15% 25%, rgba(0, 223, 154, 0.16) 0%, transparent 50%),
            linear-gradient(rgba(0, 223, 154, 0.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 223, 154, 0.035) 1px, transparent 1px),
            linear-gradient(135deg, #06141a 0%, #0b222c 60%, #081b23 100%);
        background-size: 100% 100%, 38px 38px, 38px 38px, 100% 100%;
        padding: 4.5rem 2rem 3.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 223, 154, 0.28);
        box-shadow: 0 22px 48px -12px rgba(4, 13, 18, 0.75), 0 0 28px rgba(0, 223, 154, 0.08);
    }
    
    .top-pill {
        display: inline-block;
        background: rgba(0, 223, 154, 0.12);
        color: #00df9a;
        padding: 6px 18px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 223, 154, 0.3);
    }
    
    .hero-title {
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 1.5rem;
        letter-spacing: -0.03em;
        line-height: 1.3;
        color: #ffffff;
    }
    
    .hero-title span {
        color: #00df9a;
    }
    
    .hero-subtitle {
        font-weight: 400;
        font-size: 1.08rem;
        color: rgba(248, 250, 252, 0.76);
        max-width: 800px;
        margin: 0 auto 2.25rem auto;
        line-height: 1.6;
    }
    
    .hero-features {
        display: flex;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
        max-width: 780px;
        margin: 0 auto;
    }
    
    .feature-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(0, 223, 154, 0.09);
        color: #f8fafc;
        padding: 8px 18px;
        border-radius: 999px;
        font-size: 0.88rem;
        font-weight: 600;
        border: 1px solid rgba(0, 223, 154, 0.3);
        box-shadow: 0 0 16px rgba(0, 223, 154, 0.08);
        transition: all 0.22s ease;
    }
    .feature-pill::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #00df9a;
        box-shadow: 0 0 8px rgba(0, 223, 154, 0.85);
        flex-shrink: 0;
    }
    
    .feature-pill:hover {
        background: rgba(0, 223, 154, 0.16);
        border-color: rgba(0, 223, 154, 0.55);
        transform: translateY(-1px);
    }
    
    .result-box {
        padding: 24px;
        border-radius: 16px;
        background: linear-gradient(135deg, #06141a 0%, #0b222c 60%, #081b23 100%);
        color: #f8fafc;
        border: 1px solid rgba(0, 223, 154, 0.28);
        box-shadow: 0 14px 32px -10px rgba(4, 13, 18, 0.6);
        border-left: 5px solid #00df9a;
    }
    .result-box h3, .result-box p {
        color: #f8fafc;
    }
    
    /* === CSS Untuk Section Fitur AI === */
    .features-section {
        padding: 3.25rem 1rem 0.5rem 1rem;
        text-align: center;
        border-top: 1px solid rgba(0, 223, 154, 0.16);
        margin-top: 2.75rem;
    }
    .features-title {
        font-weight: 800;
        font-size: 2.2rem;
        color: #ffffff;
        line-height: 1.3;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }
    .features-title span {
        color: #00df9a;
    }
    .features-subtitle {
        color: rgba(248, 250, 252, 0.72);
        font-size: 1.02rem;
        margin-bottom: 2.75rem;
    }
    .features-section-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: clamp(1.55rem, 3vw, 2rem);
        color: #00df9a;
        line-height: 1.35;
        margin: 0 auto 0.85rem !important;
        letter-spacing: 0;
        max-width: 900px;
        text-align: center;
    }
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 1.25rem;
        max-width: 950px;
        margin: 0 auto;
    }
    .feature-card {
        position: relative;
        background: linear-gradient(145deg, rgba(6, 20, 26, 0.82) 0%, rgba(11, 34, 44, 0.68) 100%);
        border: 1px solid rgba(0, 223, 154, 0.24);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.65rem 1.75rem;
        text-align: left;
        transition: all 0.28s ease;
        box-shadow: 0 12px 28px -12px rgba(4, 13, 18, 0.6), inset 0 1px 0 rgba(0, 223, 154, 0.12);
    }
    .feature-card:hover {
        transform: translateY(-4px);
        background: linear-gradient(145deg, rgba(8, 27, 35, 0.92) 0%, rgba(13, 41, 53, 0.82) 100%);
        border-color: rgba(0, 223, 154, 0.48);
        box-shadow: 0 16px 34px -10px rgba(4, 13, 18, 0.75), 0 0 20px rgba(0, 223, 154, 0.12);
    }
    .feature-card-title {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-weight: 700;
        font-size: 1.1rem;
        color: #ffffff;
        margin-bottom: 0.65rem;
    }
    .feature-card-title::before {
        content: "";
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00df9a;
        box-shadow: 0 0 10px rgba(0, 223, 154, 0.85);
        flex-shrink: 0;
    }
    .feature-card-desc {
        color: rgba(248, 250, 252, 0.72);
        font-size: 0.92rem;
        line-height: 1.6;
    }
    /* === Kotak Container Utama, Tabel, Tombol & Kontrol Senada AI Scanner === */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #06141a 0%, #0b222c 60%, #081b23 100%) !important;
        border: 1px solid rgba(0, 223, 154, 0.28) !important;
        border-radius: 18px !important;
        padding: 1.4rem 1.5rem !important;
        box-shadow: 0 18px 40px -12px rgba(4, 13, 18, 0.65), 0 0 24px rgba(0, 223, 154, 0.07) !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(4, 13, 18, 0.62) !important;
        border: 1px dashed rgba(0, 223, 154, 0.38) !important;
        border-radius: 14px !important;
        color: #f8fafc !important;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #00df9a !important;
        background: rgba(0, 223, 154, 0.06) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: rgba(0, 223, 154, 0.14) !important;
        color: #00df9a !important;
        border: 1px solid rgba(0, 223, 154, 0.38) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #00df9a 0%, #00b87c 100%) !important;
        color: #041217 !important;
        border: 1px solid rgba(143, 255, 224, 0.5) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
        box-shadow: 0 8px 22px -6px rgba(0, 223, 154, 0.45) !important;
        transition: all 0.22s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 26px -4px rgba(0, 223, 154, 0.62) !important;
        filter: brightness(1.05);
    }
    .stButton > button[kind="secondary"],
    button[data-testid="stBaseButton-secondary"],
    .stDownloadButton > button,
    div[data-testid="stPopover"] > div > button {
        background: linear-gradient(135deg, #06141a 0%, #0b222c 100%) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(0, 223, 154, 0.32) !important;
        border-radius: 11px !important;
        font-weight: 600 !important;
        box-shadow: 0 6px 18px -6px rgba(4, 13, 18, 0.6), 0 0 14px rgba(0, 223, 154, 0.06) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="secondary"]:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stPopover"] > div > button:hover {
        border-color: rgba(0, 223, 154, 0.65) !important;
        color: #00df9a !important;
        box-shadow: 0 8px 22px -6px rgba(0, 223, 154, 0.22) !important;
    }
    [data-testid="stPopoverBody"],
    div[role="dialog"] {
        background: linear-gradient(135deg, #06141a 0%, #0b222c 65%, #081b23 100%) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(0, 223, 154, 0.32) !important;
        border-radius: 16px !important;
        box-shadow: 0 20px 45px -10px rgba(0, 0, 0, 0.75), 0 0 24px rgba(0, 223, 154, 0.1) !important;
    }
    [data-testid="stExpander"] {
        background: linear-gradient(135deg, #06141a 0%, #0b222c 60%, #081b23 100%) !important;
        border: 1px solid rgba(0, 223, 154, 0.26) !important;
        border-radius: 14px !important;
        overflow: hidden;
        box-shadow: 0 10px 24px -10px rgba(4, 13, 18, 0.55) !important;
    }
    [data-testid="stExpander"] summary {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    [data-testid="stTable"] {
        background: linear-gradient(135deg, #06141a 0%, #0b222c 60%, #081b23 100%) !important;
        border: 1px solid rgba(0, 223, 154, 0.26) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: 0 12px 28px -12px rgba(4, 13, 18, 0.6), 0 0 18px rgba(0, 223, 154, 0.05) !important;
        margin-bottom: 0.65rem !important;
    }
    [data-testid="stTable"] table {
        width: 100% !important;
        border-collapse: collapse !important;
        color: #f8fafc !important;
        background: transparent !important;
    }
    [data-testid="stTable"] thead tr th {
        background: rgba(0, 223, 154, 0.12) !important;
        color: #00df9a !important;
        font-weight: 700 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        border-bottom: 1px solid rgba(0, 223, 154, 0.26) !important;
        padding: 0.65rem 0.85rem !important;
    }
    [data-testid="stTable"] tbody tr td {
        background: transparent !important;
        color: rgba(248, 250, 252, 0.92) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.06) !important;
        padding: 0.6rem 0.85rem !important;
        font-size: 0.86rem !important;
    }
    [data-testid="stTable"] tbody tr:hover td {
        background: rgba(0, 223, 154, 0.05) !important;
    }
    [data-testid="stAlert"],
    [data-testid="stAlertContainer"] {
        background: linear-gradient(135deg, #06141a 0%, #0b222c 60%, #081b23 100%) !important;
        border: 1px solid rgba(0, 223, 154, 0.3) !important;
        border-left: 4px solid #00df9a !important;
        border-radius: 14px !important;
        color: #f8fafc !important;
        box-shadow: 0 12px 28px -10px rgba(4, 13, 18, 0.55) !important;
    }
    [data-testid="stMarkdownContainer"] code {
        background: rgba(0, 223, 154, 0.12) !important;
        color: #00df9a !important;
        border: 1px solid rgba(0, 223, 154, 0.28) !important;
        border-radius: 6px !important;
        padding: 0.12rem 0.42rem !important;
        font-size: 0.82rem !important;
    }
    .wide-image-card {
        background: linear-gradient(135deg, #06141a 0%, #0b222c 60%, #081b23 100%);
        border: 1px solid rgba(0, 223, 154, 0.28);
        border-radius: 14px;
        padding: 1.05rem 1.25rem;
        margin: 0.65rem 0 0.75rem 0;
        color: #f8fafc;
        box-shadow: 0 12px 28px -10px rgba(6, 20, 26, 0.5), 0 0 18px rgba(0, 223, 154, 0.06);
    }
    .wide-image-intro {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        font-size: 0.85rem;
        color: rgba(248, 250, 252, 0.88);
        line-height: 1.5;
        margin-bottom: 0.7rem;
    }
    .wide-image-intro .ai-scanner-dot {
        margin-top: 0.38rem;
        flex-shrink: 0;
    }
    .wide-image-list {
        display: flex;
        flex-direction: column;
        gap: 0.42rem;
        margin-bottom: 0.75rem;
        max-height: 210px;
        overflow-y: auto;
        padding-right: 4px;
    }
    .sample-explorer-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 0.85rem;
    }
    .sample-stat-pills {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.55rem;
    }
    .sample-stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #e2e8f0;
    }
    .sample-stat-pill.is-dyslexia {
        background: rgba(239, 68, 68, 0.14);
        border-color: rgba(239, 68, 68, 0.36);
        color: #fca5a5;
    }
    .sample-stat-pill.is-normal {
        background: rgba(0, 223, 154, 0.14);
        border-color: rgba(0, 223, 154, 0.36);
        color: #00df9a;
    }
    .wide-image-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.45rem 0.75rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(0, 223, 154, 0.22);
        font-size: 0.82rem;
    }
    .wide-image-name {
        font-weight: 600;
        color: #ffffff;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .wide-image-dim {
        padding: 2px 9px;
        border-radius: 999px;
        background: rgba(0, 223, 154, 0.14);
        border: 1px solid rgba(0, 223, 154, 0.32);
        color: #00df9a;
        font-size: 0.75rem;
        font-weight: 700;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .status-row-item {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.35rem 0;
        font-size: 0.86rem;
        color: #f8fafc;
    }
    .status-dot-ok {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00df9a;
        box-shadow: 0 0 8px rgba(0, 223, 154, 0.85);
        flex-shrink: 0;
    }
    .status-dot-err {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #ef4444;
        box-shadow: 0 0 8px rgba(239, 68, 68, 0.85);
        flex-shrink: 0;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] {
        display: none !important;
    }
    [data-testid="stCameraInputWebcamStyledBox"] {
        width: min(100%, 26.25rem) !important;
        height: auto !important;
        aspect-ratio: 3 / 4 !important;
        margin: 0 auto !important;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        background: #111827;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    [data-testid="stCameraInputWebcamStyledBox"] video {
        width: 100% !important;
        height: 100% !important;
        object-fit: contain !important;
    }
    [data-testid="stCameraInputButton"] {
        width: fit-content !important;
        min-width: 0 !important;
        height: 3.25rem !important;
        min-height: 3.25rem !important;
        padding: 0 1.25rem !important;
        border-radius: 999px !important;
        display: flex !important;
        align-items: center;
        justify-content: center;
        font-size: 0 !important;
        margin: 0 auto !important;
    }
    [data-testid="stCameraInputButton"]::before {
        content: "Ambil Gambar";
        font-size: 1rem;
        line-height: 1;
        white-space: nowrap;
    }
    div[data-testid="stPopover"] {
        position: fixed;
        top: 4rem;
        left: 1rem;
        right: auto;
        z-index: 99999;
        width: auto;
    }
    .st-key-language_toggle {
        position: fixed;
        top: 4rem;
        right: 1rem;
        left: auto;
        z-index: 99999;
        width: auto;
    }
    .st-key-language_toggle button {
        min-height: 2.5rem;
        padding: 0 0.75rem;
        white-space: nowrap;
    }
    /* === Page Refresh & Entrance Animation === */
    @keyframes aiPageRevealA {
        0% {
            opacity: 0;
            transform: translateY(14px) scale(0.992);
            filter: brightness(1.18);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
            filter: brightness(1);
        }
    }
    @keyframes aiPageRevealB {
        0% {
            opacity: 0;
            transform: translateY(14px) scale(0.992);
            filter: brightness(1.18);
        }
        100% {
            opacity: 1;
            transform: translateY(0) scale(1);
            filter: brightness(1);
        }
    }
    .feature-card:nth-child(1) { animation: aiPageRevealA 0.55s cubic-bezier(0.22, 1, 0.36, 1) 0.08s both; }
    .feature-card:nth-child(2) { animation: aiPageRevealA 0.55s cubic-bezier(0.22, 1, 0.36, 1) 0.16s both; }
    .feature-card:nth-child(3) { animation: aiPageRevealA 0.55s cubic-bezier(0.22, 1, 0.36, 1) 0.24s both; }
    .feature-card:nth-child(4) { animation: aiPageRevealA 0.55s cubic-bezier(0.22, 1, 0.36, 1) 0.32s both; }
    /* === Full-Screen Refresh Splash Overlay (Spinning UNSIKA & Statistika Logos + Credits) === */
    @keyframes aiSplashOverlayFadeA {
        0% { opacity: 1; visibility: visible; }
        90% { opacity: 1; visibility: visible; }
        100% { opacity: 0; visibility: hidden; }
    }
    @keyframes aiSplashOverlayFadeB {
        0% { opacity: 1; visibility: visible; }
        90% { opacity: 1; visibility: visible; }
        100% { opacity: 0; visibility: hidden; }
    }
    @keyframes aiSplashCardStage {
        0% { opacity: 0; transform: translateY(18px) scale(0.95); }
        9% { opacity: 1; transform: translateY(0) scale(1); }
        54% { opacity: 1; transform: translateY(0) scale(1); filter: blur(0px); }
        63% { opacity: 0; transform: translateY(-14px) scale(0.97); filter: blur(4px); visibility: hidden; }
        100% { opacity: 0; transform: translateY(-14px) scale(0.97); visibility: hidden; }
    }
    @keyframes aiCreditsContainerStage {
        0%, 58% { opacity: 0; visibility: hidden; transform: scale(0.96); }
        66% { opacity: 1; visibility: visible; transform: scale(1); }
        89% { opacity: 1; visibility: visible; transform: scale(1.02); }
        100% { opacity: 0; visibility: hidden; transform: scale(1.05); }
    }
    @keyframes aiCreditLineReveal {
        0% { opacity: 0; transform: translateY(14px); filter: blur(4px); }
        100% { opacity: 1; transform: translateY(0); filter: blur(0px); }
    }
    @keyframes aiCreditDividerExpand {
        0% { width: 0; opacity: 0; }
        100% { width: 88px; opacity: 1; }
    }
    @keyframes aiOrbitSpinCW {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes aiOrbitSpinCCW {
        0% { transform: rotate(360deg); }
        100% { transform: rotate(0deg); }
    }
    @keyframes aiLogoSpin3D {
        0% { transform: perspective(600px) rotateY(0deg) scale(0.92); }
        50% { transform: perspective(600px) rotateY(180deg) scale(1.04); }
        100% { transform: perspective(600px) rotateY(360deg) scale(1); }
    }
    @keyframes aiSplashBarFill {
        0% { width: 4%; }
        35% { width: 52%; }
        75% { width: 92%; }
        100% { width: 100%; }
    }
    .ai-refresh-splash {
        position: fixed;
        inset: 0;
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1.5rem;
        background:
            radial-gradient(circle at 50% 35%, rgba(0, 223, 154, 0.16) 0%, transparent 55%),
            linear-gradient(rgba(0, 223, 154, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 223, 154, 0.03) 1px, transparent 1px),
            #040d12;
        background-size: 100% 100%, 38px 38px, 38px 38px, 100% 100%;
        backdrop-filter: blur(14px);
        pointer-events: none;
    }
    .ai-splash-box {
        position: relative;
        overflow: hidden;
        width: min(100%, 540px);
        background: linear-gradient(135deg, #06141a 0%, #0b222c 60%, #081b23 100%);
        border: 1px solid rgba(0, 223, 154, 0.38);
        border-radius: 22px;
        padding: 2.2rem 2rem 1.9rem 2rem;
        text-align: center;
        box-shadow: 0 28px 60px -12px rgba(0, 0, 0, 0.85), 0 0 38px rgba(0, 223, 154, 0.16);
        animation: aiSplashCardStage 5.6s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    }
    .ai-splash-credits {
        position: absolute;
        inset: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 1.5rem;
        pointer-events: none;
        animation: aiCreditsContainerStage 5.6s cubic-bezier(0.22, 1, 0.36, 1) forwards;
    }
    .ai-credit-created {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.26em;
        color: #94a3b8;
        margin-bottom: 0.45rem;
        opacity: 0;
        animation: aiCreditLineReveal 0.55s cubic-bezier(0.22, 1, 0.36, 1) 3.32s forwards;
    }
    .ai-credit-team {
        font-size: clamp(1.65rem, 3.8vw, 2.35rem);
        font-weight: 800;
        letter-spacing: 0.04em;
        line-height: 1.2;
        background: linear-gradient(90deg, #ffffff 0%, #8fffe0 35%, #00df9a 55%, #ffffff 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 22px rgba(0, 223, 154, 0.38));
        margin-bottom: 0.55rem;
        opacity: 0;
        animation:
            aiCreditLineReveal 0.6s cubic-bezier(0.22, 1, 0.36, 1) 3.48s forwards,
            aiShimmer 2.4s linear 3.48s infinite;
    }
    .ai-credit-divider {
        height: 2px;
        width: 0;
        border-radius: 999px;
        background: linear-gradient(90deg, transparent, #00df9a, transparent);
        box-shadow: 0 0 12px rgba(0, 223, 154, 0.75);
        margin: 0.1rem auto 0.55rem auto;
        opacity: 0;
        animation: aiCreditDividerExpand 0.55s cubic-bezier(0.22, 1, 0.36, 1) 3.62s forwards;
    }
    .ai-credit-sub {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #00df9a;
        opacity: 0;
        animation: aiCreditLineReveal 0.55s cubic-bezier(0.22, 1, 0.36, 1) 3.68s forwards;
    }
    .ai-splash-orbit-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: clamp(1rem, 4.5vw, 2.2rem);
        margin: 1.35rem auto 1.45rem auto;
        flex-wrap: nowrap;
    }
    .ai-logo-spinner-wrap {
        position: relative;
        width: clamp(92px, 24vw, 124px);
        height: clamp(92px, 24vw, 124px);
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .ai-orbit-ring-outer {
        position: absolute;
        inset: 0;
        border-radius: 50%;
        border: 2.5px solid rgba(0, 223, 154, 0.12);
        border-top-color: #00df9a;
        border-right-color: rgba(0, 223, 154, 0.55);
        box-shadow: 0 0 16px rgba(0, 223, 154, 0.25);
        animation: aiOrbitSpinCW 1.35s linear infinite;
    }
    .ai-orbit-ring-inner {
        position: absolute;
        inset: 7px;
        border-radius: 50%;
        border: 2px dashed rgba(143, 255, 224, 0.45);
        animation: aiOrbitSpinCCW 2.1s linear infinite;
    }
    .ai-logo-medallion {
        width: clamp(70px, 18.5vw, 94px);
        height: clamp(70px, 18.5vw, 94px);
        border-radius: 50%;
        background: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: clamp(8px, 2vw, 12px);
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.4), 0 0 18px rgba(0, 223, 154, 0.28);
        animation: aiLogoSpin3D 1.8s cubic-bezier(0.34, 1.56, 0.64, 1) infinite;
    }
    .ai-logo-medallion.delay-2 {
        animation-delay: 0.22s;
    }
    .ai-logo-medallion img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    .ai-splash-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        line-height: 1.3;
        margin-bottom: 0.35rem;
    }
    .ai-splash-subtitle {
        font-size: 0.86rem;
        font-weight: 600;
        color: #00df9a;
        margin-bottom: 1.3rem;
        letter-spacing: 0.01em;
    }
    .ai-splash-track {
        width: 100%;
        height: 9px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.07);
    }
    .ai-splash-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #00b87c 0%, #00df9a 35%, #8fffe0 50%, #00df9a 65%, #00b87c 100%);
        background-size: 200% 100%;
        box-shadow: 0 0 14px rgba(0, 223, 154, 0.65);
        animation:
            aiSplashBarFill 2.95s cubic-bezier(0.22, 1, 0.36, 1) forwards,
            aiShimmer 1.6s linear infinite;
    }
    /* === AI Progress Card (Spinning Logo Animation Only) === */
    @keyframes aiShimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    @keyframes aiPulseDot {
        0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 0 0 rgba(0, 223, 154, 0.7); }
        50% { transform: scale(1.18); opacity: 0.85; box-shadow: 0 0 0 6px rgba(0, 223, 154, 0); }
    }
    .ai-scanner-card {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, #06141a 0%, #0b222c 60%, #081b23 100%);
        border: 1px solid rgba(0, 223, 154, 0.28);
        border-radius: 14px;
        padding: 1.2rem 1.35rem;
        margin: 0.65rem 0 0.15rem 0;
        color: #f8fafc;
        box-shadow: 0 12px 28px -10px rgba(6, 20, 26, 0.45), 0 0 20px rgba(0, 223, 154, 0.07);
    }
    .ai-scanner-top {
        display: flex;
        align-items: center;
        gap: 1.15rem;
    }
    .ai-scanner-logos {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        flex-shrink: 0;
    }
    .ai-logo-spinner-wrap.is-compact {
        width: 62px;
        height: 62px;
    }
    .ai-logo-spinner-wrap.is-compact .ai-orbit-ring-inner {
        inset: 5px;
    }
    .ai-logo-spinner-wrap.is-compact .ai-logo-medallion {
        width: 46px;
        height: 46px;
        padding: 6px;
    }
    .ai-scanner-logos.is-done .ai-orbit-ring-outer,
    .ai-scanner-logos.is-done .ai-orbit-ring-inner,
    .ai-scanner-logos.is-done .ai-logo-medallion {
        animation: none;
    }
    .ai-scanner-info {
        flex: 1;
        min-width: 0;
    }
    .ai-scanner-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-bottom: 0.35rem;
    }
    .ai-scanner-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 3px 11px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        background: rgba(0, 223, 154, 0.14);
        color: #00df9a;
        border: 1px solid rgba(0, 223, 154, 0.3);
    }
    .ai-scanner-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #00df9a;
        animation: aiPulseDot 1.5s infinite;
    }
    .ai-scanner-meta {
        font-size: 0.8rem;
        font-weight: 600;
        color: rgba(248, 250, 252, 0.78);
    }
    .ai-scanner-pct {
        font-size: 1.25rem;
        font-weight: 800;
        color: #00df9a;
        margin-left: 0.4rem;
    }
    .ai-scanner-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #ffffff;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 0.18rem;
    }
    .ai-scanner-subtitle {
        font-size: 0.83rem;
        color: rgba(248, 250, 252, 0.72);
        line-height: 1.4;
    }
    .ai-scanner-track {
        width: 100%;
        height: 10px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        overflow: hidden;
        margin: 0.95rem 0 0.85rem 0;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .ai-scanner-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #00b87c 0%, #00df9a 35%, #8fffe0 50%, #00df9a 65%, #00b87c 100%);
        background-size: 200% 100%;
        animation: aiShimmer 2s linear infinite;
        transition: width 0.32s cubic-bezier(0.22, 1, 0.36, 1);
        box-shadow: 0 0 14px rgba(0, 223, 154, 0.55);
    }
    .ai-scanner-steps {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.55rem;
        margin-bottom: 0.8rem;
    }
    .ai-step-pill {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.45rem 0.65rem;
        border-radius: 10px;
        font-size: 0.76rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: rgba(248, 250, 252, 0.52);
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
    }
    .ai-step-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: rgba(248, 250, 252, 0.28);
        flex-shrink: 0;
    }
    .ai-step-pill.is-active {
        background: rgba(0, 223, 154, 0.13);
        border-color: rgba(0, 223, 154, 0.48);
        color: #ffffff;
        box-shadow: 0 0 14px rgba(0, 223, 154, 0.14);
    }
    .ai-step-pill.is-active .ai-step-dot {
        background: #00df9a;
        box-shadow: 0 0 8px rgba(0, 223, 154, 0.9);
        animation: aiPulseDot 1.5s infinite;
    }
    .ai-step-pill.is-done {
        background: rgba(21, 128, 61, 0.18);
        border-color: rgba(34, 197, 94, 0.38);
        color: #86efac;
    }
    .ai-step-pill.is-done .ai-step-dot {
        background: #22c55e;
        box-shadow: 0 0 6px rgba(34, 197, 94, 0.75);
    }
    .ai-scanner-tip {
        display: flex;
        align-items: flex-start;
        gap: 0.55rem;
        padding: 0.55rem 0.75rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.035);
        border-left: 3px solid #00df9a;
        font-size: 0.77rem;
        color: rgba(248, 250, 252, 0.78);
        line-height: 1.45;
    }
    .ai-scanner-tip strong {
        color: #00df9a;
        white-space: nowrap;
    }
    /* === Simple Bottom Footer (Advisor, Team Members, Disclaimer & Copyright) === */
    .simple-research-footer {
        position: relative;
        margin: 3.2rem auto 1.2rem auto;
        padding-top: 2.2rem;
        max-width: 840px;
        text-align: center;
        color: #94a3b8;
    }
    .simple-research-footer::before {
        content: "";
        position: absolute;
        top: 0;
        left: 5%;
        right: 5%;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.16) 25%, rgba(0, 223, 154, 0.45) 50%, rgba(255, 255, 255, 0.16) 75%, transparent 100%);
    }
    .simple-footer-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: 0.01em;
        margin-bottom: 0.4rem;
    }
    .simple-footer-title span {
        color: #00df9a;
    }
    .simple-footer-sub {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748b;
        letter-spacing: 0.02em;
        margin-bottom: 2rem;
    }
    .simple-footer-cols {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 2.5rem;
        max-width: 660px;
        margin: 0 auto 2rem auto;
        text-align: left;
    }
    .simple-footer-col {
        border-left: 2px solid rgba(0, 223, 154, 0.28);
        padding-left: 1.1rem;
    }
    .simple-footer-heading {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.92rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.02em;
        margin-bottom: 0.9rem;
    }
    .simple-footer-person {
        font-size: 0.87rem;
        line-height: 1.5;
        color: #cbd5e1;
        margin: 0 0 0.75rem 0;
    }
    .simple-footer-person:last-child {
        margin-bottom: 0;
    }
    .simple-footer-person strong {
        color: #f8fafc;
        font-weight: 600;
    }
    .simple-footer-person span {
        display: block;
        font-size: 0.78rem;
        color: #64748b;
    }
    .simple-footer-disclaimer {
        font-size: 0.79rem;
        line-height: 1.65;
        color: #64748b;
        max-width: 760px;
        margin: 0 auto 1.35rem auto;
    }
    .simple-footer-disclaimer p {
        margin: 0.2rem 0;
        color: #64748b;
    }
    .simple-footer-copyright {
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.03em;
        color: #475569;
    }
    @media (max-width: 640px) {
        .ai-splash-box {
            padding: 1.75rem 1.1rem 1.5rem 1.1rem;
        }
        .ai-scanner-card {
            padding: 1rem;
        }
        .ai-scanner-steps {
            grid-template-columns: 1fr;
            gap: 0.38rem;
        }
        .ai-scanner-visual {
            width: 54px;
            height: 54px;
            min-width: 54px;
        }
        .st-key-language_toggle {
            right: 0.75rem;
        }
        .st-key-language_toggle button {
            font-size: 0.8rem;
            padding: 0 0.55rem;
        }
        .hero-container {
            padding: 4.5rem 1rem 1.5rem;
            border-radius: 16px;
            margin-bottom: 1.25rem;
        }
        .hero-container .hero-title {
            font-size: clamp(1.35rem, 5vw, 1.8rem);
            line-height: 1.25;
            letter-spacing: -0.02em;
            margin-bottom: 1rem;
        }
        .hero-container .hero-subtitle {
            font-size: clamp(0.85rem, 3.2vw, 1rem);
            line-height: 1.5;
            margin-bottom: 1.5rem;
        }
        .hero-container .hero-features {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            width: 100%;
        }
        .hero-container .feature-pill {
            padding: 8px 4px;
            font-size: clamp(0.62rem, 2.5vw, 0.78rem);
            white-space: nowrap;
        }
        .hero-container img {
            max-width: clamp(50px, 16.5vw, 84px) !important;
            height: clamp(30px, 9.5vw, 46px) !important;
        }
        .hero-container > div[style*="display:flex"] {
            flex-wrap: nowrap !important;
            gap: clamp(6px, 2vw, 12px) !important;
            padding: 8px 12px !important;
            margin-bottom: 1.5rem !important;
        }
        .hero-container .features-section {
            padding: 2rem 0.25rem 0.25rem;
            margin-top: 2rem;
        }
        .hero-container .features-section-title {
            font-size: clamp(1.2rem, 4vw, 1.5rem);
            line-height: 1.3;
        }
        .hero-container .features-subtitle {
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }
        .hero-container .features-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
        }
        .hero-container .feature-card {
            padding: 0.9rem;
        }
        .hero-container .feature-card-title {
            font-size: clamp(0.82rem, 3.4vw, 0.95rem);
            line-height: 1.35;
        }
        .hero-container .feature-card-desc {
            font-size: clamp(0.7rem, 2.8vw, 0.78rem);
            line-height: 1.45;
        }
        .simple-footer-cols {
            grid-template-columns: 1fr;
            gap: 1.25rem;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <style>
    [data-testid="stCameraInputButton"]::before {{
        content: "{tr("Ambil Gambar")}";
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
st.session_state.setdefault("page_refresh_seq", 0)
_refresh_variant = "A" if st.session_state["page_refresh_seq"] % 2 == 0 else "B"
st.markdown(
    f"""
    <style>
    .hero-container,
    [data-testid="stVerticalBlockBorderWrapper"],
    .wide-image-card,
    [data-testid="stTable"] {{
        animation: aiPageReveal{_refresh_variant} 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = APP_DIR


class CameraImageUpload(io.BytesIO):
    def __init__(self, filename: str, image_bytes: bytes) -> None:
        super().__init__(image_bytes)
        self.name = filename


TFLITE_CANDIDATES = [
    os.path.join(BASE_DIR, "dyslexia_model_v2.tflite"),
    os.path.join(BASE_DIR, "dyslexia_model.tflite"),
]
KERAS_PATH = os.path.join(BASE_DIR, "dyslexia_model.h5")

MODEL_EFFNET = "EfficientNet-B0 (Klasifikasi)"
MODEL_MEASUREMENT = "OpenCV: Pengukuran & Crop Eksperimental"
WIDE_IMAGE_WARNING_WIDTH = 448
ANALYSIS_TYPES = [MODEL_EFFNET, MODEL_MEASUREMENT]
ANALYSIS_EFFNET_ONLY = "EfficientNet saja"
ANALYSIS_OPENCV_ONLY = "OpenCV saja (crop + EfficientNet)"
ANALYSIS_BOTH = "Keduanya"

settings_col, language_col = st.columns([1, 1], gap="small")
with settings_col:
    settings_popover = st.popover(
        tr("Pengaturan"),
        use_container_width=False,
        help=tr("Atur model dan parameter analisis."),
    )
with language_col:
    flag_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 42">'
        '<rect width="64" height="21" fill="#CE1126"/>'
        '<rect y="21" width="64" height="21" fill="#FFFFFF"/>'
        '</svg>'
        if UI_LANGUAGE == "Indonesia"
        else
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 42">'
        '<rect width="64" height="42" fill="#012169"/>'
        '<path d="M0 0h8l56 34v8h-8L0 8zm64 0h-8L0 34v8h8L64 8z" fill="#FFF"/>'
        '<path d="M0 0h4l60 36v4h-4L0 4zm64 0h-4L0 36v4h4L64 4z" fill="#C8102E"/>'
        '<path d="M27 0h10v42H27zM0 16h64v10H0z" fill="#FFF"/>'
        '<path d="M29 0h6v42h-6zM0 18h64v6H0z" fill="#C8102E"/>'
        '</svg>'
    )
    flag_data_uri = base64.b64encode(flag_svg.encode("utf-8")).decode("ascii")
    st.markdown(
        f"""
        <style>
            .st-key-language_toggle button::before {{
                content: "";
                display: inline-block;
                width: 1.5rem;
                min-width: 1.5rem;
                height: 1rem;
                flex-shrink: 0;
                margin-right: 0.45rem;
                vertical-align: middle;
                background: url("data:image/svg+xml;base64,{flag_data_uri}") center / cover no-repeat;
                border-radius: 2px;
                box-shadow: 0 0 0 1px rgba(128, 128, 128, 0.35);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.button(
        "ID" if UI_LANGUAGE == "Indonesia" else "EN",
        key="language_toggle",
        help=tr("Ganti bahasa / Change language"),
        on_click=toggle_ui_language,
    )

with settings_popover:
    st.markdown(f"### {tr('Pengaturan Analisis')}")
    analysis_mode = st.radio(
        tr("Pilih alur analisis"),
        [ANALYSIS_EFFNET_ONLY, ANALYSIS_OPENCV_ONLY, ANALYSIS_BOTH],
        index=2,
        key="analysis_mode",
        help=(
            tr(
                "OpenCV menggunakan EfficientNet untuk mengklasifikasikan setiap crop; "
                "mode ini tidak menjalankan klasifikasi gambar utuh."
            )
        ),
        format_func=tr,
    )
    st.caption(
        tr(
            "OpenCV saja berarti deteksi/crop OpenCV lalu EfficientNet mengklasifikasikan "
            "crop; EfficientNet gambar utuh hanya berjalan pada mode yang menyertakannya."
        )
    )
    threshold = st.slider(
        tr("Threshold Disleksia"),
        0.30,
        0.90,
        0.60,
        0.05,
        key="threshold_dislexia",
    )
    st.caption(
        tr(
            "Gambar diprediksi Disleksia jika skor kelas Disleksia > {threshold:.2f}. "
            "Ambang ini bukan standar klinis."
        ).format(threshold=threshold)
    )
    st.markdown(
        f"**{tr('OpenCV: Pengukuran & Crop Eksperimental')}**"
    )
    st.caption(
        tr(
            "Mengukur dan memotong komponen tinta; crop diproses dengan EfficientNet "
            "yang sama. Satu komponen belum tentu satu huruf."
        )
    )
    min_component_area = st.slider(
        tr("Luas minimum komponen (piksel)"),
        min_value=1,
        max_value=500,
        value=12,
        key="measurement_min_area",
        help=tr("Komponen yang lebih kecil dari nilai ini akan diabaikan."),
    )

run_effnet = analysis_mode in {ANALYSIS_EFFNET_ONLY, ANALYSIS_BOTH}
run_opencv = analysis_mode in {ANALYSIS_OPENCV_ONLY, ANALYSIS_BOTH}


@st.cache_data
def get_base64_logos():
    logos = []
    for i in range(1, 5):
        path = os.path.join(APP_DIR, "logo", f"{i}.png")
        if not os.path.exists(path):
            continue

        with Image.open(path) as source:
            image = source.convert("RGBA")
            image.thumbnail((180, 120), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
        logos.append(
            f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"
        )
    return logos

logos = get_base64_logos()
SPLASH_VERSION = 5
show_refresh_splash = (
    st.session_state.get("splash_shown_version") != SPLASH_VERSION
    or st.session_state.pop("trigger_splash", False)
)
st.session_state["splash_shown_version"] = SPLASH_VERSION
if show_refresh_splash and len(logos) >= 4:
    unsika_logo = logos[2]
    statistika_logo = logos[3]
    st.markdown(
        f"""
<style>
.ai-refresh-splash.seq-{st.session_state["page_refresh_seq"]} {{
    animation: aiSplashOverlayFade{_refresh_variant} 5.6s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}}
</style>
<div class="ai-refresh-splash seq-{st.session_state["page_refresh_seq"]}">
<div class="ai-splash-box">
<span class="ai-scanner-badge"><span class="ai-scanner-dot"></span>{tr("Memuat Aplikasi")}</span>
<div class="ai-splash-orbit-row">
<div class="ai-logo-spinner-wrap">
<div class="ai-orbit-ring-outer"></div>
<div class="ai-orbit-ring-inner"></div>
<div class="ai-logo-medallion">
<img src="{unsika_logo}" alt="Logo UNSIKA">
</div>
</div>
<div class="ai-logo-spinner-wrap">
<div class="ai-orbit-ring-outer"></div>
<div class="ai-orbit-ring-inner"></div>
<div class="ai-logo-medallion delay-2">
<img src="{statistika_logo}" alt="Logo Program Studi Statistika UNSIKA">
</div>
</div>
</div>
<div class="ai-splash-title">{tr("Alat Deteksi Dysgraphia")}</div>
<div class="ai-splash-subtitle">{tr("Program Studi Statistika • Universitas Singaperbangsa Karawang")}</div>
<div class="ai-splash-track"><div class="ai-splash-fill"></div></div>
</div>
<div class="ai-splash-credits">
<div class="ai-credit-created">Created By</div>
<div class="ai-credit-team">Anova Outlier 4</div>
<div class="ai-credit-divider"></div>
<div class="ai-credit-sub">UNSIKA Research Team</div>
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

if show_refresh_splash:
    _disc_delay = "5.25s" if len(logos) >= 4 else "0.2s"
    st.markdown(
        f"""
<style>
@keyframes aiDisclaimerFadeIn {{
    0% {{ opacity: 0; visibility: hidden; pointer-events: none; }}
    1% {{ visibility: visible; pointer-events: auto; }}
    100% {{ opacity: 1; visibility: visible; pointer-events: auto; }}
}}
@keyframes aiDisclaimerCardPop {{
    0% {{ opacity: 0; transform: translateY(24px) scale(0.92); }}
    60% {{ opacity: 1; transform: translateY(-4px) scale(1.015); }}
    100% {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
@keyframes aiPulseAge {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(0, 223, 154, 0.45); }}
    50% {{ box-shadow: 0 0 0 8px rgba(0, 223, 154, 0); }}
}}
.ai-disclaimer-checkbox {{
    display: none !important;
}}
.ai-disclaimer-overlay {{
    position: fixed;
    inset: 0;
    z-index: 99998;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.25rem;
    background: rgba(4, 13, 18, 0.82);
    backdrop-filter: blur(10px);
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    animation: aiDisclaimerFadeIn 0.5s ease-out {_disc_delay} forwards;
    transition: opacity 0.35s ease, visibility 0.35s ease;
}}
.ai-disclaimer-checkbox:checked ~ .ai-disclaimer-overlay {{
    opacity: 0 !important;
    visibility: hidden !important;
    pointer-events: none !important;
    animation: none !important;
}}
.ai-disclaimer-card {{
    position: relative;
    width: min(100%, 560px);
    background: linear-gradient(145deg, #071820 0%, #0d2733 55%, #071921 100%);
    border: 1px solid rgba(0, 223, 154, 0.45);
    border-radius: 22px;
    padding: 2rem 1.85rem 1.75rem;
    color: #e2e8f0;
    box-shadow: 0 28px 60px -12px rgba(0, 0, 0, 0.88), 0 0 40px rgba(0, 223, 154, 0.14);
    opacity: 0;
    animation: aiDisclaimerCardPop 0.55s cubic-bezier(0.22, 1, 0.36, 1) {_disc_delay} forwards;
}}
.ai-disclaimer-x {{
    position: absolute;
    top: 14px;
    right: 14px;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.16);
    color: #cbd5e1;
    font-size: 1.35rem;
    line-height: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s ease;
    user-select: none;
}}
.ai-disclaimer-x:hover {{
    background: rgba(239, 68, 68, 0.22);
    border-color: rgba(239, 68, 68, 0.55);
    color: #ffffff;
    transform: rotate(90deg);
}}
.ai-disclaimer-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 13px;
    border-radius: 999px;
    background: rgba(245, 158, 11, 0.14);
    border: 1px solid rgba(245, 158, 11, 0.42);
    color: #fbbf24;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}}
.ai-disclaimer-title {{
    font-size: clamp(1.25rem, 2.8vw, 1.5rem);
    font-weight: 800;
    color: #ffffff;
    line-height: 1.25;
    margin-bottom: 0.4rem;
}}
.ai-disclaimer-sub {{
    font-size: 0.88rem;
    color: #94a3b8;
    margin-bottom: 1rem;
}}
.ai-age-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 12px;
    background: linear-gradient(90deg, rgba(0, 223, 154, 0.18), rgba(14, 165, 233, 0.16));
    border: 1px solid rgba(0, 223, 154, 0.52);
    color: #6ee7b7;
    font-size: 0.92rem;
    font-weight: 700;
    margin-bottom: 1.1rem;
    animation: aiPulseAge 2.4s infinite;
}}
.ai-disclaimer-list {{
    margin: 0 0 1.35rem 0;
    padding: 0;
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
    font-size: 0.88rem;
    line-height: 1.5;
    color: #cbd5e1;
}}
.ai-disclaimer-list li {{
    position: relative;
    padding-left: 1.3rem;
}}
.ai-disclaimer-list li::before {{
    content: "";
    position: absolute;
    left: 0;
    top: 0.48rem;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #00df9a;
    box-shadow: 0 0 8px rgba(0, 223, 154, 0.7);
}}
.ai-disclaimer-btn {{
    display: block;
    width: 100%;
    text-align: center;
    padding: 0.78rem 1.25rem;
    border-radius: 12px;
    background: linear-gradient(90deg, #00df9a 0%, #0ea5e9 100%);
    color: #04131a;
    font-weight: 800;
    font-size: 0.94rem;
    letter-spacing: 0.02em;
    cursor: pointer;
    box-shadow: 0 10px 24px -6px rgba(0, 223, 154, 0.45);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    user-select: none;
}}
.ai-disclaimer-btn:hover {{
    transform: translateY(-2px);
    box-shadow: 0 14px 28px -6px rgba(0, 223, 154, 0.6);
}}
</style>
<input type="checkbox" id="ai-disclaimer-close-toggle" class="ai-disclaimer-checkbox" />
<div class="ai-disclaimer-overlay">
<div class="ai-disclaimer-card" role="dialog" aria-modal="true" aria-labelledby="ai-disc-title">
<label for="ai-disclaimer-close-toggle" class="ai-disclaimer-x" title="Close">&times;</label>
<div class="ai-disclaimer-badge"><span class="ai-scanner-dot"></span><span>{tr("DISCLAIMER PENELITIAN • ALAT BANTU SKRINING")}</span></div>
<div id="ai-disc-title" class="ai-disclaimer-title">{tr("Alat Bantu Deteksi Dini Disgrafia & Disleksia")}</div>
<div class="ai-disclaimer-sub">{tr("Berbasis Indikator Tulisan Tangan (EfficientNet-B0 & OpenCV)")}</div>
<div class="ai-age-pill">{tr("Khusus Anak Usia 4 – 7 Tahun")}</div>
<ul class="ai-disclaimer-list">
<li>{tr("Aplikasi ini hanya berfungsi sebagai alat bantu skrining awal (prototipe penelitian) berdasarkan pola visual tulisan tangan, bukan alat diagnosis medis atau klinis final.")}</li>
<li>{tr("Model dioptimalkan dan dievaluasi untuk karakteristik tulisan tangan anak pada rentang usia 4–7 tahun.")}</li>
<li>{tr("Hasil prediksi dan tingkat keyakinan (confidence) tetap perlu dikonsultasikan dengan psikolog, terapis okupasi, atau tenaga profesional terkait.")}</li>
</ul>
<label for="ai-disclaimer-close-toggle" class="ai-disclaimer-btn">{tr("Saya Mengerti & Lanjutkan")}</label>
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

logo_tags = "".join(
    f'<img src="{logo}" alt="Logo mitra {index}" '
    'style="width:auto; max-width:150px; height:72px; object-fit:contain;">'
    for index, logo in enumerate(logos, start=1)
)

if logos:
    logo_container = (
        '<div style="display:flex; justify-content:center; align-items:center; '
        'flex-wrap:wrap; gap:clamp(16px,3vw,35px); width:fit-content; '
        'max-width:calc(100% - 24px); margin:0 auto 2.5rem; background:#fff; '
        'padding:14px clamp(20px,4vw,45px); border-radius:60px; '
        'box-shadow:0 10px 25px rgba(0,0,0,0.2);">'
        f'{logo_tags}</div>'
    )
else:
    logo_container = ""

# Tampilan Awal (Hero Section) persis seperti referensi GipsyAI
st.markdown(f"""
<div class="hero-container">
{logo_container}
<div class="hero-title">
{tr("Alat Deteksi Dysgraphia")} <br>
<span>{tr("Model EfficientNet-B0 & OpenCV")}</span>
</div>
<div class="hero-subtitle">
{tr("Analisis pola gambar tulisan tangan tanpa ribet teknis. Cocok untuk guru, orang tua, hingga terapis. Dilengkapi berbagai fitur pendukung:")}
</div>
<div class="hero-features">
<span class="feature-pill">{tr("Pemrosesan Banyak Gambar")}</span>
<span class="feature-pill">{tr("Ekspor Hasil CSV")}</span>
<span class="feature-pill">{tr("Visualisasi Grad-CAM")}</span>
<span class="feature-pill">{tr("Eksperimen Crop OpenCV")}</span>
</div>
<div style="margin: 1.1rem auto 0; display: inline-flex; align-items: center; gap: 8px; padding: 8px 18px; border-radius: 999px; background: rgba(0, 223, 154, 0.14); border: 1px solid rgba(0, 223, 154, 0.42); color: #6ee7b7; font-size: 0.85rem; font-weight: 700;">
{tr("Sasaran Usia 4–7 Tahun • Alat Bantu Skrining Indikator Tulisan Tangan (Bukan Diagnosis Klinis)")}
</div>

<div class="features-section">
<h2 class="features-section-title">{tr("Mendukung Analisis Awal Disleksia Berbasis AI")}</h2>
<p class="features-subtitle">{tr("Diwujudkan melalui berbagai fitur unggulan untuk membantu analisis Anda. Hasil merupakan prototipe awal, bukan diagnosis.")}</p>
<div class="features-grid">
<div class="feature-card">
<div class="feature-card-title">{tr("Klasifikasi EfficientNet-B0")}</div>
<div class="feature-card-desc">{tr("Menganalisis gambar tulisan tangan dan memprediksi kategori Disleksia atau Normal. Visualisasi Grad-CAM menunjukkan area gambar yang memengaruhi prediksi. Hasil merupakan prediksi prototipe, bukan bukti klinis atau diagnosis.")}</div>
</div>
<div class="feature-card">
<div class="feature-card-title">{tr("Eksperimen Crop OpenCV")}</div>
<div class="feature-card-desc">{tr("OpenCV memotong komponen tinta, lalu EfficientNet-B0 yang sama memberi skor kelas pada setiap crop. Crop belum tentu satu huruf dan hasilnya masih eksperimen, bukan diagnosis.")}</div>
</div>
<div class="feature-card">
<div class="feature-card-title">{tr("Pemrosesan Banyak Gambar")}</div>
<div class="feature-card-desc">{tr("Mendukung pemrosesan puluhan gambar sekaligus dalam satu waktu, mempermudah pengujian data dan riset dalam skala besar secara instan.")}</div>
</div>
<div class="feature-card">
<div class="feature-card-title">{tr("Laporan & Ekspor Data")}</div>
<div class="feature-card-desc">{tr("Dapatkan hasil analisis lengkap beserta metrik raw score dan confidence level yang siap diekspor ke format CSV untuk kebutuhan riset lanjutan.")}</div>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)


def render_research_footer() -> None:
    st.markdown(
        f"""
<div class="simple-research-footer">
<div class="simple-footer-title"><span>Dyslev AI</span> — {tr("Alat Deteksi Dysgraphia")}</div>
<div class="simple-footer-sub">{tr("EfficientNet-B0 (TFLite & Grad-CAM) | OpenCV Component Analysis | Fast & Efficient")}</div>
<div class="simple-footer-cols">
<div class="simple-footer-col">
<div class="simple-footer-heading"><span class="ai-scanner-dot"></span><span>{tr("Advisor")}</span></div>
<p class="simple-footer-person"><strong>Dr. Karunia Eka Lestari</strong><span>Universitas Singaperbangsa Karawang</span></p>
<p class="simple-footer-person"><strong>Dr. Mokhammad Ridwan Yudhanegara</strong><span>Universitas Singaperbangsa Karawang</span></p>
</div>
<div class="simple-footer-col">
<div class="simple-footer-heading"><span class="ai-scanner-dot"></span><span>{tr("Team Members")}</span></div>
<p class="simple-footer-person"><strong>Nurul Syifa Az Zahra</strong><span>Universitas Singaperbangsa Karawang</span></p>
<p class="simple-footer-person"><strong>Danish Alzam</strong><span>Universitas Singaperbangsa Karawang</span></p>
<p class="simple-footer-person"><strong>Galih Rizkia Syahlani</strong><span>Universitas Singaperbangsa Karawang</span></p>
<p class="simple-footer-person"><strong>Daffa Najwa Saputra</strong><span>Universitas Singaperbangsa Karawang</span></p>
</div>
</div>
<div class="simple-footer-disclaimer">
<p>{tr("Disclaimer: Riset Mahasiswa yang menggunakan model EfficientNet-B0 (TFLite & Grad-CAM) dan OpenCV, digunakan untuk pengembangan dan eksperimen Program Studi Statistika UNSIKA.")}</p>
<p>{tr("Diikutsertakan dalam kompetisi Malaysia Data Innovation Talent (MDIT) 2026 yang diselenggarakan oleh Universiti Teknologi MARA (UiTM) Malaysia.")}</p>
</div>
<div class="simple-footer-copyright">
{tr("© 2026 Anova Outlier 4 — Program Studi Statistika, Universitas Singaperbangsa Karawang. All Rights Reserved.")}
</div>
</div>
        """,
        unsafe_allow_html=True,
    )

# =====================================================================
#  EfficientNet-B0 dan Eksperimen Crop OpenCV
# =====================================================================
if "analysis_intro_seen" not in st.session_state:
    st.session_state["analysis_intro_seen"] = False
for obsolete_key in (
    "use_effnet",
    "use_measurement",
    "use_yolo",
    "model_setup_complete",
):
    st.session_state.pop(obsolete_key, None)
if "upload_generation" not in st.session_state:
    st.session_state["upload_generation"] = 0
if "camera_capture_generation" not in st.session_state:
    st.session_state["camera_capture_generation"] = 0
if "camera_images" not in st.session_state:
    st.session_state["camera_images"] = []
if "reload_dialog_open" not in st.session_state:
    st.session_state["reload_dialog_open"] = False

if not st.session_state["analysis_intro_seen"]:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.dialog(tr("Informasi Analisis"), dismissible=False)
def analysis_intro_dialog():
    st.write(
        tr(
            "Aplikasi memiliki dua alur eksperimen: EfficientNet-B0 mengklasifikasikan "
            "gambar utuh secara default, atau bagian gambar lebar secara mandiri "
            "setelah gambar yang direkomendasikan dibagi otomatis. OpenCV memotong komponen tinta "
            "lalu mengirim setiap crop ke EfficientNet-B0 yang sama; OpenCV bukan "
            "model AI terpisah."
        )
    )
    st.markdown(
        tr(
            "- **EfficientNet saja:** klasifikasi gambar tulisan secara utuh. "
            "Untuk gambar lebar yang direkomendasikan untuk dibagi, tiap bagian diklasifikasikan "
            "mandiri dan skornya digabung dengan rata-rata tertimbang luas piksel.\n"
            "- **OpenCV saja:** deteksi komponen, crop, lalu klasifikasi crop dengan EfficientNet.\n"
            "- **Keduanya:** jalankan dua alur dan bandingkan hasilnya.\n"
            "- Skor pada crop bersifat eksperimen dan bukan peluang diagnosis; "
            "komponen tinta belum tentu satu huruf."
        )
    )
    if st.button(
        tr("Lanjut ke Analisis"),
        type="primary",
        use_container_width=True,
    ):
        st.session_state["analysis_intro_seen"] = True
        st.session_state["trigger_splash"] = True
        st.session_state["page_refresh_seq"] = (
            st.session_state.get("page_refresh_seq", 0) + 1
        )
        st.rerun()


def open_reload_dialog():
    st.session_state["reload_dialog_open"] = True


def cancel_reload():
    st.session_state["reload_dialog_open"] = False


def clear_uploaded_data():
    st.session_state["upload_generation"] += 1
    st.session_state["camera_capture_generation"] += 1
    st.session_state["camera_images"] = []
    st.session_state["reload_dialog_open"] = False
    st.session_state["trigger_splash"] = True
    st.session_state["page_refresh_seq"] = (
        st.session_state.get("page_refresh_seq", 0) + 1
    )


def restart_from_beginning():
    st.session_state.clear()


@st.dialog(tr("Muat Ulang Aplikasi"), dismissible=False)
def reload_options_dialog():
    st.write(
        tr(
            "Pilih tindakan. Refresh data akan menghapus gambar yang diunggah, "
            "tetapi mempertahankan pengaturan analisis. Muat ulang akan "
            "mengembalikan aplikasi ke halaman awal."
        )
    )
    cancel_col, clear_col, restart_col = st.columns(3)
    cancel_col.button(
        tr("Batal"),
        key="cancel_reload",
        on_click=cancel_reload,
        use_container_width=True,
    )
    clear_col.button(
        tr("Refresh data aja"),
        key="clear_uploaded_data",
        on_click=clear_uploaded_data,
        use_container_width=True,
    )
    restart_col.button(
        tr("Muat Ulang"),
        key="restart_from_beginning",
        on_click=restart_from_beginning,
        type="primary",
        use_container_width=True,
    )


with st.sidebar:
    st.button(
        tr("Muat Ulang"),
        key="open_reload_dialog",
        on_click=open_reload_dialog,
        use_container_width=True,
        help=tr("Kosongkan gambar yang diunggah atau mulai kembali dari awal."),
    )

if st.session_state["reload_dialog_open"]:
    reload_options_dialog()


if (
    not st.session_state["analysis_intro_seen"]
):
    _, start_col, _ = st.columns([1, 1.5, 1])
    if start_col.button(
        tr("Mulai Analisis"),
        type="primary",
        use_container_width=True,
    ):
        analysis_intro_dialog()
    st.stop()

# Mapping kelas EfficientNet (alfabetis sesuai train_and_convert.py)
CLASS_NAMES = ["Disleksia", "Normal"]

@st.cache_resource(show_spinner="Memuat EfficientNet-B0...")
def load_efficientnet():
    """Load TFLite interpreter EfficientNet. Return (interpreter, path, error)."""
    try:
        import tensorflow as tf
    except Exception as e:
        return None, None, f"TensorFlow belum terpasang ({e})"
    for path in TFLITE_CANDIDATES:
        if os.path.exists(path):
            interp = tf.lite.Interpreter(
                model_path=path,
                num_threads=min(4, os.cpu_count() or 1),
            )
            interp.allocate_tensors()
            return interp, path, None
    return None, None, "File .tflite EfficientNet tidak ditemukan"


@st.cache_resource(show_spinner="Memuat model Keras untuk Grad-CAM...")
def load_keras_model():
    try:
        import tensorflow as tf
        if os.path.exists(KERAS_PATH):
            return tf.keras.models.load_model(KERAS_PATH, compile=False)
    except Exception:
        pass
    return None


# ---------------------------- Inference ------------------------------
def preprocess_effnet(image: Image.Image):
    img_resized = image.resize((224, 224))
    arr = np.asarray(img_resized, dtype=np.float32)
    return img_resized, np.expand_dims(arr, axis=0) / 255.0


def enhance_handwriting_contrast(image: Image.Image) -> Image.Image:
    gray = np.asarray(image.convert("L"))
    black_level, white_level = np.percentile(gray, (2, 98))
    if white_level <= black_level:
        enhanced = gray
    else:
        enhanced = np.clip(
            (gray.astype(np.float32) - black_level)
            * (255.0 / (white_level - black_level)),
            0,
            255,
        ).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))


def predict_efficientnet(interpreter, image: Image.Image, threshold: float):
    start = time.time()
    img_resized, input_data = preprocess_effnet(image)
    inp_idx = getattr(interpreter, "_cached_inp_idx", None)
    out_idx = getattr(interpreter, "_cached_out_idx", None)
    if inp_idx is None or out_idx is None:
        inp_idx = interpreter.get_input_details()[0]["index"]
        out_idx = interpreter.get_output_details()[0]["index"]
        try:
            setattr(interpreter, "_cached_inp_idx", inp_idx)
            setattr(interpreter, "_cached_out_idx", out_idx)
        except Exception:
            pass
    interpreter.set_tensor(inp_idx, input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(out_idx)[0]
    raw = float(preds[0])  # indeks 0 = Disleksia
    label = "Disleksia" if raw > threshold else "Normal"
    conf = calibrate_prediction_confidence(raw, threshold)
    normal_score = float(preds[1]) if len(preds) > 1 else 1 - raw
    return {
        "label": label, "confidence": conf, "raw": raw,
        "class_scores": {
            "Disleksia": raw,
            "Normal": normal_score,
        },
        "time": time.time() - start, "img_resized": img_resized,
        "input": input_data,
    }


def predict_component_crops(
    interpreter,
    image: Image.Image,
    measurements: pd.DataFrame,
    threshold: float,
    on_crop_progress: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    crop_start = time.time()
    predictions = []
    crop_previews: list[dict[str, Any]] = []
    cropped_regions = crop_connected_regions(image, measurements)
    total_regions = len(cropped_regions)
    update_interval = max(1, total_regions // 4) if total_regions else 1
    for crop_idx, component in enumerate(cropped_regions, start=1):
        if on_crop_progress is not None and (
            crop_idx == 1
            or crop_idx == total_regions
            or crop_idx % update_interval == 0
        ):
            on_crop_progress(crop_idx, total_regions)
        result = predict_efficientnet(interpreter, component["crop"], threshold)
        predictions.append(
            {
                "Komponen": component["Komponen"],
                "Baris": component["Baris"],
                "Prediksi eksperimen": result["label"],
                "Skor Disleksia (%)": f"{result['class_scores']['Disleksia'] * 100:.2f}%",
                "Skor Normal (%)": f"{result['class_scores']['Normal'] * 100:.2f}%",
                "Skor kelas terpilih (%)": f"{result['confidence'] * 100:.2f}%",
            }
        )
    crop_elapsed = time.time() - crop_start
    total_crops = len(predictions)
    dyslexia_count = sum(
        prediction["Prediksi eksperimen"] == "Disleksia"
        for prediction in predictions
    )
    normal_count = total_crops - dyslexia_count
    if total_crops > 0:
        dys_ratio = dyslexia_count / total_crops
        crop_pred = "Disleksia" if dys_ratio > threshold else "Normal"
        crop_conf = calibrate_prediction_confidence(dys_ratio, threshold)
        crop_lvl, _ = confidence_level(crop_conf)
        conf_str = f"{crop_conf * 100:.2f}%"
    else:
        crop_pred = "Data kurang"
        conf_str = "-"
        crop_lvl = "-"
    summary = {
        "Prediksi": crop_pred,
        "Tingkat Kepercayaan": conf_str,
        "Level Confidence": crop_lvl,
        "Jumlah crop dianalisis": total_crops,
        "Komponen diprediksi Disleksia (%)": (
            f"{dyslexia_count / total_crops * 100:.2f}%"
            if total_crops
            else "Data kurang"
        ),
        "Komponen diprediksi Normal (%)": (
            f"{normal_count / total_crops * 100:.2f}%"
            if total_crops
            else "Data kurang"
        ),
        "Inference Time (s)": f"{crop_elapsed:.4f}",
    }
    return predictions, crop_previews, summary


def confidence_level(c):
    return ("Rendah", "red") if c < 0.6 else ("Medium", "orange") if c < 0.8 else ("Tinggi", "green")


def get_active_model(model_name: str):
    if model_name == MODEL_EFFNET:
        m, _, err = load_efficientnet()
    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")
    if err:
        if err.startswith("TensorFlow belum terpasang (") and err.endswith(")"):
            error_details = err[len("TensorFlow belum terpasang ("):-1]
            err = tr("TensorFlow belum terpasang ({error})").format(
                error=error_details
            )
        else:
            err = tr(err)
        st.error(err)
        return None
    return m


def get_active_analyses() -> dict[str, Any]:
    model = get_active_model(MODEL_EFFNET)
    if model is None:
        return {}
    return {
        MODEL_EFFNET: model,
        MODEL_MEASUREMENT: None,
    }


def build_comparison_summary(
    effnet_results: list[dict[str, Any]],
    crop_summaries: list[dict[str, Any]],
) -> pd.DataFrame:
    effnet_columns = ["Nama File", "Prediksi", "Tingkat Kepercayaan"]
    crop_columns = [
        "Nama File",
        "Jumlah crop dianalisis",
        "Komponen diprediksi Disleksia (%)",
        "Komponen diprediksi Normal (%)",
    ]
    effnet = pd.DataFrame(effnet_results)
    crops = pd.DataFrame(crop_summaries)
    if not effnet.empty:
        effnet = effnet.reindex(columns=effnet_columns).rename(
            columns={
                "Prediksi": "Prediksi EfficientNet (gambar utuh)",
                "Tingkat Kepercayaan": "Confidence EfficientNet",
            }
        )
    if not crops.empty:
        crops = crops.reindex(columns=crop_columns)
    if effnet.empty:
        return crops
    if crops.empty:
        return effnet
    return effnet.merge(crops, on="Nama File", how="outer")


def _parse_pct(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def build_hybrid_conclusion_summary(
    effnet_results: list[dict[str, Any]],
    crop_summaries: list[dict[str, Any]],
    threshold: float,
) -> pd.DataFrame:
    """Gabungkan skor EfficientNet-B0 (global) dan OpenCV Crop menjadi satu kesimpulan hybrid."""
    eff_by_name = {
        row["Nama File"]: row for row in effnet_results if "Nama File" in row
    }
    crop_by_name = {
        row["Nama File"]: row for row in crop_summaries if "Nama File" in row
    }
    all_names = list(dict.fromkeys([*eff_by_name.keys(), *crop_by_name.keys()]))

    records: list[dict[str, Any]] = []
    for fname in all_names:
        eff_row = eff_by_name.get(fname)
        crop_row = crop_by_name.get(fname)

        eff_raw_pct: float | None = None
        if eff_row is not None and "Raw Score (Disleksia)" in eff_row:
            try:
                eff_raw_pct = float(eff_row["Raw Score (Disleksia)"]) * 100.0
            except (TypeError, ValueError):
                eff_raw_pct = None

        crop_dys_pct = (
            _parse_pct(crop_row.get("Komponen diprediksi Disleksia (%)"))
            if crop_row is not None
            else None
        )

        if eff_raw_pct is not None and crop_dys_pct is not None:
            hybrid_pct = 0.70 * eff_raw_pct + 0.30 * crop_dys_pct
            eff_pred = "Disleksia" if eff_raw_pct > (threshold * 100.0) else "Normal"
            crop_pred = "Disleksia" if crop_dys_pct > (threshold * 100.0) else "Normal"
            agreement = (
                "Sepakat (Kedua Model)"
                if eff_pred == crop_pred
                else "Berbeda"
            )
        elif eff_raw_pct is not None:
            hybrid_pct = eff_raw_pct
            agreement = "Hanya EfficientNet-B0"
        elif crop_dys_pct is not None:
            hybrid_pct = crop_dys_pct
            agreement = "Hanya OpenCV Crop"
        else:
            continue

        hybrid_raw = hybrid_pct / 100.0
        final_label = "Disleksia" if hybrid_raw > threshold else "Normal"
        final_conf = calibrate_prediction_confidence(hybrid_raw, threshold)
        conf_lvl, _ = confidence_level(final_conf)

        records.append({
            "Nama File": fname,
            "Skor EfficientNet-B0 (%)": f"{eff_raw_pct:.2f}%" if eff_raw_pct is not None else "-",
            "Skor OpenCV Crop (%)": f"{crop_dys_pct:.2f}%" if crop_dys_pct is not None else "-",
            "Akumulasi Skor Hybrid (%)": f"{hybrid_pct:.2f}%",
            "Kesepakatan Model": agreement,
            "Kesimpulan Akhir Hybrid": final_label,
            "Confidence Hybrid": f"{final_conf * 100:.2f}% ({conf_lvl})",
        })

    return pd.DataFrame(records)


EXPORT_DATASET_LABELS = {
    "hybrid_conclusion": "Kesimpulan Hybrid (Gabungan 2 Model)",
    "effnet_summary": "Ringkasan EfficientNet",
    "gradcam_parts": "Skor per Bagian Grad-CAM",
    "comparison": "Ringkasan Perbandingan",
    "opencv_summary": "Pengukuran Gambar OpenCV",
    "opencv_crop_summary": "Ringkasan Crop OpenCV",
    "opencv_crop_details": "Prediksi Komponen OpenCV",
    "opencv_component_scale": "Skala Komponen",
    "opencv_line_variation": "Variasi per Baris",
}

EXPORT_CSV_FILENAMES = {
    "hybrid_conclusion": "kesimpulan_hybrid_gabungan.csv",
    "effnet_summary": "ringkasan_efficientnet.csv",
    "gradcam_parts": "skor_gradcam_per_bagian.csv",
    "comparison": "ringkasan_perbandingan.csv",
    "opencv_summary": "pengukuran_gambar_opencv.csv",
    "opencv_crop_summary": "ringkasan_crop_opencv.csv",
    "opencv_crop_details": "prediksi_komponen_opencv.csv",
    "opencv_component_scale": "skala_komponen.csv",
    "opencv_line_variation": "variasi_per_baris.csv",
}


def build_export_tables(
    batch_result: dict[str, Any],
    comparison_df: pd.DataFrame,
    hybrid_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    record_tables = (
        ("hybrid_conclusion", hybrid_df.to_dict("records") if hybrid_df is not None else []),
        ("effnet_summary", batch_result.get("results", [])),
        ("comparison", comparison_df.to_dict("records")),
        ("opencv_summary", batch_result.get("opencv_results", [])),
        ("opencv_crop_summary", batch_result.get("opencv_crop_summaries", [])),
        ("opencv_line_variation", batch_result.get("opencv_line_results", [])),
    )
    for key, records in record_tables:
        frame = pd.DataFrame(records)
        if not frame.empty:
            tables[key] = frame

    view_tables = (
        ("gradcam_parts", "gradcam_part_results", MODEL_EFFNET),
        ("opencv_crop_details", "crop_predictions", MODEL_MEASUREMENT),
        ("opencv_component_scale", "measurements", MODEL_MEASUREMENT),
    )
    for key, field, model_name in view_tables:
        records = [
            {"Nama File": output["filename"], **record}
            for output in batch_result.get("views", [])
            if output["analysis"] == model_name
            for record in output.get(field) or []
        ]
        frame = pd.DataFrame(records)
        if not frame.empty:
            tables[key] = frame
    return tables


def build_excel_export(tables: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(
        output,
        engine="xlsxwriter",
        engine_kwargs={
            "options": {
                "strings_to_formulas": False,
                "strings_to_urls": False,
            }
        },
    ) as writer:
        for key, frame in tables.items():
            frame.to_excel(
                writer,
                sheet_name=tr(EXPORT_DATASET_LABELS[key])[:31],
                index=False,
            )
    return output.getvalue()


def build_csv_export_zip(tables: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for key, frame in tables.items():
            export_frame = frame.copy()
            if "Nama File" in export_frame.columns:
                export_frame["Nama File"] = export_frame["Nama File"].map(
                    lambda value: (
                        f"'{value}"
                        if isinstance(value, str)
                        and value.lstrip(" \t\r\n")[:1] in {"=", "+", "-", "@"}
                        else value
                    )
                )
            archive.writestr(
                EXPORT_CSV_FILENAMES[key],
                export_frame.to_csv(index=False).encode("utf-8-sig"),
            )
    return output.getvalue()


@st.cache_data(show_spinner=False)
def dataframe_to_styled_png(
    df: pd.DataFrame, title: str = "", subtitle: str = ""
) -> bytes:
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(6, 2), dpi=200)
        ax.text(0.5, 0.5, "Tabel Kosong", ha="center", va="center", fontsize=12)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    display_df = df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].apply(
            lambda v: "" if pd.isna(v) else str(v)
        )

    n_rows, n_cols = display_df.shape
    headers = [str(h) for h in display_df.columns]

    wrapped_headers = [
        textwrap.fill(h, width=18) if len(h) > 20 and " " in h else h
        for h in headers
    ]

    raw_data = display_df.values.tolist()
    wrapped_data = []
    row_line_counts = []

    for row in raw_data:
        new_row = []
        max_lines_in_row = 1
        for val in row:
            s_val = str(val)
            if len(s_val) > 28 and " " in s_val:
                wrapped = textwrap.fill(s_val, width=26)
                lines = wrapped.count("\n") + 1
            else:
                wrapped = s_val
                lines = wrapped.count("\n") + 1
            if lines > max_lines_in_row:
                max_lines_in_row = lines
            new_row.append(wrapped)
        wrapped_data.append(new_row)
        row_line_counts.append(max_lines_in_row)

    col_max_lens = []
    col_is_text = []
    for c_idx, col in enumerate(wrapped_headers):
        header_len = max(len(l) for l in col.split("\n"))
        cell_lens = [
            max(len(l) for l in str(wrapped_data[r_idx][c_idx]).split("\n"))
            for r_idx in range(n_rows)
        ]
        m_len = max([header_len] + cell_lens)
        col_max_lens.append(m_len)

        sample_vals = [
            str(wrapped_data[r_idx][c_idx]) for r_idx in range(min(5, n_rows))
        ]
        is_txt = any(
            len(v) > 12 or (" " in v and not v.replace(".", "").replace("%", "").isdigit())
            for v in sample_vals
        )
        col_is_text.append(is_txt)

    total_len = max(sum(col_max_lens), 1)
    col_widths = [max(0.06, l / total_len) for l in col_max_lens]
    sum_w = sum(col_widths)
    col_widths = [w / sum_w for w in col_widths]

    base_font_size = 9.0 if (n_cols > 7 or n_rows > 30) else 10.0
    char_width_inch = 0.15 if base_font_size >= 10.0 else 0.13
    fig_width = max(8.0, min(24.0, total_len * char_width_inch + 2.5))

    has_title = bool(title or subtitle)
    title_height = 0.85 if has_title else 0.25
    header_lines = max((h.count("\n") + 1) for h in wrapped_headers)
    header_height = 0.38 + (header_lines * 0.18)

    line_h = 0.16 if base_font_size < 10.0 else 0.18
    base_r_h = 0.26 if base_font_size < 10.0 else 0.28
    row_heights = [
        base_r_h + (line_count * line_h) for line_count in row_line_counts
    ]
    content_height = sum(row_heights)
    footer_height = 0.45
    total_height = (
        title_height + header_height + content_height + footer_height
    )

    fig = plt.figure(figsize=(fig_width, total_height), dpi=250)
    fig.patch.set_facecolor("#ffffff")

    ax = fig.add_axes([0.03, 0.08, 0.94, 0.82])
    ax.axis("off")

    if title:
        fig.text(
            0.035,
            0.94,
            title,
            fontsize=13,
            fontweight="bold",
            color="#0f172a",
            ha="left",
        )
    sub = (
        subtitle
        if subtitle
        else "Dyslev AI — Research Vision Analysis & Handwriting Evaluation"
    )
    fig.text(
        0.035,
        0.90 if title else 0.94,
        sub,
        fontsize=8.8,
        color="#64748b",
        ha="left",
    )

    table = ax.table(
        cellText=wrapped_data,
        colLabels=wrapped_headers,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(base_font_size)

    header_color = "#1e293b"
    header_text_color = "#ffffff"
    zebra_even = "#ffffff"
    zebra_odd = "#f8fafc"
    border_color = "#cbd5e1"

    rel_header_h = header_height / total_height

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(border_color)
        cell.set_linewidth(0.75)

        is_col_txt = col_is_text[col]

        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(
                weight="bold",
                color=header_text_color,
                ha="left" if is_col_txt else "center",
            )
            cell.set_height(rel_header_h)
        else:
            bg = zebra_odd if row % 2 == 1 else zebra_even
            cell.set_facecolor(bg)
            cell.set_text_props(
                color="#1e293b",
                ha="left" if is_col_txt else "center",
            )
            rel_row_h = row_heights[row - 1] / total_height
            cell.set_height(rel_row_h)

    fig.text(
        0.035,
        0.02,
        "Program Studi Statistika • Universitas Singaperbangsa Karawang | Anova Outlier 4",
        fontsize=8.0,
        color="#94a3b8",
        ha="left",
    )

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=250,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(fig)
    return buf.getvalue()


COLLECTED_SAMPLES_DIR = os.path.join(BASE_DIR, "collected_samples")


def auto_save_dyslexia_sample(filename: str, raw_bytes: bytes) -> bool:
    """Simpan otomatis salinan gambar yang terdeteksi Disleksia untuk bank sampel."""
    if not raw_bytes:
        return False
    auto_dir = os.path.join(COLLECTED_SAMPLES_DIR, "auto_disleksia")
    os.makedirs(auto_dir, exist_ok=True)
    digest = hashlib.sha256(raw_bytes).hexdigest()[:10]
    stem = os.path.splitext(os.path.basename(filename))[0]
    ext = os.path.splitext(filename)[1].lower() or ".png"
    target_path = os.path.join(auto_dir, f"{stem}_{digest}{ext}")
    if os.path.exists(target_path):
        return False
    with open(target_path, "wb") as f:
        f.write(raw_bytes)
    return True


def save_manual_verified_sample(
    filename: str,
    raw_bytes: bytes,
    label: str,
    verifier_name: str,
    notes: str,
) -> str:
    """Simpan sampel terverifikasi manual 2-langkah beserta log metadata.csv."""
    label_folder = "Disleksia" if label == "Disleksia" else "Normal"
    target_dir = os.path.join(COLLECTED_SAMPLES_DIR, "manual_verified", label_folder)
    os.makedirs(target_dir, exist_ok=True)
    digest = hashlib.sha256(raw_bytes).hexdigest()[:10]
    stem = os.path.splitext(os.path.basename(filename))[0]
    ext = os.path.splitext(filename)[1].lower() or ".png"
    saved_name = f"{stem}_{digest}{ext}"
    saved_path = os.path.join(target_dir, saved_name)
    with open(saved_path, "wb") as f:
        f.write(raw_bytes)

    meta_path = os.path.join(COLLECTED_SAMPLES_DIR, "manual_verified", "metadata.csv")
    row = pd.DataFrame([{
        "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Nama File": saved_name,
        "Label Validasi": label_folder,
        "Verifikator": verifier_name.strip(),
        "Catatan Observasi": notes.strip(),
    }])
    row.to_csv(meta_path, mode="a", header=not os.path.exists(meta_path), index=False)
    return saved_path


def build_collected_samples_zip() -> bytes:
    """Kemas seluruh sampel terkumpul (auto & manual) ke dalam file ZIP."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(COLLECTED_SAMPLES_DIR):
            for root_dir, _, files in os.walk(COLLECTED_SAMPLES_DIR):
                for fn in files:
                    full_p = os.path.join(root_dir, fn)
                    rel_p = os.path.relpath(full_p, COLLECTED_SAMPLES_DIR)
                    zf.write(full_p, arcname=rel_p)
    return output.getvalue()


@st.cache_data(show_spinner=False)
def combine_image_with_table_png(
    image_bytes: bytes,
    df: pd.DataFrame,
    table_title: str = "",
) -> bytes:
    """Gabungkan gambar hasil visualisasi dengan tabel ringkasan model di bagian bawahnya."""
    if df is None or df.empty:
        return image_bytes
    with Image.open(io.BytesIO(image_bytes)) as top_im:
        top_rgb = top_im.convert("RGB")
    table_png = dataframe_to_styled_png(df, title=table_title)
    with Image.open(io.BytesIO(table_png)) as bot_im:
        bot_rgb = bot_im.convert("RGB")

    target_w = max(top_rgb.width, 960)
    if top_rgb.width != target_w:
        top_h = int(round(top_rgb.height * (target_w / top_rgb.width)))
        top_rgb = top_rgb.resize((target_w, top_h), Image.Resampling.LANCZOS)

    bot_h = int(round(bot_rgb.height * (target_w / bot_rgb.width)))
    bot_rgb = bot_rgb.resize((target_w, bot_h), Image.Resampling.LANCZOS)

    gap = 18
    combined = Image.new("RGB", (target_w, top_rgb.height + gap + bot_rgb.height), (255, 255, 255))
    combined.paste(top_rgb, (0, 0))
    combined.paste(bot_rgb, (0, top_rgb.height + gap))

    out = io.BytesIO()
    combined.save(out, format="PNG", dpi=(250, 250))
    return out.getvalue()


def build_png_export_zip(tables: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for key, frame in tables.items():
            title = tr(EXPORT_DATASET_LABELS.get(key, key))
            png_bytes = dataframe_to_styled_png(frame, title=title)
            filename = EXPORT_CSV_FILENAMES.get(
                key, f"{key}.csv"
            ).replace(".csv", ".png")
            archive.writestr(filename, png_bytes)
    return output.getvalue()


def render_table_download_row(
    df: pd.DataFrame,
    file_stem: str,
    base_name: str,
    title: str = "",
    key: str = "",
) -> None:
    if df is None or df.empty:
        return
    col_csv, col_png = st.columns(2)
    csv_filename = (
        f"{file_stem}_{base_name}.csv" if file_stem else f"{base_name}.csv"
    )
    png_filename = (
        f"{file_stem}_{base_name}.png" if file_stem else f"{base_name}.png"
    )

    with col_csv:
        st.download_button(
            tr("Unduh CSV"),
            df.to_csv(index=False).encode("utf-8"),
            file_name=csv_filename,
            mime="text/csv",
            key=f"dl_csv_{key or base_name}",
            on_click="ignore",
        )
    with col_png:
        png_bytes = dataframe_to_styled_png(df, title=title)
        st.download_button(
            tr("Unduh Gambar (PNG)"),
            png_bytes,
            file_name=png_filename,
            mime="image/png",
            key=f"dl_png_{key or base_name}",
            on_click="ignore",
        )


# -------------------------- Batch Processing -------------------------
BATCH_FULL = "Analisis Lengkap"
BATCH_TABLE = "Hanya Tabel Ringkasan"
CROP_AGGREGATION_VERSION = 1


PANEL_TITLES = [
    tr("Gambar Asli"),
    tr("Grid with Heatmap"),
    tr("Heatmap + Gambar"),
    tr("Heatmap Murni"),
]


def make_heat_panels(image: Image.Image, hm: np.ndarray, size=320):
    """Hasilkan 4 panel persegi: asli, grid+heatmap, overlay, heatmap murni."""
    base = np.array(image.resize((size, size)))
    hm = np.nan_to_num(hm)
    # Heatmap halus
    smooth = cv2.resize(hm.astype(np.float32), (size, size))
    color_s = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * smooth), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = np.clip(color_s * 0.45 + base * 0.55, 0, 255).astype(np.uint8)
    # Grid: sel Grad-CAM asli (kotak-kotak) + garis grid + nilai per sel
    n = hm.shape[0]
    blocky = cv2.resize(hm.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
    color_b = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * blocky), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    grid = np.clip(color_b * 0.45 + base * 0.55, 0, 255).astype(np.uint8)
    step = size / n
    for k in range(1, n):
        p = int(round(k * step))
        cv2.line(grid, (p, 0), (p, size), (255, 255, 255), 1)
        cv2.line(grid, (0, p), (size, p), (255, 255, 255), 1)
    for yy in range(n):
        for xx in range(n):
            v = hm[yy, xx]
            if v >= 0.5:
                cv2.putText(grid, f"{v:.1f}", (int(xx * step) + 4, int(yy * step) + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return [base, grid, overlay, color_s]


def make_row(panels, filename, status, color, size=320):
    """Satu baris: 4 panel berjudul + strip info file & status di bawahnya."""
    cells = []
    for img, title in zip(panels, PANEL_TITLES):
        head = np.full((30, size, 3), 255, np.uint8)
        cv2.putText(head, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        cells.append(cv2.copyMakeBorder(np.vstack([head, img]), 4, 4, 4, 4,
                                        cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    row = np.hstack(cells)
    info = np.full((36, row.shape[1], 3), 255, np.uint8)
    cv2.rectangle(info, (0, 0), (8, 36), color, -1)
    cv2.putText(info, filename[:60], (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(info, status, (row.shape[1] - 260, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return cv2.copyMakeBorder(np.vstack([row, info]), 0, 10, 0, 0, cv2.BORDER_CONSTANT, value=(230, 230, 230))


def heat_row_for(image, r, keras_model, filename):
    from grad_cam import make_gradcam_heatmap
    hm = make_gradcam_heatmap(
        r["input"],
        keras_model,
        "top_activation",
        pred_index=r.get("gradcam_class_index"),
    )
    col = (239, 68, 68) if r["label"] == "Disleksia" else (34, 197, 94)
    return make_row(make_heat_panels(image, hm), filename,
                    f"{r['label']} {r['confidence'] * 100:.1f}%", col)


model_usage_map = {
    MODEL_EFFNET: run_effnet,
    MODEL_MEASUREMENT: run_opencv,
}
model_rows = []
for name in ANALYSIS_TYPES:
    is_used = model_usage_map.get(name, True)
    if is_used:
        status_label = tr("Digunakan")
        badge_bg = "rgba(0,223,154,0.14)"
        badge_border = "rgba(0,223,154,0.32)"
        badge_color = "#00df9a"
        dot_bg = "#00df9a"
        dot_shadow = "0 0 8px rgba(0,223,154,0.85)"
        row_opacity = "1"
    else:
        status_label = tr("Tidak Digunakan")
        badge_bg = "rgba(148,163,184,0.12)"
        badge_border = "rgba(148,163,184,0.28)"
        badge_color = "#94a3b8"
        dot_bg = "#64748b"
        dot_shadow = "none"
        row_opacity = "0.68"

    model_rows.append(
        f"""
        <tr style="background:transparent; color:#f8fafc; opacity:{row_opacity};">
          <td style="padding:12px 16px; border-top:1px solid rgba(255,255,255,0.06); font-weight:600;">
            {tr(name)}
          </td>
          <td style="padding:12px 16px; border-top:1px solid rgba(255,255,255,0.06);">
            <span style="display:inline-flex; align-items:center; gap:6px;
                         padding:3px 10px; border-radius:999px;
                         background:{badge_bg}; border:1px solid {badge_border};
                         font-size:0.74rem; font-weight:700; color:{badge_color};">
              <span style="width:7px; height:7px; border-radius:50%; background:{dot_bg};
                           box-shadow:{dot_shadow};"></span>
              {status_label}
            </span>
          </td>
        </tr>
        """
    )

st.html(
    f"""
    <div style="max-width:680px; overflow:hidden; margin:0.35rem 0 1rem;
                background:linear-gradient(135deg, #06141a 0%, #0b222c 60%, #081b23 100%);
                border:1px solid rgba(0,223,154,0.28); border-radius:16px;
                box-shadow:0 16px 36px -12px rgba(4,13,18,0.65), 0 0 22px rgba(0,223,154,0.07);">
      <table style="width:100%; border-collapse:collapse; font-size:0.86rem;">
        <thead>
          <tr style="background:rgba(0,223,154,0.12); color:#00df9a;
                     border-bottom:1px solid rgba(0,223,154,0.24);">
            <th style="padding:11px 16px; text-align:left; font-size:0.74rem;
                       font-weight:700; letter-spacing:0.07em; text-transform:uppercase;">
              {tr("Jenis Analisis Model")}
            </th>
            <th style="width:140px; padding:11px 16px; text-align:left; font-size:0.74rem;
                       font-weight:700; letter-spacing:0.07em; text-transform:uppercase;">
              {tr("Status")}
            </th>
          </tr>
        </thead>
        <tbody>
          {''.join(model_rows)}
        </tbody>
      </table>
    </div>
    """
)

st.markdown("---")
with st.container():
    st.subheader(tr("Analisis Tulisan Tangan"))
    with st.container(border=True):
        st.markdown(
            tr(
                "**Eksperimen 1:** EfficientNet-B0 pada gambar utuh  \n"
                "**Eksperimen 2:** OpenCV crop → EfficientNet-B0 pada tiap komponen"
            )
        )
        enhance_image_contrast = st.checkbox(
            tr("Perjelas kontras tulisan"),
            value=True,
            help=(
                tr(
                    "Memekatkan bagian gelap dan mencerahkan bagian terang dengan "
                    "peregangan kontras, sambil mempertahankan tingkat abu-abu. Tidak "
                    "membalik atau membinarisasi warna. File asli tidak diubah."
                )
            ),
        )
        source_choice = st.radio(
            tr("Sumber gambar"),
            ["Upload file", "Kamera langsung"],
            horizontal=True,
            key="analysis_image_source",
            format_func=tr,
        )
        if source_choice == "Upload file":
            batch_files = st.file_uploader(
                tr("Pilih satu atau beberapa gambar"),
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"batch_{st.session_state['upload_generation']}",
            )
        else:
            st.caption(
                tr(
                    "Pratinjau berbentuk potret. Jika kamera menampilkan lanskap, "
                    "gambar akan tampak lebih kecil di tengah tanpa terpotong. "
                    "Resolusi diminta hingga 1080p sesuai dukungan perangkat; aplikasi "
                    "tidak mengubah ukuran atau mengompresi ulang foto. Streamlit "
                    "menyimpan hasil kamera dalam JPEG kualitas maksimum."
                )
            )
            camera_file = st.camera_input(
                tr("Ambil Gambar"),
                key=(
                    f"camera_capture_{st.session_state['upload_generation']}_"
                    f"{st.session_state['camera_capture_generation']}"
                ),
                resolution="1080p",
            )
            if camera_file is not None:
                camera_images = st.session_state["camera_images"]
                camera_images.append(
                    (
                        f"kamera_{len(camera_images) + 1}.jpg",
                        camera_file.getvalue(),
                    )
                )
                st.session_state["camera_capture_generation"] += 1
                st.rerun()

            camera_images = st.session_state["camera_images"]
            if camera_images:
                st.caption(
                    tr("Foto kamera siap dianalisis: {count}").format(
                        count=len(camera_images)
                    )
                )
                with st.expander(tr("Lihat foto yang sudah diambil")):
                    preview_columns = st.columns(min(len(camera_images), 4))
                    for image_index, (image_name, image_bytes) in enumerate(
                        camera_images
                    ):
                        with preview_columns[image_index % len(preview_columns)]:
                            st.image(
                                image_bytes,
                                caption=image_name,
                                use_container_width=True,
                            )
                if st.button(
                    tr("Hapus semua foto kamera"),
                    key="clear_camera_images",
                ):
                    st.session_state["camera_images"] = []
                    st.session_state["camera_capture_generation"] += 1
                    st.rerun()

            camera_fallback = st.file_uploader(
                tr("Atau tambahkan beberapa foto dari perangkat"),
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"camera_fallback_{st.session_state['upload_generation']}",
                help=(
                    tr(
                        "Pilih satu atau beberapa foto jika kamera langsung tidak "
                        "tersedia, atau untuk menambahkan foto lain."
                    )
                ),
            )
            batch_files = [
                CameraImageUpload(image_name, image_bytes)
                for image_name, image_bytes in camera_images
            ] + list(camera_fallback or [])

        wide_images = images_over_width(batch_files, WIDE_IMAGE_WARNING_WIDTH)

        if st.session_state.get("batch_mode") in {
            "Statistik Variasi Tinggi (CV)",
            "Tampilkan Heatmap + Grid PNG",
        }:
            del st.session_state["batch_mode"]
        batch_output_options = [BATCH_FULL, BATCH_TABLE]
        batch_mode = st.radio(
            tr("Format output"),
            batch_output_options,
            horizontal=True,
            key="batch_mode",
            format_func=tr,
        )
        can_split_wide_images = (
            bool(wide_images)
            and run_effnet
            and batch_mode == BATCH_FULL
            and os.path.exists(KERAS_PATH)
        )
        if wide_images:
            import html as _html_notice

            image_items_html = "".join(
                f'<div class="wide-image-item">'
                f'<span class="wide-image-name">{_html_notice.escape(filename)}</span>'
                f'<span class="wide-image-dim">{width} × {height} px</span>'
                f'</div>'
                for _, filename, width, height, _ in wide_images
            )
            warning_intro = _html_notice.escape(
                tr(
                    "Gambar berikut memiliki lebar lebih dari {threshold} piksel. "
                    "Input model diubah menjadi 224 × 224 piksel, sehingga detail "
                    "tulisan mungkin sulit terlihat:"
                ).format(threshold=WIDE_IMAGE_WARNING_WIDTH)
            )
            warning_guidance = _html_notice.escape(
                tr(
                    "Gambar lebar yang tercantum direkomendasikan untuk dibagi "
                    "otomatis dengan pengaturan di bawah."
                    if can_split_wide_images
                    else "Pembagian per gambar hanya tersedia pada Analisis Lengkap "
                    "saat model Grad-CAM EfficientNet tersedia."
                )
            )
            st.markdown(
                f'<div class="wide-image-card">'
                f'<div class="wide-image-intro">'
                f'<span class="ai-scanner-dot"></span>'
                f'<span>{warning_intro}</span>'
                f'</div>'
                f'<div class="wide-image-list">{image_items_html}</div>'
                f'<div class="ai-scanner-tip"><span>{warning_guidance}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        split_wide_file_indices: set[int] = set()
        if can_split_wide_images:
            batch_images_key = hashlib.sha256(
                "|".join(
                    f"{file_index}:{uploaded_image_fingerprint(uploaded_file)}"
                    for file_index, uploaded_file in enumerate(batch_files)
                ).encode("utf-8")
            ).hexdigest()[:16]
            split_recommended_wide_images = st.checkbox(
                tr("Disarankan: bagi otomatis semua gambar lebar untuk Grad-CAM"),
                value=True,
                key=f"split_wide_gradcam_{batch_images_key}",
                help=tr(
                    "Semua gambar lebar yang tercantum akan dibagi otomatis "
                    "untuk Grad-CAM. Matikan centang ini jika ingin memprosesnya "
                    "tanpa pembagian."
                ),
            )
            if split_recommended_wide_images:
                split_wide_file_indices = {
                    file_index
                    for file_index, _, _, _, _ in wide_images
                }
        generate_batch = False
        if batch_files:
            generate_batch = st.button(
                tr("Mulai Analisis"),
                type="primary",
                key="generate_multiple_processing",
            )
        scanner_placeholder = st.empty()
    batch_signature = (
        (
            tuple(
                (bf.name, _get_uploaded_file_meta(bf)[2])
                for bf in batch_files
            ),
            analysis_mode,
            batch_mode,
            threshold,
            min_component_area,
            enhance_image_contrast,
            tuple(sorted(split_wide_file_indices)),
            CROP_AGGREGATION_VERSION,
        )
        if batch_files
        else None
    )

    if batch_files and generate_batch:
        active_models = get_active_analyses()
        if active_models:
            import html as _html

            results = []
            opencv_results = []
            opencv_line_results = []
            opencv_crop_summaries = []
            tiles = []
            output_views = []
            keras_model = None
            scan_start_time = time.time()
            last_ui_update = [0.0]

            if run_effnet and batch_mode == BATCH_FULL:
                keras_model = load_keras_model()
                if keras_model is None:
                    st.warning(
                        tr("Model .h5 tidak tersedia, heatmap dilewati.")
                    )

            if run_effnet and run_opencv:
                workflow_label = tr("Alur Aktif: EfficientNet-B0 & OpenCV")
            elif run_effnet:
                workflow_label = tr(
                    "Alur Aktif: EfficientNet-B0 (Klasifikasi & Grad-CAM)"
                )
            else:
                workflow_label = tr(
                    "Alur Aktif: OpenCV Crop & EfficientNet-B0"
                )

            scanner_tips = [
                tr(
                    "Peregangan kontras (persentil 2–98%) mempertegas goresan pena "
                    "tanpa menghilangkan gradasi abu-abu alami."
                ),
                tr(
                    "Grad-CAM memetakan aktivasi layer konvolusi terakhir untuk "
                    "menampilkan area tulisan yang paling memengaruhi prediksi."
                ),
                tr(
                    "Gambar lebar (>448 px) dibagi menjadi beberapa segmen vertikal "
                    "agar detail goresan tidak menyusut saat di-resize ke 224×224 px."
                ),
                tr(
                    "OpenCV mengelompokkan komponen tinta per baris dan menghitung "
                    "koefisien variasi (CV) tinggi serta lebar secara otomatis."
                ),
            ]

            def update_scanner_card(
                progress_fraction: float,
                filename: str,
                sample_idx: int,
                total_samples: int,
                detail_text: str,
                tip_index: int = 0,
                is_done: bool = False,
                force: bool = False,
            ) -> None:
                now_ts = time.time()
                if (
                    not force
                    and not is_done
                    and (now_ts - last_ui_update[0]) < 0.2
                ):
                    return
                last_ui_update[0] = now_ts
                pct = max(2, min(100, int(round(progress_fraction * 100))))
                if is_done:
                    pct = 100
                elapsed = now_ts - scan_start_time
                safe_filename = _html.escape(filename)
                safe_detail = _html.escape(detail_text)
                tip_text = _html.escape(
                    scanner_tips[tip_index % len(scanner_tips)]
                )
                badge_label = (
                    tr("Analisis Selesai")
                    if is_done
                    else tr("Pemindaian AI Aktif")
                )
                counter_label = tr("Sampel {current}/{total}").format(
                    current=sample_idx,
                    total=total_samples,
                )
                pill_state_cls = "is-done" if is_done else "is-active"
                workflow_pill_html = (
                    f'<div class="ai-step-pill {pill_state_cls}" style="grid-column: 1 / -1;">'
                    f'<span class="ai-step-dot"></span>'
                    f'<span>{_html.escape(workflow_label)} &bull; {_html.escape(counter_label)}</span>'
                    f'</div>'
                )

                visual_cls = "ai-scanner-logos is-done" if is_done else "ai-scanner-logos"
                unsika_uri = logos[2] if len(logos) >= 4 else ""
                statistika_uri = logos[3] if len(logos) >= 4 else ""
                card_html = (
                    f'<div class="ai-scanner-card">'
                    f'<div class="ai-scanner-top">'
                    f'<div class="{visual_cls}">'
                    f'<div class="ai-logo-spinner-wrap is-compact">'
                    f'<div class="ai-orbit-ring-outer"></div>'
                    f'<div class="ai-orbit-ring-inner"></div>'
                    f'<div class="ai-logo-medallion">'
                    f'<img src="{unsika_uri}" alt="UNSIKA">'
                    f'</div>'
                    f'</div>'
                    f'<div class="ai-logo-spinner-wrap is-compact">'
                    f'<div class="ai-orbit-ring-outer"></div>'
                    f'<div class="ai-orbit-ring-inner"></div>'
                    f'<div class="ai-logo-medallion delay-2">'
                    f'<img src="{statistika_uri}" alt="Statistika UNSIKA">'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                    f'<div class="ai-scanner-info">'
                    f'<div class="ai-scanner-header-row">'
                    f'<span class="ai-scanner-badge">'
                    f'<span class="ai-scanner-dot"></span>{_html.escape(badge_label)}'
                    f'</span>'
                    f'<span class="ai-scanner-meta">'
                    f'{_html.escape(counter_label)} &bull; {elapsed:.1f}s'
                    f'<span class="ai-scanner-pct">{pct}%</span>'
                    f'</span>'
                    f'</div>'
                    f'<div class="ai-scanner-title">{safe_filename}</div>'
                    f'<div class="ai-scanner-subtitle">{safe_detail}</div>'
                    f'</div>'
                    f'</div>'
                    f'<div class="ai-scanner-track">'
                    f'<div class="ai-scanner-fill" style="width:{pct}%;"></div>'
                    f'</div>'
                    f'<div class="ai-scanner-steps">{workflow_pill_html}</div>'
                    f'<div class="ai-scanner-tip">'
                    f'<strong>{_html.escape(tr("Info AI"))}:</strong>'
                    f'<span>{tip_text}</span>'
                    f'</div>'
                    f'</div>'
                )
                scanner_placeholder.markdown(card_html, unsafe_allow_html=True)

            total_files = len(batch_files)
            png_encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
            for i, bf in enumerate(batch_files):
                sample_num = i + 1
                update_scanner_card(
                    (sample_num - 0.5) / total_files,
                    bf.name,
                    sample_num,
                    total_files,
                    detail_text=tr(
                        "Memproses sampel {current} dari {total} pada seluruh model aktif..."
                    ).format(
                        current=sample_num,
                        total=total_files,
                    ),
                    tip_index=i,
                    force=(i == 0),
                )
                image = Image.open(bf).convert('RGB')
                analysis_image = (
                    enhance_handwriting_contrast(image)
                    if enhance_image_contrast
                    else image
                )
                w, h = image.size
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for model_name, model in active_models.items():
                    if model_name == MODEL_EFFNET and not run_effnet:
                        continue
                    if model_name == MODEL_MEASUREMENT and not run_opencv:
                        continue
                    batch_visual = np.array(analysis_image)
                    component_measurements = None
                    line_variability_records = None
                    component_predictions = []
                    crop_previews = []
                    crop_summary = None
                    crop_error = None
                    gradcam_part_results = []
                    if model_name == MODEL_EFFNET:
                        r = None
                        if (
                            i in split_wide_file_indices
                            and w > WIDE_IMAGE_WARNING_WIDTH
                            and keras_model is not None
                        ):
                            try:
                                image_parts = split_image_into_equal_width_parts(
                                    analysis_image,
                                    WIDE_IMAGE_WARNING_WIDTH,
                                )
                                gradcam_rows = []
                                part_score_rows = []
                                part_predictions = []
                                part_areas = []
                                for part_index, image_part in enumerate(
                                    image_parts,
                                    start=1,
                                ):
                                    part_result = predict_efficientnet(
                                        model,
                                        image_part,
                                        threshold,
                                    )
                                    part_result["gradcam_class_index"] = (
                                        CLASS_NAMES.index(part_result["label"])
                                    )
                                    part_predictions.append(part_result)
                                    part_areas.append(
                                        image_part.width * image_part.height
                                    )
                                    part_score_rows.append({
                                        "Bagian": f"{part_index}/{len(image_parts)}",
                                        "Prediksi": part_result["label"],
                                        "Tingkat Kepercayaan": (
                                            f"{part_result['confidence'] * 100:.2f}%"
                                        ),
                                        "Raw Score (Disleksia)": (
                                            f"{part_result['raw']:.6f}"
                                        ),
                                        "Ukuran Gambar": (
                                            f"{image_part.width}x{image_part.height}"
                                        ),
                                        "Inference Time (s)": (
                                            f"{part_result['time']:.4f}"
                                        ),
                                    })
                                    part_name = tr(
                                        "Bagian {index}/{count} - {filename}"
                                    ).format(
                                        index=part_index,
                                        count=len(image_parts),
                                        filename=bf.name,
                                    )
                                    gradcam_rows.append(
                                        heat_row_for(
                                            image_part,
                                            part_result,
                                            keras_model,
                                            part_name,
                                        )
                                    )
                                r = aggregate_efficientnet_predictions(
                                    part_predictions,
                                    part_areas,
                                    threshold,
                                )
                                gradcam_part_results = part_score_rows
                                batch_visual = np.vstack(gradcam_rows)
                                tiles.extend(gradcam_rows)
                            except Exception as e:
                                st.warning(
                                    tr("Heatmap gagal untuk {filename}: {error}").format(
                                        filename=bf.name,
                                        error=e,
                                    )
                                )

                        if r is None:
                            r = predict_efficientnet(
                                model,
                                analysis_image,
                                threshold,
                            )
                            if batch_mode == BATCH_FULL and keras_model is not None:
                                try:
                                    batch_visual = heat_row_for(
                                        analysis_image, r, keras_model, bf.name
                                    )
                                    tiles.append(batch_visual)
                                except Exception as e:
                                    st.warning(
                                        tr("Heatmap gagal untuk {filename}: {error}").format(
                                            filename=bf.name,
                                            error=e,
                                        )
                                    )
                        lvl, _ = confidence_level(r["confidence"])
                        results.append({
                            "Nama File": bf.name,
                            "Model": model_name,
                            "Metode Prediksi": (
                                "Rata-rata tertimbang skor per bagian"
                                if gradcam_part_results
                                else "Gambar utuh"
                            ),
                            "Prediksi": r["label"],
                            "Tingkat Kepercayaan": f"{r['confidence'] * 100:.2f}%",
                            "Level Confidence": lvl,
                            "Raw Score (Disleksia)": f"{r['raw']:.6f}",
                            "Ukuran Gambar": f"{w}x{h}",
                            "Inference Time (s)": f"{r['time']:.4f}",
                            "Waktu Analisis": now,
                        })
                        if r["label"] == "Disleksia":
                            try:
                                auto_save_dyslexia_sample(bf.name, bf.getvalue())
                            except Exception:
                                pass
                        caption_data = {
                            "label": r["label"],
                            "confidence": r["confidence"],
                        }
                    else:
                        _, overlay, measurements = measure_connected_regions(
                            analysis_image,
                            min_area=min_component_area,
                        )
                        line_variability = summarize_line_variability(measurements)
                        crop_model = active_models.get(MODEL_EFFNET)
                        if crop_model is None:
                            crop_error = tr(
                                "EfficientNet tidak tersedia; crop tidak dapat "
                                "diklasifikasikan."
                            )
                            crop_summary = {
                                "Jumlah crop dianalisis": 0,
                                "Komponen diprediksi Disleksia (%)": "Model tidak tersedia",
                                "Komponen diprediksi Normal (%)": "Model tidak tersedia",
                            }
                        else:
                            component_predictions, crop_previews, crop_summary = (
                                predict_component_crops(
                                    crop_model,
                                    analysis_image,
                                    measurements,
                                    threshold,
                                    on_crop_progress=None,
                                )
                            )
                        opencv_crop_summaries.append(
                            {"Nama File": bf.name, **crop_summary}
                        )
                        line_variability_records = line_variability.to_dict("records")
                        opencv_line_results.extend(
                            {
                                "Nama File": bf.name,
                                **line_summary,
                            }
                            for line_summary in line_variability_records
                        )
                        valid_line_indices = line_variability[
                            "Indeks variasi ukuran (%)"
                        ].dropna()
                        opencv_results.append({
                            "Nama File": bf.name,
                            "Jumlah Komponen": len(measurements),
                            "Jumlah Baris": len(line_variability),
                            "Rata-rata indeks variasi ukuran (%)": (
                                f"{valid_line_indices.mean():.2f}%"
                                if not valid_line_indices.empty
                                else "Data kurang"
                            ),
                            "Ukuran Gambar": f"{w}x{h}",
                            "Waktu Analisis": now,
                        })
                        batch_visual = overlay
                        component_measurements = measurements
                        caption_data = {"component_count": len(measurements)}

                    encoded_ok, encoded_image = cv2.imencode(
                        ".png",
                        cv2.cvtColor(batch_visual, cv2.COLOR_RGB2BGR),
                        png_encode_params,
                    )
                    if not encoded_ok:
                        raise RuntimeError(
                            tr("Gagal menyiapkan hasil gambar: {filename}").format(
                                filename=bf.name
                            )
                        )
                    output_views.append({
                        "analysis": model_name,
                        "filename": bf.name,
                        "caption_data": caption_data,
                        "gradcam_part_results": gradcam_part_results,
                        "image": encoded_image.tobytes(),
                        "is_measurement": component_measurements is not None,
                        "measurements": (
                            component_measurements.to_dict("records")
                            if component_measurements is not None
                            else None
                        ),
                        "line_variability": line_variability_records,
                        "crop_predictions": component_predictions,
                        "crop_previews": crop_previews,
                        "crop_summary": crop_summary,
                        "crop_error": crop_error,
                    })

                update_scanner_card(
                    sample_num / total_files,
                    bf.name,
                    sample_num,
                    total_files,
                    detail_text=tr(
                        "Memproses sampel {current} dari {total} pada seluruh model aktif..."
                    ).format(
                        current=sample_num,
                        total=total_files,
                    ),
                    tip_index=i,
                )

            update_scanner_card(
                1.0,
                batch_files[-1].name,
                total_files,
                total_files,
                detail_text=tr(
                    "Seluruh tahapan analisis gambar telah selesai diproses."
                ),
                tip_index=0,
                is_done=True,
                force=True,
            )

            st.session_state["batch_analysis_result"] = {
                "signature": batch_signature,
                "results": results,
                "opencv_results": opencv_results,
                "opencv_line_results": opencv_line_results,
                "opencv_crop_summaries": opencv_crop_summaries,
                "views": output_views,
                "tiles": tiles,
                "grid_png": None,
            }

    def get_or_build_combined_grid_png(result_dict: dict[str, Any]) -> bytes | None:
        if result_dict.get("grid_png") is not None:
            return result_dict["grid_png"]
        raw_tiles = result_dict.get("tiles") or []
        if not raw_tiles:
            return None
        num_tiles = len(raw_tiles)
        max_rows_per_col = 25
        if num_tiles <= max_rows_per_col:
            grid = np.vstack(raw_tiles)
        else:
            num_cols = (num_tiles + max_rows_per_col - 1) // max_rows_per_col
            rows_per_col = (num_tiles + num_cols - 1) // num_cols
            tile_h, tile_w = raw_tiles[0].shape[:2]
            blank_tile = np.full((tile_h, tile_w, 3), 230, dtype=np.uint8)
            col_strips = []
            for col_idx in range(num_cols):
                chunk = list(
                    raw_tiles[col_idx * rows_per_col : (col_idx + 1) * rows_per_col]
                )
                while len(chunk) < rows_per_col:
                    chunk.append(blank_tile)
                col_strips.append(np.vstack(chunk))
            grid = np.hstack(col_strips)
        ok, png = cv2.imencode(
            ".png",
            cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )
        if not ok:
            raise RuntimeError(
                tr("Gagal menyiapkan grid heatmap untuk diunduh.")
            )
        result_dict["grid_png"] = png.tobytes()
        return result_dict["grid_png"]

    batch_result = st.session_state.get("batch_analysis_result")
    if batch_files and batch_result and batch_result["signature"] == batch_signature:
        show_batch_comparison = len(batch_files) > 1
        hybrid_df = build_hybrid_conclusion_summary(
            batch_result["results"],
            batch_result["opencv_crop_summaries"],
            threshold,
        )
        if not hybrid_df.empty:
            st.markdown(
                f"## {tr('Kesimpulan Akhir & Akumulasi Skor Hybrid (EfficientNet-B0 + OpenCV)')}"
            )
            st.caption(
                tr(
                    "Akumulasi Skor Hybrid menggabungkan kedua model secara tertimbang "
                    "(70% Skor EfficientNet-B0 + 30% Skor OpenCV Crop) menjadi satu "
                    "keputusan akhir dengan kalibrasi confidence."
                )
            )
            st.markdown(
                """
<style>
@keyframes hybPulseRed {
    0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.55); transform: scale(1); }
    50% { box-shadow: 0 0 0 9px rgba(239, 68, 68, 0); transform: scale(1.12); }
}
@keyframes hybPulseGreen {
    0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.55); transform: scale(1); }
    50% { box-shadow: 0 0 0 9px rgba(16, 185, 129, 0); transform: scale(1.12); }
}
.hyb-hero-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
    gap: 14px;
    margin: 0.5rem 0 1.25rem;
}
.hyb-hero-card {
    position: relative;
    border-radius: 16px;
    padding: 1.05rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 14px;
    border: 1px solid;
    backdrop-filter: blur(8px);
}
.hyb-hero-card.dys {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.42), rgba(153, 27, 27, 0.22));
    border-color: rgba(248, 113, 113, 0.55);
}
.hyb-hero-card.norm {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.42), rgba(4, 120, 87, 0.22));
    border-color: rgba(52, 211, 153, 0.55);
}
.hyb-shape-orb {
    width: 44px;
    height: 44px;
    min-width: 44px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 1.15rem;
}
.hyb-hero-card.dys .hyb-shape-orb {
    background: rgba(239, 68, 68, 0.25);
    border: 2px solid #f87171;
    color: #fecaca;
    animation: hybPulseRed 2.1s infinite;
}
.hyb-hero-card.norm .hyb-shape-orb {
    background: rgba(16, 185, 129, 0.25);
    border: 2px solid #34d399;
    color: #a7f3d0;
    animation: hybPulseGreen 2.4s infinite;
}
.hyb-table-wrap {
    width: 100%;
    overflow-x: auto;
    border-radius: 14px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    background: rgba(15, 23, 42, 0.55);
    margin-bottom: 0.75rem;
}
.hyb-styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
}
.hyb-styled-table th {
    background: rgba(30, 41, 59, 0.9);
    color: #e2e8f0;
    font-weight: 700;
    text-align: left;
    padding: 11px 14px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.25);
    white-space: nowrap;
}
.hyb-styled-table td {
    padding: 10px 14px;
    color: #f1f5f9;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    vertical-align: middle;
}
.hyb-styled-table tr:last-child td {
    border-bottom: none;
}
.hyb-row-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 11px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.82rem;
    white-space: nowrap;
}
.hyb-row-pill.dys {
    background: rgba(239, 68, 68, 0.18);
    border: 1px solid rgba(248, 113, 113, 0.5);
    color: #fca5a5;
}
.hyb-row-pill.norm {
    background: rgba(16, 185, 129, 0.18);
    border: 1px solid rgba(52, 211, 153, 0.5);
    color: #6ee7b7;
}
.hyb-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
}
.hyb-dot.dys {
    background: #ef4444;
    animation: hybPulseRed 1.8s infinite;
}
.hyb-dot.norm {
    background: #10b981;
    animation: hybPulseGreen 2.2s infinite;
}
</style>
                """,
                unsafe_allow_html=True,
            )

            # Shape khusus di atas tabel hybrid jika gambar <= 10 (bukan mode list besar)
            if len(batch_files) <= 10:
                cards_html = ['<div class="hyb-hero-grid">']
                for _, hrow in hybrid_df.iterrows():
                    is_dys = hrow["Kesimpulan Akhir Hybrid"] == "Disleksia"
                    cls_mod = "dys" if is_dys else "norm"
                    orb_sym = "!" if is_dys else "✓"
                    title_lbl = tr("Disleksia") if is_dys else tr("Normal")
                    cards_html.append(
                        f'<div class="hyb-hero-card {cls_mod}">'
                        f'<div class="hyb-shape-orb">{orb_sym}</div>'
                        f'<div style="min-width:0;">'
                        f'<div style="font-size:0.76rem; color:#94a3b8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{hrow["Nama File"]}</div>'
                        f'<div style="font-size:1.08rem; font-weight:800; color:#ffffff; margin:2px 0;">{title_lbl} &bull; {hrow["Akumulasi Skor Hybrid (%)"]}</div>'
                        f'<div style="font-size:0.78rem; color:#cbd5e1;">EffNet: {hrow["Skor EfficientNet-B0 (%)"]} | OpenCV: {hrow["Skor OpenCV Crop (%)"]} | {tr(hrow["Kesepakatan Model"])}</div>'
                        f'</div></div>'
                    )
                cards_html.append('</div>')
                st.markdown("".join(cards_html), unsafe_allow_html=True)

            translated_hybrid_df = translate_dataframe(hybrid_df)

            def _build_animated_hybrid_html(df_in: pd.DataFrame) -> str:
                rows_html = [
                    '<div class="hyb-table-wrap"><table class="hyb-styled-table"><thead><tr>',
                    f'<th>{tr("Kesimpulan Akhir Hybrid")}</th>',
                    f'<th>{tr("Nama File")}</th>',
                    f'<th>{tr("Skor EfficientNet-B0 (%)")}</th>',
                    f'<th>{tr("Skor OpenCV Crop (%)")}</th>',
                    f'<th>{tr("Akumulasi Skor Hybrid (%)")}</th>',
                    f'<th>{tr("Kesepakatan Model")}</th>',
                    f'<th>{tr("Confidence Hybrid")}</th>',
                    '</tr></thead><tbody>',
                ]
                for _, r_in in df_in.iterrows():
                    raw_pred = str(r_in["Kesimpulan Akhir Hybrid"])
                    is_dys = raw_pred in ("Disleksia", "Dyslexic")
                    c_cls = "dys" if is_dys else "norm"
                    rows_html.append(
                        f'<tr>'
                        f'<td><span class="hyb-row-pill {c_cls}"><span class="hyb-dot {c_cls}"></span>{raw_pred}</span></td>'
                        f'<td style="font-weight:600;">{r_in["Nama File"]}</td>'
                        f'<td>{r_in["Skor EfficientNet-B0 (%)"]}</td>'
                        f'<td>{r_in["Skor OpenCV Crop (%)"]}</td>'
                        f'<td style="font-weight:700; color:#38bdf8;">{r_in["Akumulasi Skor Hybrid (%)"]}</td>'
                        f'<td>{r_in["Kesepakatan Model"]}</td>'
                        f'<td>{r_in["Confidence Hybrid"]}</td>'
                        f'</tr>'
                    )
                rows_html.append('</tbody></table></div>')
                return "".join(rows_html)

            if len(batch_files) > 10:
                with st.expander(
                    tr(
                        "Lihat tabel kesimpulan hybrid seluruh sampel ({count} sampel)"
                    ).format(count=len(batch_files)),
                    expanded=True,
                ):
                    st.markdown(
                        _build_animated_hybrid_html(translated_hybrid_df),
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    _build_animated_hybrid_html(translated_hybrid_df),
                    unsafe_allow_html=True,
                )
            render_table_download_row(
                translated_hybrid_df,
                file_stem="",
                base_name="kesimpulan_hybrid_gabungan",
                title=tr("Kesimpulan Akhir & Akumulasi Skor Hybrid (EfficientNet-B0 + OpenCV)"),
                key="hybrid_conclusion_top",
            )
        comparison_df = build_comparison_summary(
            batch_result["results"],
            batch_result["opencv_crop_summaries"],
        )
        effnet_views_by_file = {
            output["filename"]: output
            for output in batch_result["views"]
            if output["analysis"] == MODEL_EFFNET
        }
        opencv_views_by_file = {
            output["filename"]: output
            for output in batch_result["views"]
            if output["analysis"] == MODEL_MEASUREMENT
        }
        effnet_results_df = pd.DataFrame(batch_result["results"])
        opencv_summary_df = pd.DataFrame(batch_result["opencv_results"])
        crop_summary_df = pd.DataFrame(batch_result["opencv_crop_summaries"])
        if analysis_mode == ANALYSIS_BOTH and show_batch_comparison:
            st.markdown(
                f"""
<div style="margin: 1.5rem 0 0.75rem; padding: 0.9rem 1.2rem; border-radius: 14px; background: linear-gradient(90deg, rgba(30, 58, 138, 0.35), rgba(15, 23, 42, 0.6)); border-left: 4px solid #60a5fa;">
<div style="font-size: 1.15rem; font-weight: 800; color: #f8fafc;">{tr('Hasil Perbandingan Model EfficientNet dan OpenCV')}</div>
<div style="font-size: 0.82rem; color: #cbd5e1; margin-top: 2px;">{tr('Komparasi metrik mentah antar gambar antara klasifikasi EfficientNet-B0 utuh dan segmentasi komponen OpenCV.')}</div>
</div>
                """,
                unsafe_allow_html=True,
            )
            translated_comparison_df = translate_dataframe(comparison_df)
            if len(batch_files) > 10:
                with st.expander(
                    tr(
                        "Lihat tabel ringkasan perbandingan seluruh sampel ({count} sampel)"
                    ).format(count=len(batch_files)),
                    expanded=False,
                ):
                    st.table(
                        translated_comparison_df,
                        hide_index=True,
                    )
            else:
                st.table(
                    translated_comparison_df,
                    hide_index=True,
                )
            render_table_download_row(
                translated_comparison_df,
                file_stem="",
                base_name="ringkasan_perbandingan_efficientnet_opencv",
                title=tr("Hasil Perbandingan Model EfficientNet dan OpenCV"),
                key="comparison_summary",
            )

        effnet_rows_by_file = {
            filename: effnet_results_df[
                effnet_results_df["Nama File"] == filename
            ]
            for filename in effnet_results_df["Nama File"].unique()
        } if not effnet_results_df.empty else {}
        opencv_rows_by_file = {
            filename: opencv_summary_df[
                opencv_summary_df["Nama File"] == filename
            ]
            for filename in opencv_summary_df["Nama File"].unique()
        } if not opencv_summary_df.empty else {}
        crop_summary_by_file = {
            filename: crop_summary_df[
                crop_summary_df["Nama File"] == filename
            ]
            for filename in crop_summary_df["Nama File"].unique()
        } if not crop_summary_df.empty else {}

        def render_sample_detail_view(filename: str) -> None:
            file_stem = os.path.splitext(filename)[0]
            st.markdown(
                f"### {tr('Hasil gambar — {filename}').format(filename=filename)}"
            )

            effnet_output = effnet_views_by_file.get(filename)
            if run_effnet:
                st.markdown(f"#### {tr('Model EfficientNet-B0')}")
                effnet_rows = translate_dataframe(
                    effnet_rows_by_file.get(filename, pd.DataFrame())
                )
                st.table(
                    effnet_rows,
                    hide_index=True,
                )
                if batch_mode == BATCH_FULL and effnet_output is not None:
                    st.markdown(f"#### {tr('Grid heatmap dan panel hasil')}")
                    st.image(
                        effnet_output["image"],
                        caption=localized_result_caption(effnet_output),
                        use_container_width=True,
                    )
                    effnet_img_with_table = combine_image_with_table_png(
                        effnet_output["image"],
                        effnet_rows,
                        table_title=f"{tr('Model EfficientNet-B0')} — {filename}",
                    )
                    st.download_button(
                        tr("Unduh Gambar + Tabel Hasil"),
                        effnet_img_with_table,
                        file_name=f"{file_stem}_grid_panel_hasil.png",
                        mime="image/png",
                        key=f"download_effnet_grid_{filename}",
                        on_click="ignore",
                    )
                    gradcam_part_frame = pd.DataFrame(
                        effnet_output.get("gradcam_part_results", [])
                    )
                    if not gradcam_part_frame.empty:
                        st.markdown(
                            f"#### {tr('Skor Klasifikasi per Bagian Grad-CAM')}"
                        )
                        st.caption(
                            tr(
                                "Setiap bagian diklasifikasikan secara mandiri. "
                                "Prediksi gambar pada tabel di atas adalah rata-rata "
                                "tertimbang skor Disleksia dari seluruh bagian."
                            )
                        )
                        gradcam_part_frame = translate_dataframe(
                            gradcam_part_frame
                        )
                        st.table(
                            gradcam_part_frame,
                            hide_index=True,
                        )
                        render_table_download_row(
                            gradcam_part_frame,
                            file_stem=file_stem,
                            base_name="skor_gradcam_per_bagian",
                            title=f"{tr('Skor Klasifikasi per Bagian Grad-CAM')} — {filename}",
                            key=f"gradcam_parts_{filename}",
                        )

            opencv_output = opencv_views_by_file.get(filename)
            if run_opencv and opencv_output is not None:
                st.markdown(f"#### {tr('Deteksi OpenCV')}")
                st.markdown(f"##### {tr('Model Crop Komponen + EfficientNet')}")
                crop_summary_frame = translate_dataframe(
                    crop_summary_by_file.get(filename, pd.DataFrame())
                )
                st.table(
                    crop_summary_frame,
                    hide_index=True,
                )
                st.markdown(f"##### {tr('Komponen terdeteksi')}")
                st.image(
                    opencv_output["image"],
                    caption=localized_result_caption(opencv_output),
                    use_container_width=True,
                )
                opencv_img_with_table = combine_image_with_table_png(
                    opencv_output["image"],
                    crop_summary_frame,
                    table_title=f"{tr('Deteksi OpenCV')} — {filename}",
                )
                st.download_button(
                    tr("Unduh Gambar + Tabel Hasil"),
                    opencv_img_with_table,
                    file_name=f"{file_stem}_komponen_opencv_hasil.png",
                    mime="image/png",
                    key=f"download_opencv_image_{filename}",
                    on_click="ignore",
                )
                if opencv_output["crop_error"]:
                    st.warning(opencv_output["crop_error"])

                file_crop_details = pd.DataFrame(
                    opencv_output["crop_predictions"]
                )
                st.markdown(f"##### {tr('Rincian prediksi per huruf')}")
                crop_details_display = translate_dataframe(file_crop_details)
                st.table(
                    crop_details_display,
                    hide_index=True,
                )
                render_table_download_row(
                    crop_details_display,
                    file_stem=file_stem,
                    base_name="rincian_prediksi_crop",
                    title=f"{tr('Rincian Prediksi per Huruf')} — {filename}",
                    key=f"crop_details_{filename}",
                )
                file_measurements = pd.DataFrame(
                    opencv_output["measurements"] or []
                )
                file_line_variability = pd.DataFrame(
                    opencv_output["line_variability"] or []
                )
                if st.toggle(
                    tr("Skala Komponen & Variasi Huruf"),
                    value=False,
                    key=f"show_component_scale_v2_{filename}",
                ):
                    st.table(
                        translate_dataframe(file_measurements),
                        hide_index=True,
                    )
                    render_table_download_row(
                        translate_dataframe(file_measurements),
                        file_stem=file_stem,
                        base_name="skala_komponen",
                        title=f"{tr('Skala Komponen & Variasi Huruf')} — {filename}",
                        key=f"component_scale_{filename}",
                    )
                    st.markdown(f"##### {tr('Hasil pengukuran gambar')}")
                    st.table(
                        translate_dataframe(
                            opencv_rows_by_file.get(filename, pd.DataFrame())
                        ),
                        hide_index=True,
                    )
                    render_table_download_row(
                        translate_dataframe(
                            opencv_rows_by_file.get(filename, pd.DataFrame())
                        ),
                        file_stem=file_stem,
                        base_name="hasil_pengukuran",
                        title=f"{tr('Hasil Pengukuran Gambar OpenCV')} — {filename}",
                        key=f"opencv_summary_{filename}",
                    )
                    st.markdown(f"##### {tr('Variasi per baris')}")
                    st.table(
                        translate_dataframe(
                            file_line_variability.fillna("Data kurang")
                        ),
                        hide_index=True,
                    )
                    render_table_download_row(
                        translate_dataframe(
                            file_line_variability.fillna("Data kurang")
                        ),
                        file_stem=file_stem,
                        base_name="variasi_per_baris",
                        title=f"{tr('Variasi per Baris')} — {filename}",
                        key=f"line_variability_{filename}",
                    )

        def classify_sample_entry(filename: str) -> tuple[str, str]:
            """Return (category_key, score_summary) where category_key is 'Disleksia' or 'Non-Disleksia'."""
            eff_df = effnet_rows_by_file.get(filename)
            if run_effnet and eff_df is not None and not eff_df.empty:
                pred_val = str(eff_df.iloc[0]["Prediksi"])
                conf_val = str(eff_df.iloc[0]["Tingkat Kepercayaan"])
                category = "Disleksia" if pred_val == "Disleksia" else "Non-Disleksia"
                summary_str = f"EfficientNet: {tr(pred_val)} ({conf_val})"
                if run_opencv:
                    c_df = crop_summary_by_file.get(filename)
                    if c_df is not None and not c_df.empty:
                        dys_crop_pct = str(
                            c_df.iloc[0]["Komponen diprediksi Disleksia (%)"]
                        )
                        summary_str += f" | Crop Disleksia: {dys_crop_pct}"
                return category, summary_str

            ocv_view = opencv_views_by_file.get(filename)
            if ocv_view is not None:
                crop_preds = ocv_view.get("crop_predictions") or []
                total_c = len(crop_preds)
                dys_c = sum(
                    1
                    for cp in crop_preds
                    if cp.get("Prediksi eksperimen") == "Disleksia"
                )
                dys_ratio = (dys_c / total_c) if total_c > 0 else 0.0
                category = "Disleksia" if dys_ratio >= 0.5 else "Non-Disleksia"
                return (
                    category,
                    f"OpenCV Crop Disleksia: {dys_ratio * 100:.2f}% ({dys_c}/{total_c})",
                )
            return "Non-Disleksia", "-"

        if len(batch_files) <= 10:
            for batch_file in batch_files:
                render_sample_detail_view(batch_file.name)
        else:
            all_filenames = list(
                dict.fromkeys(batch_file.name for batch_file in batch_files)
            )
            sample_entries = []
            for idx, fname in enumerate(all_filenames, start=1):
                cat_key, score_info = classify_sample_entry(fname)
                sample_entries.append(
                    {
                        "index": idx,
                        "filename": fname,
                        "category": cat_key,
                        "score_info": score_info,
                    }
                )

            total_samples_count = len(sample_entries)
            dyslexia_samples_count = sum(
                1 for item in sample_entries if item["category"] == "Disleksia"
            )
            non_dyslexia_samples_count = (
                total_samples_count - dyslexia_samples_count
            )

            st.markdown(f"## {tr('Daftar & Filter Sampel Hasil Analisis')}")
            with st.container(border=True):
                st.caption(
                    tr(
                        "Menampilkan menu daftar sampel karena jumlah gambar lebih dari 10 ({count} sampel). Gunakan pencarian nama gambar dan filter kategori di bawah untuk melihat detail setiap sampel."
                    ).format(count=total_samples_count)
                )
                st.markdown(
                    f'<div class="sample-explorer-header">'
                    f'<div class="sample-stat-pills">'
                    f'<span class="sample-stat-pill">'
                    f'{tr("Total Sampel")}: {total_samples_count}</span>'
                    f'<span class="sample-stat-pill is-dyslexia">'
                    f'{tr("Sampel Disleksia")}: {dyslexia_samples_count}</span>'
                    f'<span class="sample-stat-pill is-normal">'
                    f'{tr("Sampel Non-Disleksia")}: {non_dyslexia_samples_count}</span>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                filter_col, search_col = st.columns([1.25, 1], gap="medium")
                with filter_col:
                    filter_options = ["Semua Sampel", "Disleksia", "Non-Disleksia"]
                    filter_counts = {
                        "Semua Sampel": total_samples_count,
                        "Disleksia": dyslexia_samples_count,
                        "Non-Disleksia": non_dyslexia_samples_count,
                    }
                    selected_category_filter = st.radio(
                        tr("Filter kategori sampel"),
                        filter_options,
                        horizontal=True,
                        key="sample_explorer_category_filter",
                        format_func=lambda opt: (
                            f"{tr(opt)} ({filter_counts.get(opt, 0)})"
                        ),
                    )
                with search_col:
                    search_filename_query = st.text_input(
                        tr("Cari berdasarkan nama gambar"),
                        value="",
                        placeholder=tr("Ketik nama gambar..."),
                        key="sample_explorer_search_query",
                    ).strip()

                filtered_entries = [
                    item
                    for item in sample_entries
                    if (
                        selected_category_filter == "Semua Sampel"
                        or item["category"] == selected_category_filter
                    )
                    and (
                        not search_filename_query
                        or search_filename_query.lower()
                        in item["filename"].lower()
                    )
                ]

                if not filtered_entries:
                    st.info(
                        tr(
                            "Tidak ada sampel yang cocok dengan pencarian atau filter yang dipilih."
                        )
                    )
                else:
                    filtered_summary_records = [
                        {
                            "No": item["index"],
                            "Nama File": item["filename"],
                            "Kategori": tr(item["category"]),
                            "Ringkasan Skor": item["score_info"],
                        }
                        for item in filtered_entries
                    ]
                    filtered_summary_df = pd.DataFrame(filtered_summary_records)
                    with st.expander(
                        tr(
                            "Lihat tabel daftar sampel terfilter ({count} sampel)"
                        ).format(count=len(filtered_entries)),
                        expanded=False,
                    ):
                        st.table(
                            translate_dataframe(filtered_summary_df),
                            hide_index=True,
                        )
                        render_table_download_row(
                            translate_dataframe(filtered_summary_df),
                            file_stem="",
                            base_name="daftar_sampel_terfilter",
                            title=tr("Daftar Sampel Terfilter"),
                            key="filtered_sample_list",
                        )

                    entry_by_filename = {
                        item["filename"]: item for item in filtered_entries
                    }
                    selected_sample_filename = st.selectbox(
                        tr(
                            "Pilih sampel dari daftar ({count} hasil cocok)"
                        ).format(count=len(filtered_entries)),
                        options=[item["filename"] for item in filtered_entries],
                        format_func=lambda fname: (
                            f"{entry_by_filename[fname]['index']}. {fname} — "
                            f"[{tr(entry_by_filename[fname]['category'])}] "
                            f"{entry_by_filename[fname]['score_info']}"
                        ),
                        key="sample_explorer_selected_filename",
                    )
                    st.markdown("---")
                    if selected_sample_filename:
                        render_sample_detail_view(selected_sample_filename)

        export_tables = build_export_tables(batch_result, comparison_df, hybrid_df)
        combined_grid_available = (
            show_batch_comparison
            and bool(batch_result.get("tiles") or batch_result.get("grid_png"))
            and batch_mode == BATCH_FULL
        )
        if export_tables or combined_grid_available:
            st.markdown(f"### {tr('Unduh Data Pilihan')}")
            export_selection_id = hashlib.sha256(
                repr(batch_result["signature"]).encode("utf-8")
            ).hexdigest()[:12]
            with st.expander(tr("Pilih tabel dan format")):
                selected_export_keys = []
                if export_tables:
                    export_format = st.radio(
                        tr("Format unduhan"),
                        [
                            "Excel (beberapa sheet)",
                            "ZIP (CSV per tabel)",
                            "ZIP (Gambar PNG per tabel)",
                        ],
                        horizontal=True,
                        key=f"batch_export_format_{export_selection_id}",
                        format_func=tr,
                    )
                    export_scope_options = [None] + list(
                        dict.fromkeys(batch_file.name for batch_file in batch_files)
                    )
                    selected_export_filename = st.selectbox(
                        tr("Cakupan data"),
                        export_scope_options,
                        format_func=lambda filename: (
                            tr("Semua gambar")
                            if filename is None
                            else filename
                        ),
                        key=f"batch_export_scope_{export_selection_id}",
                    )
                    st.caption(
                        tr("Cakupan ini berlaku untuk semua tabel yang dipilih.")
                    )
                    export_columns = st.columns(2)
                    with export_columns[0]:
                        st.markdown(f"#### {tr('EfficientNet')}")
                        for key in ("effnet_summary", "gradcam_parts"):
                            if key in export_tables and st.checkbox(
                                tr(EXPORT_DATASET_LABELS[key]),
                                value=False,
                                key=f"batch_export_{export_selection_id}_{key}",
                            ):
                                selected_export_keys.append(key)
                    with export_columns[1]:
                        st.markdown(f"#### {tr('OpenCV')}")
                        for key in (
                            "opencv_summary",
                            "opencv_crop_summary",
                            "opencv_crop_details",
                            "opencv_component_scale",
                            "opencv_line_variation",
                        ):
                            if key in export_tables and st.checkbox(
                                tr(EXPORT_DATASET_LABELS[key]),
                                value=False,
                                key=f"batch_export_{export_selection_id}_{key}",
                            ):
                                selected_export_keys.append(key)
                    if "hybrid_conclusion" in export_tables or "comparison" in export_tables:
                        st.markdown(f"#### {tr('Keseluruhan')}")
                        for key in ("hybrid_conclusion", "comparison"):
                            if key in export_tables and st.checkbox(
                                tr(EXPORT_DATASET_LABELS[key]),
                                value=False,
                                key=f"batch_export_{export_selection_id}_{key}",
                            ):
                                selected_export_keys.append(key)

                include_combined_grid = False
                if combined_grid_available:
                    st.markdown(f"#### {tr('Gambar perbandingan')}")
                    include_combined_grid = st.checkbox(
                        tr("Grid panel hasil gabungan"),
                        value=False,
                        key=f"batch_export_{export_selection_id}_combined_grid",
                    )
                    if (
                        export_tables
                        and selected_export_filename is not None
                    ):
                        st.caption(
                            tr("Grid panel gabungan selalu mencakup semua gambar.")
                        )

                if selected_export_keys:
                    selected_tables = {}
                    for key in selected_export_keys:
                        frame = export_tables[key]
                        if (
                            selected_export_filename is not None
                            and "Nama File" in frame.columns
                        ):
                            frame = frame[
                                frame["Nama File"] == selected_export_filename
                            ]
                        selected_tables[key] = translate_dataframe(frame)
                    if export_format == "Excel (beberapa sheet)":
                        export_data = build_excel_export(selected_tables)
                        export_filename = "hasil_analisis_pilihan.xlsx"
                        export_mime = (
                            "application/vnd.openxmlformats-officedocument."
                            "spreadsheetml.sheet"
                        )
                        download_label = tr("Unduh Excel")
                    elif export_format == "ZIP (Gambar PNG per tabel)":
                        export_data = build_png_export_zip(selected_tables)
                        export_filename = "tabel_gambar_pilihan.zip"
                        export_mime = "application/zip"
                        download_label = tr("Unduh ZIP (Gambar PNG)")
                    else:
                        export_data = build_csv_export_zip(selected_tables)
                        export_filename = "hasil_analisis_pilihan.zip"
                        export_mime = "application/zip"
                        download_label = tr("Unduh ZIP (CSV)")
                    st.download_button(
                        download_label,
                        export_data,
                        file_name=export_filename,
                        mime=export_mime,
                        key=f"download_selected_tables_{export_selection_id}",
                        on_click="ignore",
                    )
                if include_combined_grid:
                    combined_grid_bytes = get_or_build_combined_grid_png(
                        batch_result
                    )
                    if combined_grid_bytes is not None:
                        st.download_button(
                            tr("Unduh Grid Panel Gabungan"),
                            combined_grid_bytes,
                            file_name="grid_panel_hasil_gabungan.png",
                            mime="image/png",
                            key=f"download_combined_grid_{export_selection_id}",
                            on_click="ignore",
                        )
                if not selected_export_keys and not include_combined_grid:
                    st.caption(
                        tr("Pilih minimal satu data atau grid panel untuk diunduh.")
                    )

        st.markdown("---")
        st.markdown(f"### {tr('Bank Sampel & Kontribusi Dataset (Keberagaman Model)')}")
        auto_dir = os.path.join(COLLECTED_SAMPLES_DIR, "auto_disleksia")
        man_dys_dir = os.path.join(COLLECTED_SAMPLES_DIR, "manual_verified", "Disleksia")
        man_norm_dir = os.path.join(COLLECTED_SAMPLES_DIR, "manual_verified", "Normal")
        auto_cnt = len(os.listdir(auto_dir)) if os.path.exists(auto_dir) else 0
        man_cnt = (
            (len(os.listdir(man_dys_dir)) if os.path.exists(man_dys_dir) else 0)
            + (len(os.listdir(man_norm_dir)) if os.path.exists(man_norm_dir) else 0)
        )
        col_stat1, col_stat2 = st.columns(2)
        col_stat1.metric(tr("Auto-Collect (Terdeteksi Disleksia)"), f"{auto_cnt} {tr('gambar')}")
        col_stat2.metric(tr("Input Manual (Terverifikasi 2 Langkah)"), f"{man_cnt} {tr('gambar')}")

        if auto_cnt + man_cnt > 0:
            st.download_button(
                tr("Unduh Seluruh Bank Sampel (.zip)"),
                build_collected_samples_zip(),
                file_name="bank_sampel_dyslev_ai.zip",
                mime="application/zip",
                key="download_collected_samples_zip",
                on_click="ignore",
            )

        with st.expander(tr("Tambah Sampel Manual (Verifikasi 2 Langkah)"), expanded=False):
            if "manual_sample_step" not in st.session_state:
                st.session_state["manual_sample_step"] = 1
            if "sample_verify_code" not in st.session_state:
                st.session_state["sample_verify_code"] = hashlib.sha256(
                    str(time.time()).encode("utf-8")
                ).hexdigest()[:6].upper()
            if "sample_math_pair" not in st.session_state:
                st.session_state["sample_math_pair"] = (7, 5)

            if st.session_state["manual_sample_step"] == 1:
                st.markdown(f"#### {tr('Langkah 1: Validasi Identitas & Kelayakan Sampel')}")
                batch_name_map = {bf.name: bf for bf in batch_files}
                selected_src_name = st.selectbox(
                    tr("Pilih gambar dari hasil analisis saat ini"),
                    options=list(batch_name_map.keys()),
                    key="manual_sample_src_select",
                )
                manual_extra_upload = st.file_uploader(
                    tr("Atau unggah gambar sampel baru secara terpisah (opsional)"),
                    type=["png", "jpg", "jpeg"],
                    key="manual_sample_extra_upload",
                )
                target_label = st.radio(
                    tr("Label Validasi Sampel"),
                    ["Disleksia", "Normal"],
                    horizontal=True,
                    key="manual_sample_label",
                )
                chk_age = st.checkbox(
                    tr("Saya memastikan sampel ini adalah tulisan tangan anak usia 4–7 tahun."),
                    key="manual_sample_chk_age",
                )
                chk_quality = st.checkbox(
                    tr("Saya memastikan gambar jelas, tidak terpotong, dan layak untuk pelatihan model."),
                    key="manual_sample_chk_quality",
                )
                verifier_name = st.text_input(
                    tr("Nama Lengkap Verifikator / Peneliti (min. 3 karakter)"),
                    key="manual_sample_verifier",
                )
                obs_notes = st.text_area(
                    tr("Catatan Observasi Visual / Klinis (min. 10 karakter)"),
                    placeholder=tr("Contoh: Terdapat huruf b/d terbalik dan spasi antar kata tidak konsisten..."),
                    key="manual_sample_notes",
                )

                if st.button(tr("Lanjut ke Verifikasi Langkah 2"), type="primary", key="btn_sample_step1"):
                    if not (chk_age and chk_quality):
                        st.error(tr("Harap centang kedua pernyataan kelayakan sampel (usia 4–7 tahun & kualitas gambar)."))
                    elif len(verifier_name.strip()) < 3:
                        st.error(tr("Nama verifikator minimal 3 karakter."))
                    elif len(obs_notes.strip()) < 10:
                        st.error(tr("Catatan observasi minimal 10 karakter."))
                    else:
                        chosen_file = manual_extra_upload if manual_extra_upload is not None else batch_name_map[selected_src_name]
                        st.session_state["manual_sample_payload"] = {
                            "filename": chosen_file.name,
                            "bytes": chosen_file.getvalue(),
                            "label": target_label,
                            "verifier": verifier_name.strip(),
                            "notes": obs_notes.strip(),
                        }
                        st.session_state["sample_verify_code"] = hashlib.sha256(
                            f"{time.time()}_{chosen_file.name}".encode("utf-8")
                        ).hexdigest()[:6].upper()
                        a = (int(time.time() * 100) % 8) + 2
                        b = (int(time.time() * 10) % 7) + 3
                        st.session_state["sample_math_pair"] = (a, b)
                        st.session_state["manual_sample_step"] = 2
                        st.rerun()

            else:
                payload = st.session_state.get("manual_sample_payload", {})
                if not payload:
                    st.session_state["manual_sample_step"] = 1
                    st.rerun()
                st.markdown(f"#### {tr('Langkah 2: Verifikasi Keamanan & Konfirmasi Akhir')}")
                prev_col, info_col = st.columns([1, 2])
                with prev_col:
                    st.image(payload["bytes"], caption=payload["filename"], use_container_width=True)
                with info_col:
                    st.markdown(f"- **File:** `{payload['filename']}`")
                    st.markdown(f"- **Label Validasi:** `{payload['label']}`")
                    st.markdown(f"- **Verifikator:** `{payload['verifier']}`")
                    st.markdown(f"- **Observasi:** {payload['notes']}")

                v_code = st.session_state["sample_verify_code"]
                ma, mb = st.session_state["sample_math_pair"]
                st.info(tr("Kode Verifikasi Keamanan: **{code}** | Tantangan Hitung: **{a} + {b} = ?**").format(code=v_code, a=ma, b=mb))

                c1, c2 = st.columns(2)
                with c1:
                    inp_code = st.text_input(tr("Masukkan Kode Verifikasi (6 karakter)"), key="manual_sample_inp_code")
                with c2:
                    inp_math = st.text_input(tr("Jawaban Tantangan Hitung"), key="manual_sample_inp_math")
                inp_phrase = st.text_input(
                    tr("Ketik tepat kata berikut untuk konfirmasi akhir: KONFIRMASI SAMPEL"),
                    key="manual_sample_inp_phrase",
                )

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if st.button(tr("Kembali ke Langkah 1"), key="btn_sample_back"):
                        st.session_state["manual_sample_step"] = 1
                        st.rerun()
                with b_col2:
                    if st.button(tr("Simpan Sampel Terverifikasi"), type="primary", key="btn_sample_save"):
                        if inp_code.strip().upper() != v_code:
                            st.error(tr("Kode verifikasi 6 karakter tidak sesuai."))
                        elif inp_math.strip() != str(ma + mb):
                            st.error(tr("Jawaban tantangan hitung salah."))
                        elif inp_phrase.strip().upper() != "KONFIRMASI SAMPEL":
                            st.error(tr("Frasa konfirmasi harus persis: KONFIRMASI SAMPEL"))
                        else:
                            saved_p = save_manual_verified_sample(
                                payload["filename"],
                                payload["bytes"],
                                payload["label"],
                                payload["verifier"],
                                payload["notes"],
                            )
                            st.session_state["manual_sample_step"] = 1
                            st.session_state.pop("manual_sample_payload", None)
                            st.success(tr("Sampel berhasil diverifikasi 2 langkah dan disimpan ke bank dataset: {name}").format(name=os.path.basename(saved_p)))

render_research_footer()
