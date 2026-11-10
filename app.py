import base64
import hashlib
import io
import os
import time
import zipfile
from datetime import datetime
from typing import Any

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
    "Alat Deteksi Dysgraphia & Dyseleksia": "Dysgraphia & Dyslexia Detection Tool",
    "Model EfficientNet-B0 & OpenCV": "EfficientNet-B0 & OpenCV Models",
    "EfficientNet-B0 (Klasifikasi)": "EfficientNet-B0 (Classification)",
    "Aktif": "Active",
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
    "Unduh hasil EfficientNet — {filename} (CSV)": "Download EfficientNet results — {filename} (CSV)",
    "Unduh grid panel — {filename} (PNG)": "Download panel grid — {filename} (PNG)",
    "Unduh ringkasan perbandingan (CSV)": "Download comparison summary (CSV)",
    "Unduh gambar kedua alur (ZIP)": "Download images from both workflows (ZIP)",
    "Rincian crop (CSV)": "Crop Details (CSV)",
    "Skala komponen (CSV)": "Component Scale (CSV)",
    "Variasi per baris (CSV)": "Variation by Line (CSV)",
    "Pengukuran gambar (CSV)": "Image Measurements (CSV)",
    "Ringkasan crop (CSV)": "Crop Summary (CSV)",
    "Gambar komponen (PNG)": "Component Image (PNG)",
    "Komponen terdeteksi": "Detected Components",
    "Rincian prediksi per huruf": "Per-Component Prediction Details",
    "Skala Komponen & Variasi Huruf": "Component Scale & Letter Variation",
    "Hasil pengukuran gambar": "Image Measurement Results",
    "Variasi per baris": "Variation by Line",
    "Analisis Tulisan Tangan": "Handwriting Analysis",
    "Unduh semua crop — {filename} (ZIP)": "Download all crops — {filename} (ZIP)",
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
    "Aplikasi memiliki dua alur eksperimen: EfficientNet-B0 mengklasifikasikan gambar utuh, sedangkan OpenCV memotong komponen tinta lalu mengirim setiap crop ke EfficientNet-B0 yang sama. OpenCV bukan model AI terpisah.": (
        "The app offers two experimental workflows: EfficientNet-B0 classifies "
        "the full image, while OpenCV crops ink components and sends each crop to "
        "the same EfficientNet-B0 model. OpenCV is not a separate AI model."
    ),
    "- **EfficientNet saja:** klasifikasi gambar tulisan secara utuh.\n- **OpenCV saja:** deteksi komponen, crop, lalu klasifikasi crop dengan EfficientNet.\n- **Keduanya:** jalankan dua alur dan bandingkan hasilnya.\n- Skor pada crop bersifat eksperimen dan bukan peluang diagnosis; komponen tinta belum tentu satu huruf.": (
        "- **EfficientNet only:** classify the full handwriting image.\n"
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
}

st.session_state.setdefault("ui_language", "Indonesia")
UI_LANGUAGE = st.session_state["ui_language"]


def tr(text: str) -> str:
    if UI_LANGUAGE == "English":
        return ENGLISH_TRANSLATIONS.get(text, text)
    return text


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
    }
    /* Memaksa background Streamlit menyesuaikan tema gelap agar lebih menyatu (opsional, tapi membuat border lebih baik) */
    .hero-container {
        position: relative;
        background-color: #06141a; /* Dark navy/teal seperti referensi */
        background-image: 
            radial-gradient(circle at 10% 30%, rgba(0, 223, 154, 0.15) 0%, transparent 50%),
            linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px);
        background-size: 100% 100%, 40px 40px, 40px 40px;
        padding: 5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    .top-pill {
        display: inline-block;
        background: rgba(0, 223, 154, 0.1);
        color: #00df9a;
        padding: 6px 18px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 2rem;
        border: 1px solid rgba(0, 223, 154, 0.2);
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
        color: #00df9a; /* Warna aksen Mint Green */
    }
    
    .hero-subtitle {
        font-weight: 400;
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.7);
        max-width: 800px;
        margin: 0 auto 2.5rem auto;
        line-height: 1.6;
    }
    
    .hero-features {
        display: flex;
        justify-content: center;
        gap: 16px;
        flex-wrap: wrap;
        max-width: 700px;
        margin: 0 auto;
    }
    
    .feature-pill {
        background: transparent;
        color: rgba(255, 255, 255, 0.85);
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 0.95rem;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: all 0.2s;
    }
    
    .feature-pill:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.3);
    }
    
    /* Styling result box agar tetap rapi jika di bawah hero */
    .result-box {
        padding: 24px;
        border-radius: 16px;
        background-color: white;
        color: black;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-left: 6px solid #4f46e5;
    }
    .result-box h3, .result-box p {
        color: black;
    }
    
    /* === CSS Untuk Section Fitur AI === */
    .features-section {
        padding: 4rem 1rem 1rem 1rem;
        text-align: center;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 3rem;
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
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.1rem;
        margin-bottom: 3.5rem;
    }
    .features-section-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: clamp(1.55rem, 3vw, 2rem);
        color: #00d9a0;
        line-height: 1.35;
        margin: 0 auto 1rem !important;
        letter-spacing: 0;
        max-width: 900px;
        text-align: center;
    }
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 1.5rem;
        max-width: 950px;
        margin: 0 auto;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        text-align: left;
        transition: all 0.3s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.15);
    }
    .feature-icon {
        background-color: rgba(0, 223, 154, 0.1);
        color: #00df9a;
        width: 52px;
        height: 52px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0, 223, 154, 0.2);
    }
    .feature-card-title {
        font-weight: 700;
        font-size: 1.15rem;
        color: #ffffff;
        margin-bottom: 0.75rem;
    }
    .feature-card-desc {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.95rem;
        line-height: 1.6;
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
    @media (max-width: 640px) {
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
            max-width: 90px !important;
            height: 48px !important;
        }
        .hero-container > div[style*="display:flex"] {
            gap: 8px !important;
            padding: 8px 14px !important;
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
ANALYSIS_TYPES = [MODEL_EFFNET, MODEL_MEASUREMENT]
ANALYSIS_EFFNET_ONLY = "EfficientNet saja"
ANALYSIS_OPENCV_ONLY = "OpenCV saja (crop + EfficientNet)"
ANALYSIS_BOTH = "Keduanya"

with st.popover(
    f"⚙️ {tr('Pengaturan')}",
    use_container_width=False,
    help=tr("Atur model dan parameter analisis."),
):
    st.radio(
        tr("Bahasa / Language"),
        ["Indonesia", "English"],
        horizontal=True,
        key="ui_language",
    )
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
    st.markdown("---")
    st.markdown(f"**{tr('Status Komponen')}**")
    eff_ok = any(os.path.exists(path) for path in TFLITE_CANDIDATES)
    st.markdown(f"{'🟢' if eff_ok else '🔴'} {tr('EfficientNet-B0 (Klasifikasi)')} (.tflite)")
    st.markdown(f"🟢 {tr(MODEL_MEASUREMENT)}")

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
{tr("Alat Deteksi Dysgraphia & Dyseleksia")} <br>
<span>{tr("Model EfficientNet-B0 & OpenCV")}</span>
</div>
<div class="hero-subtitle">
{tr("Analisis pola gambar tulisan tangan tanpa ribet teknis. Cocok untuk guru, orang tua, hingga terapis. Dilengkapi berbagai fitur pendukung:")}
</div>
<div class="hero-features">
<span class="feature-pill">✓ {tr("Pemrosesan Banyak Gambar")}</span>
<span class="feature-pill">✓ {tr("Ekspor Hasil CSV")}</span>
<span class="feature-pill">✓ {tr("Visualisasi Grad-CAM")}</span>
<span class="feature-pill">✓ {tr("Eksperimen Crop OpenCV")}</span>
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
            "gambar utuh, sedangkan OpenCV memotong komponen tinta lalu mengirim "
            "setiap crop ke EfficientNet-B0 yang sama. OpenCV bukan model AI terpisah."
        )
    )
    st.markdown(
        tr(
            "- **EfficientNet saja:** klasifikasi gambar tulisan secara utuh.\n"
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
            interp = tf.lite.Interpreter(model_path=path)
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
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    interpreter.set_tensor(inp[0]['index'], input_data)
    interpreter.invoke()
    preds = interpreter.get_tensor(out[0]['index'])[0]
    raw = float(preds[0])  # indeks 0 = Disleksia
    if raw > threshold:
        label, conf = "Disleksia", raw
    else:
        label, conf = "Normal", float(preds[1]) if len(preds) > 1 else 1 - raw
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    predictions = []
    crop_previews = []
    for component in crop_connected_regions(image, measurements):
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
        encoded_ok, encoded_crop = cv2.imencode(
            ".png",
            cv2.cvtColor(np.asarray(component["crop"]), cv2.COLOR_RGB2BGR),
        )
        if not encoded_ok:
            raise RuntimeError(
                f"Gagal menyiapkan crop komponen {component['Komponen']}."
            )
        crop_previews.append(
            {
                "Komponen": component["Komponen"],
                "Baris": component["Baris"],
                "Prediksi": result["label"],
                "Skor": f"{result['confidence'] * 100:.2f}%",
                "image": encoded_crop.tobytes(),
            }
        )
    total_crops = len(predictions)
    dyslexia_count = sum(
        prediction["Prediksi eksperimen"] == "Disleksia"
        for prediction in predictions
    )
    normal_count = total_crops - dyslexia_count
    summary = {
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


def build_comparison_images_zip(images: list[tuple[str, bytes]]) -> bytes:
    """Package named output images into a downloadable ZIP archive."""
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, image_bytes in images:
            archive.writestr(name, image_bytes)
    return archive_buffer.getvalue()


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
    hm = make_gradcam_heatmap(r["input"], keras_model, "top_activation")
    col = (239, 68, 68) if r["label"] == "Disleksia" else (34, 197, 94)
    return make_row(make_heat_panels(image, hm), filename,
                    f"{r['label']} {r['confidence'] * 100:.1f}%", col)


model_rows = []
for name in ANALYSIS_TYPES:
    model_rows.append(
        f"""
        <tr style="background:#ffffff; color:#172b2a;">
          <td style="padding:10px 14px; border-top:1px solid #e7eef0; font-weight:600;">
            {tr(name)}
          </td>
          <td style="padding:10px 14px; border-top:1px solid #e7eef0;">
            <span style="font-size:0.75rem; font-weight:700; color:#15803d;">{tr("Aktif")}</span>
          </td>
        </tr>
        """
    )

st.html(
    f"""
    <div style="max-width:640px; overflow:hidden; margin:0.35rem 0 0.9rem;
                border:1px solid #dbe4e8; border-radius:14px;
                box-shadow:0 4px 14px rgba(15,23,42,0.07);">
      <table style="width:100%; border-collapse:collapse; font-size:0.84rem;">
        <thead>
          <tr style="background:linear-gradient(110deg,#102b35,#164e49); color:#f8fafc;">
            <th style="padding:10px 14px; text-align:left; font-size:0.7rem;
                       font-weight:700; letter-spacing:0.08em; text-transform:uppercase;">
              {tr("Jenis Analisis Model")}
            </th>
            <th style="width:130px; padding:10px 14px; text-align:left; font-size:0.7rem;
                       font-weight:700; letter-spacing:0.08em; text-transform:uppercase;">
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
        generate_batch = False
        if batch_files:
            generate_batch = st.button(
                tr("Mulai Analisis"),
                type="primary",
                key="generate_multiple_processing",
            )
    batch_signature = (
        (
            tuple(
                (bf.name, hashlib.sha256(bf.getvalue()).hexdigest())
                for bf in batch_files
            ),
            analysis_mode,
            batch_mode,
            threshold,
            min_component_area,
            enhance_image_contrast,
            CROP_AGGREGATION_VERSION,
        )
        if batch_files
        else None
    )

    if batch_files and generate_batch:
        active_models = get_active_analyses()
        if active_models:
            results = []
            opencv_results = []
            opencv_line_results = []
            opencv_crop_summaries = []
            tiles = []
            output_views = []
            keras_model = None
            gradcam_load_attempted = False
            selected_operation_count = int(run_effnet) + int(run_opencv)
            total_operations = len(batch_files) * selected_operation_count
            completed_operations = 0
            progress = st.progress(0.0, text=tr("Memulai analisis..."))

            for i, bf in enumerate(batch_files):
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
                    component_predictions = []
                    crop_previews = []
                    crop_summary = None
                    crop_error = None
                    if model_name == MODEL_EFFNET:
                        r = predict_efficientnet(model, analysis_image, threshold)
                        lvl, _ = confidence_level(r["confidence"])
                        results.append({
                            "Nama File": bf.name,
                            "Model": model_name,
                            "Status": "🔴" if r["label"] == "Disleksia" else "🟢",
                            "Prediksi": r["label"],
                            "Tingkat Kepercayaan": f"{r['confidence'] * 100:.2f}%",
                            "Level Confidence": lvl,
                            "Raw Score (Disleksia)": f"{r['raw']:.6f}",
                            "Ukuran Gambar": f"{w}x{h}",
                            "Inference Time (s)": f"{r['time']:.4f}",
                            "Waktu Analisis": now,
                        })
                        caption_data = {
                            "label": r["label"],
                            "confidence": r["confidence"],
                        }
                        if batch_mode == BATCH_FULL:
                            if not gradcam_load_attempted:
                                keras_model = load_keras_model()
                                gradcam_load_attempted = True
                                if keras_model is None:
                                    st.warning(
                                        tr("Model .h5 tidak tersedia, heatmap dilewati.")
                                    )
                            if keras_model is not None:
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
                                )
                            )
                        opencv_crop_summaries.append(
                            {"Nama File": bf.name, **crop_summary}
                        )
                        opencv_line_results.extend(
                            {
                                "Nama File": bf.name,
                                **line_summary,
                            }
                            for line_summary in line_variability.to_dict("records")
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

                    completed_operations += 1
                    progress.progress(
                        completed_operations / total_operations,
                        text=f"{tr(model_name)}: {bf.name}",
                    )
                    encoded_ok, encoded_image = cv2.imencode(
                        ".png",
                        cv2.cvtColor(batch_visual, cv2.COLOR_RGB2BGR),
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
                        "image": encoded_image.tobytes(),
                        "is_measurement": component_measurements is not None,
                        "measurements": (
                            component_measurements.to_dict("records")
                            if component_measurements is not None
                            else None
                        ),
                        "line_variability": (
                            summarize_line_variability(
                                component_measurements
                            ).to_dict("records")
                            if component_measurements is not None
                            else None
                        ),
                        "crop_predictions": component_predictions,
                        "crop_previews": crop_previews,
                        "crop_summary": crop_summary,
                        "crop_error": crop_error,
                        "filename": bf.name,
                    })
            progress.progress(1.0, text=tr("Selesai"))

            grid_png = None
            if tiles:
                grid = np.vstack(tiles)
                ok, png = cv2.imencode(".png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError(
                        tr("Gagal menyiapkan grid heatmap untuk diunduh.")
                    )
                grid_png = png.tobytes()

            st.session_state["batch_analysis_result"] = {
                "signature": batch_signature,
                "results": results,
                "opencv_results": opencv_results,
                "opencv_line_results": opencv_line_results,
                "opencv_crop_summaries": opencv_crop_summaries,
                "views": output_views,
                "grid_png": grid_png,
            }

    batch_result = st.session_state.get("batch_analysis_result")
    if batch_files and batch_result and batch_result["signature"] == batch_signature:
        show_batch_comparison = len(batch_files) > 1
        comparison_df = build_comparison_summary(
            batch_result["results"],
            batch_result["opencv_crop_summaries"],
        )
        effnet_views = [
            output
            for output in batch_result["views"]
            if output["analysis"] == MODEL_EFFNET
        ]
        opencv_views = [
            output
            for output in batch_result["views"]
            if output["analysis"] == MODEL_MEASUREMENT
        ]
        comparison_images = [
            (
                f"EfficientNet/{os.path.splitext(output['filename'])[0]}_hasil.png",
                output["image"],
            )
            for output in effnet_views
        ] + [
            (
                f"OpenCV/{os.path.splitext(output['filename'])[0]}_komponen.png",
                output["image"],
            )
            for output in opencv_views
        ]
        effnet_results_df = pd.DataFrame(batch_result["results"])
        opencv_summary_df = pd.DataFrame(batch_result["opencv_results"])
        crop_summary_df = pd.DataFrame(batch_result["opencv_crop_summaries"])
        if (
            show_batch_comparison
            and batch_result["grid_png"] is not None
            and batch_mode == BATCH_FULL
        ):
            st.markdown(f"#### {tr('Grid panel hasil gabungan')}")
            st.image(
                batch_result["grid_png"],
                caption=tr("Panel gabungan hasil EfficientNet dan Grad-CAM"),
                use_container_width=True,
            )
        if analysis_mode == ANALYSIS_BOTH and show_batch_comparison:
            st.markdown(
                f"## {tr('Hasil Perbandingan Model EfficientNet dan OpenCV')}"
            )
            st.dataframe(
                translate_dataframe(comparison_df),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                tr(
                    "Perbandingan ini menampilkan klasifikasi EfficientNet pada gambar "
                    "utuh dan proporsi label Disleksia/Normal dari crop OpenCV yang "
                    "diklasifikasikan EfficientNet. Persentase crop adalah proporsi "
                    "label komponen, bukan kemungkinan diagnosis; segmentasi komponen "
                    "belum tentu memotong huruf secara tepat."
                )
            )
            comparison_downloads = st.columns(2)
            comparison_downloads[0].download_button(
                tr("Unduh ringkasan perbandingan (CSV)"),
                translate_dataframe(comparison_df).to_csv(index=False).encode("utf-8"),
                file_name="ringkasan_perbandingan_efficientnet_opencv.csv",
                mime="text/csv",
                on_click="ignore",
            )
            comparison_downloads[1].download_button(
                tr("Unduh gambar kedua alur (ZIP)"),
                build_comparison_images_zip(comparison_images),
                file_name="gambar_perbandingan_efficientnet_opencv.zip",
                mime="application/zip",
                on_click="ignore",
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

        for batch_file in batch_files:
            filename = batch_file.name
            file_stem = os.path.splitext(filename)[0]
            st.markdown(
                tr("Hasil gambar — {filename}").format(filename=filename)
            )

            effnet_output = next(
                (
                    output
                    for output in effnet_views
                    if output["filename"] == filename
                ),
                None,
            )
            if run_effnet:
                st.markdown(f"### {tr('Model EfficientNet-B0')}")
                effnet_rows = translate_dataframe(
                    effnet_rows_by_file.get(filename, pd.DataFrame())
                )
                st.dataframe(
                    effnet_rows,
                    use_container_width=True,
                    hide_index=True,
                )
                if batch_mode == BATCH_FULL and effnet_output is not None:
                    st.markdown(f"#### {tr('Grid heatmap dan panel hasil')}")
                    st.image(
                        effnet_output["image"],
                        caption=localized_result_caption(effnet_output),
                        use_container_width=True,
                    )
                effnet_downloads = st.columns(2)
                effnet_downloads[0].download_button(
                    tr("Unduh hasil EfficientNet — {filename} (CSV)").format(
                        filename=filename
                    ),
                    effnet_rows.to_csv(index=False).encode("utf-8"),
                    file_name=f"{file_stem}_hasil_efficientnet.csv",
                    mime="text/csv",
                    key=f"download_effnet_{filename}",
                    on_click="ignore",
                )
                if batch_mode == BATCH_FULL and effnet_output is not None:
                    effnet_downloads[1].download_button(
                        tr("Unduh grid panel — {filename} (PNG)").format(
                            filename=filename
                        ),
                        effnet_output["image"],
                        file_name=f"{file_stem}_grid_panel.png",
                        mime="image/png",
                        key=f"download_effnet_grid_{filename}",
                        on_click="ignore",
                    )

            opencv_output = next(
                (
                    output
                    for output in opencv_views
                    if output["filename"] == filename
                ),
                None,
            )
            if run_opencv and opencv_output is not None:
                st.markdown(f"### {tr('Deteksi OpenCV')}")
                st.markdown(f"#### {tr('Model Crop Komponen + EfficientNet')}")
                st.caption(
                    tr(
                        "Persentase merupakan proporsi crop yang diprediksi Disleksia "
                        "atau Normal pada gambar ini, bukan peluang diagnosis. Crop "
                        "belum tentu satu huruf; model EfficientNet dilatih untuk gambar "
                        "utuh sehingga hasil crop masih eksperimen."
                    )
                )
                crop_summary_frame = translate_dataframe(
                    crop_summary_by_file.get(filename, pd.DataFrame())
                )
                st.dataframe(
                    crop_summary_frame,
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown(f"#### {tr('Komponen terdeteksi')}")
                st.image(
                    opencv_output["image"],
                    caption=localized_result_caption(opencv_output),
                    use_container_width=True,
                )
                if opencv_output["crop_error"]:
                    st.warning(opencv_output["crop_error"])

                file_crop_details = pd.DataFrame(
                    opencv_output["crop_predictions"]
                )
                st.markdown(f"#### {tr('Rincian prediksi per huruf')}")
                crop_details_display = translate_dataframe(file_crop_details)
                st.dataframe(
                    crop_details_display,
                    use_container_width=True,
                    hide_index=True,
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
                    st.dataframe(
                        translate_dataframe(file_measurements),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.markdown(f"#### {tr('Hasil pengukuran gambar')}")
                    st.dataframe(
                        translate_dataframe(
                            opencv_rows_by_file.get(filename, pd.DataFrame())
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.markdown(f"#### {tr('Variasi per baris')}")
                    st.dataframe(
                        translate_dataframe(
                            file_line_variability.fillna("Data kurang")
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                file_crops_zip = build_comparison_images_zip(
                    [
                        (f"crop_{crop['Komponen']}.png", crop["image"])
                        for crop in opencv_output["crop_previews"]
                    ]
                )
                opencv_downloads = st.columns(6)
                opencv_downloads[0].download_button(
                    tr("Rincian crop (CSV)"),
                    crop_details_display.to_csv(index=False).encode("utf-8"),
                    file_name=f"{file_stem}_rincian_prediksi_crop.csv",
                    mime="text/csv",
                    key=f"download_crop_details_{filename}",
                    on_click="ignore",
                )
                opencv_downloads[1].download_button(
                    tr("Skala komponen (CSV)"),
                    translate_dataframe(file_measurements).to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name=f"{file_stem}_skala_komponen.csv",
                    mime="text/csv",
                    key=f"download_component_scale_{filename}",
                    on_click="ignore",
                )
                opencv_downloads[2].download_button(
                    tr("Variasi per baris (CSV)"),
                    translate_dataframe(file_line_variability).to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name=f"{file_stem}_variasi_per_baris.csv",
                    mime="text/csv",
                    key=f"download_line_variability_{filename}",
                    on_click="ignore",
                )
                opencv_downloads[3].download_button(
                    tr("Pengukuran gambar (CSV)"),
                    translate_dataframe(
                        opencv_rows_by_file.get(filename, pd.DataFrame())
                    ).to_csv(index=False).encode("utf-8"),
                    file_name=f"{file_stem}_hasil_pengukuran.csv",
                    mime="text/csv",
                    key=f"download_opencv_summary_{filename}",
                    on_click="ignore",
                )
                opencv_downloads[4].download_button(
                    tr("Ringkasan crop (CSV)"),
                    crop_summary_frame.to_csv(index=False).encode("utf-8"),
                    file_name=f"{file_stem}_ringkasan_crop.csv",
                    mime="text/csv",
                    key=f"download_crop_summary_{filename}",
                    on_click="ignore",
                )
                opencv_downloads[5].download_button(
                    "Gambar komponen (PNG)",
                    opencv_output["image"],
                    file_name=f"{file_stem}_komponen_opencv.png",
                    mime="image/png",
                    key=f"download_opencv_image_{filename}",
                    on_click="ignore",
                )
                if opencv_output["crop_previews"]:
                    st.download_button(
                        tr("Unduh semua crop — {filename} (ZIP)").format(
                            filename=filename
                        ),
                        file_crops_zip,
                        file_name=f"{file_stem}_crop_opencv.zip",
                        mime="application/zip",
                        key=f"download_opencv_crops_{filename}",
                        on_click="ignore",
                    )
