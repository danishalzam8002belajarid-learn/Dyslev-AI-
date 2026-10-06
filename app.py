import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import os
import pandas as pd
import time
from datetime import datetime

from grad_cam import make_gradcam_heatmap, overlay_heatmap

# Setup Halaman
st.set_page_config(
    page_title="Dyslexia Analysis AI",
    page_icon="🧠",
    layout="wide"
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
</style>
""", unsafe_allow_html=True)

import base64
@st.cache_data
def get_base64_logos():
    # Memaksa reset cache agar logo baru terbaca
    try:
        import cv2
        logos = []
        for i in range(1, 5):
            path = os.path.join(os.getcwd(), "logo", f"{i}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    h, w = img.shape[:2]
                    new_h = 120
                    new_w = int((new_h / h) * w)
                    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    _, buffer = cv2.imencode('.png', resized)
                    logos.append(f"data:image/png;base64,{base64.b64encode(buffer).decode()}")
        return logos
    except Exception as e:
        return []

logos = get_base64_logos()
logo_tags = "".join([f'<img src="{l}" style="height: 85px; object-fit: contain;">' for l in logos])

if logos:
    logo_container = f"""<div style="display: inline-flex; justify-content: center; align-items: center; gap: 35px; margin-bottom: 2.5rem; background: white; padding: 16px 45px; border-radius: 60px; box-shadow: 0 10px 25px rgba(0,0,0,0.2);">{logo_tags}</div><br>"""
else:
    logo_container = ""

# Tampilan Awal (Hero Section) persis seperti referensi GipsyAI
st.markdown(f"""
<div class="hero-container">
{logo_container}
<div class="hero-title">
Dyslev AI (Dyslexia Evaluation & Vision Analysis) <br>
<span>Alat Deteksi Dini Disleksia Berbasis AI</span>
</div>
<div class="hero-subtitle">
Analisis pola gambar tulisan tangan tanpa ribet teknis. Cocok untuk guru, orang tua, hingga terapis. Dilengkapi berbagai fitur pendukung:
</div>
<div class="hero-features">
<span class="feature-pill">✓ Batch Processing</span>
<span class="feature-pill">✓ Ekspor Hasil CSV</span>
<span class="feature-pill">✓ Grad-CAM Visualizer</span>
<span class="feature-pill">✓ Offline TFLite Inference</span>
</div>

<div class="features-section">
<h2 class="features-title">Menjadi Alat Deteksi Dini Disleksia Berbasis AI yang <br><span>Cepat & Akurat</span></h2>
<p class="features-subtitle">Diwujudkan melalui berbagai fitur unggulan untuk membantu analisis Anda.</p>
<div class="features-grid">
<div class="feature-card">
<div class="feature-icon">🧠</div>
<div class="feature-card-title">Model AI Teroptimasi</div>
<div class="feature-card-desc">Didukung oleh arsitektur EfficientNet-B0 yang ringan, menghadirkan presisi tinggi dengan kecepatan komputasi maksimal meskipun dijalankan secara offline.</div>
</div>
<div class="feature-card">
<div class="feature-icon">👁️</div>
<div class="feature-card-title">Visualisasi Grad-CAM</div>
<div class="feature-card-desc">Tidak hanya memberikan skor, tapi juga menyorot area visual spesifik pada tulisan yang menjadi indikasi utama pola disleksia (reversal).</div>
</div>
<div class="feature-card">
<div class="feature-icon">📊</div>
<div class="feature-card-title">Analisis Batch Cepat</div>
<div class="feature-card-desc">Mendukung pemrosesan puluhan gambar sekaligus dalam satu waktu, mempermudah pengujian data dan riset dalam skala besar secara instan.</div>
</div>
<div class="feature-card">
<div class="feature-icon">📑</div>
<div class="feature-card-title">Laporan & Ekspor Data</div>
<div class="feature-card-desc">Dapatkan hasil analisis lengkap beserta metrik raw score dan confidence level yang siap diekspor ke format CSV untuk kebutuhan riset lanjutan.</div>
</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# Cache loading model agar tidak di-load berulang kali
@st.cache_resource
def load_models():
    # Load TFLite Model (Untuk Prediksi Utama)
    tflite_model_path = "dyslexia_model_v2.tflite"
    keras_model_path = "dyslexia_model.h5"
    
    interpreter = None
    keras_model = None
    
    if os.path.exists(tflite_model_path):
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
    
    if os.path.exists(keras_model_path):
        keras_model = tf.keras.models.load_model(keras_model_path)
        
    return interpreter, keras_model

interpreter, keras_model = load_models()

# Mapping Kelas
CLASS_NAMES = ["Disleksia", "Normal"] # Sesuaikan dengan mapping train_and_convert.py alfabetis

st.markdown("---")
with st.expander("📂 Batch Processing - Analisis Multiple Gambar"):
    st.markdown("""
    **Fitur Batch Processing memungkinkan Anda:**
    - ✅ Upload multiple gambar sekaligus (maksimal 20 file)
    - ✅ Dapatkan hasil analisis dalam tabel terstruktur
    - ✅ Ekspor hasil ke format CSV
    - ✅ Bandingkan hasil across multiple samples
    - ✅ Statistik summary untuk analisis cepat
    """)
    
    batch_files = st.file_uploader("Pilih multiple gambar untuk analisis batch", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch")
    
    if batch_files:
        if st.button("Jalankan Batch Analisis", type="primary"):
            if interpreter is None:
                st.error("Model TFLite tidak ditemukan! Pastikan telah menjalankan script train_and_convert.py")
            else:
                with st.spinner("Memproses gambar batch..."):
                    results = []
                    for bf in batch_files:
                        start_time = time.time()
                        
                        # Membaca gambar
                        image = Image.open(bf).convert('RGB')
                        width, height = image.size
                        img_resized = image.resize((224, 224))
                        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                        img_array_expanded = np.expand_dims(img_array, axis=0) / 255.0
                        
                        input_details = interpreter.get_input_details()
                        output_details = interpreter.get_output_details()
                        input_data = np.array(img_array_expanded, dtype=np.float32)
                        
                        interpreter.set_tensor(input_details[0]['index'], input_data)
                        interpreter.invoke()
                        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
                        
                        inference_time = time.time() - start_time
                        
                        # Thresholding
                        threshold_dyslexia = 0.6
                        raw_prediction = float(predictions[0])
                        
                        if raw_prediction > threshold_dyslexia:
                            status_icon = "🔴"
                            level_risiko = "Tinggi"
                            conf = raw_prediction
                            pred_class = "Disleksia"
                        else:
                            status_icon = "🟢"
                            level_risiko = "Rendah"
                            conf = float(predictions[1])
                            pred_class = "Normal"
                            
                        # Level Confidence
                        if conf < 0.6:
                            conf_level = "Rendah"
                        elif conf < 0.8:
                            conf_level = "Medium"
                        else:
                            conf_level = "Tinggi"
                            
                        results.append({
                            "Nama File": bf.name,
                            "Status": status_icon,
                            "Level Risiko": level_risiko,
                            "Tingkat Kepercayaan": f"{conf * 100:.2f}%",
                            "Level Confidence": conf_level,
                            "Ukuran Gambar": f"{width}x{height}",
                            "Waktu Analisis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Raw Score (Dyslexia)": f"{raw_prediction:.6f}",
                            "Inference Time (s)": f"{inference_time:.4f}",
                            "Prediksi Asli": pred_class # Hidden class for summary
                        })
                    
                    # Buat DataFrame
                    df = pd.DataFrame(results)
                    
                    # Styling Function untuk warna baris
                    def highlight_rows(row):
                        if row['Level Risiko'] == 'Tinggi':
                            return ['background-color: #ffcccc; color: #900000; font-weight: bold'] * len(row)
                        else:
                            return ['background-color: #d1fae5; color: #064e3b; font-weight: normal'] * len(row)
                    
                    # Terapkan styling, sembunyikan kolom Prediksi Asli agar lebih rapi
                    styled_df = df.drop(columns=["Prediksi Asli"]).style.apply(highlight_rows, axis=1)
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Statistik Summary
                    dyslexia_count = len(df[df["Prediksi Asli"] == "Disleksia"])
                    normal_count = len(df[df["Prediksi Asli"] == "Normal"])
                    st.info(f"📊 **Statistik Summary:** Terdeteksi **{dyslexia_count} Disleksia** dan **{normal_count} Normal** dari total {len(batch_files)} gambar.")
                    
                    # Export CSV
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Hasil (CSV)",
                        data=csv,
                        file_name="batch_analysis_results.csv",
                        mime="text/csv",
                    )

st.markdown("---")
st.subheader("Analisis Gambar Tunggal")

uploaded_file = st.file_uploader("Pilih Gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca gambar menggunakan Pillow
    image = Image.open(uploaded_file).convert('RGB')
    
    # Preprocessing
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0) / 255.0  # Normalisasi
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_column_width=True)
        
    if st.button("Analisis Gambar", type="primary"):
        if interpreter is None:
            st.error("Model TFLite tidak ditemukan! Pastikan telah menjalankan script train_and_convert.py")
        else:
            with st.spinner('Sedang menganalisis...'):
                # TFLite Inference
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                # Konversi data ke tipe yang diharapkan (Float32)
                input_data = np.array(img_array_expanded, dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                
                interpreter.invoke()
                
                predictions = interpreter.get_tensor(output_details[0]['index'])[0]
                
                # Logika Threshold (Batas Ambang)
                threshold_dyslexia = 0.6
                raw_prediction_dyslexia = float(predictions[0]) # Indeks 0 = Disleksia
                
                if raw_prediction_dyslexia > threshold_dyslexia:
                    predicted_class = "Disleksia"
                    confidence = raw_prediction_dyslexia
                    status_color = "#ef4444" # Merah
                else:
                    predicted_class = "Normal"
                    confidence = float(predictions[1]) # Indeks 1 = Normal
                    status_color = "#22c55e" # Hijau
                
                # Penentuan Label Keyakinan (Confidence Level)
                if confidence < 0.6:
                    confidence_label = "Low confidence"
                    conf_color = "red"
                elif confidence < 0.8:
                    confidence_label = "Medium confidence"
                    conf_color = "orange"
                else:
                    confidence_label = "High confidence"
                    conf_color = "green"
                
                st.markdown(f"""
                <div class="result-box">
                    <h3>Hasil Analisis</h3>
                    <p>Status: <strong style="color: {status_color};">{predicted_class}</strong></p>
                    <p>Tingkat Kepercayaan: <strong><span style="color: {conf_color};">{confidence * 100:.2f}% - {confidence_label}</span></strong></p>
                </div>
                <br>
                """, unsafe_allow_html=True)
                
                # Menampilkan Detail Teknis seperti di aplikasi referensi
                with st.expander("📊 Detail Teknis Prediksi"):
                    st.markdown(f"""
                    - **Raw Prediction Value:** {raw_prediction_dyslexia:.6f}
                    - **Threshold untuk Dyslexic:** > {threshold_dyslexia}
                    - **Kelas Prediksi:** {predicted_class}
                    - **Model Type:** TFLite (Optimized)
                    - **Model Size:** ~15.3 MB
                    """)
                
                # Grad-CAM (Hanya jika Keras Model tersedia)
                if keras_model is not None:
                    with col2:
                        st.subheader("Area Potensi (Heatmap)")
                        # Nama layer konvolusi terakhir EfficientNetB0
                        last_conv_layer_name = "top_activation"
                        
                        try:
                            heatmap = make_gradcam_heatmap(img_array_expanded, keras_model, last_conv_layer_name)
                            
                            # Ubah gambar PIL ke numpy array (RGB) untuk di-overlay
                            img_np = np.array(img_resized)
                            
                            # Buat overlay
                            superimposed = overlay_heatmap(img_np, heatmap)
                            
                            st.image(superimposed, use_column_width=True, caption="Warna merah menunjukkan area fokus utama AI")
                        except Exception as e:
                            st.warning(f"Tidak dapat menghasilkan visualisasi Grad-CAM: {e}")
                else:
                    st.info("Keras Model (.h5) tidak ditemukan untuk menjalankan Grad-CAM. Prediksi selesai tanpa heatmap.")
