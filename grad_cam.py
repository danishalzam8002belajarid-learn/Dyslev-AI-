import numpy as np
import tensorflow as tf
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Menghasilkan heatmap Grad-CAM untuk gambar.
    """
    # Buat model yang memetakan gambar input ke aktivasi layer konvolusi terakhir 
    # serta ke prediksi akhir
    grad_model = tf.keras.models.Model(
        model.inputs, 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Hitung gradien top predicted class terhadap aktivasi layer konvolusi terakhir
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradien layer output terhadap feature map layer konvolusi
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vektor dengan bentuk (channels,), di mana setiap elemen 
    # adalah intensitas rata-rata dari gradien pada feature map channel tersebut
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Kalikan setiap channel dalam array feature map dengan "pentingnya" 
    # channel tersebut terhadap prediksi
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalisasi heatmap antara 0 dan 1 untuk visualisasi
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Menambahkan heatmap di atas gambar asli.
    """
    # Resize heatmap agar sesuai ukuran gambar asli
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Ubah format heatmap menjadi 8-bit warna (0-255)
    heatmap = np.uint8(255 * heatmap)
    
    # Terapkan colormap untuk membuat warna jet
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Konversi dari BGR (OpenCV) ke RGB (Streamlit / Matplotlib)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Gabungkan gambar asli dengan heatmap
    superimposed_img = heatmap_rgb * alpha + img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return superimposed_img
