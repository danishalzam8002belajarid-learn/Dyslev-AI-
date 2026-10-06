import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Konfigurasi
DATASET_DIR = "data" # Struktur: data/train/dyslexic, data/train/normal
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2 # Asumsi: normal vs dyslexic

def create_model():
    # Load EfficientNetB0 pre-trained pada ImageNet tanpa top (classification) layer
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Jangan di-freeze sepenuhnya agar layer konvolusi bisa belajar fitur tulisan tangan (penting untuk Grad-CAM)
    base_model.trainable = True
    
    # Opsional: Kita bisa freeze sebagian layer awal jika dataset kecil, 
    # tapi untuk hasil Grad-CAM terbaik, kita biarkan seluruhnya trainable.

    # Tambahkan custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x) # Dropout untuk mencegah overfitting
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Gunakan learning rate yang sangat kecil karena kita melakukan fine-tuning pada seluruh model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def train():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Direktori '{DATASET_DIR}' tidak ditemukan.")
        print("Silakan buat struktur folder berikut sebelum menjalankan script:")
        print("data/")
        print("  ├── train/")
        print("  │     ├── dyslexic/  (Isi dengan gambar-gambar tulisan disleksia)")
        print("  │     └── normal/    (Isi dengan gambar-gambar tulisan normal)")
        return None

    train_dir = os.path.join(DATASET_DIR, "train")
    if not os.path.exists(train_dir):
        print(f"Error: Direktori latih '{train_dir}' tidak ditemukan.")
        return None

    # Augmentasi data ringan untuk tulisan tangan
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        validation_split=0.2 # 20% data digunakan untuk validasi
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    if train_generator.samples == 0:
         print("Error: Tidak ada gambar ditemukan di dalam folder dataset.")
         return None

    print(f"Mapping kelas: {train_generator.class_indices}")

    model = create_model()
    
    print("Mulai melatih model...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    # Simpan model Keras (Diperlukan untuk Grad-CAM nanti)
    model.save('dyslexia_model.h5')
    print("Model Keras berhasil disimpan sebagai 'dyslexia_model.h5'")
    return model

def convert_to_tflite():
    # Load model Keras yang baru dilatih
    if not os.path.exists('dyslexia_model.h5'):
        print("Error: 'dyslexia_model.h5' tidak ditemukan. Jalankan pelatihan terlebih dahulu.")
        return

    model = tf.keras.models.load_model('dyslexia_model.h5')
    
    # Konversi ke TFLite (Float32)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Default optimization untuk tflite tanpa kuantisasi int8 akan mempertahankan Float32
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Memastikan input dan output tipe data float32
    converter.target_spec.supported_types = [tf.float32]
    
    tflite_model = converter.convert()

    with open('dyslexia_model_v2.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Model TFLite berhasil disimpan sebagai 'dyslexia_model_v2.tflite' (Format Float32)")

if __name__ == "__main__":
    print("=== Pipeline Pelatihan Model Disleksia ===")
    model = train()
    if model is not None:
        convert_to_tflite()
        print("Pipeline selesai!")
