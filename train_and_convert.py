import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Konfigurasi
DATASET_DIR = "data"  # Struktur: data/train/Disleksia, data/train/Normal
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 2
LABEL_SMOOTHING = 0.1  # Mencegah model overconfident saat prediksi salah


def create_model(learning_rate: float = 1e-4):
    """Membangun arsitektur EfficientNet-B0 dengan Label Smoothing."""
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    # Biarkan layer konvolusi trainable agar mendukung Grad-CAM & fitur tulisan tangan
    base_model.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)  # Dropout ditingkatkan ke 0.3 untuk regularisasi & kalibrasi
    predictions = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    return model


def compute_balanced_class_weights(generator) -> dict[int, float]:
    """Menghitung class weight seimbang agar kelas Disleksia tidak terabaikan (menekan FN)."""
    counts = np.bincount(generator.classes)
    total = float(np.sum(counts))
    num_classes = len(counts)
    weights = {
        idx: float(total / (num_classes * max(count, 1)))
        for idx, count in enumerate(counts)
    }
    return weights


def train(fine_tune_from_existing: bool = False, epochs: int = EPOCHS):
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Direktori '{DATASET_DIR}' tidak ditemukan.")
        return None

    train_dir = os.path.join(DATASET_DIR, "train")
    if not os.path.exists(train_dir):
        print(f"Error: Direktori latih '{train_dir}' tidak ditemukan.")
        return None

    # Augmentasi data ringan untuk tulisan tangan (tanpa horizontal flip)
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        validation_split=0.2,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    validation_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    if train_generator.samples == 0:
        print("Error: Tidak ada gambar ditemukan di dalam folder dataset.")
        return None

    print(f"Mapping kelas: {train_generator.class_indices}")
    class_weights = compute_balanced_class_weights(train_generator)
    print(f"Class weights seimbang: {class_weights}")

    if fine_tune_from_existing and os.path.exists("dyslexia_model.h5"):
        print("Memuat bobot 'dyslexia_model.h5' untuk fine-tuning lanjutan...")
        model = tf.keras.models.load_model("dyslexia_model.h5", compile=False)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            metrics=["accuracy"],
        )
    else:
        model = create_model(learning_rate=1e-4)

    callbacks = [
        ModelCheckpoint(
            "dyslexia_model.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("Mulai melatih model...")
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    model.save("dyslexia_model.h5")
    print("Model Keras terbaik berhasil disimpan sebagai 'dyslexia_model.h5'")
    return model


def convert_to_tflite():
    if not os.path.exists("dyslexia_model.h5"):
        print("Error: 'dyslexia_model.h5' tidak ditemukan. Jalankan pelatihan terlebih dahulu.")
        return

    model = tf.keras.models.load_model("dyslexia_model.h5", compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]

    tflite_model = converter.convert()

    with open("dyslexia_model_v2.tflite", "wb") as f:
        f.write(tflite_model)

    print("Model TFLite berhasil disimpan sebagai 'dyslexia_model_v2.tflite' (Format Float32)")


if __name__ == "__main__":
    print("=== Pipeline Pelatihan Model Disleksia (Calibrated + Balanced) ===")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Perangkat GPU terdeteksi: {gpus if gpus else 'Tidak ada (CPU mode)'}")
    # Gunakan fine_tune_from_existing=True jika melanjutkan dari dyslexia_model.h5
    model = train(fine_tune_from_existing=False, epochs=EPOCHS)
    if model is not None:
        convert_to_tflite()
        print("Pipeline selesai!")
