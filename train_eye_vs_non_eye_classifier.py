import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse

# Cấu hình tham số
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

def train_eye_vs_non_eye_classifier(input_folder, output_folder):
    # Chuẩn bị các generator cho train và validation
    # Không dùng rescale vì ảnh đã được preprocess bằng preprocess_input
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 20% dữ liệu sẽ dùng cho validation
    )

    train_generator = datagen.flow_from_directory(
        input_folder,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',  # Sử dụng binary classification (valid / invalid)
        subset='training'  # Sử dụng cho training
    )

    validation_generator = datagen.flow_from_directory(
        input_folder,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',  # Sử dụng binary classification (valid / invalid)
        subset='validation'  # Sử dụng cho validation
    )

    # Dùng EfficientNet làm feature extractor
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Freeze base model để chỉ học classifier

    # Trích xuất đặc trưng từ EfficientNet
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)  # Dropout để tránh overfitting
    output = Dense(1, activation='sigmoid')(x)  # Sigmoid cho binary classification

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',  # Binary crossentropy cho 2 lớp
                  metrics=['accuracy'])

    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(output_folder, 'best_model.h5'), 
        monitor='val_loss', 
        save_best_only=True, 
        mode='min', 
        verbose=1
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    # Huấn luyện mô hình
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop]
    )

    # Lưu mô hình sau khi huấn luyện
    model.save(os.path.join(output_folder, 'final_model.h5'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet model to classify 'eye' vs 'non-eye' images.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the dataset folder")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the trained model")
    args = parser.parse_args()

    train_eye_vs_non_eye_classifier(args.input_folder, args.output_folder)
