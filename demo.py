import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image

# Cấu hình tham số
IMG_SIZE = 224
FEATURE_DIM = 1280  # Dimension của feature vector từ EfficientNet
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "dataset_train/final_model.h5"  # Mô hình phân loại bệnh mắt
FEATURE_EXTRACTOR_MODEL_PATH = "eye_classifier/final_model.h5"  # Mô hình phân biệt "eye vs non-eye"

# Hàm tải ảnh và kiểm tra ảnh mắt hay không
def preprocess_image(image_path):
    """Chuẩn bị ảnh đầu vào, thay đổi kích thước và chuẩn hóa theo yêu cầu của EfficientNet."""
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalization
    return img

def check_eye_or_not(image_path, eye_model):
    """Kiểm tra xem ảnh có phải là ảnh mắt hay không bằng model phân biệt."""
    img = preprocess_image(image_path)
    prediction = eye_model.predict(img, verbose=0)
    return prediction[0][0] > 0.5

# Xây dựng mô hình EfficientNet cho phân loại bệnh mắt
def build_eye_disease_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)  # Sigmoid cho binary classification

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',  # Binary crossentropy cho 2 lớp
                  metrics=['accuracy'])
    return model

def load_eye_model():
    """Tải mô hình phân biệt ảnh mắt và không phải ảnh mắt."""
    if os.path.exists(FEATURE_EXTRACTOR_MODEL_PATH):
        model = load_model(FEATURE_EXTRACTOR_MODEL_PATH)
        print("Mô hình 'Eye vs Non-Eye' đã được tải thành công.")
        return model
    else:
        raise Exception(f"Mô hình phân biệt ảnh mắt '{FEATURE_EXTRACTOR_MODEL_PATH}' không tìm thấy!")

def load_eye_disease_model():
    """Tải mô hình phân loại bệnh mắt."""
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Mô hình phân loại bệnh mắt đã được tải thành công.")
        return model
    else:
        raise Exception(f"Mô hình phân loại bệnh mắt '{MODEL_PATH}' không tìm thấy!")

def train_eye_disease_model(train_folder, output_folder):
    """Huấn luyện mô hình phân loại bệnh mắt với EfficientNet."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    validation_generator = val_datagen.flow_from_directory(
        train_folder,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    model = build_eye_disease_model()

    checkpoint = ModelCheckpoint(os.path.join(output_folder, 'best_model.h5'), monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop]
    )

    model.save(os.path.join(output_folder, 'final_model.h5'))

# GUI để tải ảnh và kiểm tra
class EyeClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Disease Classification & Eye Detection")

        self.eye_model = load_eye_model()
        self.eye_disease_model = load_eye_disease_model()

        self.label = tk.Label(root, text="Chọn ảnh để kiểm tra:", font=("Arial", 16))
        self.label.pack(pady=20)

        self.select_button = tk.Button(root, text="Chọn ảnh", command=self.select_image, font=("Arial", 14))
        self.select_button.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=20)

    def select_image(self):
        """Chọn ảnh và kiểm tra xem có phải ảnh mắt không, nếu có thì phân loại bệnh mắt."""
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        try:
            # Kiểm tra ảnh có phải mắt không
            if check_eye_or_not(file_path, self.eye_model):
                self.result_label.config(text="Ảnh là mắt, tiến hành phân loại bệnh mắt...")
                # Tiến hành phân loại bệnh mắt
                img = preprocess_image(file_path)
                prediction = self.eye_disease_model.predict(img, verbose=0)
                if prediction[0][0] > 0.5:
                    self.result_label.config(text="Bệnh mắt: Dương tính")
                else:
                    self.result_label.config(text="Bệnh mắt: Âm tính")
            else:
                self.result_label.config(text="Không phải ảnh mắt, dừng chương trình.")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = EyeClassifierApp(root)
    root.mainloop()
