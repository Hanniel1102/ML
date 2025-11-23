import os
import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
import argparse

class EfficientNetPreprocessor:
    def __init__(self, img_size=224):
        self.img_size = img_size

    def load_image(self, img_path):
        """Load ảnh và chuyển sang RGB."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh: {img_path}. Kiểm tra đường dẫn!")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def resize_and_pad(self, img):
        """
        Resize ảnh về đúng kích thước EfficientNet yêu cầu (224x224).
        Giữ tỷ lệ bằng cách padding (không méo hình).
        """
        h, w = img.shape[:2]
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (new_w, new_h))

        # Tạo nền đen -> padding
        padded = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized

        return padded

    def prepare(self, img_path):
        """Xử lý ảnh hoàn chỉnh cho EfficientNet."""
        img = self.load_image(img_path)
        img = self.resize_and_pad(img)
        img = img.astype(np.float32)

        # Chuẩn hóa theo EfficientNet
        img = preprocess_input(img)

        # Thêm batch dimension: (1, 224, 224, 3)
        img = np.expand_dims(img, axis=0)
        return img

    def save_processed_image(self, img, save_path):
        """Lưu ảnh đã xử lý vào file."""
        img = np.squeeze(img, axis=0)  # Remove batch dimension
        img = np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert RGB back to BGR for saving

def process_folder(input_folder, output_folder, img_size=224):
    # Tạo thư mục đích nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    preprocessor = EfficientNetPreprocessor(img_size)

    # Duyệt qua tất cả thư mục con (cataract, diabetic_retinopathy, glaucoma, normal)
    for category in os.listdir(input_folder):
        category_path = os.path.join(input_folder, category)
        
        if os.path.isdir(category_path):
            category_output_folder = os.path.join(output_folder, category)
            if not os.path.exists(category_output_folder):
                os.makedirs(category_output_folder)

            # Duyệt qua tất cả ảnh trong thư mục con
            for filename in os.listdir(category_path):
                img_path = os.path.join(category_path, filename)

                # Kiểm tra nếu đó là ảnh
                if os.path.isfile(img_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    print(f"Đang xử lý: {filename} trong thư mục {category}")
                    try:
                        # Xử lý ảnh và lưu vào thư mục đích
                        processed_img = preprocessor.prepare(img_path)
                        save_path = os.path.join(category_output_folder, filename)
                        preprocessor.save_processed_image(processed_img, save_path)
                    except Exception as e:
                        print(f"Không thể xử lý {filename}: {e}")

    print(f"\nĐã xử lý xong toàn bộ ảnh từ {input_folder}. Kết quả đã được lưu tại {output_folder}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xử lý toàn bộ ảnh trong thư mục để chuẩn hóa cho EfficientNet.")
    parser.add_argument("--input_folder", type=str, required=True, help="Thư mục chứa ảnh đầu vào")
    parser.add_argument("--output_folder", type=str, required=True, help="Thư mục lưu ảnh đã xử lý")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)
