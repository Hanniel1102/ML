import numpy as np
from PIL import Image
import random
import os
from pathlib import Path

def add_noise_to_image(image_path, noise_factor=0.2):
    """
    Thêm nhiễu vào ảnh.
    Args:
        image_path (str): đường dẫn đến ảnh gốc.
        noise_factor (float): mức độ nhiễu.
    Returns:
        Image: ảnh đã được thêm nhiễu.
    """
    img = Image.open(image_path)
    img = np.array(img) / 255.0  # Chuyển đổi ảnh sang dãy [0, 1]
    
    noise = np.random.normal(0, noise_factor, img.shape)  # Tạo nhiễu ngẫu nhiên
    noisy_img = img + noise  # Thêm nhiễu vào ảnh
    noisy_img = np.clip(noisy_img, 0, 1)  # Giới hạn giá trị để ảnh không vượt quá [0, 1]
    
    noisy_img = (noisy_img * 255).astype(np.uint8)  # Chuyển về dãy [0, 255]
    return Image.fromarray(noisy_img)

def add_blur_to_image(image_path, blur_factor=5):
    """
    Thêm làm mờ vào ảnh.
    Args:
        image_path (str): đường dẫn đến ảnh gốc.
        blur_factor (int): mức độ làm mờ.
    Returns:
        Image: ảnh đã được làm mờ.
    """
    from PIL import ImageFilter
    img = Image.open(image_path)
    blurred_img = img.filter(ImageFilter.GaussianBlur(blur_factor))
    return blurred_img

def adjust_brightness(image_path, brightness_factor=0.3):
    """
    Điều chỉnh độ sáng của ảnh (tối hoặc sáng).
    Args:
        image_path (str): đường dẫn đến ảnh gốc.
        brightness_factor (float): hệ số điều chỉnh (< 1 = tối, > 1 = sáng).
    Returns:
        Image: ảnh đã điều chỉnh độ sáng.
    """
    from PIL import ImageEnhance
    img = Image.open(image_path)
    enhancer = ImageEnhance.Brightness(img)
    bright_img = enhancer.enhance(brightness_factor)
    return bright_img

def create_dataset_with_bad_images(train_dir, output_dir, num_bad_images_per_class=5):
    """
    Copy tất cả ảnh gốc + tạo thêm ảnh xấu vào folder mới.
    Args:
        train_dir (str): Thư mục chứa dataset Train gốc.
        output_dir (str): Thư mục đích để lưu ảnh (gốc + xấu).
        num_bad_images_per_class (int): Số ảnh xấu cần tạo cho mỗi class (4-5 ảnh).
    """
    import shutil
    
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    print(f"🚀 Bắt đầu tạo dataset với ảnh gốc + {num_bad_images_per_class} ảnh xấu/class...")
    print(f"📁 Tổng số classes: {len(classes)}")
    print(f"💾 Lưu vào: {output_dir}\n")
    
    total_copied = 0
    total_bad_created = 0
    
    for cls in classes:
        class_path = os.path.join(train_dir, cls)
        output_class_path = os.path.join(output_dir, cls)
        
        # Tạo folder cho class trong output_dir
        os.makedirs(output_class_path, exist_ok=True)
        
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if len(image_files) == 0:
            print(f"⚠️  {cls}: Không có ảnh")
            continue
        
        # 1. Copy tất cả ảnh gốc
        copied_count = 0
        for img_file in image_files:
            src_path = os.path.join(class_path, img_file)
            dst_path = os.path.join(output_class_path, img_file)
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                total_copied += 1
            except Exception as e:
                print(f"❌ Lỗi copy {img_file}: {e}")
        
        # 2. Tạo ảnh xấu
        num_to_create = min(num_bad_images_per_class, len(image_files))
        selected_images = random.sample(image_files, num_to_create)
        
        bad_created = 0
        for i, img_file in enumerate(selected_images):
            img_path = os.path.join(class_path, img_file)
            base_name = os.path.splitext(img_file)[0]
            
            # Tạo 3 loại ảnh xấu khác nhau
            degradation_types = [
                ('noisy', lambda p: add_noise_to_image(p, noise_factor=0.35)),
                ('blurry', lambda p: add_blur_to_image(p, blur_factor=8)),
                ('dark', lambda p: adjust_brightness(p, brightness_factor=0.25))
            ]
            
            # Chọn loại degradation ngẫu nhiên
            deg_type, deg_func = random.choice(degradation_types)
            
            try:
                bad_img = deg_func(img_path)
                output_path = os.path.join(output_class_path, f"{base_name}_bad_{deg_type}.jpg")
                bad_img.save(output_path, quality=85)
                bad_created += 1
                total_bad_created += 1
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {img_file}: {e}")
        
        print(f"✅ {cls:35s}: {copied_count} ảnh gốc + {bad_created} ảnh xấu")
    
    print(f"\n🎉 Hoàn tất!")
    print(f"   📋 Tổng ảnh gốc: {total_copied}")
    print(f"   🔧 Tổng ảnh xấu: {total_bad_created}")
    print(f"   📊 Tổng cộng: {total_copied + total_bad_created} ảnh")

# ==========================================
# MAIN - Chạy script
# ==========================================
if __name__ == "__main__":
    train_dir = "Tomato/Train"
    output_dir = "Tomato/Bad_Dataset"
    
    if not os.path.exists(train_dir):
        print(f"❌ Không tìm thấy thư mục: {train_dir}")
        exit(1)
    
    # Copy tất cả ảnh gốc + thêm 4-5 ảnh xấu cho mỗi class
    create_dataset_with_bad_images(train_dir, output_dir, num_bad_images_per_class=5)
    
    print(f"\n💡 Dataset mới (gốc + xấu) đã được lưu vào: {output_dir}/")
