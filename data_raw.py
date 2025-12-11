"""
Data Augmentation - Tạo ảnh xấu cho dataset
Thêm các ảnh có chất lượng kém (noise, blur, dark, bright) để model học robust hơn
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
import os
import cv2
from pathlib import Path
from tqdm import tqdm


def add_noise_to_image(image_path, noise_factor=0.2):
    """
    Thêm Gaussian noise vào ảnh để mô phỏng ảnh chụp trong điều kiện ánh sáng yếu.
    
    Args:
        image_path (str): Đường dẫn đến ảnh gốc
        noise_factor (float): Mức độ nhiễu (0.1-0.5, khuyến nghị 0.2-0.35)
    
    Returns:
        PIL.Image: Ảnh đã thêm nhiễu
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Tạo Gaussian noise
        noise = np.random.normal(0, noise_factor, img_array.shape)
        noisy_img = img_array + noise
        noisy_img = np.clip(noisy_img, 0, 1)
        
        # Convert về uint8
        noisy_img = (noisy_img * 255).astype(np.uint8)
        return Image.fromarray(noisy_img)
    except Exception as e:
        print(f"❌ Lỗi add_noise: {e}")
        return None

def add_blur_to_image(image_path, blur_factor=5):
    """
    Thêm Gaussian blur để mô phỏng ảnh chụp bị rung hoặc mất focus.
    
    Args:
        image_path (str): Đường dẫn đến ảnh gốc
        blur_factor (int): Mức độ làm mờ (1-15, khuyến nghị 5-10)
    
    Returns:
        PIL.Image: Ảnh đã làm mờ
    """
    try:
        img = Image.open(image_path).convert('RGB')
        blurred_img = img.filter(ImageFilter.GaussianBlur(blur_factor))
        return blurred_img
    except Exception as e:
        print(f"❌ Lỗi add_blur: {e}")
        return None


def adjust_brightness(image_path, brightness_factor=0.3):
    """
    Điều chỉnh độ sáng để mô phỏng ảnh chụp ban đêm hoặc quá sáng.
    
    Args:
        image_path (str): Đường dẫn đến ảnh gốc
        brightness_factor (float): Hệ số điều chỉnh
            - < 1.0: Làm tối (0.2-0.5 cho ảnh ban đêm)
            - > 1.0: Làm sáng (1.2-1.5 cho ảnh quá sáng)
    
    Returns:
        PIL.Image: Ảnh đã điều chỉnh độ sáng
    """
    try:
        img = Image.open(image_path).convert('RGB')
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(brightness_factor)
        return bright_img
    except Exception as e:
        print(f"❌ Lỗi adjust_brightness: {e}")
        return None


def add_motion_blur(image_path, kernel_size=15):
    """
    Thêm motion blur để mô phỏng ảnh chụp khi camera di chuyển.
    
    Args:
        image_path (str): Đường dẫn đến ảnh gốc
        kernel_size (int): Kích thước kernel (7-25, khuyến nghị 15)
    
    Returns:
        PIL.Image: Ảnh có motion blur
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Tạo motion blur kernel (horizontal)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Apply kernel
        blurred = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(blurred)
    except Exception as e:
        print(f"❌ Lỗi add_motion_blur: {e}")
        return None


def adjust_contrast(image_path, contrast_factor=0.5):
    """
    Điều chỉnh độ tương phản để mô phỏng ảnh phẳng hoặc quá contrasty.
    
    Args:
        image_path (str): Đường dẫn đến ảnh gốc
        contrast_factor (float): Hệ số tương phản
            - < 1.0: Giảm contrast (0.3-0.7)
            - > 1.0: Tăng contrast (1.2-1.5)
    
    Returns:
        PIL.Image: Ảnh đã điều chỉnh contrast
    """
    try:
        img = Image.open(image_path).convert('RGB')
        enhancer = ImageEnhance.Contrast(img)
        contrasted_img = enhancer.enhance(contrast_factor)
        return contrasted_img
    except Exception as e:
        print(f"❌ Lỗi adjust_contrast: {e}")
        return None


def add_jpeg_compression(image_path, quality=20):
    """
    Thêm JPEG compression artifacts để mô phỏng ảnh bị nén mạnh.
    
    Args:
        image_path (str): Đường dẫn đến ảnh gốc
        quality (int): Chất lượng JPEG (10-50, thấp = nhiễu hơn)
    
    Returns:
        PIL.Image: Ảnh bị nén JPEG
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Save và reload với quality thấp
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        return compressed_img
    except Exception as e:
        print(f"❌ Lỗi add_jpeg_compression: {e}")
        return None

def create_augmented_dataset(input_dir, output_dir, 
                            num_augmented_per_class=10,
                            copy_original=True,
                            degradation_types='all'):
    """
    Tạo dataset augmentation với nhiều loại ảnh xấu để model học robust hơn.
    
    Args:
        input_dir (str): Thư mục dataset gốc (Train/Val/Test)
        output_dir (str): Thư mục đích để lưu ảnh (gốc + augmented)
        num_augmented_per_class (int): Số ảnh augmented cần tạo/class (10-20 khuyến nghị)
        copy_original (bool): Có copy ảnh gốc không (True khuyến nghị)
        degradation_types (str|list): 'all' hoặc list các loại ['noise', 'blur', 'dark', ...]
    
    Returns:
        dict: Thống kê augmentation
    """
    import shutil
    
    # Định nghĩa các loại degradation
    all_degradations = {
        'noise_light': lambda p: add_noise_to_image(p, noise_factor=0.20),
        'noise_heavy': lambda p: add_noise_to_image(p, noise_factor=0.35),
        'blur_light': lambda p: add_blur_to_image(p, blur_factor=5),
        'blur_heavy': lambda p: add_blur_to_image(p, blur_factor=10),
        'dark': lambda p: adjust_brightness(p, brightness_factor=0.3),
        'very_dark': lambda p: adjust_brightness(p, brightness_factor=0.15),
        'bright': lambda p: adjust_brightness(p, brightness_factor=1.4),
        'motion_blur': lambda p: add_motion_blur(p, kernel_size=15),
        'low_contrast': lambda p: adjust_contrast(p, contrast_factor=0.5),
        'jpeg_compress': lambda p: add_jpeg_compression(p, quality=25),
    }
    
    # Chọn degradations
    if degradation_types == 'all':
        degradations = list(all_degradations.items())
    else:
        degradations = [(k, v) for k, v in all_degradations.items() if k in degradation_types]
    
    if not degradations:
        print("❌ Không có degradation nào được chọn!")
        return None
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ Không tìm thấy thư mục: {input_path}")
        return None
    
    print(f"\n{'='*70}")
    print(f"🚀 BẮT ĐẦU TẠO AUGMENTED DATASET")
    print(f"{'='*70}")
    print(f"📁 Input: {input_path}")
    print(f"💾 Output: {output_path}")
    print(f"🎲 Số ảnh augmented/class: {num_augmented_per_class}")
    print(f"📋 Copy ảnh gốc: {'CÓ' if copy_original else 'KHÔNG'}")
    print(f"🔧 Degradation types: {len(degradations)} loại")
    for deg_name, _ in degradations:
        print(f"   - {deg_name}")
    print(f"{'='*70}\n")
    
    # Tìm tất cả classes
    classes = [d for d in input_path.iterdir() if d.is_dir()]
    
    stats = {
        'total_copied': 0,
        'total_augmented': 0,
        'classes': {}
    }
    
    for class_folder in classes:
        class_name = class_folder.name
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Tìm tất cả ảnh
        image_files = list(class_folder.glob("*.jpg")) + \
                     list(class_folder.glob("*.jpeg")) + \
                     list(class_folder.glob("*.png"))
        
        if len(image_files) == 0:
            print(f"⚠️  {class_name}: Không có ảnh")
            continue
        
        print(f"\n📂 Đang xử lý {class_name}:")
        print(f"   📊 Tổng ảnh gốc: {len(image_files)}")
        
        copied_count = 0
        augmented_count = 0
        
        # 1. Copy ảnh gốc (nếu cần)
        if copy_original:
            print(f"   📋 Đang copy ảnh gốc...", end=" ")
            for img_file in image_files:
                try:
                    dst_path = output_class_dir / img_file.name
                    if not dst_path.exists():  # Tránh copy trùng
                        shutil.copy2(img_file, dst_path)
                        copied_count += 1
                        stats['total_copied'] += 1
                except Exception as e:
                    print(f"\n   ❌ Lỗi copy {img_file.name}: {e}")
            print(f"✅ {copied_count} ảnh")
        
        # 2. Tạo augmented images
        num_to_create = min(num_augmented_per_class, len(image_files))
        selected_images = random.sample(image_files, num_to_create)
        
        print(f"   🎨 Đang tạo {num_to_create} ảnh augmented...")
        
        for img_file in tqdm(selected_images, desc=f"   {class_name[:25]:25s}", 
                            ncols=70, leave=False):
            base_name = img_file.stem
            
            # Chọn degradation ngẫu nhiên
            deg_name, deg_func = random.choice(degradations)
            
            try:
                augmented_img = deg_func(str(img_file))
                
                if augmented_img is not None:
                    output_filename = f"{base_name}_aug_{deg_name}.jpg"
                    output_filepath = output_class_dir / output_filename
                    augmented_img.save(output_filepath, quality=90, optimize=True)
                    augmented_count += 1
                    stats['total_augmented'] += 1
            except Exception as e:
                print(f"\n   ❌ Lỗi augment {img_file.name}: {e}")
        
        stats['classes'][class_name] = {
            'original': len(image_files),
            'copied': copied_count,
            'augmented': augmented_count
        }
        
        print(f"   ✅ Hoàn tất: {copied_count} gốc + {augmented_count} augmented")
    
    # Tổng kết
    print(f"\n{'='*70}")
    print(f"🎉 HOÀN TẤT TẠO AUGMENTED DATASET")
    print(f"{'='*70}")
    print(f"📊 Tổng kết:")
    print(f"   📋 Tổng ảnh gốc copied: {stats['total_copied']}")
    print(f"   🎨 Tổng ảnh augmented: {stats['total_augmented']}")
    print(f"   📈 Tổng cộng: {stats['total_copied'] + stats['total_augmented']} ảnh")
    print(f"\n📁 Dataset mới đã lưu tại: {output_path.absolute()}")
    print(f"{'='*70}\n")
    
    return stats

# ==========================================
# MAIN - Chạy script
# ==========================================
if __name__ == "__main__":
    """
    Cách sử dụng:
    
    1. Tạo augmented dataset cho Train:
       python data_raw.py
    
    2. Tùy chỉnh:
       - num_augmented_per_class: 10-20 (khuyến nghị 15)
       - copy_original: True (giữ ảnh gốc) / False (chỉ tạo augmented)
       - degradation_types: 'all' hoặc ['noise_light', 'blur_heavy', 'dark']
    """
    
    # ============ CẤU HÌNH ============
    INPUT_DIR = "Tomato/Train"           # Thư mục dataset gốc
    OUTPUT_DIR = "Tomato/Augmented_Train"  # Thư mục lưu augmented dataset
    NUM_AUGMENTED = 15                    # Số ảnh augmented/class (10-20)
    COPY_ORIGINAL = True                  # Copy ảnh gốc sang output
    DEGRADATION_TYPES = 'all'             # 'all' hoặc list cụ thể
    
    # Hoặc chỉ chọn 1 vài loại degradation:
    # DEGRADATION_TYPES = ['noise_heavy', 'blur_heavy', 'dark', 'very_dark', 'motion_blur']
    
    # ============ CHẠY ============
    stats = create_augmented_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        num_augmented_per_class=NUM_AUGMENTED,
        copy_original=COPY_ORIGINAL,
        degradation_types=DEGRADATION_TYPES
    )
    
    if stats:
        print("\n✅ Augmentation hoàn tất!")
        print(f"💡 Sử dụng dataset mới để train model: {OUTPUT_DIR}/")
        print(f"💡 Trong notebook: train_datagen.flow_from_directory('{OUTPUT_DIR}')")
    else:
        print("\n❌ Augmentation thất bại!")
        exit(1)
