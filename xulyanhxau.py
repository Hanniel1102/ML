import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
from scipy.ndimage import gaussian_filter
import cv2

def denoise_image(image_path, sigma=1):
    """
    Xử lý ảnh xấu bằng cách giảm nhiễu (sử dụng Gaussian blur).
    Args:
        image_path (str): đường dẫn đến ảnh bị nhiễu.
        sigma (float): độ mờ của bộ lọc Gaussian.
    Returns:
        Image: ảnh sau khi xử lý.
    """
    img = Image.open(image_path)
    img = np.array(img) / 255.0  # Chuyển ảnh thành dãy [0, 1]
    
    # Áp dụng bộ lọc Gaussian
    denoised_img = gaussian_filter(img, sigma=sigma)
    denoised_img = np.clip(denoised_img, 0, 1)
    
    denoised_img = (denoised_img * 255).astype(np.uint8)  # Chuyển lại sang dãy [0, 255]
    return Image.fromarray(denoised_img)

def enhance_image(image_path):
    """
    Tăng cường chất lượng ảnh: giảm nhiễu, tăng độ sắc nét, điều chỉnh độ sáng.
    Args:
        image_path (str): đường dẫn đến ảnh xấu.
    Returns:
        Image: ảnh sau khi xử lý.
    """
    img = Image.open(image_path)
    
    # 1. Giảm nhiễu với Non-local Means Denoising
    img_array = np.array(img)
    if len(img_array.shape) == 3:  # RGB
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
    else:  # Grayscale
        denoised = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
    
    img = Image.fromarray(denoised)
    
    # 2. Tăng độ sắc nét
    sharpness_enhancer = ImageEnhance.Sharpness(img)
    img = sharpness_enhancer.enhance(1.5)
    
    # 3. Điều chỉnh độ sáng nếu ảnh quá tối
    brightness_enhancer = ImageEnhance.Brightness(img)
    img = brightness_enhancer.enhance(1.2)
    
    # 4. Tăng độ tương phản
    contrast_enhancer = ImageEnhance.Contrast(img)
    img = contrast_enhancer.enhance(1.1)
    
    return img

def process_bad_images_dataset(input_dir, output_dir):
    """
    Xử lý tất cả ảnh xấu trong dataset và lưu vào folder mới.
    Args:
        input_dir (str): Thư mục chứa ảnh xấu (có cấu trúc class).
        output_dir (str): Thư mục lưu ảnh đã xử lý.
    """
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"🔧 Bắt đầu xử lý ảnh xấu...")
    print(f"📁 Input: {input_dir}")
    print(f"💾 Output: {output_dir}\n")
    
    total_processed = 0
    
    for cls in classes:
        class_input_path = os.path.join(input_dir, cls)
        class_output_path = os.path.join(output_dir, cls)
        
        os.makedirs(class_output_path, exist_ok=True)
        
        # Chỉ xử lý ảnh xấu (có "_bad_" trong tên)
        image_files = [f for f in os.listdir(class_input_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) 
                      and '_bad_' in f]
        
        if len(image_files) == 0:
            print(f"⚠️  {cls}: Không có ảnh xấu")
            continue
        
        processed_count = 0
        for img_file in image_files:
            img_path = os.path.join(class_input_path, img_file)
            
            try:
                # Xử lý ảnh
                enhanced_img = enhance_image(img_path)
                
                # Lưu với tên mới (bỏ "_bad_")
                new_name = img_file.replace('_bad_noisy', '_enhanced').replace('_bad_blurry', '_enhanced').replace('_bad_dark', '_enhanced')
                output_path = os.path.join(class_output_path, new_name)
                
                enhanced_img.save(output_path, quality=95)
                processed_count += 1
                total_processed += 1
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {img_file}: {e}")
        
        print(f"✅ {cls:35s}: Đã xử lý {processed_count} ảnh xấu")
    
    print(f"\n🎉 Hoàn tất! Tổng cộng đã xử lý {total_processed} ảnh")

def delete_bad_images_from_dataset(dataset_dir):
    """
    Xóa tất cả ảnh xấu (có "_bad_" trong tên) khỏi dataset.
    Args:
        dataset_dir (str): Thư mục chứa dataset (có cấu trúc class).
    """
    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    print(f"🗑️  Bắt đầu xóa ảnh xấu từ {dataset_dir}...\n")
    
    total_deleted = 0
    
    for cls in classes:
        class_path = os.path.join(dataset_dir, cls)
        
        # Tìm tất cả ảnh xấu
        bad_images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) 
                     and '_bad_' in f]
        
        deleted_count = 0
        for img_file in bad_images:
            img_path = os.path.join(class_path, img_file)
            try:
                os.remove(img_path)
                deleted_count += 1
                total_deleted += 1
            except Exception as e:
                print(f"❌ Lỗi khi xóa {img_file}: {e}")
        
        if deleted_count > 0:
            print(f"🗑️  {cls:35s}: Đã xóa {deleted_count} ảnh xấu")
    
    print(f"\n✅ Hoàn tất! Tổng cộng đã xóa {total_deleted} ảnh xấu")

# ==========================================
# MAIN - Chạy script
# ==========================================
if __name__ == "__main__":
    import sys
    
    # Chọn chế độ: 1 = Xử lý ảnh xấu, 2 = Xóa ảnh xấu
    print("="*60)
    print("CHỌN CHỨC NĂNG:")
    print("1. Xử lý ảnh xấu (denoise, sharpen, enhance) và lưu vào folder mới")
    print("2. Xóa tất cả ảnh xấu khỏi dataset")
    print("="*60)
    
    choice = input("Nhập lựa chọn (1 hoặc 2): ").strip()
    
    if choice == "1":
        # Xử lý ảnh xấu
        input_dir = "Tomato/Bad_Dataset"
        output_dir = "Train_Enhanced"
        
        if not os.path.exists(input_dir):
            print(f"❌ Không tìm thấy thư mục: {input_dir}")
            exit(1)
        
        process_bad_images_dataset(input_dir, output_dir)
        print(f"\n💡 Ảnh đã xử lý được lưu tại: {output_dir}/")
        
    elif choice == "2":
        # Xóa ảnh xấu
        dataset_dir = "Tomato/Bad_Dataset"
        
        if not os.path.exists(dataset_dir):
            print(f"❌ Không tìm thấy thư mục: {dataset_dir}")
            exit(1)
        
        confirm = input(f"⚠️  Bạn có chắc muốn xóa TẤT CẢ ảnh xấu từ '{dataset_dir}'? (yes/no): ").strip().lower()
        
        if confirm == "yes":
            delete_bad_images_from_dataset(dataset_dir)
            print(f"\n💡 Các ảnh xấu đã được xóa khỏi: {dataset_dir}/")
        else:
            print("❌ Đã hủy thao tác xóa.")
    else:
        print("❌ Lựa chọn không hợp lệ!")
