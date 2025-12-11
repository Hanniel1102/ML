"""
Data Preprocessing Pipeline - Phục hồi ảnh xấu trong dataset
Sửa chữa các ảnh có chất lượng kém (noise, blur, dark, low contrast) 
để trả về dataset sạch, chất lượng tốt cho training
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil


class DatasetPreprocessor:
    """
    PHỤC HỒI ảnh xấu trong dataset về chất lượng tốt
    
    Chức năng:
    - Tự động phát hiện ảnh xấu (tối, mờ, nhiễu, low contrast)
    - Áp dụng enhancement mạnh mẽ để sửa chữa
    - Trả về dataset sạch, đồng nhất để train model
    
    Khác với efficientnet_preprocessor.py (chỉ áp dụng khi cần),
    file này ÁP DỤNG CHO TẤT CẢ ẢNH để đảm bảo chất lượng đồng nhất.
    """
    
    def __init__(self, target_size=(256, 256), aggressive_fix=True):
        """
        Args:
            target_size: Kích thước đầu ra (width, height)
            aggressive_fix: Sửa mạnh tay (True) hay chỉ sửa khi cần (False)
        """
        self.target_size = target_size
        self.aggressive_fix = aggressive_fix
        
        # Ngưỡng phát hiện ảnh xấu (thấp hơn = sửa nhiều hơn)
        self.brightness_low = 100    # < 100 = tối
        self.brightness_high = 180   # > 180 = sáng
        self.contrast_low = 35       # < 35 = low contrast
        self.noise_high = 500        # < 500 = nhiễu cao
        self.sharpness_low = 40      # < 40 = mờ
        
    def fix_bad_image(self, image_path):
        """
        PHỤC HỒI ảnh xấu về chất lượng tốt:
        1. Load ảnh
        2. Phân tích vấn đề (tối/mờ/nhiễu/low contrast)
        3. Áp dụng sửa chữa MẠNH TẤY:
           - Denoise TRƯỚC (giảm nhiễu)
           - Brightness adjustment (sửa tối/sáng)
           - CLAHE (tăng contrast)
           - Sharpen (làm nét)
        4. Resize về target_size
        5. Trả về ảnh đã phục hồi
        
        Args:
            image_path: Đường dẫn ảnh xấu
            
        Returns:
            numpy array: Ảnh đã phục hồi (chất lượng tốt)
        """
        try:
            # 1. Load ảnh
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            
            # 2. Phân tích vấn đề
            metrics = self._analyze_image(img_array)
            issues = self._detect_issues(metrics)
            
            # 3. Sửa chữa theo thứ tự tối ưu
            fixed_img = img_array.copy()
            
            # Bước 1: DENOISE TRƯỚC (quan trọng!)
            # Phải khử nhiễu trước khi làm các thao tác khác
            if issues['has_noise'] or self.aggressive_fix:
                fixed_img = self._fix_noise(fixed_img, metrics)
            
            # Bước 2: Fix brightness (sửa tối/sáng)
            if issues['too_dark'] or issues['too_bright'] or self.aggressive_fix:
                fixed_img = self._fix_brightness(fixed_img, metrics)
            
            # Bước 3: Fix contrast (CLAHE)
            if issues['low_contrast'] or self.aggressive_fix:
                fixed_img = self._fix_contrast(fixed_img, metrics)
            
            # Bước 4: Sharpen (làm nét)
            if issues['blurry'] or self.aggressive_fix:
                fixed_img = self._fix_sharpness(fixed_img, metrics)
            
            # 5. Resize về target size (bước cuối)
            fixed_img = self._resize_image(fixed_img)
            
            return fixed_img
            
        except Exception as e:
            print(f"❌ Lỗi xử lý {image_path}: {e}")
            return None
    
    def _detect_issues(self, metrics):
        """
        Phát hiện vấn đề của ảnh
        
        Returns:
            dict: {
                'too_dark': bool,
                'too_bright': bool,
                'low_contrast': bool,
                'has_noise': bool,
                'blurry': bool
            }
        """
        return {
            'too_dark': metrics['brightness'] < self.brightness_low,
            'too_bright': metrics['brightness'] > self.brightness_high,
            'low_contrast': metrics['contrast'] < self.contrast_low,
            'has_noise': metrics['noise_variance'] < self.noise_high,
            'blurry': metrics['edge_strength'] < self.sharpness_low
        }
    
    def _resize_image(self, img_array):
        """Resize ảnh về target_size"""
        h, w = img_array.shape[:2]
        target_w, target_h = self.target_size
        
        # Chọn interpolation method phù hợp
        if h > target_h or w > target_w:
            interpolation = cv2.INTER_AREA  # Shrink - chất lượng tốt
        else:
            interpolation = cv2.INTER_CUBIC  # Upscale - smooth hơn
        
        resized = cv2.resize(img_array, (target_w, target_h), interpolation=interpolation)
        return resized
    
    def _analyze_image(self, img_array):
        """
        Phân tích chất lượng ảnh
        Returns:
            dict: brightness, contrast, noise_variance, edge_strength
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 1. Brightness
        brightness = np.mean(gray)
        
        # 2. Contrast (std deviation)
        contrast = np.std(gray)
        
        # 3. Noise (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_variance = laplacian.var()
        
        # 4. Sharpness (Sobel gradient magnitude)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'noise_variance': noise_variance,
            'edge_strength': edge_strength
        }
    
    def _fix_noise(self, img_array, metrics):
        """
        Khử nhiễu MẠNH để phục hồi ảnh nhiễu
        """
        noise_var = metrics['noise_variance']
        
        if noise_var < 200:
            # Nhiễu RẤT nặng - bilateral filter cực mạnh
            denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
            print("      → Khử nhiễu RẤT MẠNH (bilateral d=9)")
        elif noise_var < 400:
            # Nhiễu nặng - bilateral filter mạnh
            denoised = cv2.bilateralFilter(img_array, 7, 60, 60)
            print("      → Khử nhiễu MẠNH (bilateral d=7)")
        else:
            # Nhiễu nhẹ - bilateral filter vừa
            denoised = cv2.bilateralFilter(img_array, 5, 40, 40)
            print("      → Khử nhiễu (bilateral d=5)")
        
        return denoised
    
    def _fix_brightness(self, img_array, metrics):
        """
        Sửa độ sáng về mức chuẩn (120-150)
        """
        brightness = metrics['brightness']
        target_brightness = 135  # Target brightness chuẩn
        
        # Convert to LAB
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        if brightness < self.brightness_low:
            # Ảnh TỐI - tăng sáng mạnh
            alpha = target_brightness / brightness if brightness > 0 else 1.8
            alpha = min(alpha, 2.5)  # Giới hạn tối đa
            beta = 30
            l = np.clip(l * alpha + beta, 0, 255).astype(np.uint8)
            print(f"      → Tăng sáng: {brightness:.0f} → {target_brightness}")
            
        elif brightness > self.brightness_high:
            # Ảnh SÁNG - giảm sáng
            alpha = target_brightness / brightness
            beta = -15
            l = np.clip(l * alpha + beta, 0, 255).astype(np.uint8)
            print(f"      → Giảm sáng: {brightness:.0f} → {target_brightness}")
        
        # Merge back
        lab = cv2.merge([l, a, b])
        fixed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return fixed
    
    def _fix_contrast(self, img_array, metrics):
        """
        Tăng contrast bằng CLAHE MẠNH
        """
        contrast = metrics['contrast']
        
        # Convert to LAB
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        if contrast < 25:
            # Contrast RẤT thấp - CLAHE rất mạnh
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            print("      → Tăng contrast RẤT MẠNH (CLAHE 3.5)")
        elif contrast < self.contrast_low:
            # Contrast thấp - CLAHE mạnh
            clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
            print("      → Tăng contrast MẠNH (CLAHE 2.8)")
        else:
            # Contrast vừa - CLAHE nhẹ
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            print("      → Tăng contrast (CLAHE 2.0)")
        
        l = clahe.apply(l)
        
        # Merge back
        lab = cv2.merge([l, a, b])
        fixed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return fixed
    
    def _fix_sharpness(self, img_array, metrics):
        """
        Làm nét ảnh mờ bằng unsharp masking MẠNH
        """
        edge_strength = metrics['edge_strength']
        
        if edge_strength < 25:
            # Ảnh RẤT mờ - sharpen cực mạnh
            kernel = np.array([[-1, -1, -1],
                             [-1, 10, -1],
                             [-1, -1, -1]])
            print("      → Làm nét RẤT MẠNH (kernel 10)")
        elif edge_strength < self.sharpness_low:
            # Ảnh mờ - sharpen mạnh
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            print("      → Làm nét MẠNH (kernel 9)")
        else:
            # Ảnh hơi mờ - sharpen vừa
            kernel = np.array([[0, -1, 0],
                             [-1,  6, -1],
                             [0, -1, 0]])
            print("      → Làm nét (kernel 6)")
        
        sharpened = cv2.filter2D(img_array, -1, kernel)
        return sharpened
    
    def process_dataset(self, input_dir, output_dir, mode='fix'):
        """
        PHỤC HỒI toàn bộ dataset xấu về chất lượng tốt
        
        Args:
            input_dir: Thư mục dataset XẤU (Train_Bad/Test_Bad/Val_Bad)
            output_dir: Thư mục lưu dataset ĐÃ SỬA (Train_Fixed/Test_Fixed/Val_Fixed)
            mode: 'fix' (sửa ảnh xấu) hoặc 'resize' (chỉ resize)
            
        Returns:
            dict: Thống kê xử lý
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            print(f"❌ Không tìm thấy: {input_path}")
            return None
        
        print(f"\n{'='*70}")
        print(f"🔧 BẮT ĐẦU PHỤC HỒI DATASET XẤU")
        print(f"{'='*70}")
        print(f"📁 Input (ảnh XẤU): {input_path}")
        print(f"💾 Output (ảnh ĐÃ SỬA): {output_path}")
        print(f"🎯 Target size: {self.target_size}")
        print(f"⚡ Mode: {'FIX (sửa mạnh)' if mode == 'fix' else 'RESIZE (chỉ resize)'}")
        print(f"💪 Aggressive fix: {'ON' if self.aggressive_fix else 'OFF'}")
        print(f"{'='*70}\n")
        
        stats = {
            'total_processed': 0,
            'total_failed': 0,
            'classes': {}
        }
        
        # Tìm tất cả các class folders
        class_folders = [d for d in input_path.iterdir() if d.is_dir()]
        
        for class_folder in class_folders:
            class_name = class_folder.name
            output_class_dir = output_path / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Tìm tất cả ảnh trong class
            image_files = list(class_folder.glob("*.jpg")) + \
                         list(class_folder.glob("*.jpeg")) + \
                         list(class_folder.glob("*.png"))
            
            if len(image_files) == 0:
                print(f"⚠️  {class_name}: Không có ảnh")
                continue
            
            print(f"📂 Đang phục hồi {class_name}: {len(image_files)} ảnh xấu...")
            
            processed_count = 0
            failed_count = 0
            fixed_count = 0
            
            # Xử lý từng ảnh với progress bar
            for img_path in tqdm(image_files, desc=f"   {class_name[:28]:28s}", ncols=80):
                try:
                    if mode == 'fix':
                        # PHỤC HỒI ảnh xấu
                        fixed_img = self.fix_bad_image(img_path)
                        fixed_count += 1
                    else:
                        # Chỉ resize
                        img = Image.open(img_path).convert('RGB')
                        img_array = np.array(img)
                        fixed_img = self._resize_image(img_array)
                    
                    if fixed_img is not None:
                        # Lưu ảnh đã sửa
                        output_img_path = output_class_dir / img_path.name
                        img_pil = Image.fromarray(fixed_img.astype(np.uint8))
                        img_pil.save(output_img_path, quality=95, optimize=True)
                        
                        processed_count += 1
                        stats['total_processed'] += 1
                    else:
                        failed_count += 1
                        stats['total_failed'] += 1
                        
                except Exception as e:
                    print(f"\n   ❌ Lỗi: {img_path.name} - {e}")
                    failed_count += 1
                    stats['total_failed'] += 1
            
            stats['classes'][class_name] = {
                'total': len(image_files),
                'fixed': fixed_count if mode == 'fix' else 0,
                'processed': processed_count,
                'failed': failed_count
            }
            
            print(f"   ✅ Đã sửa: {processed_count}, ❌ Lỗi: {failed_count}\n")
        
        # Tổng kết
        print(f"{'='*70}")
        print(f"🎉 HOÀN TẤT PHỤC HỒI DATASET")
        print(f"{'='*70}")
        print(f"📊 Tổng kết:")
        print(f"   ✅ Ảnh đã phục hồi thành công: {stats['total_processed']}")
        print(f"   ❌ Ảnh lỗi: {stats['total_failed']}")
        print(f"\n📁 Dataset sạch đã lưu tại: {output_path.absolute()}")
        print(f"💡 Bây giờ có thể train model với dataset này!")
        print(f"{'='*70}\n")
        
        return stats
    
    def process_all_splits(self, root_dir, output_root_dir, 
                          splits=['Train', 'Val', 'Test'], mode='fix'):
        """
        PHỤC HỒI tất cả splits (Train/Val/Test) từ dataset xấu
        
        Args:
            root_dir: Thư mục gốc chứa Train_Bad/Val_Bad/Test_Bad
            output_root_dir: Thư mục đích (Train_Fixed/Val_Fixed/Test_Fixed)
            splits: List các splits cần xử lý
            mode: 'fix' (sửa ảnh xấu) hoặc 'resize' (chỉ resize)
        """
        root_path = Path(root_dir)
        output_root_path = Path(output_root_dir)
        
        print(f"\n{'='*70}")
        print(f"🔥 BẮT ĐẦU PHỤC HỒI TOÀN BỘ DATASET XẤU")
        print(f"{'='*70}")
        print(f"📁 Input: Dataset XẤU từ {root_path}")
        print(f"💾 Output: Dataset SẠCH vào {output_root_path}")
        print(f"{'='*70}\n")
        
        all_stats = {}
        
        for split in splits:
            split_dir = root_path / split
            if not split_dir.exists():
                print(f"⚠️  Không tìm thấy split: {split}")
                continue
            
            output_split_dir = output_root_path / split
            stats = self.process_dataset(split_dir, output_split_dir, mode=mode)
            
            if stats:
                all_stats[split] = stats
        
        # Tổng kết toàn bộ
        print(f"\n{'='*70}")
        print(f"🏆 TỔNG KẾT PHỤC HỒI TOÀN BỘ DATASET")
        print(f"{'='*70}")
        
        total_all = sum(stats['total_processed'] for stats in all_stats.values())
        failed_all = sum(stats['total_failed'] for stats in all_stats.values())
        
        for split, stats in all_stats.items():
            print(f"\n📦 {split}:")
            print(f"   ✅ Đã sửa thành công: {stats['total_processed']}")
            print(f"   ❌ Lỗi: {stats['total_failed']}")
            print(f"   📋 Classes:")
            for class_name, class_stats in stats['classes'].items():
                print(f"      - {class_name:30s}: {class_stats['processed']:4d} ảnh")
        
        print(f"\n{'='*70}")
        print(f"🎯 TỔNG CỘNG:")
        print(f"   ✅ {total_all} ảnh đã phục hồi thành công")
        print(f"   ❌ {failed_all} ảnh lỗi")
        print(f"\n💡 Dataset sạch đã sẵn sàng để train model!")
        print(f"{'='*70}\n")
        
        return all_stats


# ==========================================
# MAIN - Chạy preprocessing
# ==========================================
if __name__ == "__main__":
    """
    PHỤC HỒI dataset xấu về chất lượng tốt để train model
    
    Luồng sử dụng:
    1. Dùng data_raw.py để tạo dataset xấu (noise/blur/dark)
    2. Dùng file này để SỬA dataset xấu về chất lượng tốt
    3. Train model với dataset đã sửa
    
    Cách chạy:
        python data_preprocessing.py
    """
    
    # ============ CẤU HÌNH ============
    INPUT_DIR = "Tomato/Augmented_Train"  # Dataset XẤU (từ data_raw.py)
    OUTPUT_DIR = "Tomato/Fixed_Train"     # Dataset ĐÃ SỬA (sạch, chất lượng tốt)
    TARGET_SIZE = (256, 256)              # Kích thước đầu ra
    AGGRESSIVE_FIX = True                 # Sửa mạnh tay (True) hay chỉ sửa khi cần (False)
    MODE = 'fix'                          # 'fix' (sửa ảnh xấu) hoặc 'resize' (chỉ resize)
    
    # ============ KHỞI TẠO ============
    preprocessor = DatasetPreprocessor(
        target_size=TARGET_SIZE,
        aggressive_fix=AGGRESSIVE_FIX
    )
    
    # ============ CHẠY PHỤC HỒI ============
    print("\n🔧 XỬ LÝ DATASET XẤU VỀ CHẤT LƯỢNG TỐT")
    print(f"📂 Input: Dataset XẤU từ data_raw.py ({INPUT_DIR})")
    print(f"💾 Output: Dataset SẠCH để train ({OUTPUT_DIR})")
    print("="*70)
    
    stats = preprocessor.process_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        mode=MODE
    )
    
    if stats:
        print("\n✅ Xử lý dataset hoàn tất!")
        print(f"💡 Dataset sạch đã sẵn sàng tại: {OUTPUT_DIR}/")
        print(f"💡 Train model: train_datagen.flow_from_directory('{OUTPUT_DIR}')")
        print(f"\n📊 Thống kê:")
        print(f"   - Tổng ảnh đã sửa: {stats['total_processed']}")
        print(f"   - Tổng ảnh lỗi: {stats['total_failed']}")
        
        # Hiển thị chi tiết từng class
        print(f"\n📋 Chi tiết từng class:")
        for class_name, class_stats in stats['classes'].items():
            print(f"   {class_name:30s}: {class_stats['processed']}/{class_stats['total']} ảnh")
    else:
        print("\n❌ Phục hồi thất bại!")
        exit(1)
