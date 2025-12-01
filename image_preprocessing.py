"""
Module tiền xử lý và kiểm tra ảnh cho hệ thống chẩn đoán bệnh cà chua
Bao gồm: làm nét ảnh, phân tích chất lượng, kiểm tra ảnh lá
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import Tuple, Dict, Any


class ImagePreprocessor:
    """Xử lý tiền xử lý ảnh trước khi dự đoán"""
    
    def __init__(self):
        self.min_sharpness = 30  # Ngưỡng độ nét tối thiểu
        
        # === NGƯỠNG CÂN BẰNG: Chấp nhận lá bệnh NHƯNG từ chối động vật ===
        self.min_green_ratio = 0.05  # 5% - rất thấp cho lá bị bệnh nặng
        self.min_leaf_ratio = 0.12   # 12% - vegetation tổng (xanh + vàng + nâu + bóng)
        
        # Shape score - QUAN TRỌNG để phân biệt lá và động vật
        self.min_leaf_shape_score = 0.42  # Lá dài (aspect 1.5-3), động vật tròn (aspect ~1)
        
        # Texture score - Gân lá vs lông động vật
        self.min_texture_score = 0.28     # Texture cơ bản
        self.excellent_texture_score = 0.50  # Texture xuất sắc (bù đắp cho màu thấp)
        
        self.adaptive_mode = True  # Tự động điều chỉnh ngưỡng dựa trên điều kiện ảnh
        
    def auto_adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        Tự động điều chỉnh độ sáng cho ảnh tối hoặc quá sáng
        
        Args:
            image: Ảnh đầu vào (BGR)
            
        Returns:
            Ảnh đã điều chỉnh độ sáng
        """
        # Chuyển sang HSV để điều chỉnh brightness
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Tính độ sáng trung bình
        mean_brightness = np.mean(v)
        
        # Nếu ảnh quá tối (mean < 80), tăng sáng
        if mean_brightness < 80:
            # Gamma correction để tăng sáng
            gamma = 1.5 if mean_brightness < 50 else 1.3
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            v = cv2.LUT(v, table)
            
        # Nếu ảnh quá sáng (mean > 180), giảm sáng
        elif mean_brightness > 180:
            gamma = 0.7
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            v = cv2.LUT(v, table)
        
        # Merge và convert về BGR
        hsv_adjusted = cv2.merge([h, s, v])
        adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
        
        return adjusted
    
    def enhance_image(self, image: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """
        Tăng cường chất lượng ảnh: làm nét, cân bằng màu sắc
        
        Args:
            image: Ảnh đầu vào (numpy array BGR)
            aggressive: Nếu True, áp dụng xử lý mạnh hơn cho ảnh chất lượng kém
            
        Returns:
            Ảnh đã được tăng cường
        """
        # 0. Tự động điều chỉnh độ sáng trước (quan trọng cho ảnh tối)
        brightness_adjusted = self.auto_adjust_brightness(image)
        
        # 1. Khử nhiễu (tăng cường cho ảnh chất lượng kém)
        h_param = 15 if aggressive else 10
        denoised = cv2.fastNlMeansDenoisingColored(brightness_adjusted, None, h_param, h_param, 7, 21)
        
        # 2. Cân bằng histogram (CLAHE) cho từng kênh màu - mạnh hơn cho ảnh tối
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clip_limit = 3.0 if aggressive else 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. Làm nét ảnh (Unsharp Masking)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return sharpened
    
    def detect_leaf_veins(self, image: np.ndarray, enhanced: np.ndarray = None) -> Dict[str, float]:
        """
        PHÁT HIỆN GÂN LÁ - ĐẶC TRƯNG QUAN TRỌNG NHẤT
        Gân lá là cấu trúc ổn định không thay đổi dù lá bị bệnh/rách/đổi màu
        
        Args:
            image: Ảnh gốc (BGR)
            enhanced: Ảnh đã tăng cường (dùng để phát hiện tốt hơn)
            
        Returns:
            Dict với các điểm phân tích gân lá
        """
        # Sử dụng ảnh tăng cường nếu có
        work_img = enhanced if enhanced is not None else image
        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        
        # === BƯỚC 1: TĂNG CƯỜNG GÂN LÁ ===
        # Morphological operations để làm nổi gân lá
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Top-hat transform - làm nổi gân lá (vùng sáng hơn nền)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Black-hat transform - làm nổi gân lá (vùng tối hơn nền)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Kết hợp cả 2 (gân lá có thể sáng hoặc tối hơn nền)
        veins_enhanced = cv2.add(tophat, blackhat)
        
        # Adaptive threshold để binarize gân lá
        veins_binary = cv2.adaptiveThreshold(
            veins_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # === BƯỚC 2: PHÁT HIỆN CẠNH - GÂN LÁ TẠO CẠNH RÕ ===
        # Canny với nhiều ngưỡng để bắt gân lá mờ
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        edge_density = np.count_nonzero(edges) / edges.size
        vein_density = np.count_nonzero(veins_binary) / veins_binary.size
        
        # === BƯỚC 3: PHÁT HIỆN ĐƯỜNG GÂN (HOUGH LINES) ===
        # Gân lá = đường thẳng, phân nhánh, không ngẫu nhiên
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                                minLineLength=15, maxLineGap=8)
        
        num_lines = len(lines) if lines is not None else 0
        
        # Phân tích góc của các đường gân
        if lines is not None and len(lines) > 5:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                angles.append(angle)
            
            # Gân lá có nhiều góc khác nhau (phân nhánh)
            angle_variance = np.std(angles) if len(angles) > 0 else 0
            # Normalize variance (cao = phân nhánh tốt)
            angle_diversity = min(angle_variance / 30.0, 1.0)
        else:
            angle_diversity = 0
        
        # === BƯỚC 4: SOBEL GRADIENT - GÂN LÁ CÓ GRADIENT MẠNH ===
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize
        mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-6)
        gradient_mean = np.mean(mag_norm)
        gradient_std = np.std(mag_norm)
        
        # === BƯỚC 5: PHÂN TÍCH CẤU TRÚC PHÂN NHÁNH ===
        # Skeleton - trích xuất cấu trúc gân lá
        try:
            if hasattr(cv2, 'ximgproc'):
                skeleton = cv2.ximgproc.thinning(veins_binary)
            else:
                # Fallback: dùng morphological thinning đơn giản
                skeleton = veins_binary
        except:
            skeleton = veins_binary
        
        skeleton_density = np.count_nonzero(skeleton) / skeleton.size
        
        # === TÍNH ĐIỂM TỔNG HỢP ===
        # Lá: edge_density 0.03-0.15, vein_density 0.05-0.25, nhiều lines, góc đa dạng
        # Động vật: edge_density thấp, không có cấu trúc phân nhánh rõ
        
        scores = {
            'edge_density': edge_density,
            'vein_density': vein_density,
            'num_lines': num_lines,
            'angle_diversity': angle_diversity,
            'gradient_mean': gradient_mean,
            'gradient_std': gradient_std,
            'skeleton_density': skeleton_density,
        }
        
        # ĐIỂM TỔNG: Trọng số cao cho đặc trưng gân lá
        vein_score = (
            min(edge_density / 0.10, 1.0) * 0.20 +      # 20% - Cạnh
            min(vein_density / 0.15, 1.0) * 0.30 +      # 30% - Gân lá trực tiếp
            min(num_lines / 40.0, 1.0) * 0.25 +         # 25% - Số đường gân
            angle_diversity * 0.15 +                    # 15% - Phân nhánh
            min(gradient_mean / 0.3, 1.0) * 0.10        # 10% - Gradient
        )
        
        scores['vein_score'] = min(vein_score, 1.0)
        
        return scores
    
    def detect_leaf_texture(self, image: np.ndarray) -> float:
        """
        DEPRECATED: Sử dụng detect_leaf_veins() thay thế
        Giữ lại để tương thích ngược
        """
        veins_data = self.detect_leaf_veins(image)
        return veins_data['vein_score']
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Tính độ nét của ảnh sử dụng Laplacian variance
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Điểm độ nét (càng cao càng nét)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        return float(sharpness)
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        """
        Tính độ sáng trung bình của ảnh
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Độ sáng (0-255)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(np.mean(gray))
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """
        Tính độ tương phản của ảnh
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Độ tương phản (std của pixel)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(np.std(gray))
    
    def analyze_color_distribution(self, image: np.ndarray, enhanced_image: np.ndarray = None) -> Dict[str, float]:
        """
        Phân tích phân bố màu sắc trong ảnh - BAO GỒM LÁ BỆNH
        
        Args:
            image: Ảnh đầu vào (BGR)
            enhanced_image: Ảnh đã tăng cường (dùng để phân tích chính xác hơn)
            
        Returns:
            Dictionary chứa tỷ lệ màu xanh lá, vàng nâu, đen (bệnh)
        """
        # Sử dụng ảnh đã tăng cường nếu có (tốt hơn cho ảnh tối)
        analysis_image = enhanced_image if enhanced_image is not None else image
        
        # Chuyển sang HSV để phân tích màu sắc tốt hơn
        hsv = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Tính độ sáng trung bình để điều chỉnh ngưỡng
        mean_brightness = np.mean(v)
        is_dark = mean_brightness < 80
        
        # Điều chỉnh ngưỡng cho ảnh tối (giảm yêu cầu saturation và value)
        if is_dark:
            # Ngưỡng thấp hơn cho ảnh tối
            green_mask = cv2.inRange(hsv, (35, 15, 15), (85, 255, 255))
            yellow_mask = cv2.inRange(hsv, (15, 15, 15), (35, 255, 255))  # Vàng
            brown_mask = cv2.inRange(hsv, (5, 15, 15), (25, 255, 200))    # Nâu
            sat_threshold = 25
        else:
            # Ngưỡng thông thường
            green_mask = cv2.inRange(hsv, (35, 30, 30), (85, 255, 255))
            yellow_mask = cv2.inRange(hsv, (15, 30, 30), (35, 255, 255))
            brown_mask = cv2.inRange(hsv, (5, 30, 20), (25, 255, 200))
            sat_threshold = 40
        
        # PHÁT HIỆN MẢNG ĐEN/TỐI VÀ BÓNG (QUAN TRọNG)
        # Mảng đen: value thấp, bất kể hue/saturation
        dark_spots_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))  # Vùng rất tối
        
        # Bóng: value thấp (60-120) nhưng saturation trung bình (20-80) - màu xám tối
        shadow_mask = cv2.inRange(hsv, (0, 20, 60), (180, 80, 120))
        
        # Tính các tỷ lệ
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        yellow_ratio = np.sum(yellow_mask > 0) / yellow_mask.size
        brown_ratio = np.sum(brown_mask > 0) / brown_mask.size
        dark_spots_ratio = np.sum(dark_spots_mask > 0) / dark_spots_mask.size
        shadow_ratio = np.sum(shadow_mask > 0) / shadow_mask.size
        
        # Tổng hợp: VEGETATION = xanh + vàng + nâu + đen + bóng
        yellow_brown_ratio = yellow_ratio + brown_ratio
        leaf_ratio = green_ratio + yellow_brown_ratio + min(dark_spots_ratio, 0.4) + shadow_ratio * 0.5
        
        # Tính tỷ lệ độ bão hòa cao (màu sắc rõ ràng)
        high_saturation_ratio = np.sum(s > sat_threshold) / s.size
        
        return {
            'green_ratio': float(green_ratio),
            'yellow_ratio': float(yellow_ratio),
            'brown_ratio': float(brown_ratio),
            'yellow_brown_ratio': float(yellow_brown_ratio),
            'dark_spots_ratio': float(dark_spots_ratio),
            'shadow_ratio': float(shadow_ratio),  # NEW: Tỷ lệ bóng
            'leaf_ratio': float(leaf_ratio),  # Tổng vegetation bao gồm bệnh + bóng
            'high_saturation_ratio': float(high_saturation_ratio),
            'is_dark_image': is_dark,
            'mean_brightness': float(mean_brightness)
        }
    
    def is_leaf_image(self, image: np.ndarray, verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """
        Kiểm tra xem ảnh có phải là ảnh lá cây không
        
        Args:
            image: Ảnh đầu vào (numpy array BGR)
            verbose: In chi tiết kết quả
            
        Returns:
            (is_leaf, details): Tuple gồm boolean và dictionary chi tiết
        """
        details = {}
        
        # Kiểm tra độ sáng trước để quyết định có cần xử lý aggressive không
        brightness_initial = self.calculate_brightness(image)
        is_dark = brightness_initial < 80
        is_very_dark = brightness_initial < 50
        
        # Tăng cường ảnh trước khi phân tích (quan trọng cho ảnh tối)
        enhanced = self.enhance_image(image, aggressive=is_dark)
        
        # 1. Kiểm tra độ nét (sử dụng ảnh đã tăng cường)
        sharpness = self.calculate_sharpness(enhanced)
        details['sharpness'] = sharpness
        # Giảm ngưỡng cho ảnh tối
        min_sharpness = 20 if is_dark else self.min_sharpness
        details['is_sharp_enough'] = sharpness >= min_sharpness
        
        # 2. Kiểm tra độ sáng (ảnh gốc)
        brightness = brightness_initial
        details['brightness'] = brightness
        details['original_brightness'] = brightness_initial
        # Chấp nhận ảnh tối hơn
        details['is_brightness_ok'] = 20 < brightness < 240
        
        # 3. Phân tích màu sắc (sử dụng CẢ ảnh gốc VÀ ảnh tăng cường)
        color_dist = self.analyze_color_distribution(image, enhanced)
        details.update(color_dist)
        
        # === CHIẾN LƯỢC MỚI: ƯU TIÊN GÂN LÁ ===
        # Gân lá là đặc trưng ổn định nhất, không đổi dù lá bị bệnh/rách/đổi màu
        
        # Phát hiện gân lá chi tiết
        vein_analysis = self.detect_leaf_veins(image, enhanced)
        details.update(vein_analysis)
        
        # Giữ texture_score cho tương thích
        texture_score = vein_analysis['vein_score']
        details['texture_score'] = texture_score
        
        # Adaptive thresholds dựa trên điều kiện ảnh
        min_green = 0.03 if is_very_dark else (0.05 if is_dark else self.min_green_ratio)
        min_leaf = 0.10 if is_very_dark else (0.12 if is_dark else self.min_leaf_ratio)
        
        # === TÍNH CONTOURS CHO SHAPE DETECTION ===
        gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        canny_low = 30 if is_dark else 50
        canny_high = 100 if is_dark else 150
        edges = cv2.Canny(gray_enhanced, canny_low, canny_high)
        contours_detected, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # === CHIẾN LƯỢC PHÂN TẦNG ===
        # Mục tiêu: Chấp nhận lá bệnh (màu thấp) NHƯNG từ chối động vật (shape + texture khác)
        
        # Bước 1: Tính điểm hình dạng lá (QUAN TRỌNG)
        leaf_shape_score = self._calculate_leaf_shape_score(enhanced, contours_detected)
        details['leaf_shape_score'] = leaf_shape_score
        
        # Bước 2: Phân loại dựa trên GÂN LÁ (ưu tiên cao nhất)
        has_shadow = color_dist.get('shadow_ratio', 0) >= 0.08
        
        # Gân lá - tiêu chí chính
        has_veins = texture_score >= 0.25              # Có gân lá cơ bản
        has_strong_veins = texture_score >= 0.40       # Gân lá rõ ràng
        has_excellent_veins = texture_score >= 0.60    # Gân lá xuất sắc
        
        # Gân lá chi tiết
        has_vein_structure = (vein_analysis['vein_density'] >= 0.05 and 
                             vein_analysis['num_lines'] >= 10)
        
        # Shape - bổ trợ
        has_good_shape = leaf_shape_score >= self.min_leaf_shape_score
        has_excellent_shape = leaf_shape_score >= 0.60
        
        # Texture (legacy) 
        has_texture = texture_score >= self.min_texture_score
        has_strong_texture = texture_score >= self.excellent_texture_score
        
        # === 7 TRƯỜNG HỢP CHẤP NHẬN - ƯU TIÊN GÂN LÁ ===
        
        # 1. LÁ KHỎE: Xanh + gân lá hoặc shape
        case_healthy = (color_dist['green_ratio'] >= min_green and 
                       (has_veins or has_good_shape))
        
        # 2. LÁ BỆNH NHẸ: Vegetation + gân lá hoặc shape
        case_diseased = (color_dist['leaf_ratio'] >= min_leaf and 
                        color_dist['green_ratio'] >= 0.02 and
                        (has_veins or has_good_shape))
        
        # 3. LÁ BỆNH NẶNG/RÁCH: ƯU TIÊN GÂN LÁ
        # KEY: Chỉ cần gân lá rõ + một chút vegetation, bỏ qua shape
        case_severely_diseased = (has_strong_veins and  # Gân lá rõ ràng
                                 color_dist['leaf_ratio'] >= 0.10 and  # Vegetation thấp OK
                                 (color_dist['green_ratio'] >= 0.02 or  # Chút xanh
                                  color_dist['yellow_brown_ratio'] >= 0.05))  # Hoặc vàng/nâu
        
        # 4. LÁ CÓ BÓNG: Gân lá + bóng + chút màu
        case_shadow = (has_shadow and 
                      has_veins and  # Chỉ cần gân lá cơ bản
                      color_dist['green_ratio'] >= 0.02)
        
        # 5. LÁ BỊ SÂU ĂN: Gân lá xuất sắc (bù đắp mọi thứ)
        # Gân lá rất rõ → chắc chắn là lá
        case_damaged = (has_excellent_veins and  # Vein score ≥ 0.60
                       color_dist['green_ratio'] >= 0.01)  # Chỉ cần 1% xanh
        
        # 6. CÓ CẤU TRÚC GÂN: Phát hiện được nhiều đường gân phân nhánh
        # Trường hợp mới dựa hoàn toàn vào gân lá
        case_vein_structure = (has_vein_structure and  # Vein density + lines
                              color_dist['leaf_ratio'] >= 0.08 and  # Vegetation tối thiểu
                              vein_analysis['angle_diversity'] >= 0.3)  # Phân nhánh
        
        # 7. LÁ NHỎ/MẢM LÁ: Shape xuất sắc + gân lá
        case_small = (has_excellent_shape and 
                     has_veins and
                     color_dist['green_ratio'] >= 0.02)
        
        # === QUYẾT ĐỊNH CUỐI CÙNG ===
        is_valid_leaf = (case_healthy or case_diseased or case_severely_diseased or 
                        case_shadow or case_damaged or case_vein_structure or case_small)
        
        # Lưu chi tiết
        details['has_enough_green'] = is_valid_leaf
        details['is_diseased_leaf'] = case_diseased or case_severely_diseased or (case_shadow and color_dist['yellow_brown_ratio'] > 0.05)
        details['has_shadow'] = case_shadow
        details['is_damaged_leaf'] = case_damaged or case_severely_diseased
        details['is_small_leaf'] = case_small
        details['has_vein_structure'] = case_vein_structure
        details['validation_case'] = (
            'healthy' if case_healthy else
            'diseased' if case_diseased else
            'severely_diseased' if case_severely_diseased else
            'vein_structure' if case_vein_structure else
            'shadow' if case_shadow else
            'damaged' if case_damaged else
            'small' if case_small else
            'none'
        )
        details['adaptive_green_threshold'] = min_green
        details['adaptive_leaf_threshold'] = min_leaf
        
        # 4. Kiểm tra kích thước
        height, width = image.shape[:2]
        details['resolution'] = f"{width}x{height}"
        details['is_resolution_ok'] = width >= 100 and height >= 100
        
        # 5. Phát hiện edge và contour (đã tính ở trên cho shape detection)
        edge_ratio = np.sum(edges > 0) / edges.size
        details['edge_ratio'] = float(edge_ratio)
        # Giảm yêu cầu edge cho ảnh tối
        min_edge = 0.03 if is_dark else 0.05
        details['has_enough_edges'] = edge_ratio > min_edge
        
        # 6. Tính main_object_ratio từ contours đã detect
        if contours_detected:
            # Lấy contour lớn nhất
            largest_contour = max(contours_detected, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            image_area = width * height
            contour_ratio = contour_area / image_area
            details['main_object_ratio'] = float(contour_ratio)
            # Giảm yêu cầu cho ảnh tối
            min_contour = 0.05 if is_dark else 0.1
            details['has_main_object'] = contour_ratio > min_contour
        else:
            details['main_object_ratio'] = 0.0
            details['has_main_object'] = False
        
        # 7. Phân tích texture (lá có texture đặc trưng) - dùng ảnh đã tăng cường
        contrast = self.calculate_contrast(enhanced)
        details['contrast'] = contrast
        # Giảm yêu cầu contrast cho ảnh tối
        min_contrast = 15 if is_dark else 20
        details['has_good_texture'] = contrast > min_contrast
        
        # 8. Xác nhận has_leaf_shape từ leaf_shape_score đã tính ở trên
        details['has_leaf_shape'] = details.get('leaf_shape_score', 0) >= self.min_leaf_shape_score
        
        # DEBUG: In ra thông tin kiểm tra
        if verbose:
            print(f"  DEBUG - Leaf shape score: {leaf_shape_score:.3f} (threshold: {self.min_leaf_shape_score})")
            print(f"  DEBUG - Has leaf shape: {details['has_leaf_shape']}")
        
        # KẾT LUẬN TỔNG HỢP
        # Lưu thông tin ảnh đã tăng cường để sử dụng sau
        details['enhanced_image'] = enhanced
        details['is_dark_detected'] = is_dark
        details['is_very_dark'] = is_very_dark
        
        # === ĐIỀU KIỆN BẮT BUỘC (SIMPLIFIED) ===
        # Đã validate bằng logic 5 trường hợp ở trên
        # Core: PHẢI là 1 trong 5 trường hợp
        core_check_passed = details['has_enough_green']  # is_valid_leaf
        
        # Điều kiện bổ trợ
        supporting_checks = [
            details['is_sharp_enough'],          # Đủ nét
            details['is_brightness_ok'],         # Độ sáng hợp lý
            details['is_resolution_ok'],         # Độ phân giải đủ
            details['has_enough_edges'],         # Có đường viền
            details['has_main_object'],          # Có đối tượng chính
        ]
        
        # Tính điểm
        supporting_passed = sum(supporting_checks)
        total_supporting = len(supporting_checks)
        
        # Confidence dựa trên case match và supporting
        if core_check_passed:
            base_confidence = 70  # Base cho việc match được case
            supporting_bonus = (supporting_passed / total_supporting) * 30
            confidence = base_confidence + supporting_bonus
            
            # Bonus cho texture/shape score cao
            if details.get('texture_score', 0) >= 0.50:
                confidence = min(100, confidence + 5)
            if details.get('leaf_shape_score', 0) >= 0.60:
                confidence = min(100, confidence + 5)
        else:
            confidence = (supporting_passed / total_supporting) * 40  # Thấp nếu không match case
        
        details['passed_checks'] = 1 if core_check_passed else 0
        details['total_checks'] = 1
        details['core_passed'] = 1 if core_check_passed else 0
        details['supporting_passed'] = supporting_passed
        details['confidence'] = confidence
        
        # Acceptance: Core PHẢI pass + ít nhất 40% supporting (ảnh tối) hoặc 50% (ảnh sáng)
        acceptance_threshold = 0.40 if is_dark else 0.50
        supporting_check_passed = supporting_passed >= (len(supporting_checks) * acceptance_threshold)
        
        is_leaf = core_check_passed and supporting_check_passed
        
        details['is_leaf'] = is_leaf
        details['core_check_passed'] = core_check_passed
        details['acceptance_threshold'] = acceptance_threshold
        details['recommendation'] = self._get_recommendation(details)
        
        if verbose:
            print("\n" + "="*70)
            print("📊 KẾT QUẢ PHÂN TÍCH ẢNH")
            print("="*70)
            print(f"✓ Độ nét: {sharpness:.2f} {'✅' if details['is_sharp_enough'] else '❌'}")
            print(f"✓ Độ sáng: {brightness:.2f} {'✅' if details['is_brightness_ok'] else '❌'}")
            print(f"✓ Tỷ lệ màu xanh lá: {color_dist['green_ratio']*100:.2f}% {'✅' if details['has_enough_green'] else '❌'}")
            print(f"✓ Độ phân giải: {width}x{height} {'✅' if details['is_resolution_ok'] else '❌'}")
            print(f"✓ Tỷ lệ đường viền: {edge_ratio*100:.2f}% {'✅' if details['has_enough_edges'] else '❌'}")
            print(f"✓ Đối tượng chính: {details['main_object_ratio']*100:.2f}% {'✅' if details['has_main_object'] else '❌'}")
            print(f"✓ Texture/Tương phản: {contrast:.2f} {'✅' if details['has_good_texture'] else '❌'}")
            print(f"\n📈 Điểm tin cậy: {confidence:.2f}%")
            print(f"🎯 Kết luận: {'ĐÂY LÀ ẢNH LÁ ✅' if is_leaf else 'KHÔNG PHẢI ẢNH LÁ ❌'}")
            print(f"💡 Khuyến nghị: {details['recommendation']}")
            print("="*70)
        
        return is_leaf, details
    
    def _calculate_leaf_shape_score(self, image: np.ndarray, contours: list) -> float:
        """
        Tính điểm hình dạng lá dựa trên các đặc trưng:
        - Tỷ lệ aspect ratio (lá thường dài hơn rộng)
        - Độ phức tạp của contour (lá có răng cưa)
        - Solidity (tỷ lệ fill)
        - HỖ TRỢ LÁ RÁCH/BỆNH: Xem xét nhiều contours
        
        Returns:
            Điểm từ 0-1, càng cao càng giống lá
        """
        if not contours or len(contours) == 0:
            return 0.0
        
        # === CHIẾN LƯỢC MỚI: Xử lý lá rách/bệnh ===
        # Nếu có nhiều contours (lá rách), tổng hợp chúng lại
        total_area = sum(cv2.contourArea(c) for c in contours)
        
        # Lấy top 3 contours lớn nhất (trường hợp lá rách thành nhiều mảnh)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        top_contours = sorted_contours[:min(3, len(sorted_contours))]
        
        # Merge top contours để tính bounding box tổng thể
        if len(top_contours) == 1:
            largest_contour = top_contours[0]
            area = cv2.contourArea(largest_contour)
        else:
            # Lá rách: merge nhiều contours
            all_points = np.vstack([c for c in top_contours])
            largest_contour = cv2.convexHull(all_points)
            area = total_area  # Dùng tổng area thực tế
        
        if area < 50:  # Giảm ngưỡng cho lá nhỏ/rách
            return 0.0
        
        # Tính bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
        
        # Tính solidity (tỷ lệ giữa area và convex hull)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-5) if hull_area > 0 else 0
        
        # Tính perimeter complexity
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)
        
        # Lá cây thường có:
        # - Aspect ratio: 1.3 - 3.5 (dài rõ rệt, KHÔNG gần vuông như chó/mèo)
        # - Solidity: 0.75 - 0.92 (không quá lồi lỗm, không quá phẳng)
        # - Circularity: 0.25 - 0.65 (KHÔNG tròn/vuông)
        # - Area ratio: chiếm 15-80% ảnh (loại background lớn)
        
        image_area = image.shape[0] * image.shape[1]
        area_ratio = area / image_area
        
        score = 0.0
        
        # Điểm aspect ratio (CHẶT HƠN)
        if 1.4 <= aspect_ratio <= 3.5:
            score += 0.35  # Lá thường dài rõ rệt
        elif 1.2 <= aspect_ratio < 1.4:
            score += 0.10  # Gần vuông quá (có thể là động vật)
        elif aspect_ratio > 3.5:
            score += 0.20  # Rất dài, có thể là lá hẹp
        else:
            score += 0.0   # Quá vuông, không phải lá
        
        # Điểm solidity (CHẶT HƠN)
        if 0.75 <= solidity <= 0.92:
            score += 0.30  # Lá có độ đặc hợp lý
        elif 0.92 < solidity <= 0.98:
            score += 0.05  # Quá đặc - có thể là động vật/đồ vật
        
        # Điểm circularity (CHẶT HƠN)
        if 0.25 <= circularity <= 0.65:
            score += 0.25  # Không tròn, có hình dạng bất đối xứng
        elif circularity > 0.70:
            score += 0.0   # Quá tròn - không phải lá
        
        # Điểm area ratio
        if 0.15 <= area_ratio <= 0.80:
            score += 0.10  # Chiếm diện tích hợp lý
        
        return min(score, 1.0)
    
    def _get_recommendation(self, details: Dict[str, Any]) -> str:
        """Đưa ra khuyến nghị dựa trên kết quả phân tích"""
        recommendations = []
        is_dark = details.get('is_dark_detected', False)
        is_diseased = details.get('is_diseased_leaf', False)
        
        # Kiểm tra có phải ảnh lá không
        if not details.get('core_check_passed', False):
            green_pct = details.get('green_ratio', 0) * 100
            leaf_pct = details.get('leaf_ratio', 0) * 100
            shape_score = details.get('leaf_shape_score', 0)
            texture_sc = details.get('texture_score', 0)
            shadow_pct = details.get('shadow_ratio', 0) * 100
            
            # Tính lại biến này để sử dụng trong error message
            has_color_and_shape_check = details['has_enough_green'] and details['has_leaf_shape']
            has_texture_and_shape_check = details.get('texture_score', 0) >= self.min_texture_score and shape_score >= 0.40
            
            reasons = []
            if not has_color_and_shape_check and not has_texture_and_shape_check:
                if texture_sc < 0.30:
                    reasons.append(f"không có texture gân lá (score: {texture_sc:.2f}, cần ≥0.30)")
                if green_pct < 8 and shadow_pct < 10:
                    reasons.append(f"màu xanh quá thấp ({green_pct:.1f}%) và không có bóng")
                if shape_score < 0.40:
                    reasons.append(f"hình dạng không giống lá (score: {shape_score:.2f})")
            
            recommendations.append(
                f"KHÔNG PHẢI ẢNH LÁ CÂY - Lý do: {', '.join(reasons)}. "
                f"Vui lòng chụp ảnh lá cà chua rõ ràng (chấp nhận lá có bóng, lá bị sâu ăn)"
            )
            return " | ".join(recommendations)
        
        if not details['is_sharp_enough']:
            recommendations.append("Ảnh bị mờ, hãy chụp lại với camera ổn định hơn")
        
        if not details['is_brightness_ok']:
            if details['brightness'] < 30:
                if is_dark:
                    recommendations.append("Ảnh rất tối, hệ thống đã tự động tăng sáng nhưng kết quả có thể kém chính xác. Khuyến nghị chụp lại ở nơi sáng hơn")
                else:
                    recommendations.append("Ảnh quá tối, hãy chụp ở nơi có ánh sáng tốt hơn")
            else:
                recommendations.append("Ảnh quá sáng, tránh ánh sáng trực tiếp")
        
        if not details['has_main_object']:
            recommendations.append("Không phát hiện đối tượng rõ ràng, hãy chụp gần lá hơn")
        
        if recommendations:
            return " | ".join(recommendations)
        else:
            has_shadow_detected = details.get('has_shadow', False)
            is_damaged = details.get('is_damaged_leaf', False)
            
            case = details.get('validation_case', 'unknown')
            vein_score = details.get('vein_score', 0)
            num_veins = details.get('num_lines', 0)
            
            if case == 'severely_diseased':
                return f"Ảnh lá bị bệnh nặng đã được nhận diện nhờ phát hiện gân lá (score: {vein_score:.2f}, {num_veins} đường gân), có thể dự đoán"
            elif case == 'vein_structure':
                return f"Ảnh lá được xác nhận nhờ cấu trúc gân lá rõ ràng ({num_veins} đường gân phân nhánh), có thể dự đoán"
            elif is_diseased or case == 'diseased':
                return "Ảnh lá bệnh đã được nhận diện (có mảng vàng/nâu/đen), có thể dự đoán"
            elif has_shadow_detected or case == 'shadow':
                return f"Ảnh lá có bóng đen đã được nhận diện nhờ phát hiện gân lá (score: {vein_score:.2f}), có thể dự đoán"
            elif is_damaged or case == 'damaged':
                return f"Ảnh lá bị sâu ăn/rách đã được nhận diện nhờ gân lá xuất sắc (score: {vein_score:.2f}), có thể dự đoán"
            elif case == 'small':
                return "Ảnh lá nhỏ/mảng lá đã được nhận diện nhờ hình dạng đặc trưng, có thể dự đoán"
            elif is_dark:
                return "Ảnh hơi tối nhưng đã được tự động tăng cường, có thể dự đoán (khuyến nghị chụp ở nơi sáng hơn)"
            else:
                return "Ảnh đạt chất lượng tốt, có thể dự đoán"
    
    def preprocess_for_prediction(self, image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Tiền xử lý ảnh đầy đủ trước khi đưa vào model
        
        Args:
            image: Ảnh đầu vào (numpy array BGR)
            target_size: Kích thước mục tiêu (width, height)
            
        Returns:
            Ảnh đã được xử lý và resize
        """
        # Kiểm tra độ sáng để quyết định mức độ xử lý
        brightness = self.calculate_brightness(image)
        is_dark = brightness < 80
        
        # 1. Tăng cường chất lượng (aggressive mode cho ảnh tối)
        enhanced = self.enhance_image(image, aggressive=is_dark)
        
        # 2. Resize với interpolation phù hợp
        resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return resized
    
    def process_pil_image(self, pil_image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Xử lý ảnh PIL Image
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            (original_cv2, enhanced_cv2): Tuple của ảnh gốc và ảnh đã tăng cường
        """
        # Convert PIL to numpy array (RGB)
        rgb_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        # Enhance
        enhanced = self.enhance_image(bgr_array)
        
        return bgr_array, enhanced


class LeafDetector:
    """Phát hiện và phân đoạn lá trong ảnh"""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
    
    def segment_leaf(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Phân đoạn lá từ nền
        
        Args:
            image: Ảnh đầu vào (BGR)
            
        Returns:
            (masked_image, mask): Ảnh đã loại bỏ nền và mask
        """
        # Chuyển sang HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Tạo mask cho màu xanh lá
        # Range 1: Xanh lá nhạt đến đậm
        lower_green1 = np.array([35, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # Range 2: Màu vàng/nâu (lá bệnh)
        lower_green2 = np.array([20, 40, 40])
        upper_green2 = np.array([35, 255, 255])
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        
        # Kết hợp masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations để làm sạch mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Áp dụng mask lên ảnh gốc
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        return masked_image, mask
    
    def detect_leaf_region(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Phát hiện vùng chứa lá và trả về thông tin chi tiết
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Dictionary chứa thông tin về vùng lá
        """
        # Segment lá
        masked_image, mask = self.segment_leaf(image)
        
        # Tìm contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                'found': False,
                'message': 'Không phát hiện được lá trong ảnh'
            }
        
        # Lấy contour lớn nhất (giả định là lá)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Tính các thông số
        area = cv2.contourArea(largest_contour)
        x, y, w, h = cv2.boundingRect(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Tính circularity (độ tròn)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        return {
            'found': True,
            'area': float(area),
            'bounding_box': (int(x), int(y), int(w), int(h)),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'contour': largest_contour,
            'masked_image': masked_image,
            'mask': mask
        }


# Hàm tiện ích để sử dụng nhanh
def quick_check_leaf(image_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Kiểm tra nhanh xem file ảnh có phải là ảnh lá không
    
    Args:
        image_path: Đường dẫn đến file ảnh
        
    Returns:
        (is_leaf, details): Kết quả kiểm tra
    """
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        return False, {'error': 'Không thể đọc file ảnh'}
    
    # Kiểm tra
    preprocessor = ImagePreprocessor()
    return preprocessor.is_leaf_image(image, verbose=True)


def preprocess_and_check(pil_image: Image.Image, target_size: Tuple[int, int] = (256, 256)) -> Dict[str, Any]:
    """
    Hàm tổng hợp: kiểm tra và tiền xử lý ảnh
    
    Args:
        pil_image: PIL Image object
        target_size: Kích thước đích
        
    Returns:
        Dictionary chứa tất cả thông tin
    """
    preprocessor = ImagePreprocessor()
    
    # Convert PIL to CV2
    original_cv2, _ = preprocessor.process_pil_image(pil_image)
    
    # Kiểm tra có phải ảnh lá không
    is_leaf, details = preprocessor.is_leaf_image(original_cv2)
    
    # Nếu là ảnh lá, tiền xử lý
    if is_leaf:
        enhanced = preprocessor.preprocess_for_prediction(original_cv2, target_size)
        
        # Convert back to RGB for display
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        enhanced_pil = Image.fromarray(enhanced_rgb)
        
        return {
            'is_leaf': True,
            'details': details,
            'enhanced_image': enhanced_pil,
            'ready_for_prediction': True
        }
    else:
        return {
            'is_leaf': False,
            'details': details,
            'enhanced_image': None,
            'ready_for_prediction': False
        }


if __name__ == "__main__":
    """Test module"""
    print("="*70)
    print("🧪 TEST MODULE TIỀN XỬ LÝ ẢNH")
    print("="*70)
    
    # Test với ảnh mẫu
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\n📸 Đang kiểm tra ảnh: {image_path}")
        is_leaf, details = quick_check_leaf(image_path)
        
        if is_leaf:
            print("\n✅ ẢNH HỢP LỆ - Có thể sử dụng để dự đoán")
        else:
            print("\n❌ ẢNH KHÔNG HỢP LỆ")
            print(f"💡 Lý do: {details.get('recommendation', 'Không xác định')}")
    else:
        print("\n💡 Cách sử dụng:")
        print("   python image_preprocessing.py <đường_dẫn_ảnh>")
        print("\nVí dụ:")
        print("   python image_preprocessing.py test_image.jpg")
