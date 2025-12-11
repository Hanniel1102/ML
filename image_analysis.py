"""
Image Analysis Module
Xử lý phân tích ảnh lá: shape, color, texture/vein detection
"""

import numpy as np
import cv2
from PIL import Image
import io


def analyze_image(image_bytes):
    """
    Phân tích toàn diện ảnh lá
    
    Args:
        image_bytes: Bytes của ảnh
        
    Returns:
        dict: Kết quả phân tích bao gồm shape, color, texture, finalScore
    """
    # Convert bytes to numpy array
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    img_array = np.array(image)
    
    # 1. Phân tích texture và tạo edge mask trước (Texture/Vein Features)
    texture_result = analyze_texture(img_array)
    texture_features = texture_result['features']
    edge_mask = texture_result['edge_mask']
    raw_edge_mask = texture_result['raw_edge_mask']
    
    # 2. Phân tích hình dạng với edge mask (Shape Features)
    shape_features = analyze_shape(img_array, edge_mask)
    
    # 3. Phân tích màu sắc chỉ trong vùng có edge (Color Features)
    color_features = analyze_color(img_array, edge_mask)
    
    # 4. Tạo ảnh xử lý để hiển thị
    processed_images = generate_processed_images(img_array, edge_mask, texture_result)
    
    # 5. Tính điểm tổng hợp
    final_score = calculate_leaf_score(shape_features, color_features, texture_features)
    
    # 6. Kiểm tra ĐƠN GIẢN: Chỉ cần đủ điểm VÀ đủ màu xanh
    green_ratio = float(color_features['greenRatio'])
    
    # Logic đơn giản: Score >= 60% VÀ Green >= 20%
    is_leaf = final_score['score'] >= 0.60 and green_ratio >= 0.20
    
    return {
        'shape': shape_features,
        'color': color_features,
        'texture': texture_features,
        'finalScore': final_score,
        'isLeaf': is_leaf,
        'processedImages': processed_images
    }


def analyze_shape(img_array, edge_mask=None):
    """
    Phân tích đặc trưng hình dạng - chỉ trong vùng có edge (foreground)
    
    Returns:
        dict: aspectRatio, mainObjectRatio, roundness, eccentricity, elongation, greenDensity
    """
    height, width = img_array.shape[:2]
    
    # Aspect Ratio (Tỷ lệ khung hình)
    aspect_ratio = width / height
    
    # Nếu có edge mask, chỉ phân tích vùng có edge
    if edge_mask is not None:
        foreground_pixels = np.sum(edge_mask)
        
        # Tính số pixel xanh trong vùng foreground
        green_pixels = 0
        for y in range(height):
            for x in range(width):
                if edge_mask[y, x]:
                    r, g, b = img_array[y, x]
                    if g > r and g > b and g > 50:  # Màu xanh rõ ràng
                        green_pixels += 1
        
        green_density = green_pixels / foreground_pixels if foreground_pixels > 0 else 0
        
        # KIỂM TRA BẮT BUỘC: Nếu không đủ xanh, đây không phải lá
        if green_density < 0.20:  # Ít nhất 20% phải là xanh lá
            green_density = 0.0
    else:
        # Fallback: phân tích toàn bộ ảnh
        object_pixels = 0
        green_pixels = 0
        
        for y in range(height):
            for x in range(width):
                r, g, b = img_array[y, x]
                
                # Phát hiện vùng không phải background trắng/đen
                brightness = (int(r) + int(g) + int(b)) / 3
                if 20 < brightness < 235:
                    object_pixels += 1
                    if g > r and g > b and g > 50:
                        green_pixels += 1
        
        foreground_pixels = object_pixels
        green_density = green_pixels / foreground_pixels if foreground_pixels > 0 else 0
    
    total_pixels = width * height
    main_object_ratio = foreground_pixels / total_pixels
    
    # Roundness (Độ tròn) - ước lượng từ contour
    if edge_mask is not None:
        contours, _ = cv2.findContours(edge_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            roundness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        else:
            roundness = 0
    else:
        perimeter = np.sqrt(foreground_pixels) * 4
        area = foreground_pixels
        roundness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Eccentricity (Độ lệch tâm)
    eccentricity = abs(aspect_ratio - 1)
    
    return {
        'aspectRatio': f"{aspect_ratio:.3f}",
        'mainObjectRatio': f"{main_object_ratio:.3f}",
        'roundness': f"{min(roundness, 1):.3f}",
        'eccentricity': f"{eccentricity:.3f}",
        'elongation': f"{(aspect_ratio - 1):.3f}" if aspect_ratio > 1 else f"{(1/aspect_ratio - 1):.3f}",
        'greenDensity': f"{green_density:.3f}"
    }


def get_gabor_vein_response(img_gray):
    """Tính tổng phản hồi Gabor ở nhiều hướng để làm nổi bật gân lá."""
    orientations = [0, 45, 90, 135] # Quét các hướng chính
    vein_map = np.zeros_like(img_gray, dtype=np.float32)

    for theta in orientations:
        # Kernel Gabor (Kích thước 9x9, sigma=1.5, lambda=5.0, gamma=0.5)
        # Các tham số này cần được tinh chỉnh theo độ phân giải ảnh
        kernel = cv2.getGaborKernel((9, 9), 1.5, np.deg2rad(theta), 5.0, 0.5, 0, ktype=cv2.CV_32F)
        
        # Lọc ảnh
        filtered = cv2.filter2D(img_gray, cv2.CV_8UC1, kernel)
        
        # Gộp kết quả: Giữ phản hồi mạnh nhất từ các hướng
        np.maximum(vein_map, filtered, out=vein_map)
        
    return vein_map

def morphological_thinning(binary_img, iterations=10):
    """Thinning algorithm to extract skeleton"""
    # ... [Code gốc cho morphological_thinning giữ nguyên]
    skeleton = np.zeros_like(binary_img)
    img = binary_img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    for _ in range(iterations):
        eroded = cv2.erode(img, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        
        if cv2.countNonZero(img) == 0:
            break
            
    return skeleton


def analyze_texture(img_array):
    """
    Phân tích texture và tạo viền gân lá với cải tiến sử dụng Gabor filter, CLAHE, và cải tiến tiền xử lý.
    
    Returns:
        dict: features, edge_mask (outer contour filled), raw_edge_mask, magnitude, leaf_contour
    """
    height, width = img_array.shape[:2]
    
    # Chuyển đổi không gian màu sang HSV
    hsv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Tạo mask cho màu xanh và vàng
    lower_green = np.array([35, 20, 20])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    lower_yellow = np.array([15, 20, 20])
    upper_yellow = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    
    leaf_mask = cv2.bitwise_or(green_mask, yellow_mask)

    # Tiền xử lý bằng CLAHE để tăng cường độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 
    hsv_image[..., 1] = clahe.apply(hsv_image[..., 1])  # Tăng cường kênh Saturation

    # Morphological transformation để làm sạch
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_contour = None
    outer_contour_mask = np.zeros((height, width), dtype=np.uint8)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.003 * cv2.arcLength(largest_contour, True)
        leaf_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        cv2.drawContours(outer_contour_mask, [leaf_contour], -1, 255, -1)
    
    final_edge_mask = (outer_contour_mask > 0).astype(np.uint8)
    
    # Phát hiện gân lá sử dụng Gabor filter
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gabor_response = get_gabor_vein_response(gray_image)

    # Threshold với percentile để làm nổi bật gân lá
    threshold_value = np.percentile(gabor_response[final_edge_mask], 75) if np.sum(final_edge_mask) > 0 else 127
    _, vein_binary = cv2.threshold(gabor_response, threshold_value, 255, cv2.THRESH_BINARY)

    # Morphological closing để nối các gân đứt đoạn
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    vein_connected = cv2.morphologyEx(vein_binary, cv2.MORPH_CLOSE, kernel_connect, iterations=1)

    # Sử dụng thinning để làm mảnh các đường gân
    vein_skeleton = morphological_thinning(vein_connected, iterations=8)
    
    # Đảm bảo vein_skeleton là uint8
    if vein_skeleton.dtype != np.uint8:
        vein_skeleton = vein_skeleton.astype(np.uint8)

    # Áp dụng vào vùng lá (đảm bảo mask là uint8)
    vein_in_leaf = cv2.bitwise_and(vein_skeleton, vein_skeleton, mask=outer_contour_mask.astype(np.uint8))
    
    # Loại bỏ các thành phần nhiễu (blob) nhỏ
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(vein_in_leaf.astype(np.uint8), connectivity=8)
    
    min_vein_area = 5  # Điều chỉnh để phát hiện gân nhỏ hơn
    internal_edges = np.zeros_like(vein_in_leaf)
    kept_components = 0
    
    print(f"[DEBUG Step 1-3] vein_binary={np.sum(vein_binary > 0)}, vein_connected={np.sum(vein_connected > 0)}, vein_skeleton={np.sum(vein_skeleton > 0)}")
    print(f"[DEBUG Step 4-5] vein_in_leaf={np.sum(vein_in_leaf > 0)}, num_labels={num_labels-1}")
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Điều kiện chọn gân đủ lớn hoặc dài
        if area >= min_vein_area or max(width, height) > 5:
            internal_edges[labels == i] = 255
            kept_components += 1
    
    print(f"[DEBUG Step 6] kept_components={kept_components}, internal_edges={np.sum(internal_edges > 0)}")
    
    # FALLBACK 1: Nếu không phát hiện được gì, dùng toàn bộ vein_in_leaf
    if kept_components == 0 or np.sum(internal_edges) < 10:
        print("[DEBUG] FALLBACK 1: Using all vein_in_leaf")
        internal_edges = vein_in_leaf.copy()
    
    # FALLBACK 2: Nếu vẫn rỗng, dùng Sobel edge detection
    if np.sum(internal_edges) < 10:
        print("[DEBUG] FALLBACK 2: Using Sobel edge detection")
        
        # Chỉ làm trong vùng lá (đảm bảo mask là uint8)
        mask_uint8 = outer_contour_mask.astype(np.uint8) if outer_contour_mask.dtype != np.uint8 else outer_contour_mask
        leaf_region = cv2.bitwise_and(img_array, img_array, mask=mask_uint8)
        gray_leaf = cv2.cvtColor(leaf_region, cv2.COLOR_RGB2GRAY)
        
        # Sobel edge detection
        sobelx = cv2.Sobel(gray_leaf, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_leaf, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize và threshold
        magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edges_simple = cv2.threshold(magnitude_norm, 30, 255, cv2.THRESH_BINARY)
        
        # Làm mảnh
        edges_thin = morphological_thinning(edges_simple, iterations=3)
        internal_edges = edges_thin
    
    # Dilate slightly để làm rõ hơn
    kernel_vis = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    internal_edges = cv2.dilate(internal_edges, kernel_vis, iterations=1)
    
    # Tính metrics
    leaf_area = np.sum(final_edge_mask)
    edge_count = np.sum(internal_edges > 0)
    
    print(f"[DEBUG Vein Detection] leaf_area={leaf_area}, edge_count={edge_count}, kept_components={kept_components}")
    
    if leaf_area > 0 and edge_count > 0:
        # Vein density thường từ 2-15% diện tích lá
        vein_density = edge_count / leaf_area
        
        # Scale để 5% density = score 0.5, 10% = 1.0
        vein_score = min(vein_density * 10, 1.0)
        edge_density = vein_density
        
        print(f"[DEBUG Vein Detection] vein_density={vein_density:.4f}, vein_score={vein_score:.3f}")
    else:
        vein_density = 0.0
        vein_score = 0.0
        edge_density = 0.0
        print(f"[DEBUG Vein Detection] WARNING: No veins detected!")
        
    # Tính Contrast (độ phức tạp bề mặt)
    if leaf_area > 0 and np.sum(final_edge_mask) > 0:
        contrast = min(np.std(gray_image[final_edge_mask]) / 60, 1.0)
    else:
        contrast = 0.0
    
    # [Code cho raw_edge_mask và magnitude - để tương thích ngược]
    denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
    sobelx = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edge_mask_adaptive = cv2.threshold(magnitude_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw_edge_mask = edge_mask_adaptive > 0
    
    return {
        'features': {
            'edgeDensity': f"{edge_density:.4f}",
            'veinScore': f"{vein_score:.3f}",
            'contrast': f"{contrast:.3f}"
        },
        'edge_mask': final_edge_mask,
        'raw_edge_mask': raw_edge_mask,
        'magnitude': magnitude,
        'leaf_contour': leaf_contour,
        'internal_edges': internal_edges
    }


def dilate_edge_mask(edge_mask, radius=5):
    """
    Mở rộng edge mask để bao gồm vùng xung quanh
    
    Args:
        edge_mask: Boolean array
        radius: Bán kính mở rộng
        
    Returns:
        numpy.ndarray: Dilated mask
    """
    kernel = np.ones((radius*2+1, radius*2+1), np.uint8)
    dilated = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)


def analyze_color(img_array, edge_mask):
    """
    Phân tích màu sắc chỉ trong vùng có edge (vùng lá)
    
    Args:
        img_array: RGB image array
        edge_mask: Boolean mask của vùng edge
        
    Returns:
        dict: Thông tin màu sắc
    """
    height, width = img_array.shape[:2]
    
    total_r = 0
    total_g = 0
    total_b = 0
    total_h = 0
    total_s = 0
    total_v = 0
    green_pixels = 0
    yellow_pixels = 0
    brown_pixels = 0
    analyzed_pixel_count = 0
    
    # Convert to HSV
    hsv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    for y in range(height):
        for x in range(width):
            # Chỉ phân tích pixel trong vùng có edge
            if edge_mask is None or edge_mask[y, x]:
                analyzed_pixel_count += 1
                
                r, g, b = img_array[y, x]
                h, s, v = hsv_image[y, x]
                
                total_r += int(r)
                total_g += int(g)
                total_b += int(b)
                
                # HSV values in OpenCV: H(0-179), S(0-255), V(0-255)
                # Convert H to degrees (0-360)
                h_degrees = int(h) * 2
                s_norm = int(s) / 255.0
                v_norm = int(v) / 255.0
                
                total_h += h_degrees
                total_s += s_norm
                total_v += v_norm
                
                # Phân loại màu
                if 60 <= h_degrees <= 180 and s_norm > 0.2 and v_norm > 0.2:
                    green_pixels += 1
                elif 30 <= h_degrees < 60 and s_norm > 0.3:
                    yellow_pixels += 1
                elif (h_degrees < 30 or h_degrees > 330) and v_norm < 0.5:
                    brown_pixels += 1
    
    # Nếu không có pixel nào được phân tích, dùng toàn bộ ảnh
    final_count = analyzed_pixel_count if analyzed_pixel_count > 0 else (height * width)
    
    return {
        'avgRed': f"{(total_r / final_count):.2f}",
        'avgGreen': f"{(total_g / final_count):.2f}",
        'avgBlue': f"{(total_b / final_count):.2f}",
        'avgHue': f"{(total_h / final_count):.2f}",
        'avgSaturation': f"{(total_s / final_count):.3f}",
        'avgValue': f"{(total_v / final_count):.3f}",
        'greenRatio': f"{(green_pixels / final_count):.3f}",
        'yellowRatio': f"{(yellow_pixels / final_count):.3f}",
        'brownRatio': f"{(brown_pixels / final_count):.3f}",
        'analyzedPixels': analyzed_pixel_count
    }


def calculate_leaf_score(shape, color, texture):
    """
    Tính điểm tổng hợp để xác định ảnh có phải là lá - nghiêm ngặt hơn
    
    Args:
        shape (dict): Đặc trưng hình dạng.
        color (dict): Đặc trưng màu sắc.
        texture (dict): Đặc trưng texture/gân lá.
        
    Returns:
        dict: Kết quả chấm điểm tổng hợp.
    """
    # 1. Định nghĩa trọng số
    weights = {
        'shape': 0.35,  # Shape: quan trọng để phân biệt vật thể dẹp
        'color': 0.50,  # Color: quan trọng nhất (để xác định lá xanh)
        'texture': 0.15  # Texture: quan trọng để xác định gân lá
    }
    
    # --- 2. Tính Shape Score ---
    aspect_ratio = float(shape['aspectRatio'])
    main_object_ratio = float(shape['mainObjectRatio'])
    green_density = float(shape.get('greenDensity', 0))
    
    # Aspect Ratio Score (0.3-3.0 là lý tưởng)
    if 0.3 <= aspect_ratio <= 3.0:
        aspect_score = 0.9
    elif 0.2 <= aspect_ratio <= 5.0:
        aspect_score = 0.5
    else:
        aspect_score = 0.1
    
    # Object Ratio Score (Object phải chiếm ít nhất 20% ảnh)
    object_score = min(main_object_ratio * 2.5, 1.0) if main_object_ratio > 0.2 else 0.1
    
    # Green Density Score (Mật độ xanh trong vùng vật thể)
    green_score = min(green_density * 5, 1.0) if green_density > 0.1 else 0.0
    
    shape_score = (aspect_score * 0.3 + object_score * 0.3 + green_score * 0.4)
    
    # --- 3. Tính Color Score ---
    green_ratio = float(color['greenRatio'])
    yellow_ratio = float(color.get('yellowRatio', 0))
    avg_saturation = float(color.get('avgSaturation', 0))
    
    # Logic chấm điểm màu sắc (Hoàn thiện logic bị thiếu)
    # Lá phải có ít nhất 20% màu xanh VÀ độ bão hòa đủ cao (avgSaturation > 0.2)
    
    if green_ratio < 0.20 or avg_saturation < 0.25:
        color_score = 0.0  # Loại bỏ nếu không đủ xanh hoặc quá nhạt
    elif green_ratio >= 0.50:
        color_score = 1.0
    else:
        # Scale từ 0.20 (score 0.5) đến 0.50 (score 1.0)
        color_score = 0.5 + (green_ratio - 0.20) * (5/3)
        color_score = min(color_score, 1.0) # Đảm bảo không vượt quá 1.0
        
    # --- 4. Tính Texture Score ---
    vein_score = float(texture['veinScore'])
    
    # Vein score (Mật độ gân) được chuẩn hóa: 0.0 -> 1.0
    # Lá có gân rõ ràng: vein_score > 0.15 là tốt
    if vein_score >= 0.25:
        texture_score = 1.0
    elif vein_score >= 0.10:
        # Scale từ 0.10 (score 0.5) đến 0.25 (score 1.0)
        texture_score = 0.5 + (vein_score - 0.10) * (1 / 0.15) * 0.5 
    else:
        texture_score = vein_score * 5.0 # Tối đa 0.5 nếu vein_score < 0.1

    # --- 5. Tính điểm tổng hợp (Weighted Score) ---
    final_score = (
        shape_score * weights['shape'] +
        color_score * weights['color'] +
        texture_score * weights['texture']
    )
    
    # --- 6. Áp dụng Điều kiện Bắt buộc (Hard Constraints) ---
    
    # Điều kiện BẮT BUỘC để ảnh được xem là lá:
    # 1. Màu xanh (green_ratio) phải đủ cao: ≥ 20%
    # 2. Mật độ xanh trong object (green_density) phải đủ: ≥ 15%
    # 3. Mật độ gân (vein_score) phải có tối thiểu: ≥ 0.05
    
    failed_conditions = 0
    if green_ratio < 0.20:
        failed_conditions += 1
    if green_density < 0.15:
        failed_conditions += 1
    if vein_score < 0.05: # Gân lá phải có tối thiểu
        failed_conditions += 1
        
    # Nếu thiếu quá nhiều điều kiện cốt lõi, giới hạn điểm tối đa
    if failed_conditions == 3:
        final_score = min(final_score, 0.20)  # Tối đa 20%
    elif failed_conditions == 2:
        final_score = min(final_score, 0.40)  # Tối đa 40%
    elif failed_conditions == 1:
        # Giảm nhẹ điểm nếu thiếu 1 yếu tố cốt lõi (ví dụ: gân mờ)
        final_score = final_score * 0.8 
    
    final_score = max(0.0, final_score) # Đảm bảo score không âm

    # --- 7. Xác định Confidence và Recommendation ---
    
    if final_score >= 0.80:
        confidence = 'Rất cao'
        recommendation = 'Đây chắc chắn là ảnh lá cây (Đặc điểm hình học, màu sắc, và cấu trúc gân rõ ràng).'
    elif final_score >= 0.65:
        confidence = 'Cao'
        recommendation = 'Đây có khả năng cao là ảnh lá cây (Đạt ngưỡng cơ bản, sẵn sàng phân loại bệnh).'
    elif final_score >= 0.45:
        confidence = 'Trung bình'
        recommendation = 'Có một số đặc điểm của lá nhưng không rõ ràng, có thể là lá bị che khuất hoặc có nền phức tạp.'
    else:
        confidence = 'Thấp'
        recommendation = 'Không đủ đặc trưng cốt lõi của lá cây (thiếu màu xanh hoặc cấu trúc cơ bản không rõ ràng).'
    
    return {
        'score': final_score,
        'shapeScore': f"{shape_score:.3f}",
        'colorScore': f"{color_score:.3f}",
        'textureScore': f"{texture_score:.3f}",
        'confidence': confidence,
        'recommendation': recommendation
    }

def generate_processed_images(img_array, edge_mask, texture_result):
    """
    Tạo các ảnh xử lý để hiển thị - cải thiện hiển thị gân lá
    
    Returns:
        dict: Base64 encoded images
    """
    import base64
    import io # Import io nếu chưa có
    from PIL import Image # Import Image nếu chưa có

    height, width = img_array.shape[:2]
    magnitude = texture_result['magnitude']
    raw_edge_mask = texture_result.get('raw_edge_mask', edge_mask)
    leaf_contour = texture_result.get('leaf_contour', None)
    internal_edges = texture_result.get('internal_edges', None) # Mask gân lá (255)

    # --- 1. Original image ---
    original_with_mask = img_array.copy()
    original_pil = Image.fromarray(original_with_mask)
    original_buffer = io.BytesIO()
    original_pil.save(original_buffer, format='PNG')
    original_base64 = base64.b64encode(original_buffer.getvalue()).decode()
    
    # --- 2. Edge detection ---
    edge_display = np.zeros((height, width, 3), dtype=np.uint8)
    edge_display[:] = [0, 0, 0] # Background đen
    if leaf_contour is not None:
        cv2.drawContours(edge_display, [leaf_contour], -1, (255, 255, 255), 2)
    
    edge_pil = Image.fromarray(edge_display)
    edge_buffer = io.BytesIO()
    edge_pil.save(edge_buffer, format='PNG')
    edge_base64 = base64.b64encode(edge_buffer.getvalue()).decode()
    
    # --- 3. Color analysis image ---
    hsv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    color_analyzed = np.zeros((height, width, 3), dtype=np.uint8)
    color_analyzed[:] = [40, 40, 40] # Background xám tối
    
    for y in range(height):
        for x in range(width):
            if edge_mask is not None and edge_mask[y, x]:
                h, s, v = hsv_image[y, x]
                h_degrees = int(h) * 2
                s_norm = int(s) / 255.0
                
                # Logic phân loại màu sắc để trực quan hóa
                if 60 <= h_degrees <= 180 and s_norm > 0.2:
                    color_analyzed[y, x] = [0, 255, 0] # Green region
                elif 30 <= h_degrees < 60 and s_norm > 0.3:
                    color_analyzed[y, x] = [255, 255, 0] # Yellow region
                elif (h_degrees < 30 or h_degrees >= 330) and int(v) < 128:
                    color_analyzed[y, x] = [255, 100, 0] # Brown/Red region
                else:
                    color_analyzed[y, x] = img_array[y, x] # Vùng còn lại giữ nguyên
    
    color_pil = Image.fromarray(color_analyzed)
    color_buffer = io.BytesIO()
    color_pil.save(color_buffer, format='PNG')
    color_base64 = base64.b64encode(color_buffer.getvalue()).decode()
    
    # --- 4. Vein detection (CẢI TIẾN TRỰC QUAN HÓA GÂN LÁ) ---
    vein_image = img_array.copy()
    
    # Làm tối nền để gân nổi bật hơn
    for y in range(height):
        for x in range(width):
            if edge_mask is None or not edge_mask[y, x]:
                # Làm tối Background hoàn toàn
                vein_image[y, x] = vein_image[y, x] // 5
            else:
                # Giảm độ sáng của phần lá (Lamina) để gân nổi bật
                r, g, b = vein_image[y, x]
                # Giảm xuống 40% để tạo contrast với gân
                vein_image[y, x] = [int(r * 0.4), int(g * 0.4), int(b * 0.4)]
    
    # Vẽ internal edges (gân lá) màu ĐỎ RỰC RỠ
    if internal_edges is not None:
        # Dilate nhẹ gân để dễ nhìn hơn (tùy chọn)
        kernel_display = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        internal_edges_display = cv2.dilate(internal_edges, kernel_display, iterations=1)
        
        # Overlay gân màu đỏ rực
        vein_mask = internal_edges_display > 0
        vein_image[vein_mask] = [255, 0, 0]  # Đỏ thuần rực rỡ
        
    # Vẽ VIỀN NGOÀI CÙNG màu CYAN (xanh lơ) để phân biệt với gân
    if leaf_contour is not None:
        cv2.drawContours(vein_image, [leaf_contour], -1, (0, 255, 255), 2)
    
    vein_pil = Image.fromarray(vein_image)
    vein_buffer = io.BytesIO()
    vein_pil.save(vein_buffer, format='PNG')
    vein_base64 = base64.b64encode(vein_buffer.getvalue()).decode()
    
    return {
        'original': f"data:image/png;base64,{original_base64}",
        'edge': f"data:image/png;base64,{edge_base64}",
        'color': f"data:image/png;base64,{color_base64}",
        'vein': f"data:image/png;base64,{vein_base64}"
    }