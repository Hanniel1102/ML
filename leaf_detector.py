"""
Leaf Detection Module
Phát hiện xem ảnh đầu vào có phải là ảnh lá hay không
"""
import cv2
import numpy as np
from typing import Tuple, Dict

class LeafDetector:
    """
    Detector để phân biệt ảnh lá và ảnh không phải lá
    Sử dụng các đặc điểm: màu sắc, texture, hình dạng
    """
    
    def __init__(self):
        # Ngưỡng cho các tiêu chí
        self.GREEN_THRESHOLD = 0.15  # Tỷ lệ pixel màu xanh tối thiểu
        self.EDGE_THRESHOLD = 0.05   # Tỷ lệ cạnh tối thiểu
        self.MIN_CONFIDENCE = 0.3    # Confidence tối thiểu để coi là lá
        
    def is_leaf(self, image: np.ndarray, return_details: bool = False) -> Tuple[bool, float] or Tuple[bool, float, Dict]:
        """
        Kiểm tra xem ảnh có phải là lá hay không
        
        Args:
            image: Ảnh đầu vào (BGR hoặc RGB)
            return_details: Trả về chi tiết phân tích
            
        Returns:
            (is_leaf, confidence) hoặc (is_leaf, confidence, details)
        """
        # Chuyển sang RGB nếu cần
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        # 1. Phân tích màu sắc (Green dominance)
        green_score = self._analyze_green_content(image)
        
        # 2. Phân tích texture (Edge density)
        texture_score = self._analyze_texture(image)
        
        # 3. Phân tích hình dạng (Organic shape)
        shape_score = self._analyze_shape(image)
        
        # 4. Phân tích độ sáng (Brightness distribution)
        brightness_score = self._analyze_brightness(image)
        
        # Tính confidence tổng hợp (weighted average)
        weights = {
            'green': 0.4,      # Màu xanh quan trọng nhất
            'texture': 0.25,   # Texture lá có đặc trưng
            'shape': 0.20,     # Hình dạng organic
            'brightness': 0.15 # Phân bố sáng
        }
        
        confidence = (
            green_score * weights['green'] +
            texture_score * weights['texture'] +
            shape_score * weights['shape'] +
            brightness_score * weights['brightness']
        )
        
        is_leaf = confidence >= self.MIN_CONFIDENCE
        
        if return_details:
            details = {
                'green_score': float(green_score),
                'texture_score': float(texture_score),
                'shape_score': float(shape_score),
                'brightness_score': float(brightness_score),
                'confidence': float(confidence),
                'threshold': self.MIN_CONFIDENCE
            }
            return is_leaf, confidence, details
        
        return is_leaf, confidence
    
    def _analyze_green_content(self, image: np.ndarray) -> float:
        """Phân tích tỷ lệ màu xanh trong ảnh"""
        # Chuyển sang HSV để phân tích màu tốt hơn
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Định nghĩa range màu xanh lá (green)
        # Hue: 35-85 (xanh lá), Saturation: 40-255, Value: 40-255
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Tạo mask cho màu xanh
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Tính tỷ lệ pixel xanh
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        
        # Normalize score (0-1)
        score = min(green_ratio / 0.4, 1.0)  # 40% xanh = điểm tối đa
        
        return score
    
    def _analyze_texture(self, image: np.ndarray) -> float:
        """Phân tích texture bằng edge detection"""
        # Chuyển sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Tính tỷ lệ cạnh
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Lá có texture vừa phải (không quá smooth, không quá nhiều edge)
        # Optimal range: 5-20%
        if edge_ratio < 0.05:
            score = edge_ratio / 0.05  # Quá smooth
        elif edge_ratio > 0.20:
            score = 1.0 - (edge_ratio - 0.20) / 0.30  # Quá nhiều edge
        else:
            score = 1.0  # Vùng tối ưu
        
        return max(0, min(score, 1.0))
    
    def _analyze_shape(self, image: np.ndarray) -> float:
        """Phân tích hình dạng organic"""
        # Chuyển sang grayscale và threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Tìm contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.5  # Neutral score nếu không tìm thấy contour
        
        # Lấy contour lớn nhất
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 100:
            return 0.3  # Quá nhỏ
        
        # Tính circularity và elongation
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            return 0.5
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Lá thường có circularity từ 0.3-0.7 (không quá tròn, không quá dài)
        if 0.3 <= circularity <= 0.7:
            score = 1.0
        elif circularity < 0.3:
            score = circularity / 0.3
        else:
            score = 1.0 - (circularity - 0.7) / 0.3
        
        return max(0, min(score, 1.0))
    
    def _analyze_brightness(self, image: np.ndarray) -> float:
        """Phân tích phân bố độ sáng"""
        # Chuyển sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Tính histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Lá thường có phân bố sáng ở range 50-200
        mid_range_ratio = np.sum(hist[50:200])
        
        # Kiểm tra không quá tối, không quá sáng
        too_dark = np.sum(hist[0:50]) > 0.3  # >30% quá tối
        too_bright = np.sum(hist[200:256]) > 0.3  # >30% quá sáng
        
        if too_dark or too_bright:
            score = mid_range_ratio * 0.7
        else:
            score = mid_range_ratio
        
        return score
    
    def get_rejection_reason(self, image: np.ndarray) -> str:
        """Lấy lý do tại sao ảnh không phải là lá"""
        _, _, details = self.is_leaf(image, return_details=True)
        
        reasons = []
        
        if details['green_score'] < 0.3:
            reasons.append(f"Không đủ màu xanh lá ({details['green_score']:.1%})")
        
        if details['texture_score'] < 0.3:
            reasons.append(f"Texture không phù hợp ({details['texture_score']:.1%})")
        
        if details['shape_score'] < 0.3:
            reasons.append(f"Hình dạng không organic ({details['shape_score']:.1%})")
        
        if details['brightness_score'] < 0.3:
            reasons.append(f"Độ sáng không phù hợp ({details['brightness_score']:.1%})")
        
        if not reasons:
            return f"Confidence thấp ({details['confidence']:.1%} < {self.MIN_CONFIDENCE:.1%})"
        
        return "; ".join(reasons)


# Singleton instance
_detector = None

def get_leaf_detector() -> LeafDetector:
    """Lấy instance của LeafDetector (singleton)"""
    global _detector
    if _detector is None:
        _detector = LeafDetector()
    return _detector


# Convenience functions
def is_leaf_image(image: np.ndarray) -> Tuple[bool, float]:
    """
    Kiểm tra nhanh xem ảnh có phải là lá không
    
    Returns:
        (is_leaf, confidence)
    """
    detector = get_leaf_detector()
    return detector.is_leaf(image)


def analyze_leaf_image(image: np.ndarray) -> Dict:
    """
    Phân tích chi tiết ảnh lá
    
    Returns:
        Dictionary chứa is_leaf, confidence và các scores chi tiết
    """
    detector = get_leaf_detector()
    is_leaf, confidence, details = detector.is_leaf(image, return_details=True)
    
    return {
        'is_leaf': is_leaf,
        'confidence': confidence,
        'details': details,
        'reason': detector.get_rejection_reason(image) if not is_leaf else None
    }
