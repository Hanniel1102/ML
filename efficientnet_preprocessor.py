"""
EfficientNet Preprocessing Module
Xử lý ảnh theo chuẩn EfficientNetB0 (224x224)
Luồng tối ưu: Resize -> Analyze -> Conditional Processing (CLAHE/Denoise/Sharpen) -> Normalize
"""

import numpy as np
import cv2
from PIL import Image
import io
import base64


class EfficientNetPreprocessor:
    """
    Xử lý ảnh cho EfficientNetB0 (256x256 - model đã train)
    Chỉ áp dụng xử lý KHI CẦN THIẾT dựa trên phân tích ảnh
    """
    
    def __init__(self, target_size=(256, 256)):
        """
        Args:
            target_size: Kích thước đầu ra (width, height) - MẶC ĐỊNH 256x256
        """
        self.target_size = target_size
        self.processing_steps = []
        
        # EfficientNet normalization parameters (ImageNet pretrained)
        self.mean = np.array([0.485, 0.456, 0.406])  # RGB mean
        self.std = np.array([0.229, 0.224, 0.225])   # RGB std
        
    def process(self, image_input):
        """
        Xử lý ảnh qua tất cả các bước và thu thập kết quả
        
        Args:
            image_input: PIL Image hoặc numpy array hoặc bytes
            
        Returns:
            dict: {
                'final_image': PIL Image,
                'final_array': numpy array (224, 224, 3),
                'steps': [
                    {
                        'name': str,
                        'description': str,
                        'image_base64': str,
                        'metrics': dict
                    }
                ],
                'summary': dict
            }
        """
        self.processing_steps = []
        
        # Convert input to numpy array
        if isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
            img = img.convert('RGB')
            img_array = np.array(img)
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
            img_array = np.array(img)
        else:
            img_array = image_input
            
        # BƯỚC 1: RESIZE to 224x224 (yêu cầu của EfficientNetB0)
        print(f"[EfficientNet Preprocessing] Step 1: Resize to {self.target_size}")
        step1_result = self._step1_resize(img_array)
        processed = step1_result['image']
        self.processing_steps.append(step1_result)
        
        # BƯỚC 2: ANALYZE - Phân tích chất lượng ảnh để quyết định xử lý
        print(f"[EfficientNet Preprocessing] Step 2: Analyze image quality")
        step2_result = self._step2_analyze(processed)
        metrics = step2_result['metrics']
        self.processing_steps.append(step2_result)
        
        # BƯỚC 3: CLAHE - Tăng cường contrast (conditional)
        print(f"[EfficientNet Preprocessing] Step 3: CLAHE (Contrast Enhancement)")
        step3_result = self._step3_clahe(processed, metrics)
        processed = step3_result['image']
        self.processing_steps.append(step3_result)
        
        # BƯỚC 4: DENOISE - Khử nhiễu (conditional)
        print(f"[EfficientNet Preprocessing] Step 4: Denoise")
        step4_result = self._step4_denoise(processed, metrics)
        processed = step4_result['image']
        self.processing_steps.append(step4_result)
        
        # BƯỚC 5: SHARPEN - Làm nét ảnh (conditional)
        print(f"[EfficientNet Preprocessing] Step 5: Sharpen")
        step5_result = self._step5_sharpen(processed, metrics)
        processed = step5_result['image']
        self.processing_steps.append(step5_result)
        
        # BƯỚC 6: NORMALIZE - Chuẩn hóa theo EfficientNet (ImageNet mean/std)
        print(f"[EfficientNet Preprocessing] Step 6: Normalize for EfficientNet")
        step6_result = self._step6_normalize(processed)
        normalized = step6_result['image']
        self.processing_steps.append(step6_result)
        
        # Convert final result to PIL Image (trước khi normalize)
        final_pil = Image.fromarray(processed.astype(np.uint8))
        
        print(f"[EfficientNet Preprocessing] ✅ Completed - Final size: {normalized.shape}")
        
        return {
            'final_image': final_pil,
            'final_array': normalized,  # Đã normalize, sẵn sàng cho model
            'steps': self.processing_steps,
            'summary': self._generate_summary()
        }
    
    def _step1_resize(self, img_array):
        """BƯỚC 1: Resize về kích thước EfficientNetB0"""
        height, width = img_array.shape[:2]
        target_w, target_h = self.target_size
        
        # Resize với INTER_AREA (tốt cho downscale)
        if width > target_w or height > target_h:
            resized = cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_AREA)
            method = 'INTER_AREA (downscale)'
        else:
            # Upscale với INTER_CUBIC
            resized = cv2.resize(img_array, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            method = 'INTER_CUBIC (upscale)'
        
        print(f"  → Resized {width}x{height} → {target_w}x{target_h} using {method}")
        
        return {
            'name': 'Resize',
            'description': f'Resize từ {width}x{height} → {target_w}x{target_h} (EfficientNetB0)',
            'image_base64': self._array_to_base64(resized),
            'image': resized,
            'metrics': {
                'original_size': f'{width}x{height}',
                'resized_size': f'{target_w}x{target_h}',
                'scale_ratio': f'{width/target_w:.2f}x',
                'method': method
            }
        }
    
    def _step2_analyze(self, img_array):
        """BƯỚC 2: Phân tích brightness, contrast, noise level"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 1. Brightness (mean intensity)
        brightness = np.mean(gray)
        brightness_level = self._classify_brightness(brightness)
        
        # 2. Contrast (standard deviation)
        contrast = np.std(gray)
        contrast_level = self._classify_contrast(contrast)
        
        # 3. Noise level (estimate using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_level = self._classify_noise(laplacian_var)
        
        # 4. Sharpness (edge strength)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
        sharpness_level = self._classify_sharpness(edge_strength)
        
        print(f"  → Brightness: {brightness:.1f} ({brightness_level})")
        print(f"  → Contrast: {contrast:.1f} ({contrast_level})")
        print(f"  → Noise: {laplacian_var:.1f} ({noise_level})")
        print(f"  → Sharpness: {edge_strength:.1f} ({sharpness_level})")
        
        return {
            'name': 'Analyze',
            'description': f'{brightness_level} | {contrast_level} | {noise_level} | {sharpness_level}',
            'image_base64': self._array_to_base64(img_array),
            'image': img_array,
            'metrics': {
                'brightness': float(brightness),
                'brightness_level': brightness_level,
                'contrast': float(contrast),
                'contrast_level': contrast_level,
                'noise_variance': float(laplacian_var),
                'noise_level': noise_level,
                'edge_strength': float(edge_strength),
                'sharpness_level': sharpness_level,
                'needs_enhancement': brightness < 120 or brightness > 180,
                'needs_denoising': laplacian_var < 200,
                'needs_sharpening': edge_strength < 25
            }
        }
    
    def _step3_clahe(self, img_array, metrics):
        """BƯỚC 3: CLAHE - Tăng cường contrast (chỉ khi cần)"""
        contrast = metrics['contrast']
        
        if contrast < 40:
            print(f"  → Áp dụng CLAHE (contrast={contrast:.1f} < 40)")
            processed = img_array.copy()
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            if contrast < 25:
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                description = f'Tăng contrast mạnh (CLAHE 2.5) - Contrast={contrast:.1f}'
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                description = f'Tăng contrast (CLAHE 2.0) - Contrast={contrast:.1f}'
            
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            print(f"  → Bỏ qua CLAHE (contrast={contrast:.1f} đã tốt)")
            processed = img_array
            description = f'Không áp dụng - Contrast đã tốt ({contrast:.1f})'
        
        return {
            'name': 'CLAHE',
            'description': description,
            'image_base64': self._array_to_base64(processed),
            'image': processed,
            'metrics': {
                'clahe_applied': contrast < 40,
                'contrast': float(contrast)
            }
        }
    
    def _step4_denoise(self, img_array, metrics):
        """BƯỚC 4: DENOISE - Khử nhiễu (chỉ khi cần)"""
        noise_var = metrics['noise_variance']
        
        if noise_var < 500:
            print(f"  → Áp dụng Denoise (noise={noise_var:.1f} < 500)")
            processed = cv2.bilateralFilter(img_array, 5, 30, 30)
            description = f'Khử nhiễu (Bilateral Filter d=5) - Noise={noise_var:.1f}'
        else:
            print(f"  → Bỏ qua Denoise (noise={noise_var:.1f} đã tốt)")
            processed = img_array
            description = f'Không áp dụng - Ảnh sạch ({noise_var:.1f})'
        
        return {
            'name': 'Denoise',
            'description': description,
            'image_base64': self._array_to_base64(processed),
            'image': processed,
            'metrics': {
                'denoise_applied': noise_var < 500,
                'noise_variance': float(noise_var)
            }
        }
    
    def _step5_sharpen(self, img_array, metrics):
        """BƯỚC 5: SHARPEN - Làm nét ảnh (chỉ khi cần)"""
        edge_strength = metrics['edge_strength']
        
        if edge_strength < 50:
            print(f"  → Áp dụng Sharpen (edge={edge_strength:.1f} < 50)")
            kernel = np.array([[0, -1, 0],
                             [-1,  5, -1],
                             [0, -1, 0]])
            processed = cv2.filter2D(img_array, -1, kernel)
            description = f'Làm nét ảnh (Kernel 5) - Edge={edge_strength:.1f}'
        else:
            print(f"  → Bỏ qua Sharpen (edge={edge_strength:.1f} đã sắc nét)")
            processed = img_array
            description = f'Không áp dụng - Ảnh đã sắc nét ({edge_strength:.1f})'
        
        return {
            'name': 'Sharpen',
            'description': description,
            'image_base64': self._array_to_base64(processed),
            'image': processed,
            'metrics': {
                'sharpen_applied': edge_strength < 50,
                'edge_strength': float(edge_strength)
            }
        }
    
    def _step6_normalize(self, img_array):
        """BƯỚC 6: NORMALIZE đơn giản /255 (như lúc train model)"""
        print(f"  → Normalize: [0-255] → [0-1] (rescale=1./255)")
        
        # Convert to float32 and scale to [0, 1] - ĐÚNG như lúc train
        normalized = img_array.astype(np.float32) / 255.0
        
        # KHÔNG dùng ImageNet mean/std vì model train với rescale=1./255
        
        # For display, convert back to uint8
        display_img = img_array.copy()
        
        return {
            'name': 'Normalize',
            'description': 'Rescale to [0-1]: pixel / 255.0',
            'image_base64': self._array_to_base64(display_img),
            'image': normalized,  # Normalized array for model
            'metrics': {
                'normalized': True,
                'method': 'rescale',
                'formula': 'x / 255.0',
                'range': f'[{normalized.min():.3f}, {normalized.max():.3f}]'
            }
        }
    
    # === HELPER METHODS ===
    
    def _array_to_base64(self, img_array):
        """Convert numpy array to base64 string"""
        img_pil = Image.fromarray(img_array.astype(np.uint8))
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=95)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f'data:image/jpeg;base64,{img_base64}'
    
    def _classify_brightness(self, value):
        """Phân loại độ sáng"""
        if value < 80:
            return 'Quá tối'
        elif value < 120:
            return 'Hơi tối'
        elif value < 180:
            return 'Sáng vừa'
        elif value < 220:
            return 'Hơi sáng'
        else:
            return 'Quá sáng'
    
    def _classify_contrast(self, value):
        """Phân loại độ tương phản"""
        if value < 30:
            return 'Contrast thấp'
        elif value < 50:
            return 'Contrast trung bình'
        elif value < 70:
            return 'Contrast tốt'
        else:
            return 'Contrast cao'
    
    def _classify_noise(self, value):
        """Phân loại mức độ nhiễu"""
        if value < 50:
            return 'Nhiễu cao'
        elif value < 150:
            return 'Nhiễu trung bình'
        elif value < 300:
            return 'Nhiễu thấp'
        else:
            return 'Ảnh sạch'
    
    def _classify_sharpness(self, value):
        """Phân loại độ sắc nét"""
        if value < 15:
            return 'Rất mờ'
        elif value < 25:
            return 'Hơi mờ'
        elif value < 35:
            return 'Sắc nét vừa'
        else:
            return 'Rất sắc nét'
    
    def _generate_summary(self):
        """Tạo tóm tắt quá trình xử lý (6 bước: Resize → Analyze → CLAHE → Denoise → Sharpen → Normalize)"""
        if not self.processing_steps:
            return {}
        
        # Step 1: Resize - always applied
        # Step 2: Analyze - always applied
        analyze_step = self.processing_steps[1]
        metrics = analyze_step['metrics']
        
        # Step 3-5: Check what was applied
        actions_taken = []
        
        # Step 3: CLAHE
        if len(self.processing_steps) > 2:
            clahe_step = self.processing_steps[2]
            if clahe_step.get('metrics', {}).get('clahe_applied'):
                actions_taken.append('Tăng cường contrast (CLAHE)')
        
        # Step 4: Denoise
        if len(self.processing_steps) > 3:
            denoise_step = self.processing_steps[3]
            if denoise_step.get('metrics', {}).get('denoise_applied'):
                actions_taken.append('Khử nhiễu (Bilateral Filter)')
        
        # Step 5: Sharpen
        if len(self.processing_steps) > 4:
            sharpen_step = self.processing_steps[4]
            if sharpen_step.get('metrics', {}).get('sharpen_applied'):
                actions_taken.append('Làm nét ảnh (Sharpen)')
        
        # Step 6: Normalize - always applied (ImageNet mean/std)
        actions_taken.append('Chuẩn hóa ImageNet (mean/std)')
        
        if len(actions_taken) == 1:  # Only normalization was applied
            actions_taken.insert(0, 'Không cần xử lý thêm - Ảnh chất lượng tốt')
        
        return {
            'total_steps': len(self.processing_steps),
            'actions_taken': actions_taken,
            'final_quality': {
                'brightness': metrics['brightness_level'],
                'contrast': metrics['contrast_level'],
                'noise': metrics['noise_level'],
                'sharpness': metrics['sharpness_level']
            },
            'preprocessing_type': 'EfficientNet (Conditional + ImageNet Normalization)'
        }


def preprocess_for_efficientnet(image_input, target_size=(256, 256)):
    """
    Wrapper function để xử lý ảnh cho EfficientNetB0
    
    Args:
        image_input: PIL Image, numpy array, hoặc bytes
        target_size: Kích thước đầu ra (width, height) - MẶC ĐỊNH 256x256
        
    Returns:
        dict: Kết quả xử lý với final_image, final_array và các bước
    """
    preprocessor = EfficientNetPreprocessor(target_size=target_size)
    result = preprocessor.process(image_input)
    return result


if __name__ == '__main__':
    """Test preprocessing với ảnh mẫu"""
    print("=== Testing EfficientNet Preprocessor ===\n")
    
    # Tạo ảnh test (giả lập ảnh tối, ít contrast)
    test_img = np.random.randint(50, 100, (300, 400, 3), dtype=np.uint8)
    
    # Xử lý
    result = preprocess_for_efficientnet(test_img, target_size=(224, 224))
    
    # In kết quả
    print("\n=== PROCESSING STEPS ===")
    for i, step in enumerate(result['steps'], 1):
        print(f"\nStep {i}: {step['name']}")
        print(f"  Description: {step['description']}")
        if 'metrics' in step:
            print(f"  Metrics: {step['metrics']}")
    
    print("\n=== FINAL SUMMARY ===")
    print(f"Total steps: {result['summary']['total_steps']}")
    print(f"Actions taken: {', '.join(result['summary']['actions_taken'])}")
    print(f"Final quality:")
    for key, value in result['summary']['final_quality'].items():
        print(f"  - {key}: {value}")
    
    print(f"\n✅ Final image shape: {result['final_array'].shape}")
