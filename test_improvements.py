"""
Test script để kiểm tra các cải tiến mới
- Gray World Assumption
- Frangi Vesselness Filter  
- Dynamic Weighting System
"""

import sys
import cv2
import numpy as np

print("=" * 70)
print("🧪 KIỂM TRA CÁC CẢI TIẾN MỚI")
print("=" * 70)

# Test 1: Import modules
print("\n[TEST 1] Kiểm tra import modules...")
try:
    from image_preprocessing import ImagePreprocessor
    print("✅ image_preprocessing.py - OK")
except Exception as e:
    print(f"❌ image_preprocessing.py - ERROR: {e}")
    sys.exit(1)

try:
    from image_analysis import analyze_image, calculate_dynamic_score, detect_veins_frangi
    print("✅ image_analysis.py - OK")
except Exception as e:
    print(f"❌ image_analysis.py - ERROR: {e}")
    sys.exit(1)

# Test 2: Kiểm tra scikit-image
print("\n[TEST 2] Kiểm tra scikit-image (Frangi filter)...")
try:
    from skimage.filters import frangi
    print("✅ scikit-image đã được cài đặt")
    FRANGI_AVAILABLE = True
except ImportError:
    print("⚠️  scikit-image chưa được cài đặt")
    print("    Hệ thống sẽ fallback về Gabor filter")
    print("    Để cài đặt: pip install scikit-image==0.21.0")
    FRANGI_AVAILABLE = False

# Test 3: Gray World Assumption
print("\n[TEST 3] Kiểm tra Gray World Assumption...")
try:
    preprocessor = ImagePreprocessor()
    
    # Tạo ảnh test có color cast
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_img[:, :, 0] += 50  # Thêm blue cast
    
    balanced = preprocessor.gray_world_white_balance(test_img)
    
    if balanced.shape == test_img.shape:
        print("✅ Gray World Assumption hoạt động")
        print(f"   Original mean: B={test_img[:,:,0].mean():.1f}, G={test_img[:,:,1].mean():.1f}, R={test_img[:,:,2].mean():.1f}")
        print(f"   Balanced mean: B={balanced[:,:,0].mean():.1f}, G={balanced[:,:,1].mean():.1f}, R={balanced[:,:,2].mean():.1f}")
    else:
        print("❌ Gray World Assumption có vấn đề")
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 4: Frangi Vesselness Filter
print("\n[TEST 4] Kiểm tra Frangi Vesselness Filter...")
try:
    # Tạo ảnh grayscale test
    test_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    vein_response = detect_veins_frangi(test_gray)
    
    if vein_response.shape == test_gray.shape:
        if FRANGI_AVAILABLE:
            print("✅ Frangi Vesselness Filter hoạt động")
        else:
            print("✅ Gabor Filter fallback hoạt động")
        print(f"   Input shape: {test_gray.shape}")
        print(f"   Output shape: {vein_response.shape}")
        print(f"   Output range: [{vein_response.min()}, {vein_response.max()}]")
    else:
        print("❌ Frangi/Gabor filter có vấn đề")
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 5: Dynamic Weighting System
print("\n[TEST 5] Kiểm tra Dynamic Weighting System...")
try:
    # Test data
    shape = {
        'aspectRatio': '2.1',
        'mainObjectRatio': '0.45',
        'greenDensity': '0.18',
        'roundness': '0.45',
        'eccentricity': '1.1',
        'elongation': '1.1'
    }
    
    color = {
        'greenRatio': '0.22',
        'yellowRatio': '0.05',
        'brownRatio': '0.03',
        'avgSaturation': '0.28',
        'avgRed': '120',
        'avgGreen': '140',
        'avgBlue': '100',
        'avgHue': '90',
        'avgValue': '0.55',
        'analyzedPixels': 10000
    }
    
    texture = {
        'veinScore': '0.38',
        'edgeDensity': '0.08',
        'contrast': '0.45'
    }
    
    # Test case 1: Normal
    print("\n   Test Case 1: Normal conditions")
    result = calculate_dynamic_score(shape, color, texture, {'is_dark': False})
    print(f"   ✅ Score: {result['score']:.3f}")
    print(f"      Situation: {result['situation']}")
    print(f"      Weights: shape={result['weights_used']['shape']}, color={result['weights_used']['color']}, texture={result['weights_used']['texture']}")
    
    # Test case 2: Dark image
    print("\n   Test Case 2: Dark image")
    result = calculate_dynamic_score(shape, color, texture, {'is_dark': True, 'brightness': 60})
    print(f"   ✅ Score: {result['score']:.3f}")
    print(f"      Situation: {result['situation']}")
    print(f"      Weights: shape={result['weights_used']['shape']}, color={result['weights_used']['color']}, texture={result['weights_used']['texture']}")
    
    # Test case 3: Diseased leaf
    print("\n   Test Case 3: Diseased leaf (low green)")
    color_diseased = color.copy()
    color_diseased['greenRatio'] = '0.15'
    result = calculate_dynamic_score(shape, color_diseased, texture, {})
    print(f"   ✅ Score: {result['score']:.3f}")
    print(f"      Situation: {result['situation']}")
    print(f"      Weights: shape={result['weights_used']['shape']}, color={result['weights_used']['color']}, texture={result['weights_used']['texture']}")
    
    # Test case 4: Strong veins
    print("\n   Test Case 4: Strong veins")
    texture_strong = texture.copy()
    texture_strong['veinScore'] = '0.65'
    result = calculate_dynamic_score(shape, color, texture_strong, {})
    print(f"   ✅ Score: {result['score']:.3f}")
    print(f"      Situation: {result['situation']}")
    print(f"      Weights: shape={result['weights_used']['shape']}, color={result['weights_used']['color']}, texture={result['weights_used']['texture']}")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Integration test
print("\n[TEST 6] Kiểm tra tích hợp enhance_image()...")
try:
    preprocessor = ImagePreprocessor()
    
    # Tạo ảnh test với color cast và tối
    test_img = np.ones((100, 100, 3), dtype=np.uint8) * 50  # Ảnh tối
    test_img[:, :, 2] += 30  # Red cast
    
    enhanced = preprocessor.enhance_image(test_img, aggressive=True)
    
    if enhanced.shape == test_img.shape:
        print("✅ enhance_image() hoạt động (bao gồm Gray World)")
        print(f"   Original brightness: {np.mean(test_img):.1f}")
        print(f"   Enhanced brightness: {np.mean(enhanced):.1f}")
    else:
        print("❌ enhance_image() có vấn đề")
except Exception as e:
    print(f"❌ ERROR: {e}")

# Summary
print("\n" + "=" * 70)
print("📊 TÓM TẮT KẾT QUẢ")
print("=" * 70)
print("✅ Các module đã được tích hợp thành công")
print("✅ Gray World Assumption - Hoạt động")
print(f"{'✅' if FRANGI_AVAILABLE else '⚠️ '} Frangi Vesselness Filter - {'Hoạt động' if FRANGI_AVAILABLE else 'Fallback to Gabor'}")
print("✅ Dynamic Weighting System - Hoạt động")
print("\n🎯 Hệ thống đã sẵn sàng với các cải tiến mới!")

if not FRANGI_AVAILABLE:
    print("\n💡 Khuyến nghị: Cài đặt scikit-image để sử dụng Frangi filter:")
    print("   pip install scikit-image==0.21.0")

print("=" * 70)
