# 🍅 Tomato Disease Detection - AI Web Application

Ứng dụng web sử dụng Deep Learning (EfficientNetB0 + Spatial Attention) để phát hiện bệnh trên lá cà chua với độ chính xác **95-96%**.

---

## 📋 Mục Lục

1. [✨ Tính Năng](#-tính-năng)
2. [🏆 Model v2.0](#-model-v20---cải-tiến)
3. [🔄 Luồng Xử Lý Chi Tiết](#-luồng-xử-lý-chi-tiết)
4. [🖼️ Kỹ Thuật Xử Lý Ảnh](#️-kỹ-thuật-xử-lý-ảnh)
5. [🚀 Quick Start](#-quick-start)
6. [📊 Dataset và Training](#-dataset-và-training)

---

## ✨ Tính năng

- 📤 Upload ảnh từ máy tính hoặc 📷 chụp từ camera
- 🍃 **Phát hiện tự động ảnh lá** - Từ chối ảnh không phải lá cây
- 🔬 Dự đoán 6 loại bệnh với độ tin cậy cao (**95-96% accuracy**)
- 📊 Top 5 dự đoán + phân tích chi tiết chất lượng ảnh
- 🏥 **KHUYẾN NGHỊ CHĂM SÓC** - Hướng dẫn điều trị và phòng ngừa chi tiết cho từng bệnh
- 💊 Gợi ý sản phẩm điều trị cụ thể (tên thuốc, liều lượng)
- 🎨 Giao diện đẹp, responsive, thân thiện người dùng
- 📜 Lịch sử dự đoán (100 kết quả gần nhất) + **Xem chi tiết từng lần dự đoán**
- 🖼️ Tiền xử lý ảnh thông minh (auto brightness, sharpening, CLAHE)
- ✅ Kiểm tra đa tầng: gân lá, màu sắc, hình dạng, texture

## 🏆 Model v2.0 - Cải tiến

- ✅ **Accuracy tăng**: 92% → **95-96%** (+3-4%)
- ✅ MixUp Augmentation (α=0.2)
- ✅ Spatial Attention Mechanism (7x7 kernel)
- ✅ Two-stage Training (frozen → fine-tune)
- ✅ Class Weighting cho imbalanced data
- ✅ Enhanced Architecture (512→256 dense layers)
- ✅ Test-Time Augmentation (TTA)

---

## 🔄 Luồng Xử Lý Chi Tiết

Khi người dùng upload ảnh, hệ thống thực hiện 5 bước xử lý tuần tự:

### **BƯỚC 1: TIỀN XỬ LÝ ẢNH CƠ BẢN (Basic Preprocessing)**

**File:** `app.py` - endpoint `/predict`

```
Ảnh gốc → Convert RGB → Resize 256x256 → Giữ nguyên range [0-255]
```

**Chi tiết:**
- Chuyển đổi ảnh sang RGB nếu là RGBA, grayscale, hoặc format khác
- Resize về 256x256 pixels (kích thước model được train)
- Sử dụng `Image.Resampling.BICUBIC` cho chất lượng tốt
- **QUAN TRỌNG:** Giữ nguyên pixel values trong range [0, 255] (không rescale)
- Model có data augmentation layer bên trong, tự xử lý normalization

**Lý do không dùng preprocessing phức tạp:**
- Model được train với input đơn giản (resize + rescale)
- Áp dụng CLAHE/Sharpen sẽ làm sai lệch so với training data
- Data augmentation layer trong model đã xử lý các biến đổi

---

### **BƯỚC 2: PHÂN TÍCH VÀ XÁC THỰC ẢNH LÁ (Image Validation)**

**File:** `image_analysis.py` - function `analyze_image()`

Hệ thống phân tích 3 nhóm đặc trưng để xác định ảnh có phải lá cây không:

#### **2.1. Phân Tích Texture và Gân Lá (Texture Analysis)**

**Function:** `analyze_texture()` trong `image_analysis.py`

**Pipeline:**

```
Ảnh RGB → HSV → Tạo Leaf Mask → CLAHE Enhancement → Frangi Filter → 
Gabor Filter → Threshold → Morphological Thinning → Remove Noise → Gân lá
```

**Các kỹ thuật:**

1. **HSV Color Space + Masking**
   - Tạo mask cho màu xanh lá: `H=[35-85], S=[20-255], V=[20-255]`
   - Tạo mask cho màu vàng (lá bệnh): `H=[15-45], S=[20-255], V=[20-255]`
   - Combine masks để detect cả lá khỏe và lá bệnh

2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Tăng cường kênh Saturation trong HSV
   - `clipLimit=2.0, tileGridSize=(8,8)`
   - Làm nổi bật gân lá và texture

3. **Morphological Operations**
   - Opening: Loại bỏ noise nhỏ
   - Closing: Lấp đầy khoảng trống
   - Tạo contour của lá

4. **Frangi Vesselness Filter** (Kỹ thuật chính)
   - Phát hiện cấu trúc dạng mạch máu/gân lá
   - Multi-scale detection: sigmas=[2,3,4] pixels
   - Chỉ giữ top 30% response mạnh nhất
   - **Tại sao dùng Frangi?** 
     - Chuyên phát hiện cấu trúc phân nhánh (gân lá)
     - Hiệu quả hơn Canny/Sobel cho vein detection
     - Robust với noise và lighting variations

5. **Gabor Filter Bank** (Hỗ trợ)
   - Quét 4 hướng: 0°, 45°, 90°, 135°
   - Kernel 9x9, σ=1.5, λ=5.0, γ=0.5
   - Phát hiện texture định hướng (gân lá có hướng)

6. **Weighted Combination**
   - `vein_response = 0.5*frangi + 0.5*gabor`
   - Cân bằng giữa cấu trúc và texture

7. **Morphological Thinning**
   - Làm mảnh các đường gân về 1 pixel
   - Dễ dàng đếm và phân tích cấu trúc

8. **Connected Components Analysis**
   - Loại bỏ noise: chỉ giữ components có `area ≥ 5` hoặc `length > 5`
   - Đếm số đường gân và phân tích phân nhánh

**Fallback Mechanisms:**
- Nếu không detect được gân: dùng Sobel edge detection
- Nếu scikit-image không có: fallback về Gabor filter

**Metrics tính toán:**
- `vein_density`: Tỷ lệ pixels gân / diện tích lá (2-15% là tốt)
- `vein_score`: Scale to [0,1] (5% density = 0.5, 10% = 1.0)
- `edge_density`: Mật độ cạnh trong vùng lá
- `contrast`: Độ phức tạp bề mặt (std of grayscale)

---

#### **2.2. Phân Tích Hình Dạng (Shape Analysis)**

**Function:** `analyze_shape()` trong `image_analysis.py`

**Các đặc trưng tính toán:**

1. **Aspect Ratio**
   - `aspect_ratio = width / height`
   - Lá cây thường có tỷ lệ gần 1 (vuông) hoặc dài (1.5-2.0)

2. **Main Object Ratio**
   - `main_object_ratio = foreground_pixels / total_pixels`
   - Lá nên chiếm 30-70% ảnh (0.3-0.7)

3. **Green Density** (Quan trọng nhất)
   - Đếm pixels xanh trong vùng foreground
   - `green_density = green_pixels / foreground_pixels`
   - **Hard constraint:** Phải ≥ 20% để là lá

4. **Roundness**
   - `roundness = (4π × area) / perimeter²`
   - Range [0,1]: 1=tròn hoàn hảo, <0.5=dài

5. **Eccentricity**
   - `eccentricity = |aspect_ratio - 1|`
   - Đo độ lệch khỏi hình vuông

**Điều kiện pass:**
- `green_density ≥ 0.20` (20% màu xanh)
- `main_object_ratio ≥ 0.08` (8% diện tích)
- Shape score tổng hợp ≥ 0.35

---

#### **2.3. Phân Tích Màu Sắc (Color Analysis)**

**Function:** `analyze_color()` trong `image_analysis.py`

**Chỉ phân tích trong vùng có edge** (không phân tích background)

**Metrics tính toán:**

1. **Color Distribution**
   - `green_ratio`: H=60-180°, S>0.2, V>0.2 (lá xanh khỏe)
   - `yellow_ratio`: H=30-60°, S>0.3 (lá bệnh vàng)
   - `brown_ratio`: H<30° or H>330°, V<0.5 (lá bệnh nâu)

2. **Average HSV Values**
   - `avgHue`: Màu chủ đạo (90-120° là xanh lá)
   - `avgSaturation`: Độ bão hòa màu (>0.3 là tốt)
   - `avgValue`: Độ sáng (>0.3 là đủ sáng)

**Điều kiện pass:**
- `green_ratio ≥ 0.20` (20% xanh lá)
- `avgSaturation ≥ 0.25` (màu đủ rõ, không xám)

---

#### **2.4. Hệ Thống Chấm Điểm Động (Dynamic Scoring)**

**Function:** `calculate_dynamic_score()` trong `image_analysis.py`

**Trọng số thay đổi theo tình huống:**

| Tình huống | Shape | Color | Texture | Lý do |
|------------|-------|-------|---------|-------|
| **Normal** | 35% | 50% | 15% | Tin màu sắc nhất |
| **Dark Image** | 40% | 35% | 25% | Màu không tin cậy, tăng texture |
| **Diseased Leaf** | 35% | 30% | 35% | Lá bệnh mất màu, tin texture |
| **Strong Veins** | 30% | 40% | 30% | Gân rõ = chắc chắn là lá |

**Công thức:**
```
final_score = shape_score × w_shape + color_score × w_color + texture_score × w_texture
```

**Hard Constraints:**
- `green_ratio ≥ 0.20` HOẶC `(green_ratio ≥ 0.02 VÀ vein_score ≥ 0.30)`
- `overall_score ≥ 0.60` (60%)

**Kết quả:**
- `isLeaf = True`: Pass validation → Tiếp tục predict
- `isLeaf = False`: Reject với detailed analysis

---

### **BƯỚC 3: DỰ ĐOÁN BỆNH (Disease Prediction)**

**File:** `app.py` - sử dụng TensorFlow model

**Pipeline:**

```
Ảnh [0-255] → Add batch dimension [1, 256, 256, 3] → 
Model (data aug + EfficientNetB0 + Spatial Attention) → 
Softmax probabilities [6 classes]
```

**Model Architecture:**

1. **Data Augmentation Layer** (trong model)
   - Random flip horizontal/vertical
   - Random rotation ±10°
   - Random zoom ±10%
   - **Tự động normalize** về ImageNet range

2. **EfficientNetB0 Backbone**
   - Pretrained on ImageNet
   - Feature extraction: 1280 features
   - Frozen trong stage 1, fine-tuned trong stage 2

3. **Spatial Attention Module**
   - Conv2D 7×7 kernel → Sigmoid
   - Học vùng quan trọng (lá bệnh)
   - Multiply với features: `attended = features × attention_map`

4. **Classification Head**
   - GlobalAveragePooling2D
   - Dense(256) + Dropout(0.5) + BatchNorm
   - Dense(6, softmax)

**Output:**
- 6 probabilities (1 cho mỗi class)
- Predicted class = argmax(probabilities)
- Confidence = max(probabilities) × 100%

**6 Classes:**
1. Bacterial Spot (Đốm Lá Vi Khuẩn)
2. Early Blight (Bệnh Héo Sớm)
3. Healthy (Lá Khỏe Mạnh)
4. Late Blight (Bệnh Mốc Sương)
5. Septoria Leaf Spot (Đốm Lá Septoria)
6. Yellow Leaf Curl Virus (Virus Cuộn Lá Vàng)

---

### **BƯỚC 4: TẠO KHUYẾN NGHỊ CHĂM SÓC (Care Recommendations)**

**File:** `app.py` - function `get_disease_recommendation()`

**Database:** `DISEASE_INFO` dictionary chứa đầy đủ thông tin cho 6 classes

**Nội dung cho mỗi bệnh:**

1. **Basic Info**
   - Tên tiếng Việt
   - Mức độ nghiêm trọng (Cao/Trung bình/Thấp)
   - Mô tả bệnh

2. **Symptoms** (Triệu chứng)
   - Danh sách triệu chứng trực quan
   - Giúp người dùng xác nhận chẩn đoán

3. **Causes** (Nguyên nhân)
   - Điều kiện môi trường gây bệnh
   - Nhiệt độ, độ ẩm, thời tiết

4. **Treatment** (Điều trị)
   - **Immediate**: Biện pháp khẩn cấp (24h)
   - **Short-term**: 1-4 tuần
   - **Long-term**: 3 tháng - 3 năm

5. **Prevention** (Phòng ngừa)
   - Các biện pháp tránh tái phát

6. **Products** (Sản phẩm điều trị)
   - Tên thuốc cụ thể
   - Hoạt chất

**Phân loại theo confidence:**

| Confidence | Certainty | Action Level |
|-----------|-----------|--------------|
| ≥ 90% | RẤT CAO | Áp dụng ngay tất cả biện pháp |
| ≥ 75% | CAO | Áp dụng biện pháp khuyến nghị |
| ≥ 60% | TRUNG BÌNH | Theo dõi + phòng ngừa |
| < 60% | THẤP | Chụp ảnh rõ hơn |

---

### **BƯỚC 5: TRẢ VỀ KẾT QUẢ (Response)**

**JSON Response Structure:**

```json
{
  "success": true,
  "prediction": {
    "class": "Early Blight",
    "confidence": 94.5,
    "name_vi": "Bệnh Héo Sớm",
    "severity": "Trung bình - Cao"
  },
  "top_predictions": [
    {"class": "Early Blight", "confidence": 94.5},
    {"class": "Late Blight", "confidence": 3.2},
    ...
  ],
  "recommendations": {
    "description": "...",
    "symptoms": [...],
    "treatment": {
      "immediate": [...],
      "shortterm": [...],
      "longterm": [...]
    },
    "prevention": [...],
    "products": [...]
  },
  "image_analysis": {
    "score": 87.5,
    "shapeScore": 0.82,
    "colorScore": 0.91,
    "textureScore": 0.75,
    "greenRatio": "0.654",
    "veinScore": "0.432"
  },
  "processed_images": {
    "original": "data:image/jpeg;base64,...",
    "resized": "data:image/jpeg;base64,..."
  }
}
```

---

## 🖼️ Kỹ Thuật Xử Lý Ảnh

### **Chi Tiết Các Kỹ Thuật Computer Vision**

#### **1. Frangi Vesselness Filter**

**Mục đích:** Phát hiện cấu trúc dạng mạch máu, gân lá

**Nguyên lý:**
- Sử dụng Hessian matrix để phân tích độ cong tại mỗi pixel
- Tính 2 eigenvalues (λ1, λ2) từ Hessian matrix
- Gân lá có λ1 ≈ 0 và λ2 lớn (cong theo 1 chiều)

**Công thức Frangi Filter:**

```
V(x,y) = {
  0,                                if λ2 > 0
  exp(-Rb²/2β²) × (1 - exp(-S²/2γ²)), otherwise
}

where:
  Rb = |λ1| / |λ2|  (blobness measure)
  S = √(λ1² + λ2²)  (structure strength)
```

**Tham số trong code:**
- `sigmas=[2,3,4]`: Multi-scale detection (gân to và nhỏ)
- `alpha=0.5`, `beta=0.5`: Sensitivity parameters
- `gamma=25`: Background suppression
- `black_ridges=False`: Gân sáng hơn nền

**Output:** Grayscale image với intensity = khả năng là gân lá

---

#### **2. Gabor Filter Bank**

**Mục đích:** Phát hiện texture và edges theo hướng

**Nguyên lý:**
- Tích chập của Gaussian với sinusoid
- Nhạy với texture ở tần số và hướng cụ thể

**Công thức Gabor Kernel:**

```
g(x,y,θ,λ,σ,γ) = exp(-(x'² + γ²y'²)/(2σ²)) × cos(2πx'/λ)

where:
  x' = x cos(θ) + y sin(θ)
  y' = -x sin(θ) + y cos(θ)
```

**Tham số trong code:**
- `θ = [0°, 45°, 90°, 135°]`: 4 hướng quét
- `kernel_size = 9×9`
- `σ = 1.5`: Độ rộng Gaussian
- `λ = 5.0`: Wavelength (tần số)
- `γ = 0.5`: Spatial aspect ratio

**Output:** 4 filtered images → Max pooling → Vein response map

---

#### **3. CLAHE (Contrast Limited Adaptive Histogram Equalization)**

**Mục đích:** Tăng cường độ tương phản cục bộ

**Vấn đề của Histogram Equalization thường:**
- Tăng noise ở vùng sáng/tối
- Không phù hợp với ảnh có lighting không đều

**CLAHE giải quyết:**
- Chia ảnh thành tiles (8×8)
- Equalization riêng cho từng tile
- Clip histogram ở `clipLimit=2.0` để tránh over-amplification
- Interpolate giữa các tiles để smooth

**Code:**
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_channel = clahe.apply(image_channel)
```

**Khi nào áp dụng:**
- Contrast < 40 (ảnh phẳng)
- Ảnh tối/sáng quá
- **KHÔNG dùng** trước khi đưa vào model (vì model không train với CLAHE)

---

#### **4. Morphological Operations**

**Mục đích:** Làm sạch và tăng cường cấu trúc

**Các operations:**

1. **Opening** = Erosion + Dilation
   - Loại bỏ noise nhỏ
   - Giữ nguyên shape chính

2. **Closing** = Dilation + Erosion
   - Lấp đầy khoảng trống
   - Kết nối các vùng gần nhau

3. **Top-hat** = Original - Opening
   - Làm nổi vùng sáng hơn nền
   - Phát hiện gân lá sáng

4. **Black-hat** = Closing - Original
   - Làm nổi vùng tối hơn nền
   - Phát hiện gân lá tối

**Structuring Elements:**
- `MORPH_ELLIPSE (5,5)`: Loại noise
- `MORPH_ELLIPSE (11,11)`: Fill holes
- `MORPH_RECT (2,2)`: Connect veins

---

#### **5. Morphological Thinning (Skeletonization)**

**Mục đích:** Làm mảnh đường gân về 1 pixel

**Thuật toán:**
```python
while True:
    eroded = erode(image, kernel)
    temp = opening(eroded, kernel)
    skeleton = skeleton OR (image - temp)
    image = eroded
    if image is empty: break
```

**Ứng dụng:**
- Đếm số đường gân
- Phân tích branching pattern
- Tính length/area ratio

---

#### **6. HSV Color Space Analysis**

**Tại sao dùng HSV thay vì RGB?**

| Aspect | RGB | HSV |
|--------|-----|-----|
| **Lighting sensitivity** | Cao (3 channels cùng thay đổi) | Thấp (chỉ V thay đổi) |
| **Color separation** | Khó (màu trộn 3 channels) | Dễ (H là màu thuần) |
| **Intuitive** | Không (255,0,0 = đỏ?) | Có (H=120° = xanh lá) |

**HSV trong code:**
- **H (Hue):** 0-179 trong OpenCV (0-360° / 2)
  - Green: 60-90 (120-180°)
  - Yellow: 15-30 (30-60°)
  - Brown: 5-15 (10-30°)

- **S (Saturation):** 0-255
  - >50: Màu rõ ràng
  - <30: Gần màu xám

- **V (Value):** 0-255
  - Brightness
  - <80: Tối
  - >180: Sáng

**Ứng dụng:**
```python
# Detect green leaves (khỏe hoặc bệnh nhẹ)
green_mask = cv2.inRange(hsv, (35, 20, 20), (85, 255, 255))

# Detect yellow (bệnh vàng lá)
yellow_mask = cv2.inRange(hsv, (15, 20, 20), (45, 255, 255))
```

---

#### **7. Edge Detection Methods Comparison**

| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| **Canny** | Sharp edges, non-max suppression | Sensitive to noise | General edge detection |
| **Sobel** | Simple, directional | Thick edges | Gradient calculation |
| **Frangi** | Detects vessels/veins | Slower, needs tuning | Vein structure |
| **Gabor** | Orientation + frequency | Multiple filters needed | Texture analysis |

**Trong project:**
- **Frangi + Gabor**: Vein detection (primary)
- **Sobel**: Fallback khi Frangi fail
- **Canny**: Không dùng (quá nhạy với noise)

---

#### **8. Gray World White Balance**

**File:** `image_preprocessing.py` - function `gray_world_white_balance()`

**Vấn đề:** Ảnh bị lệch màu do ánh sáng (đèn vàng, nắng chiều, đèn xanh)

**Giả định:** Trung bình các màu trong ảnh nên là xám (neutral)

**Công thức:**

```
avg_r = mean(R_channel)
avg_g = mean(G_channel)
avg_b = mean(B_channel)

avg_gray = (avg_r + avg_g + avg_b) / 3

R' = R × (avg_gray / avg_r)
G' = G × (avg_gray / avg_g)
B' = B × (avg_gray / avg_b)
```

**Khi nào dùng:**
- Ảnh chụp dưới đèn vàng (ảnh vàng toàn bộ)
- Ảnh chụp ban đêm với flash (lệch màu)
- **KHÔNG dùng** nếu ảnh chủ yếu màu xanh (sẽ làm sai màu)

---

#### **9. Connected Components Analysis**

**Mục đích:** Phân tích các vùng liên thông trong binary image

**OpenCV function:**
```python
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    binary_image, 
    connectivity=8
)
```

**Output:**
- `num_labels`: Số components (include background)
- `labels`: Mảng same size, mỗi pixel có label (0, 1, 2, ...)
- `stats`: [x, y, width, height, area] của mỗi component
- `centroids`: (cx, cy) tâm của mỗi component

**Ứng dụng trong vein detection:**
```python
for i in range(1, num_labels):  # Skip background (label 0)
    area = stats[i, cv2.CC_STAT_AREA]
    width = stats[i, cv2.CC_STAT_WIDTH]
    height = stats[i, cv2.CC_STAT_HEIGHT]
    
    # Chỉ giữ components đủ lớn/dài (là gân, không phải noise)
    if area >= 5 or max(width, height) > 5:
        valid_veins[labels == i] = 255
```

---

#### **10. Image Normalization Strategies**

**3 phương pháp:**

| Method | Formula | Range | Use Case |
|--------|---------|-------|----------|
| **Rescale** | `x / 255` | [0, 1] | Simple models |
| **Standardize** | `(x - mean) / std` | [-3, 3] | ML models |
| **ImageNet Norm** | `(x/255 - mean) / std` | [-2.5, 2.5] | Transfer learning |

**ImageNet mean/std:**
```
mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]   # RGB
```

**Trong project:**
- **Training:** Rescale only (`rescale=1./255`)
- **Inference:** Giữ nguyên [0, 255] → Model tự normalize

---

### **Luồng Xử Lý Ảnh - Sơ Đồ Tổng Quan**

```
┌─────────────────────────────────────────────────────────────────┐
│                    UPLOAD ẢNH (User Input)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────▼────────────┐
                │   BƯỚC 1: PREPROCESSING │
                │  - Convert RGB          │
                │  - Resize 256x256       │
                │  - Keep [0-255] range   │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │ BƯỚC 2: IMAGE VALIDATION│
                └────────────┬────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼─────┐      ┌──────▼──────┐      ┌─────▼──────┐
   │ TEXTURE  │      │   SHAPE     │      │   COLOR    │
   │ ANALYSIS │      │  ANALYSIS   │      │  ANALYSIS  │
   │          │      │             │      │            │
   │ - Frangi │      │ - Aspect    │      │ - HSV      │
   │ - Gabor  │      │ - Green     │      │ - Green    │
   │ - Veins  │      │   Density   │      │   Ratio    │
   └────┬─────┘      └──────┬──────┘      └─────┬──────┘
        │                   │                    │
        └───────────┬───────┴───────┬────────────┘
                    │               │
           ┌────────▼────────┐      │
           │ DYNAMIC SCORING │      │
           │ - Adjust weights│      │
           │ - Calculate score│     │
           └────────┬────────┘      │
                    │               │
                    ▼               │
              ┌──────────┐          │
              │ isLeaf?  │          │
              └─┬─────┬──┘          │
                │     │             │
            Yes │     │ No          │
                │     └──────────► REJECT
                │               (detailed analysis)
                │
    ┌───────────▼───────────┐
    │  BƯỚC 3: MODEL        │
    │  - EfficientNetB0     │
    │  - Spatial Attention  │
    │  - Softmax (6 class)  │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │ BƯỚC 4: RECOMMENDATIONS│
    │  - Match disease info  │
    │  - Treatment plans     │
    │  - Prevention tips     │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │  BƯỚC 5: RESPONSE     │
    │  - JSON with results   │
    │  - Images (base64)     │
    │  - Detailed analysis   │
    └───────────────────────┘
```

---

### **Điểm Mạnh và Hạn Chế**

#### **✅ Điểm Mạnh:**

1. **Robust Validation**
   - 3 layers kiểm tra (texture, shape, color)
   - Dynamic scoring thích ứng với điều kiện ảnh
   - Hard constraints ngăn false positives

2. **Advanced Vein Detection**
   - Frangi filter (state-of-the-art cho vein/vessel)
   - Gabor filter hỗ trợ
   - Fallback mechanisms đảm bảo không crash

3. **Color Robustness**
   - HSV space (ít nhạy với lighting)
   - Gray World white balance
   - Chấp nhận lá bệnh (vàng, nâu, đen)

4. **High Accuracy Model**
   - EfficientNetB0 (efficient + accurate)
   - Spatial Attention (focus on disease areas)
   - 95-96% accuracy

5. **Comprehensive Care Guide**
   - 6 classes với thông tin chi tiết
   - Treatment plans (immediate + long-term)
   - Sản phẩm điều trị cụ thể

#### **⚠️ Hạn Chế:**

1. **Yêu cầu scikit-image**
   - Frangi filter cần `scikit-image`
   - Fallback về Gabor nếu không có (kém hơn)

2. **Tốc độ xử lý**
   - Frangi + Gabor + Morphology: ~1-2 giây/ảnh
   - Trade-off giữa accuracy và speed

3. **Sensitivity to Image Quality**
   - Ảnh quá tối/mờ có thể reject
   - Cần ảnh rõ ràng, lá chiếm ≥30% frame

4. **6 Classes Only**
   - Không detect sâu bệnh, thiếu dinh dưỡng khác
   - Cần training data mở rộng

5. **No Multi-leaf Detection**
   - Chỉ phân tích 1 lá/ảnh
   - Ảnh nhiều lá có thể confuse model

---

## 📂 Cấu Trúc Module và Chi Tiết Kỹ Thuật

### **Tổng Quan Luồng Xử Lý**

```
📤 USER UPLOAD
    ↓
┌───────────────────────────────────────────────────────────┐
│ BƯỚC 1: BASIC PREPROCESSING (app.py)                      │
│ - Convert RGB                                              │
│ - Resize 256×256 (BICUBIC)                                │
│ - Giữ nguyên [0-255] range                                │
└───────────────┬───────────────────────────────────────────┘
                ↓
┌───────────────────────────────────────────────────────────┐
│ BƯỚC 2: IMAGE VALIDATION (image_analysis.py)              │
│                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   TEXTURE    │  │    SHAPE     │  │    COLOR     │   │
│  │              │  │              │  │              │   │
│  │ • Frangi     │  │ • Aspect     │  │ • HSV        │   │
│  │ • Gabor      │  │   Ratio      │  │ • Green      │   │
│  │ • Morphology │  │ • Green      │  │   Ratio      │   │
│  │ • Thinning   │  │   Density    │  │ • Saturation │   │
│  │              │  │ • Roundness  │  │              │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         │                 │                  │            │
│         └─────────┬───────┴──────────────────┘            │
│                   ↓                                        │
│         ┌─────────────────────┐                           │
│         │  DYNAMIC SCORING    │                           │
│         │  Adjust weights by  │                           │
│         │  image conditions   │                           │
│         └─────────┬───────────┘                           │
│                   ↓                                        │
│              isLeaf = ?                                    │
└───────────────┬───────────────────────────────────────────┘
                │
        ┌───────┴────────┐
        │                │
     FALSE            TRUE
        │                │
        ↓                ↓
   REJECT        ┌───────────────────────────────┐
   (detailed     │ BƯỚC 3: PREDICTION (model)    │
   analysis)     │ - Data augmentation layer     │
                 │ - EfficientNetB0              │
                 │ - Spatial Attention           │
                 │ - Softmax (6 classes)         │
                 └───────────┬───────────────────┘
                             ↓
                 ┌───────────────────────────────┐
                 │ BƯỚC 4: RECOMMENDATIONS       │
                 │ - Match DISEASE_INFO database │
                 │ - Treatment plans             │
                 │ - Products                    │
                 └───────────┬───────────────────┘
                             ↓
                 ┌───────────────────────────────┐
                 │ BƯỚC 5: RESPONSE              │
                 │ - JSON results                │
                 │ - Base64 images               │
                 │ - Analysis scores             │
                 └───────────────────────────────┘
                             ↓
                        💻 CLIENT
```

---

### **Tham Số Kỹ Thuật Quan Trọng**

#### **Validation Thresholds**

| Parameter | Value | Meaning | Why |
|-----------|-------|---------|-----|
| `min_green_ratio` | 0.02 (2%) | Tối thiểu % xanh trong HSV | Chấp nhận lá bệnh nặng |
| `min_leaf_ratio` | 0.08 (8%) | Tối thiểu % vegetation | Lá phải chiếm đủ diện tích |
| `min_leaf_shape_score` | 0.30 | Điểm hình dạng tối thiểu | Phân biệt lá và động vật |
| `min_texture_score` | 0.20 | Điểm texture cơ bản | Phải có gân lá |
| `excellent_texture_score` | 0.40 | Texture xuất sắc | Gân rõ, chắc chắn là lá |

#### **Color Detection (HSV Range)**

| Color | Hue (H) | Saturation (S) | Value (V) | Target |
|-------|---------|----------------|-----------|--------|
| **Green** | 35-85 | 20-255 | 20-255 | Lá khỏe mạnh |
| **Yellow** | 15-45 | 20-255 | 20-255 | Lá bệnh vàng |
| **Brown** | 5-25 | 30-255 | 20-200 | Lá bệnh nâu |
| **Dark/Shadow** | 0-180 | 0-255 | 0-60 | Mảng đen, bóng |

#### **Frangi Filter Parameters**

```python
frangi(
    image,
    sigmas=range(2, 5, 1),    # Multi-scale: 2, 3, 4 pixels
    black_ridges=False,        # Gân sáng hơn nền
    alpha=0.5,                 # Plate-like sensitivity
    beta=0.5,                  # Blobness sensitivity
    gamma=25                   # Background suppression
)
```

#### **Gabor Filter Parameters**

```python
cv2.getGaborKernel(
    ksize=(9, 9),              # Kernel size
    sigma=1.5,                 # Gaussian standard deviation
    theta=np.deg2rad(angle),   # Orientation: 0°, 45°, 90°, 135°
    lambd=5.0,                 # Wavelength (frequency)
    gamma=0.5,                 # Spatial aspect ratio
    psi=0,                     # Phase offset
    ktype=cv2.CV_32F           # Float32 kernel
)
```

#### **CLAHE Parameters**

```python
cv2.createCLAHE(
    clipLimit=2.0,             # Max histogram slope (prevent over-amplification)
    tileGridSize=(8, 8)        # Size of tiles (8x8 pixels each)
)
```

**Khi nào áp dụng CLAHE:**
- Contrast < 40: Áp dụng CLAHE với clipLimit=2.0
- Contrast < 25: Áp dụng CLAHE mạnh hơn với clipLimit=2.5
- Contrast ≥ 40: Bỏ qua (ảnh đã tốt)

#### **Morphological Structuring Elements**

```python
# Loại bỏ noise nhỏ
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Lấp đầy khoảng trống lớn
kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# Kết nối các đường gân
kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
```

#### **Dynamic Weighting Table**

| Condition | Shape Weight | Color Weight | Texture Weight | Rationale |
|-----------|--------------|--------------|----------------|-----------|
| **Normal** | 35% | 50% | 15% | Color most reliable |
| **Dark Image** | 40% | 35% | 25% | Color unreliable, trust shape/texture |
| **Diseased Leaf** | 35% | 30% | 35% | Lost color, trust vein structure |
| **Strong Veins** | 30% | 40% | 30% | Clear veins = definitely leaf |

#### **Model Architecture**

```
Input: [batch, 256, 256, 3] (0-255 range)
    ↓
Data Augmentation Layer (trong model)
    - RandomFlip(horizontal + vertical)
    - RandomRotation(±10°)
    - RandomZoom(±10%)
    - Rescaling(1./255)
    ↓
EfficientNetB0 (pretrained on ImageNet)
    - Input: [batch, 256, 256, 3] (normalized)
    - Output: [batch, 8, 8, 1280]
    ↓
Spatial Attention Module
    - Conv2D(1, kernel=7x7) → Sigmoid
    - Multiply: features × attention_map
    - Output: [batch, 8, 8, 1280] (attended)
    ↓
GlobalAveragePooling2D
    - Output: [batch, 1280]
    ↓
Dense(256) + BatchNorm + Dropout(0.5)
    ↓
Dense(6, activation='softmax')
    ↓
Output: [batch, 6] (probabilities)
```

**Training Configuration:**

| Parameter | Stage 1 (Frozen) | Stage 2 (Fine-tune) |
|-----------|------------------|---------------------|
| **Epochs** | 20 | 15 |
| **Learning Rate** | 0.001 | 0.0001 |
| **Batch Size** | 32 | 32 |
| **EfficientNet** | Frozen | Trainable |
| **Augmentation** | MixUp (α=0.2) | MixUp (α=0.2) |
| **Class Weights** | Calculated from distribution | Same |
| **Early Stopping** | patience=5 | patience=7 |
| **Reduce LR** | factor=0.5, patience=3 | factor=0.5, patience=3 |

#### **Performance Metrics**

```
Test Set Results (v2.0):
├── Overall Accuracy: 95.6%
├── Precision: 0.954
├── Recall: 0.956
├── F1-Score: 0.955
└── Per-class Accuracy:
    ├── Bacterial Spot: 94.2%
    ├── Early Blight: 96.8%
    ├── Healthy: 98.1%
    ├── Late Blight: 95.3%
    ├── Septoria Leaf Spot: 93.7%
    └── Yellow Leaf Curl Virus: 95.5%
```

---

## 🚀 Quick Start

## 🚀 Quick Start

### Bước 1: Clone hoặc tải project về
```bash
git clone https://github.com/Hanniel1102/ML.git
cd ML
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

#### **Phương pháp 1: Cài đặt tất cả (Khuyến nghị)**

```bash
# Cài đặt tất cả packages từ requirements.txt
pip install -r requirements.txt
```

#### **Phương pháp 2: Cài đặt từng package**

Nếu gặp lỗi với phương pháp 1, cài từng package:

```bash
# 1. Web Framework
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install python-multipart==0.0.6

# 2. Deep Learning
pip install tensorflow==2.15.0

# 3. Image Processing
pip install Pillow==10.1.0
pip install opencv-python==4.8.1.78

# 4. Numerical Computing
pip install numpy==1.24.3

# 5. Visualization (Optional)
pip install matplotlib==3.8.0
```

#### **Phương pháp 3: Cài đặt với pip upgrade**

Nếu có conflict về version:

```bash
# Upgrade pip trước
pip install --upgrade pip

# Cài đặt với option --upgrade
pip install -r requirements.txt --upgrade
```

#### **Chi tiết packages sẽ được cài:**

| Package | Version | Kích thước | Mô tả |
|---------|---------|------------|-------|
| `fastapi` | 0.104.1 | ~65 KB | Framework web API hiện đại |
| `uvicorn` | 0.24.0 | ~60 KB | ASGI server (chạy FastAPI) |
| `python-multipart` | 0.0.6 | ~30 KB | Xử lý file upload |
| `tensorflow` | 2.15.0 | **~450 MB** | Deep Learning framework ⚠️ |
| `Pillow` | 10.1.0 | ~3 MB | Xử lý ảnh PIL |
| `numpy` | 1.24.3 | ~15 MB | Tính toán số học |
| `opencv-python` | 4.8.1.78 | ~90 MB | Computer vision |
| `matplotlib` | 3.8.0 | ~35 MB | Visualization (optional) |

**Tổng dung lượng:** ~650-700 MB

**Thời gian cài đặt:**
- ⚡ Mạng nhanh (50+ Mbps): 5-10 phút
- 🌐 Mạng trung bình (10-50 Mbps): 15-30 phút
- 🐌 Mạng chậm (<10 Mbps): 30-60 phút

**Lưu ý quan trọng:**
- ⚠️ TensorFlow (~450 MB) là package lớn nhất
- 💾 Cần ~2 GB dung lượng trống (bao gồm dependencies)
- 🔧 Nếu có GPU NVIDIA, cài thêm: `pip install tensorflow[and-cuda]`

### Bước 4: Kiểm tra cài đặt

```bash
# Kiểm tra Python version (cần >= 3.11)
python --version

# Kiểm tra TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"

# Kiểm tra GPU (nếu có)
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

# Kiểm tra tất cả packages
python -c "import fastapi, uvicorn, tensorflow, PIL, cv2, numpy, matplotlib; print('✅ All packages installed successfully!')"
```

#### **Xử lý lỗi cài đặt thường gặp:**

**Lỗi 1: "ERROR: Could not find a version that satisfies tensorflow==2.15.0"**
```bash
# Giải pháp: Cài TensorFlow phiên bản mới nhất
pip install tensorflow
```

**Lỗi 2: "ImportError: DLL load failed" (Windows)**
```bash
# Giải pháp: Cài Visual C++ Redistributable
# Tải tại: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

**Lỗi 3: "ModuleNotFoundError: No module named 'cv2'"**
```bash
# Giải pháp: Cài lại opencv-python
pip uninstall opencv-python
pip install opencv-python==4.8.1.78
```

**Lỗi 4: Conflict giữa numpy và tensorflow**
```bash
# Giải pháp: Cài numpy tương thích
pip install numpy==1.24.3 --force-reinstall
```

**Lỗi 5: Timeout khi cài TensorFlow**
```bash
# Giải pháp: Tăng timeout và dùng cache
pip install tensorflow==2.15.0 --timeout=1000 --cache-dir ./pip_cache
```

### Bước 5: Chạy ứng dụng

**Cách 1: Chạy trực tiếp (khuyến nghị)**
```bash
python app.py
```

**Cách 2: Chạy với Uvicorn (production mode)**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Cách 3: Chạy với custom port**
```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```

Sau khi chạy thành công, bạn sẽ thấy:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
✅ Model loaded successfully: best_tomato_model.keras
```

### Bước 6: Truy cập ứng dụng

Mở trình duyệt và truy cập: **http://localhost:8000**

**Hoặc:**
- Từ máy khác trong cùng mạng: `http://<IP-máy-chủ>:8000`
- Tìm IP: `ipconfig` (Windows) hoặc `ifconfig` (macOS/Linux)

## 📁 Cấu trúc Dự án

```
Hocmaynangcao/
│
├── 🚀 PRODUCTION (Inference - Dự đoán)
│   ├── app.py                              # ⭐ FastAPI server + Disease Info Database
│   ├── efficientnet_preprocessor.py        # ⭐ Preprocessing 6 bước (Resize→Analyze→CLAHE→Denoise→Sharpen→Normalize)
│   ├── image_preprocessing.py              # ⭐ Validate ảnh lá (LeafDetector, vein detection, quality check)
│   ├── image_analysis.py                   # ⭐ Phân tích đặc trưng lá (shape, color, texture, veins)
│   ├── templates/index.html                # ⭐ Web UI (upload, camera, 6-step preprocessing display)
│   ├── best_tomato_model.keras             # ⭐ Model EfficientNetB0 + Spatial Attention (95-96%)
│   ├── best_tomato_model.h5                # Model backup format
│   ├── prediction_history.json             # Lịch sử dự đoán (100 items)
│   └── requirements.txt                    # Dependencies
│
├── 📊 TRAINING (Chuẩn bị data + Train model)
│   ├── data_raw.py                         # ⭐ Tạo ảnh XẤU (10 degradation types: noise/blur/dark/motion/contrast/jpeg)
│   ├── data_preprocessing.py               # ⭐ SỬA ảnh xấu về tốt (denoise→brightness→contrast→sharpen)
│   ├── train_model_h5.ipynb                # ⭐ Notebook train model chính
│   ├── Code_demo.ipynb                     # Notebook demo cũ
│   └── Code_demo_optimized.ipynb           # Notebook demo đã optimize
│
├── 📂 DATASET
│   └── Tomato/
│       ├── Train/                          # Dataset training (6 classes)
│       ├── Val/                            # Dataset validation
│       ├── Test/                           # Dataset testing
│       └── Augmented_Train/                # Dataset XẤU (từ data_raw.py) - Optional
│
└── 📄 DOCUMENTS
    ├── README.md                           # File này
    ├── Báo cáo Học máy nâng cao.docx       # Báo cáo project
    └── Báo cáo Học máy nâng cao.pdf        # Báo cáo PDF
```

### 🔄 Luồng hoạt động

#### **A. INFERENCE (Dự đoán - Production)**

```mermaid
User upload ảnh
    ↓
app.py (FastAPI)
    ↓
1. image_preprocessing.py
   └─→ Validate ảnh lá (LeafDetector)
   └─→ Kiểm tra: gân lá, màu sắc, hình dạng
   └─→ ✅ PASS hoặc ❌ REJECT
    ↓
2. efficientnet_preprocessor.py
   └─→ Bước 1: Resize (224x224)
   └─→ Bước 2: Analyze (brightness, contrast, noise, edge)
   └─→ Bước 3: CLAHE (nếu contrast < 40)
   └─→ Bước 4: Denoise (nếu noise < 500)
   └─→ Bước 5: Sharpen (nếu edge < 50)
   └─→ Bước 6: Normalize (ImageNet mean/std)
    ↓
3. Model Prediction
   └─→ best_tomato_model.keras
   └─→ EfficientNetB0 + Spatial Attention
   └─→ Output: 6 class probabilities
    ↓
4. image_analysis.py (Parallel)
   └─→ Phân tích shape, color, texture
   └─→ Generate visualizations
   └─→ Calculate leaf_score
    ↓
5. Response to User
   └─→ Top 5 predictions
   └─→ Disease info + Treatment recommendations
   └─→ 6-step preprocessing images
   └─→ Analysis results
```

#### **B. TRAINING (Train model mới)**

```mermaid
1. data_raw.py
   └─→ Input: Tomato/Train (ảnh gốc sạch)
   └─→ Process: Tạo 10 loại degradation
       • noise_light, noise_heavy
       • blur_light, blur_heavy
       • dark, very_dark, bright
       • motion_blur
       • low_contrast
       • jpeg_compress
   └─→ Output: Tomato/Augmented_Train (ảnh XẤU)
    ↓
2. data_preprocessing.py
   └─→ Input: Tomato/Augmented_Train (ảnh XẤU)
   └─→ Process: PHỤC HỒI ảnh xấu
       • Step 1: Denoise (khử nhiễu MẠNH)
       • Step 2: Fix brightness (sửa tối/sáng → 135)
       • Step 3: Fix contrast (CLAHE 2.8-3.5)
       • Step 4: Sharpen (làm nét kernel 9-10)
       • Step 5: Resize (256x256)
   └─→ Output: Tomato/Fixed_Train (ảnh SẠCH, chất lượng tốt)
    ↓
3. train_model_h5.ipynb
   └─→ Input: Tomato/Fixed_Train (hoặc Tomato/Train gốc)
   └─→ Process:
       • Load data with ImageDataGenerator
       • Build: EfficientNetB0 + Spatial Attention
       • Stage 1: Train frozen (15 epochs)
       • Stage 2: Fine-tune all (15 epochs)
       • Apply: MixUp, Class Weighting, TTA
   └─→ Output: best_tomato_model.keras (95-96% accuracy)
```

## 🏥 Hệ Thống Khuyến Nghị Chăm Sóc (NEW!)

### Database 6 loại bệnh với thông tin chi tiết:

1. **Bacterial Spot** (Đốm Lá Vi Khuẩn) - 🔴 Cao
2. **Early Blight** (Bệnh Héo Sớm) - 🟡 Trung bình-Cao
3. **Healthy** (Lá Khỏe Mạnh) - ✅ Không bệnh
4. **Late Blight** (Bệnh Mốc Sương) - 🔴 RẤT CAO ⚠️
5. **Septoria Leaf Spot** (Đốm Lá Septoria) - 🟡 Trung bình
6. **Yellow Leaf Curl Virus** (Virus Cuộn Lá Vàng) - 🔴 Rất Cao

### Mỗi bệnh bao gồm:

- 📖 **Mô tả chi tiết** - Nguyên nhân, đặc điểm bệnh
- 🔍 **Triệu chứng** - 4-5 dấu hiệu nhận biết
- ⚠️ **Xử lý khẩn cấp** - Hành động trong 24-48 giờ
- 📅 **Điều trị ngắn hạn** - Kế hoạch 1-4 tuần
- 🌱 **Giải pháp dài hạn** - Phòng ngừa 2-12 tháng
- 🦠 **Nguyên nhân gây bệnh** - Điều kiện thuận lợi
- 🛡️ **Biện pháp phòng ngừa** - Thực hành tốt nhất
- 💊 **Sản phẩm khuyên dùng** - Tên thuốc cụ thể (Ridomil Gold, Daconil, Imidacloprid...)

### Ví dụ khuyến nghị:

**Late Blight (Mốc Sương) - Nguy hiểm nhất:**
```
🚨 KHẨN CẤP: Nhổ bỏ cây bệnh ngay lập tức!
🔥 Đốt hoặc chôn sâu (không compost)
💊 Phun Ridomil Gold (Metalaxyl + Mancozeb) NGAY
🚧 Cách ly khu vực bệnh, không đi lại
📅 Phun thuốc 5-7 ngày/lần trong 3 tuần
🌱 Trồng giống kháng bệnh (Defiant PHR, Matt's Wild Cherry)
```

**Healthy (Khỏe mạnh):**
```
✅ Duy trì chế độ chăm sóc hiện tại
🌿 Kiểm tra định kỳ để phát hiện sớm bệnh
💧 Tưới nước đều đặn, tránh khô hạn
🌞 Đảm bảo đủ ánh sáng (6-8 giờ/ngày)
```

## 🎯 Sử dụng

### Dự đoán từ ảnh upload

1. Click nút **"📁 Chọn ảnh từ máy"**
2. Chọn ảnh lá cà chua
3. Click **"🔮 Dự đoán"**
4. Xem kết quả

### Dự đoán từ camera

1. Click nút **"📷 Chụp ảnh từ camera"**
2. Cho phép truy cập camera
3. Chụp ảnh lá cà chua
4. Click **"🔮 Dự đoán"**
5. Xem kết quả

### Xem lịch sử

1. Click tab **"📜 Lịch sử"**
2. Xem các lần dự đoán trước (thumbnail, tên bệnh, độ tin cậy, thời gian)
3. **Click vào bất kỳ item nào** để xem chi tiết đầy đủ:
   - Ảnh gốc kích thước lớn
   - Top 5 dự đoán với thanh progress
   - Thông tin bệnh + khuyến nghị chăm sóc đầy đủ
   - Metadata: thời gian, file, vein score
4. Có thể xóa từng item (nút 🗑️) hoặc xóa toàn bộ

## 🔬 Công nghệ Sử dụng

### Backend
- **FastAPI** - Framework web hiện đại, nhanh
- **TensorFlow 2.15.0** - Deep Learning framework
- **OpenCV** - Xử lý ảnh
- **Pillow** - Thao tác ảnh
- **NumPy** - Tính toán số học

### Model
- **EfficientNetB0** - Base architecture
- **6 classes**: Bacterial Spot, Early Blight, Healthy, Late Blight, Septoria Leaf Spot, Yellow Leaf Curl Virus
- **Input**: 256x256 RGB images
- **Output**: Softmax probabilities

### Frontend
- **HTML5/CSS3/JavaScript** - Giao diện responsive
- **Fetch API** - Gọi API
- **Canvas API** - Chụp ảnh từ camera

## 🛠️ Chi tiết Module Xử lý

### 📦 **1. efficientnet_preprocessor.py** (453 lines)
**Chức năng**: Preprocessing 6 bước cho model inference

**Pipeline:**
1. **Step 1: Resize** - Resize về 224x224 (EfficientNetB0 input)
2. **Step 2: Analyze** - Tính metrics: brightness, contrast, noise_variance, edge_strength
3. **Step 3: CLAHE** - Tăng contrast nếu < 40 (adaptive histogram equalization)
4. **Step 4: Denoise** - Khử nhiễu nếu variance < 500 (bilateral filter)
5. **Step 5: Sharpen** - Làm nét nếu edge < 50 (unsharp masking)
6. **Step 6: Normalize** - ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

**Output**: 6 ảnh (1 ảnh/bước) + summary JSON

---

### 📦 **2. image_preprocessing.py** (1084 lines)
**Chức năng**: Validate ảnh đầu vào có phải lá cây không

**Class ImagePreprocessor:**
- `is_leaf_image()` - Kiểm tra 8 tiêu chí:
  - ✅ Phát hiện gân lá (vein detection - quan trọng nhất)
  - ✅ Phân tích màu sắc (green, yellow, brown, dark_spots, shadow)
  - ✅ Phân tích hình dạng lá (aspect ratio, solidity, circularity)
  - ✅ Kiểm tra độ nét, độ sáng, độ tương phản
  - ✅ Edge detection (Canny)
  - ✅ Contour analysis
  - ✅ Texture score
  - ✅ Leaf shape score

**Class LeafDetector:**
- `segment_leaf()` - Tách lá khỏi background
- `quick_check_leaf()` - Kiểm tra nhanh

**Xử lý đặc biệt:**
- ✅ Chấp nhận lá bệnh (vàng/nâu/đen)
- ✅ Chấp nhận lá có bóng
- ✅ Chấp nhận lá bị sâu ăn/rách
- ✅ Auto-enhance ảnh tối

---

### 📦 **3. image_analysis.py** (566 lines)
**Chức năng**: Phân tích chi tiết đặc trưng lá

**Functions:**
- `analyze_shape()` - Phân tích hình dạng (perimeter, convexity, roughness)
- `analyze_color()` - Phân tích màu sắc HSV/LAB
- `analyze_texture()` - Phân tích texture (LBP, Gabor filters, vein patterns)
- `calculate_leaf_score()` - Tính điểm tổng hợp
- `generate_processed_images()` - Tạo ảnh visualization (edge map, veins, histogram)

**Output**: JSON với scores + 3 ảnh visualization

---

### 📦 **4. data_raw.py** (340 lines)
**Chức năng**: Tạo dataset XẤU để test độ robust

**10 degradation types:**
1. `noise_light` - Gaussian noise nhẹ (factor=0.1)
2. `noise_heavy` - Gaussian noise nặng (factor=0.3)
3. `blur_light` - Gaussian blur nhẹ (kernel=5)
4. `blur_heavy` - Gaussian blur nặng (kernel=15)
5. `dark` - Giảm brightness 50%
6. `very_dark` - Giảm brightness 70%
7. `bright` - Tăng brightness 30%
8. `motion_blur` - Motion blur (kernel=15)
9. `low_contrast` - Giảm contrast 50%
10. `jpeg_compress` - JPEG artifacts (quality=20)

**Usage:**
```python
python data_raw.py
# Input: Tomato/Train
# Output: Tomato/Augmented_Train
```

---

### 📦 **5. data_preprocessing.py** (452 lines)
**Chức năng**: PHỤC HỒI ảnh xấu về chất lượng tốt

**4 bước sửa chữa MẠNH:**
1. **Fix Noise** - Bilateral filter d=7-9 (khử nhiễu trước tiên)
2. **Fix Brightness** - Điều chỉnh về target=135 (sửa tối/sáng)
3. **Fix Contrast** - CLAHE 2.8-3.5 (tăng contrast mạnh)
4. **Fix Sharpness** - Unsharp masking kernel 9-10 (làm nét)

**Mode:**
- `aggressive_fix=True` - Sửa TẤT CẢ ảnh (đồng nhất chất lượng)
- `aggressive_fix=False` - Chỉ sửa ảnh xấu (conditional)

**Usage:**
```python
python data_preprocessing.py
# Input: Tomato/Augmented_Train (ảnh XẤU)
# Output: Tomato/Fixed_Train (ảnh SẠCH)
```

## 🌐 API Endpoints

### `GET /`
Hiển thị giao diện web

### `POST /predict`
Dự đoán bệnh từ ảnh

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
    "success": true,
    "predicted_class": "Early Blight",
    "confidence": 95.67,
    "top_predictions": [
        {"class": "Early Blight", "confidence": 95.67},
        {"class": "Late Blight", "confidence": 3.21},
        ...
    ],
    "image_analysis": {
        "type": "diseased_leaf",
        "green_ratio": 45.23,
        "recommendation": "Ảnh đạt chất lượng tốt"
    },
    "disease_info": {
        "name_vi": "Bệnh Héo Sớm",
        "severity": "Trung bình - Cao",
        "description": "Bệnh do nấm Alternaria solani...",
        "symptoms": ["Đốm tròn có vòng đồng tâm..."],
        "treatment": {
            "immediate": ["Cắt bỏ lá bệnh..."],
            "shortterm": ["Phun thuốc 7 ngày/lần..."],
            "longterm": ["Cải tạo đất..."]
        },
        "prevention": [...],
        "products": ["Daconil", "Mancozeb", "Azoxystrobin"]
    }
}
```

### `GET /history`
Lấy lịch sử dự đoán (tất cả items)

### `GET /history/{item_id}` ⭐ NEW
Lấy chi tiết một item trong lịch sử (bao gồm disease_info, top_predictions)

### `DELETE /history/{item_id}`
Xóa một item trong lịch sử

### `DELETE /history`
Xóa toàn bộ lịch sử

### `GET /health`
Kiểm tra trạng thái server

## 🎓 Train Model Mới

### Quy trình ĐẦY ĐỦ

#### **Bước 1: Chuẩn bị Dataset (Optional - nếu muốn augmentation)**

**1a. Tạo ảnh xấu (để test độ robust):**
```bash
python data_raw.py
```
- Input: `Tomato/Train/` (ảnh gốc sạch)
- Output: `Tomato/Augmented_Train/` (ảnh nhiễu/mờ/tối)
- Mục đích: Test xem model có học được từ ảnh chất lượng kém không

**1b. Phục hồi ảnh xấu (sửa về tốt để train):**
```bash
python data_preprocessing.py
```
- Input: `Tomato/Augmented_Train/` (ảnh XẤU)
- Output: `Tomato/Fixed_Train/` (ảnh đã SỬA - chất lượng tốt)
- Mục đích: Train model với data sạch, đồng nhất

**Lưu ý:**
- ⚠️ Bước 1 là **OPTIONAL** - chỉ dùng nếu bạn muốn tạo augmented dataset
- ✅ Có thể train trực tiếp với `Tomato/Train/` gốc (đã đủ tốt)
- 💡 `data_raw.py` và `data_preprocessing.py` là 2 bước ngược nhau:
  - `data_raw.py`: Làm XẤU dataset
  - `data_preprocessing.py`: SỬA dataset xấu về tốt

---

#### **Bước 2: Train Model**

```bash
# Mở notebook
jupyter notebook train_model_h5.ipynb

# Hoặc chạy tất cả cells: Cell → Run All
# Thời gian: 45-60 phút (GPU) hoặc 4-6 giờ (CPU)
```

### Cấu hình Training

```python
# Dataset paths
TRAIN_DIR = 'Tomato/Train'           # Hoặc 'Tomato/Fixed_Train' nếu dùng preprocessed
VAL_DIR = 'Tomato/Val'
TEST_DIR = 'Tomato/Test'

# Hyperparameters
IMG_SIZE = 256
BATCH_SIZE = 32                      # Giảm xuống 16 nếu GPU OOM
EPOCHS_STAGE1 = 15                   # Frozen base
EPOCHS_STAGE2 = 15                   # Fine-tuning
DROPOUT_RATE = 0.3                   # Tăng lên 0.4 nếu overfitting
LEARNING_RATE = 0.001

# Advanced features
USE_MIXUP = True                     # MixUp augmentation
USE_ATTENTION = True                 # Spatial Attention
USE_CLASS_WEIGHTS = True             # Imbalanced data
USE_TTA = True                       # Test-Time Augmentation
```

### Kết quả mong đợi
- Test Accuracy: **95-97%**
- Top-3 Accuracy: **>98%**
- Per-class F1: **>0.90**
- Loss: **<0.20**

## 🐛 Troubleshooting

### 🔴 **Inference Issues (Dự đoán)**

#### **1. Ảnh bị từ chối "KHÔNG PHẢI ẢNH LÁ CÂY"**
**Nguyên nhân:**
- Không phát hiện được gân lá (vein score < 0.15)
- Màu sắc không giống lá (green_ratio < 2%)
- Hình dạng không giống lá (shape score < 0.40)

**Giải pháp:**
- ✅ Chụp ở nơi sáng (tránh bóng tối quá nặng)
- ✅ Lá chiếm >30% diện tích ảnh
- ✅ Focus rõ ràng (tránh ảnh mờ)
- ✅ Chụp từ góc nhìn thẳng (tránh góc nghiêng quá)
- ✅ Chấp nhận lá bệnh (vàng/nâu/đen), lá có bóng, lá rách

**Kiểm tra:**
```bash
# Xem chi tiết phân tích
# Vào web → Upload ảnh → Xem phần "Phân Tích Đặc Trưng Lá Cây"
# Kiểm tra: vein_score, green_ratio, leaf_shape_score
```

#### **2. Độ tin cậy thấp (<70%)**
**Nguyên nhân:**
- Ảnh chất lượng kém (mờ, tối, nhiễu)
- Bệnh phức tạp (nhiều loại bệnh trên 1 lá)
- Model chưa học tốt trường hợp này

**Giải pháp:**
- ✅ Chụp lại với chất lượng tốt hơn
- ✅ Xem Top 5 predictions (có thể bệnh đúng ở vị trí 2-3)
- ✅ Tham khảo nhiều lá khác nhau

#### **3. Dự đoán sai**
**Nguyên nhân:**
- Model confusion giữa các bệnh tương tự (Early Blight ↔ Late Blight)
- Triệu chứng bệnh chưa rõ ràng (giai đoạn sớm)

**Giải pháp:**
- ✅ Xem Top 5 predictions
- ✅ So sánh triệu chứng với disease_info
- ✅ Chụp nhiều lá khác nhau để xác nhận

---

### 🟡 **Training Issues (Train model)**

#### **1. GPU Out of Memory**
```python
# Trong notebook
BATCH_SIZE = 16  # Giảm từ 32 xuống 16
IMG_SIZE = 224   # Giảm từ 256 xuống 224
```

#### **2. Overfitting (train acc >> val acc)**
**Ví dụ:** Train 98%, Val 85%

**Giải pháp:**
```python
DROPOUT_RATE = 0.4        # Tăng từ 0.3
USE_MIXUP = True          # Bật MixUp
L2_REGULARIZATION = 0.01  # Thêm L2 reg
AUGMENTATION_STRENGTH = 0.3  # Tăng augmentation
```

#### **3. Underfitting (cả 2 acc đều thấp)**
**Ví dụ:** Train 80%, Val 78%

**Giải pháp:**
```python
Dense(768)                # Tăng capacity (từ 512)
LEARNING_RATE = 0.002     # Tăng learning rate
EPOCHS_STAGE2 = 20        # Train lâu hơn
DROPOUT_RATE = 0.2        # Giảm dropout
```

#### **4. Convergence chậm (loss giảm chậm)**
**Giải pháp:**
```python
LEARNING_RATE = 0.002     # Tăng learning rate
BATCH_SIZE = 64           # Tăng batch size
USE_WARMUP = True         # Thêm warmup schedule
```

#### **5. Class imbalance (một vài class acc thấp)**
**Giải pháp:**
```python
USE_CLASS_WEIGHTS = True  # Bật class weighting
FOCAL_LOSS = True         # Dùng Focal Loss thay Categorical Crossentropy
OVERSAMPLE_MINORITY = True  # Oversample class thiểu số
```

---

### 🟢 **Data Preprocessing Issues**

#### **1. data_raw.py lỗi "No such file or directory"**
**Giải pháp:**
```python
# Kiểm tra đường dẫn trong file
INPUT_DIR = "Tomato/Train"  # Phải có thư mục này
OUTPUT_DIR = "Tomato/Augmented_Train"  # Sẽ tự tạo
```

#### **2. data_preprocessing.py quá chậm**
**Giải pháp:**
```python
# Giảm số lượng ảnh test
# Hoặc giảm aggressive_fix
aggressive_fix = False  # Chỉ sửa ảnh xấu thay vì tất cả
```

#### **3. Ảnh sau preprocessing quá sáng/tối**
**Giải pháp:**
```python
# Điều chỉnh ngưỡng trong data_preprocessing.py
self.brightness_low = 80     # Giảm từ 100
self.brightness_high = 200   # Tăng từ 180
```

## 📊 So sánh Model v1.0 vs v2.0 vs v2.1

| Metric | v1.0 | v2.0 | v2.1 (Current) | Cải thiện |
|--------|------|------|----------------|-----------|
| Test Accuracy | 92.3% | 95.6% | **95.6%** | **+3.3%** ⬆️ |
| Top-3 Accuracy | 97.8% | 98.9% | **98.9%** | **+1.1%** ⬆️ |
| F1-Score (avg) | 0.918 | 0.953 | **0.953** | **+0.035** ⬆️ |
| Model Size | 16 MB | 55 MB | 55 MB | +39 MB |
| Inference | ~100ms | ~150ms | ~150ms | +50ms |
| **Disease Info** | ❌ | ❌ | **✅ 6 diseases** | **NEW!** |
| **Care Recommendations** | ❌ | ❌ | **✅ Full guide** | **NEW!** |
| **History Detail View** | ❌ | ❌ | **✅ Modal popup** | **NEW!** |

**10 Tính năng nổi bật:**
1. MixUp Augmentation - Tăng tính tổng quát
2. Spatial Attention - Tập trung vào vùng bệnh
3. Two-Stage Training - Fine-tune hiệu quả
4. Class Weighting - Xử lý imbalanced data
5. Enhanced Architecture - Dense layers tốt hơn
6. Advanced Augmentation - 7 techniques thay vì 4
7. Test-Time Augmentation - Tăng accuracy inference
8. **🏥 Disease Care Database - 6 bệnh với hướng dẫn chi tiết** ⭐ NEW
9. **💊 Product Recommendations - Tên thuốc cụ thể** ⭐ NEW
10. **📋 History Detail Modal - Xem lại kết quả cũ** ⭐ NEW

## 🚀 Deploy

### Deploy lên Cloud (Heroku, AWS, GCP)
1. Chuẩn bị `Procfile`:
   ```
   web: uvicorn app:app --host 0.0.0.0 --port $PORT
   ```

2. Thêm vào `requirements.txt`:
   ```
   gunicorn==21.2.0
   ```

3. Deploy theo hướng dẫn của platform

## 📄 License

MIT License - Free to use for educational and research purposes

## 🔍 So sánh Các File Xử lý

| File | Chức năng | Khi nào dùng | Input | Output |
|------|-----------|--------------|-------|--------|
| **efficientnet_preprocessor.py** | 6-step preprocessing cho inference | Dự đoán realtime | Ảnh user upload | Ảnh chuẩn 224x224 + 6 bước |
| **image_preprocessing.py** | Validate ảnh lá + enhance | Kiểm tra trước khi dự đoán | Ảnh bất kỳ | True/False + details |
| **image_analysis.py** | Phân tích đặc trưng lá | Hiển thị thông tin chi tiết | Ảnh lá | Shape/color/texture scores |
| **data_raw.py** | Làm XẤU dataset | Tạo augmented data (optional) | Dataset gốc | Dataset xấu |
| **data_preprocessing.py** | SỬA ảnh xấu về tốt | Fix dataset trước train | Dataset xấu | Dataset sạch |

**Lưu ý quan trọng:**
- 🚀 **Production**: Chỉ dùng `efficientnet_preprocessor.py` + `image_preprocessing.py` + `image_analysis.py`
- 📊 **Training**: Chỉ dùng `data_raw.py` + `data_preprocessing.py` (optional)
- ⚠️ **KHÔNG dùng chung**: File training ≠ File production

---

## 🎉 Acknowledgments

- EfficientNet: Tan & Le (2019)
- MixUp: Zhang et al. (2018)
- CBAM: Woo et al. (2018)
- Dataset: PlantVillage Project

## 🎁 Điểm Nổi Bật v2.1 (December 11, 2025)

### 🏥 Disease Care System
- **Database chuyên nghiệp**: 6 bệnh với 500+ dòng hướng dẫn chi tiết
- **3-tier treatment plan**: Immediate → Short-term → Long-term
- **Severity indicators**: 🔴 Cao, 🟡 Trung bình, 🟢 Thấp, ✅ Khỏe
- **Product recommendations**: Tên thương mại cụ thể (Ridomil Gold, Daconil, Actara...)
- **Visual UI**: Color-coded badges, collapsible sections, responsive design

### 📋 Interactive History
- **Click-to-view**: Mỗi lịch sử giờ có thể click để xem chi tiết
- **Modal popup**: Hiển thị đầy đủ ảnh gốc, top predictions, disease info
- **Smart navigation**: ESC key, click outside, X button
- **Preserved data**: Lưu disease_info và top_predictions trong history.json

### 🔧 Advanced Preprocessing Pipeline
- **6-step conditional preprocessing**: Resize → Analyze → CLAHE → Denoise → Sharpen → Normalize
- **Smart validation**: Vein detection, color analysis, shape analysis
- **Auto-enhancement**: Adaptive brightness, contrast, sharpness adjustments
- **Robust to degradation**: Handles dark, blurry, noisy, low-contrast images

### 📊 Professional Data Pipeline
- **data_raw.py**: 10 degradation types để tạo augmented dataset
- **data_preprocessing.py**: 4-step restoration để fix ảnh xấu về tốt
- **Flexible workflow**: Có thể train với dataset gốc hoặc preprocessed

### 💡 Use Cases
1. **Nông dân**: Chụp ảnh → Nhận hướng dẫn điều trị ngay lập tức
2. **Nhà nghiên cứu**: Theo dõi diễn biến bệnh qua lịch sử + phân tích đặc trưng
3. **Giáo dục**: Học sinh/sinh viên học về bệnh cây trồng + preprocessing pipeline
4. **Cửa hàng thuốc**: Tư vấn sản phẩm phù hợp cho khách hàng

---

## 📝 Changelog

### v2.1 (December 11, 2025)
- ✅ Tách preprocessing thành 6 bước riêng biệt (luôn hiển thị)
- ✅ Thêm data_raw.py (10 augmentation types)
- ✅ Thêm data_preprocessing.py (4-step restoration)
- ✅ Cải thiện README với luồng hoạt động chi tiết
- ✅ Xóa file thừa (leaf_detector.py trùng, test.py cũ)

### v2.0 (December 5, 2025)
- ✅ Disease Care System với 6 bệnh chi tiết
- ✅ Interactive History với modal popup
- ✅ Spatial Attention Mechanism
- ✅ MixUp Augmentation
- ✅ Accuracy: 92% → 95-96%

---

**Version 2.1** - December 11, 2025 | **Status:** Production Ready ✅ | **Accuracy:** 95-96% 🎯 | **NEW:** 6-Step Preprocessing Pipeline + Professional Data Pipeline 🔧
