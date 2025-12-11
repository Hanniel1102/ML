# 🍅 Tomato Disease Detection - AI Web Application

Ứng dụng web sử dụng Deep Learning (EfficientNetB0 + Spatial Attention) để phát hiện bệnh trên lá cà chua với độ chính xác **95-96%**.

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
