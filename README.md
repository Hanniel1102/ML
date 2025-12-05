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
├── ├──app.py                          # FastAPI server + Disease Info Database
│   ├── image_preprocessing.py          # Tiền xử lý ảnh thông minh
│   ├── leaf_detector.py                # Phát hiện lá cây
│   └── requirements.txt                # Dependencies
│
├── ├── best_tomato_model.keras         # Model tối ưu v2.0 (95-96%)
│   └── models/
│       ├── class_names.json
│       └── model_info.json
│
├── ├── prediction_history.json         # Lưu lịch sử + disease_info
│   └── Tomato/                         # Dataset
│       ├── Train/
│       ├── Val/
│       └── Test/
│
├── ├── templates/
│       └── index.html                  # UI + Disease Recommendations
│
└── ├── Code_demo_optimized.ipynb       # Training notebook v2.0
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

## 🛠️ Module Tiền xử lý Thông minh

File `image_preprocessing.py` bao gồm:

### 1. Kiểm tra tính hợp lệ của ảnh
- ✅ Phát hiện gân lá (vein detection)
- ✅ Phân tích màu sắc (green, yellow, brown, shadow)
- ✅ Phân tích hình dạng lá (aspect ratio, solidity)
- ✅ Kiểm tra độ nét, độ sáng, độ tương phản

### 2. Tăng cường chất lượng ảnh
- Tự động điều chỉnh độ sáng (auto brightness)
- Làm nét ảnh (sharpening)
- Cân bằng histogram (CLAHE)
- Khử nhiễu (denoising)

### 3. Xử lý đặc biệt
- Hỗ trợ ảnh tối/quá sáng
- Phát hiện lá bệnh, lá có bóng
- Chấp nhận lá bị sâu ăn, lá rách

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

### Yêu cầu
- Python 3.11+, GPU với CUDA (khuyến nghị)
- 8GB RAM, Dataset đúng cấu trúc

### Training
```bash
# Mở notebook
jupyter notebook Code_demo_optimized.ipynb

# Hoặc chạy tất cả cells: Cell → Run All
# Thời gian: 45-60 phút (GPU) hoặc 4-6 giờ (CPU)
```

### Hyperparameters chính
```python
IMG_SIZE = 256
BATCH_SIZE = 32          # Giảm xuống 16 nếu GPU OOM
EPOCHS_STAGE1 = 15       # Frozen base
EPOCHS_STAGE2 = 15       # Fine-tuning
DROPOUT_RATE = 0.3       # Tăng lên 0.4 nếu overfitting

USE_MIXUP = True         # MixUp augmentation
USE_ATTENTION = True     # Spatial Attention
USE_CLASS_WEIGHTS = True # Imbalanced data
```

### Kết quả mong đợi
- Test Accuracy: **95-97%**
- Top-3 Accuracy: **>98%**
- Per-class F1: **>0.90**

## 🐛 Troubleshooting

### GPU Out of Memory
```python
BATCH_SIZE = 16  # Giảm xuống trong notebook
```

### Overfitting (train acc >> val acc)
```python
DROPOUT_RATE = 0.4  # Tăng regularization
USE_MIXUP = True
```

### Underfitting (cả 2 acc đều thấp)
```python
Dense(768)  # Tăng capacity
LEARNING_RATE = 0.002
EPOCHS_STAGE2 = 20
```

### Ảnh bị từ chối
- Chụp ở nơi sáng, tránh quá tối/sáng
- Lá chiếm >30% diện tích ảnh
- Focus rõ, tránh ảnh mờ
- Chỉ upload ảnh lá cây thật

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

## 🎉 Acknowledgments

- EfficientNet: Tan & Le (2019)
- MixUp: Zhang et al. (2018)
- CBAM: Woo et al. (2018)
- Dataset: PlantVillage Project

## 🎁 Điểm Nổi Bật v2.1 (December 5, 2025)

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

### 💡 Use Cases
1. **Nông dân**: Chụp ảnh → Nhận hướng dẫn điều trị ngay lập tức
2. **Nhà nghiên cứu**: Theo dõi diễn biến bệnh qua lịch sử
3. **Giáo dục**: Học sinh/sinh viên học về bệnh cây trồng
4. **Cửa hàng thuốc**: Tư vấn sản phẩm phù hợp cho khách hàng

---

**Version 2.1** - December 5, 2025 | **Status:** Production Ready ✅ | **Accuracy:** 95-96% 🎯 | **NEW:** Disease Care System 🏥
