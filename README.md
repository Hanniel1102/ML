# 🍅 Tomato Disease Detection - AI Web Application

Ứng dụng web sử dụng Deep Learning (EfficientNetB0 + Spatial Attention) để phát hiện bệnh trên lá cà chua với độ chính xác **95-96%**.

## ✨ Tính năng

- 📤 Upload ảnh từ máy tính hoặc 📷 chụp từ camera
- 🍃 **Phát hiện tự động ảnh lá** - Từ chối ảnh không phải lá cây
- 🔬 Dự đoán 6 loại bệnh với độ tin cậy cao (**95-96% accuracy**)
- 📊 Top 5 dự đoán + phân tích chi tiết chất lượng ảnh
- 🎨 Giao diện đẹp, responsive, thân thiện người dùng
- 📜 Lịch sử dự đoán (100 kết quả gần nhất)
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

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng
```bash
python app.py
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 5. Truy cập ứng dụng

Mở trình duyệt và truy cập: **http://localhost:8000**

## 📁 Cấu trúc Dự án

```
Hocmaynangcao/
├── ├── app.py                          # FastAPI server
│   ├── image_preprocessing.py          # Tiền xử lý ảnh
│   ├── leaf_detector.py                # Phát hiện lá (NEW)
│   └── requirements.txt                # Dependencies
│
├── ├── best_tomato_model.keras         # Model tối ưu v2.0
│   └── models/
│       ├── class_names.json
│       └── model_info.json
│
├── |── prediction_history.json
│   └── Tomato/                         # Dataset
│       ├── Train/
│       ├── Val/
│       └── Test/
│
├── templates/
│       └── index.html
│
└── Code_demo_optimized.ipynb       # Training notebook v2.0
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
2. Xem các lần dự đoán trước
3. Có thể xóa từng item hoặc xóa toàn bộ

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
    }
}
```

### `GET /history`
Lấy lịch sử dự đoán

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

## 📊 So sánh Model v1.0 vs v2.0

| Metric | v1.0 | v2.0 | Cải thiện |
|--------|------|------|-----------|
| Test Accuracy | 92.3% | **95.6%** | **+3.3%** ⬆️ |
| Top-3 Accuracy | 97.8% | **98.9%** | **+1.1%** ⬆️ |
| F1-Score (avg) | 0.918 | **0.953** | **+0.035** ⬆️ |
| Model Size | 16 MB | 55 MB | +39 MB |
| Inference | ~100ms | ~150ms | +50ms |

**7 Cải tiến chính:**
1. MixUp Augmentation - Tăng tính tổng quát
2. Spatial Attention - Tập trung vào vùng bệnh
3. Two-Stage Training - Fine-tune hiệu quả
4. Class Weighting - Xử lý imbalanced data
5. Enhanced Architecture - Dense layers tốt hơn
6. Advanced Augmentation - 7 techniques thay vì 4
7. Test-Time Augmentation - Tăng accuracy inference

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

---

**Version 2.0** - December 2025 | **Status:** Production Ready ✅ | **Accuracy:** 95-96% 🎯
