# 🍅 Tomato Disease Detection Web App

Ứng dụng web sử dụng FastAPI và TensorFlow để chẩn đoán bệnh trên lá cà chua.

## Tính năng

✅ Upload ảnh từ máy tính
✅ Chụp ảnh trực tiếp từ camera
✅ Dự đoán bệnh với độ tin cậy cao
✅ Hiển thị Top 5 dự đoán chi tiết
✅ Giao diện đẹp, responsive

## Cài đặt

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Đảm bảo có model

Đặt file model (`.keras`) vào thư mục gốc hoặc thư mục `models/`:
- `Tomato_EfficientNetB0_Final.keras`
- `best_tomato_model.keras`

### 3. Chạy server

```bash
python app.py
```

Hoặc:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Mở trình duyệt

Truy cập: http://localhost:8000

## Cấu trúc thư mục

```
Hocmaynangcao/
├── app.py                              # FastAPI server
├── requirements.txt                     # Dependencies
├── templates/
│   └── index.html                      # Giao diện web
├── models/                             # (Optional) Thư mục chứa model
│   ├── final_model.keras
│   └── class_names.json
├── Tomato_EfficientNetB0_Final.keras   # Model file
└── best_tomato_model.keras             # Model backup
```

## API Endpoints

### GET `/`
Hiển thị giao diện web

### POST `/predict`
Dự đoán bệnh từ ảnh upload

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
    "success": true,
    "predicted_class": "Tomato_Late_blight",
    "confidence": 98.45,
    "top_predictions": [
        {
            "class": "Tomato_Late_blight",
            "confidence": 98.45
        },
        ...
    ]
}
```

### GET `/health`
Kiểm tra trạng thái server

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "num_classes": 10
}
```

## Sử dụng

1. **Chọn ảnh từ máy:** Click nút "📁 Chọn ảnh từ máy"
2. **Chụp ảnh từ camera:** Click nút "📷 Chụp ảnh từ camera"
3. **Dự đoán:** Click nút "🔮 Dự đoán" để phân tích
4. **Xem kết quả:** Kết quả hiển thị với độ tin cậy và top 5 dự đoán

## Lưu ý

- Hỗ trợ định dạng ảnh: JPG, JPEG, PNG, BMP, TIFF
- Ảnh sẽ được tự động resize về kích thước phù hợp
- Camera yêu cầu HTTPS (trừ localhost)
- Để truy cập từ thiết bị khác trong cùng mạng: sử dụng IP máy chủ

## Troubleshooting

**Lỗi: Model not found**
- Đảm bảo file model tồn tại ở đúng đường dẫn
- Kiểm tra tên file model trong `app.py`

**Lỗi: Camera không hoạt động**
- Kiểm tra quyền truy cập camera trong trình duyệt
- Đảm bảo sử dụng HTTPS hoặc localhost

**Lỗi: Port already in use**
- Thay đổi port trong `app.py`: `uvicorn.run(app, port=8001)`
