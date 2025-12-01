from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import cv2
from datetime import datetime
import base64

# Import module tiền xử lý thông minh
from image_preprocessing import ImagePreprocessor, preprocess_and_check

app = FastAPI(title="Tomato Disease Detection API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Biến toàn cục
model = None
class_names = None
IMG_SIZE = 256
preprocessor = ImagePreprocessor()

# File lưu lịch sử
HISTORY_FILE = "prediction_history.json"

# Load lịch sử từ file
def load_history():
    """Load lịch sử dự đoán từ file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

# Lưu lịch sử vào file
def save_history(history):
    """Lưu lịch sử dự đoán vào file"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Lỗi lưu lịch sử: {e}")

# Thêm kết quả vào lịch sử
def add_to_history(result_data):
    """Thêm kết quả dự đoán vào lịch sử"""
    history = load_history()
    
    # Giới hạn 100 kết quả gần nhất
    if len(history) >= 100:
        history = history[-99:]
    
    history.append(result_data)
    save_history(history)
    return len(history)

# Load model khi khởi động
@app.on_event("startup")
async def load_model_startup():
    global model, class_names, IMG_SIZE
    
    print("🚀 Đang khởi động server...")
    
    # Tìm và load model
    model_paths = [
        "Tomato_EfficientNetB0_Final.keras",
        "best_tomato_model.keras",
        "models/final_model.keras",
        "models/best_model.keras"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"✅ Đã load model: {model_path}")
            break
    
    if model is None:
        print("❌ Không tìm thấy model!")
        raise RuntimeError("Model not found!")
    
    # Load class names
    if os.path.exists('models/class_names.json'):
        with open('models/class_names.json', 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print(f"✅ Đã load class names từ file")
    else:
        # Lấy từ dataset
        DATASET_PATH = "../Hocmaynangcao/Tomato/"
        test_dir = os.path.join(DATASET_PATH, 'Test')
        
        if os.path.exists(test_dir):
            temp_ds = tf.keras.utils.image_dataset_from_directory(
                test_dir,
                image_size=(256, 256),
                batch_size=32,
                label_mode='categorical',
                shuffle=False
            )
            class_names = temp_ds.class_names
            print(f"✅ Đã load class names từ dataset")
        else:
            # Fallback class names
            class_names = [
                "Tomato_Bacterial_spot",
                "Tomato_Early_blight",
                "Tomato_Late_blight",
                "Tomato_Leaf_Mold",
                "Tomato_Septoria_leaf_spot",
                "Tomato_Spider_mites",
                "Tomato_Target_Spot",
                "Tomato_Yellow_Leaf_Curl_Virus",
                "Tomato_mosaic_virus",
                "Tomato_healthy"
            ]
            print(f"⚠️ Sử dụng class names mặc định")
    
    IMG_SIZE = model.input_shape[1]
    print(f"📏 Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"📝 Số lượng classes: {len(class_names)}")
    print(f"✅ Server đã sẵn sàng!\n")

# API endpoint chính
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """API dự đoán bệnh từ ảnh upload - với kiểm tra thông minh"""
    global model, class_names, IMG_SIZE
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model chưa được load")
    
    try:
        # Đọc ảnh
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Chuyển sang RGB nếu cần
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # === BƯỚC 1: KIỂM TRA THÔNG MINH ===
        # Sử dụng thuật toán đa tầng: texture + shape + color
        result = preprocess_and_check(img, target_size=(IMG_SIZE, IMG_SIZE))
        
        # Nếu KHÔNG phải lá cây (chó, mèo, người, đồ vật)
        if not result['is_leaf']:
            details = result['details']
            return JSONResponse({
                "success": False,
                "error": "NOT_LEAF_IMAGE",
                "message": "Ảnh không phải là ảnh lá cây",
                "recommendation": details.get('recommendation', 'Vui lòng chọn ảnh lá cây'),
                "analysis": {
                    "green_ratio": round(details.get('green_ratio', 0) * 100, 2),
                    "shadow_ratio": round(details.get('shadow_ratio', 0) * 100, 2),
                    "texture_score": round(details.get('texture_score', 0), 2),
                    "leaf_shape_score": round(details.get('leaf_shape_score', 0), 2),
                    "is_damaged_leaf": details.get('is_damaged_leaf', False),
                    "has_shadow": details.get('has_shadow', False)
                }
            })
        
        # === BƯỚC 2: SỬ DỤNG ẢNH ĐÃ TĂNG CƯỜNG ===
        # Ảnh đã được tự động xử lý: tăng sáng, làm nét, CLAHE
        enhanced_img = result['enhanced_image']
        img_array = np.array(enhanced_img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # === BƯỚC 3: DỰ ĐOÁN BỆNH ===
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_idx] * 100)
        
        # Top 5 predictions
        num_top = min(5, len(class_names))
        top_idx = np.argsort(predictions[0])[-num_top:][::-1]
        top_predictions = [
            {
                "class": class_names[int(idx)],
                "confidence": float(predictions[0][idx] * 100)
            }
            for idx in top_idx
        ]
        
        # === BƯỚC 4: PHÂN TÍCH CHẤT LƯỢNG ẢNH ===
        details = result['details']
        image_analysis = {
            "type": "diseased_leaf" if details.get('is_diseased_leaf') else (
                    "shadow_leaf" if details.get('has_shadow') else (
                    "damaged_leaf" if details.get('is_damaged_leaf') else "healthy_leaf")),
            "green_ratio": round(details.get('green_ratio', 0) * 100, 2),
            "shadow_ratio": round(details.get('shadow_ratio', 0) * 100, 2),
            "texture_score": round(details.get('texture_score', 0), 2),
            "leaf_shape_score": round(details.get('leaf_shape_score', 0), 2),
            "brightness": round(details.get('brightness', 0), 1),
            "sharpness": round(details.get('sharpness', 0), 1),
            "recommendation": details.get('recommendation', 'Ảnh đạt chất lượng tốt')
        }
        
        # === BƯỚC 5: LƯU VÀO LỊCH SỬ ===
        # Convert ảnh sang base64 để lưu thumbnail
        img_thumbnail = img.copy()
        img_thumbnail.thumbnail((150, 150))
        buffered = io.BytesIO()
        img_thumbnail.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        history_entry = {
            "id": len(load_history()) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": file.filename,
            "predicted_class": class_names[predicted_class_idx],
            "confidence": round(confidence, 2),
            "image_type": image_analysis["type"],
            "vein_score": round(details.get('vein_score', 0), 2),
            "thumbnail": f"data:image/jpeg;base64,{img_base64}"
        }
        add_to_history(history_entry)
        
        response_data = {
            "success": True,
            "predicted_class": class_names[predicted_class_idx],
            "confidence": confidence,
            "top_predictions": top_predictions,
            "image_analysis": image_analysis,
            "preprocessing": "enhanced" if details.get('is_dark_detected') else "standard",
            "history_id": history_entry["id"]
        }
        
        return JSONResponse(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")

@app.get("/history")
async def get_history():
    """Lấy lịch sử dự đoán"""
    try:
        history = load_history()
        # Sắp xếp theo thời gian mới nhất
        history.reverse()
        return JSONResponse({
            "success": True,
            "count": len(history),
            "history": history
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.delete("/history/{item_id}")
async def delete_history_item(item_id: int):
    """Xóa một item trong lịch sử"""
    try:
        history = load_history()
        history = [h for h in history if h.get('id') != item_id]
        save_history(history)
        return JSONResponse({
            "success": True,
            "message": "Đã xóa thành công"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.delete("/history")
async def clear_history():
    """Xóa toàn bộ lịch sử"""
    try:
        save_history([])
        return JSONResponse({
            "success": True,
            "message": "Đã xóa toàn bộ lịch sử"
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái server"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "num_classes": len(class_names) if class_names else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
