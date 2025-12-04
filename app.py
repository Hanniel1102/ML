from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import os
import json
import cv2
from datetime import datetime
import base64

# Thiết lập TensorFlow trước khi import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Tắt warning

# Import TensorFlow
try:
    import tensorflow as tf
    # Với TensorFlow 2.15, sử dụng tf.keras
    from tensorflow import keras
    print(f"✅ TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"❌ Lỗi import TensorFlow: {e}")
    raise

# Import module tiền xử lý thông minh
from image_preprocessing import ImagePreprocessor, preprocess_and_check
# Import leaf detector
from leaf_detector import get_leaf_detector, analyze_leaf_image

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
    
    # Tìm và load model (ưu tiên model tối ưu mới)
    model_paths = [
        "best_tomato_model.keras",  # Model tối ưu mới nhất
        "Tomato_EfficientNetB0_Optimized.keras",  # Model tối ưu backup
        "Tomato_EfficientNetB0_Final.keras",  # Model cũ
        "test_model.keras",  # Model test
        "models/final_model.keras",
        "models/best_model.keras"
    ]
    
    # Define custom layers cho model tối ưu
    @tf.keras.utils.register_keras_serializable()
    class SpatialAttention(tf.keras.layers.Layer):
        """Spatial Attention mechanism"""
        def __init__(self, kernel_size=7, **kwargs):
            super().__init__(**kwargs)
            self.kernel_size = kernel_size
            
        def build(self, input_shape):
            self.conv = tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=self.kernel_size,
                padding='same',
                activation='sigmoid',
                use_bias=False
            )
            super().build(input_shape)
            
        def call(self, inputs):
            avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
            max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
            concat = tf.concat([avg_pool, max_pool], axis=-1)
            attention = self.conv(concat)
            return inputs * attention
        
        def get_config(self):
            config = super().get_config()
            config.update({"kernel_size": self.kernel_size})
            return config
    
    @tf.keras.utils.register_keras_serializable()
    class MixUp(tf.keras.layers.Layer):
        """MixUp augmentation layer"""
        def __init__(self, alpha=0.2, **kwargs):
            super().__init__(**kwargs)
            self.alpha = alpha
        
        def get_config(self):
            config = super().get_config()
            config.update({"alpha": self.alpha})
            return config
    
    custom_objects = {
        'SpatialAttention': SpatialAttention,
        'MixUp': MixUp
    }
    
    loaded_model = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"🔄 Đang load model từ: {model_path}")
                # Thử load với custom objects cho model tối ưu
                try:
                    loaded_model = keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
                except:
                    # Fallback: dùng tf.keras
                    loaded_model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
                
                model = loaded_model
                print(f"✅ Đã load model: {model_path}")
                print(f"📊 Model info: input_shape={model.input_shape}, output_shape={model.output_shape}")
                
                # Compile lại model
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                print(f"✅ Model đã được compile lại")
                break
            except Exception as e:
                import traceback
                print(f"⚠️ Không thể load model {model_path}:")
                print(f"   {str(e)[:300]}")
                traceback.print_exc()
                continue
    
    if model is None:
        print("❌ Không tìm thấy model nào có thể load được!")
        print("💡 Vui lòng kiểm tra lại file model hoặc train lại model với TensorFlow 2.15.0")
        raise RuntimeError("Model not found!")
    
    # Load class names
    if os.path.exists('models/class_names.json'):
        with open('models/class_names.json', 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print(f"✅ Đã load class names từ file")
    else:
        # Lấy từ dataset nếu tồn tại
        test_dirs = [
            "Tomato/Test",
            "../Hocmaynangcao/Tomato/Test",
            "H:/nam4ki1/Hocmaynangcao/Tomato/Test"
        ]
        
        class_names = None
        # Không cần load từ dataset, sẽ dùng fallback bên dưới
        
        if class_names is None:
            # Fallback: sử dụng keras.utils
            for test_dir in test_dirs:
                if os.path.exists(test_dir):
                    try:
                        temp_ds = keras.utils.image_dataset_from_directory(
                            test_dir,
                            image_size=(256, 256),
                            batch_size=32,
                            label_mode='categorical',
                            shuffle=False
                        )
                        class_names = temp_ds.class_names
                        print(f"✅ Đã load class names từ keras.utils: {test_dir}")
                        break
                    except Exception as e:
                        # Nếu không được, đọc trực tiếp từ thư mục
                        try:
                            class_names = sorted([d for d in os.listdir(test_dir) 
                                                if os.path.isdir(os.path.join(test_dir, d))])
                            print(f"✅ Đã load class names từ thư mục: {test_dir}")
                            break
                        except:
                            continue
        
        if class_names is None:
            # Fallback cuối cùng: class names mặc định
            class_names = [
                "Bacterial Spot",
                "Early Blight", 
                "Healthy",
                "Late Blight",
                "Septoria Leaf Spot",
                "Yellow Leaf Curl Virus"
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
        
        # === BƯỚC 0: KIỂM TRA NHANH - CÓ PHẢI ẢNH LÁ KHÔNG ===
        img_array_check = np.array(img)
        leaf_analysis = analyze_leaf_image(img_array_check)
        
        if not leaf_analysis['is_leaf']:
            return JSONResponse({
                "success": False,
                "error": "NOT_LEAF_IMAGE",
                "message": "⚠️ Ảnh không phải là ảnh lá cây",
                "confidence": round(leaf_analysis['confidence'] * 100, 1),
                "reason": leaf_analysis['reason'],
                "recommendation": "Vui lòng upload ảnh lá cà chua để phát hiện bệnh",
                "analysis": {
                    "green_score": round(leaf_analysis['details']['green_score'] * 100, 1),
                    "texture_score": round(leaf_analysis['details']['texture_score'] * 100, 1),
                    "shape_score": round(leaf_analysis['details']['shape_score'] * 100, 1),
                    "brightness_score": round(leaf_analysis['details']['brightness_score'] * 100, 1)
                }
            })
        
        # === BƯỚC 1: KIỂM TRA THÔNG MINH ===
        # Sử dụng thuật toán đa tầng: texture + shape + color
        result = preprocess_and_check(img, target_size=(IMG_SIZE, IMG_SIZE))
        
        # Nếu KHÔNG phải lá cây (chó, mèo, người, đồ vật)
        if not result['is_leaf']:
            details = result['details']
            # details có thể là string (lý do từ chối) hoặc dict (phân tích chi tiết)
            if isinstance(details, str):
                # Trường hợp từ chối sớm với lý do string
                return JSONResponse({
                    "success": False,
                    "error": "NOT_LEAF_IMAGE",
                    "message": "Ảnh không phải là ảnh lá cây",
                    "reason": details,
                    "recommendation": "Vui lòng chọn ảnh lá cây thật"
                })
            else:
                # Trường hợp có phân tích chi tiết
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

        # --- Additional safeguard ---
        # Kiểm tra phụ để giảm false-positives, nhưng ưu tiên vein_score hơn
        details = result.get('details', {})
        
        # Lấy các chỉ số quan trọng
        vein_score = float(details.get('vein_score', details.get('texture_score', 0)))
        main_obj_ratio = float(details.get('main_object_ratio', 0))
        green_ratio = float(details.get('green_ratio', 0))
        leaf_shape_score = float(details.get('leaf_shape_score', 0))
        
        # Configurable thresholds via env vars
        MIN_VEIN_SCORE = float(os.environ.get('MIN_VEIN_SCORE', '0.20'))
        MIN_GREEN_RATIO = float(os.environ.get('MIN_GREEN_RATIO', '0.01'))
        
        # CHIẾN LƯỢC MỚI: Chặn chỉ khi CẢ HAI điều kiện sau đều THẤT BẠI:
        # 1. Không có gân lá rõ (vein_score < 0.20)
        # 2. Không có màu xanh hoặc vegetation (green_ratio < 1%)
        # => Điều này tránh chặn lá thật có gân rõ hoặc có màu xanh
        
        has_vein_structure = vein_score >= MIN_VEIN_SCORE
        has_vegetation = green_ratio >= MIN_GREEN_RATIO
        has_reasonable_shape = leaf_shape_score >= 0.15
        
        # Chỉ từ chối nếu KHÔNG có gì giống lá cả
        is_likely_not_leaf = (not has_vein_structure and 
                              not has_vegetation and 
                              not has_reasonable_shape)
        
        # Allow override
        FORCE_PREDICT = os.environ.get('FORCE_PREDICT_ON_WEAK_LEAF', '0') == '1'
        
        if not FORCE_PREDICT and is_likely_not_leaf:
            # Return structured rejection with analysis
            return JSONResponse({
                "success": False,
                "error": "LOW_LEAF_CONFIDENCE",
                "message": "Ảnh có vẻ không phải lá cây (không có gân lá, không có màu xanh, không có hình dạng lá)",
                "recommendation": "Vui lòng chụp lại ảnh lá rõ ràng hơn",
                "analysis": {
                    "vein_score": round(vein_score, 3),
                    "green_ratio": round(green_ratio * 100, 2),
                    "leaf_shape_score": round(leaf_shape_score, 3),
                    "main_object_ratio": round(main_obj_ratio, 4),
                    "has_vein_structure": has_vein_structure,
                    "has_vegetation": has_vegetation,
                    "has_reasonable_shape": has_reasonable_shape
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
        import traceback
        traceback.print_exc()  # In ra console để debug
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
