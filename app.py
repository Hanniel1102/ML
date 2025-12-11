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
# Import image analysis (MODULE CHÍNH cho validation và analysis)
from image_analysis import analyze_image
from efficientnet_preprocessor import preprocess_for_efficientnet

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

# Database thông tin bệnh và giải pháp chăm sóc
DISEASE_INFO = {
    "Bacterial Spot": {
        "name_vi": "Đốm Lá Vi Khuẩn",
        "severity": "Cao",
        "description": "Bệnh do vi khuẩn Xanthomonas gây ra, tạo các đốm đen nhỏ trên lá và quả, ảnh hưởng nghiêm trọng đến năng suất.",
        "symptoms": [
            "Đốm nhỏ màu đen hoặc nâu trên lá, có viền vàng",
            "Lá bị vàng và rụng sớm",
            "Đốm trên quả làm giảm chất lượng",
            "Lan rộng nhanh trong điều kiện ẩm ướt"
        ],
        "causes": [
            "Độ ẩm cao (>80%)",
            "Nhiệt độ 25-30°C",
            "Mưa nhiều, tưới nước trực tiếp lên lá",
            "Vi khuẩn lây lan qua vết thương, giọt nước"
        ],
        "treatment": {
            "immediate": [
                "🔴 CẤP BÁN: Loại bỏ lá bệnh và tiêu hủy ngay (đốt hoặc chôn sâu)",
                "💧 Tránh tưới nước lên lá, chỉ tưới gốc",
                "🌿 Phun thuốc kháng sinh đồng (copper hydroxide) hoặc streptomycin",
                "🔬 Cách ly cây bệnh khỏi cây khỏe mạnh"
            ],
            "shortterm": [
                "Phun thuốc 7-10 ngày/lần trong 3-4 tuần",
                "Sử dụng phân bón giàu canxi để tăng cường sức đề kháng",
                "Cải thiện thoát nước, tránh úng nước",
                "Tỉa bớt lá để tăng thông gió"
            ],
            "longterm": [
                "Luân canh cây trồng (nghỉ 2-3 năm)",
                "Trồng giống kháng bệnh (varieties có gen kháng)",
                "Sử dụng màng phủ để giảm bắn nước lên lá",
                "Khử trùng dụng cụ làm vườn thường xuyên",
                "Xây dựng hệ thống tưới nhỏ giọt"
            ]
        },
        "prevention": [
            "Chọn giống kháng bệnh",
            "Tưới nước buổi sáng để lá khô nhanh",
            "Khoảng cách trồng rộng (60-90cm)",
            "Khử trùng hạt giống trước khi gieo"
        ],
        "products": [
            "Kocide 3000 (copper hydroxide)",
            "Streptomycin sulfate",
            "Mancozeb + copper",
            "Actigard (kích hoạt miễn dịch)"
        ]
    },
    "Early Blight": {
        "name_vi": "Bệnh Héo Sớm",
        "severity": "Trung bình - Cao",
        "description": "Bệnh do nấm Alternaria solani, gây đốm đồng tâm trên lá, thân và quả. Phổ biến nhất ở cà chua.",
        "symptoms": [
            "Đốm tròn có vòng đồng tâm (mắt bò) trên lá già",
            "Lá vàng và rụng từ dưới lên",
            "Vết thối đen trên thân gần gốc",
            "Đốm đen lõm trên cuống quả"
        ],
        "causes": [
            "Nhiệt độ ấm (24-29°C)",
            "Độ ẩm cao, mưa nhiều",
            "Dinh dưỡng thiếu hụt (đặc biệt N, K)",
            "Cây già, stress do hạn"
        ],
        "treatment": {
            "immediate": [
                "✂️ Cắt bỏ lá bệnh ngay lập tức",
                "🍄 Phun thuốc diệt nấm chlorothalonil hoặc mancozeb",
                "🌱 Bón phân NPK cân đối, tăng canxi",
                "💦 Giảm tưới nước, tránh ẩm ướt"
            ],
            "shortterm": [
                "Phun thuốc 7 ngày/lần trong 3-4 tuần",
                "Luân phiên các loại thuốc diệt nấm",
                "Bón phân hữu cơ tăng cường sức khỏe",
                "Dọn sạch lá rụng dưới gốc"
            ],
            "longterm": [
                "Cải tạo đất, tăng chất hữu cơ",
                "Trồng giống kháng bệnh (Iron Lady, Mountain Magic)",
                "Phủ mulch để tránh bắn đất lên lá",
                "Tưới nhỏ giọt thay vì tưới phun",
                "Luân canh 3-4 năm"
            ]
        },
        "prevention": [
            "Trồng xa họ cà (khoai tây, ớt, cà tím)",
            "Giữ khoảng cách 60-75cm giữa các cây",
            "Bón vôi trước khi trồng (pH 6.0-6.8)",
            "Phun phòng bệnh 2 tuần/lần"
        ],
        "products": [
            "Daconil (chlorothalonil)",
            "Dithane M-45 (mancozeb)",
            "Azoxystrobin",
            "Copper fungicide"
        ]
    },
    "Healthy": {
        "name_vi": "Lá Khỏe Mạnh",
        "severity": "Không có",
        "description": "Cây cà chua đang phát triển tốt, không có dấu hiệu bệnh. Tiếp tục duy trì chăm sóc.",
        "symptoms": [
            "Lá xanh đồng đều, không đốm",
            "Tăng trưởng mạnh mẽ",
            "Không có vết thối hoặc héo",
            "Quả phát triển bình thường"
        ],
        "causes": [],
        "treatment": {
            "immediate": [
                "✅ Duy trì chế độ chăm sóc hiện tại",
                "🌿 Kiểm tra định kỳ để phát hiện sớm bệnh",
                "💧 Tưới nước đều đặn, tránh khô hạn",
                "🌞 Đảm bảo đủ ánh sáng (6-8 giờ/ngày)"
            ],
            "shortterm": [
                "Bón phân NPK cân đối 10-14 ngày/lần",
                "Theo dõi sâu bệnh thường xuyên",
                "Tỉa cành phụ (suckers) nếu cần",
                "Đóng cọc hỗ trợ cây khi cao >60cm"
            ],
            "longterm": [
                "Xây dựng lịch trình bón phân khoa học",
                "Luân canh để duy trì độ phì đất",
                "Sử dụng phân compost định kỳ",
                "Ghi chép nhật ký chăm sóc",
                "Phun phòng bệnh sinh học"
            ]
        },
        "prevention": [
            "Tưới sáng sớm, tránh tối muộn",
            "Làm cỏ thường xuyên",
            "Bón vôi dolomite bổ sung Ca, Mg",
            "Sử dụng compost chất lượng cao"
        ],
        "products": [
            "Phân NPK 16-16-16 (tăng trưởng)",
            "Phân NPK 15-5-30 (ra hoa, quả)",
            "Phân compost hữu cơ",
            "Trichoderma (nấm đối kháng)"
        ]
    },
    "Late Blight": {
        "name_vi": "Bệnh Mốc Sương",
        "severity": "Rất Cao",
        "description": "Bệnh nguy hiểm nhất, do nấm Phytophthora infestans. Có thể tiêu diệt toàn bộ vườn trong 1-2 tuần.",
        "symptoms": [
            "Đốm lớn màu nâu xám trên lá",
            "Vệt trắng mốc ở mặt dưới lá (khi ẩm)",
            "Thân đen, chết nhanh",
            "Quả thối nhanh, mùi hôi"
        ],
        "causes": [
            "Thời tiết mát (15-25°C)",
            "Độ ẩm rất cao (>90%)",
            "Mưa liên tục, sương mù",
            "Gió lan truyền bào tử"
        ],
        "treatment": {
            "immediate": [
                "🚨 KHẨN CẤP: Nhổ bỏ cây bệnh nặng ngay lập tức!",
                "🔥 Đốt hoặc chôn sâu (không compost)",
                "💊 Phun thuốc diệt nấm Metalaxyl + Mancozeb NGAY",
                "🚧 Cách ly khu vực bệnh, không đi lại"
            ],
            "shortterm": [
                "Phun thuốc 5-7 ngày/lần, không bỏ lần nào",
                "Luân phiên 2-3 loại thuốc để tránh kháng",
                "Tăng thông gió, giảm ẩm tối đa",
                "Ngừng tưới nước 3-5 ngày nếu có thể",
                "Giám sát 24/7, phát hiện sớm"
            ],
            "longterm": [
                "Trồng giống kháng bệnh (Matt's Wild Cherry, Defiant PHR)",
                "Xây nhà lưới/nhà kính để kiểm soát ẩm",
                "Hệ thống tưới nhỏ giọt tự động",
                "Không trồng cà chua liên tục >2 mùa",
                "Khử trùng toàn bộ vườn sau thu hoạch"
            ]
        },
        "prevention": [
            "Trồng giống kháng bệnh (ưu tiên số 1)",
            "Che mưa bằng mái che hoặc mulch plastic",
            "Phun phòng trước mưa 1-2 ngày",
            "Khoảng cách >90cm, không trồng dày"
        ],
        "products": [
            "Ridomil Gold (Metalaxyl + Mancozeb) - ƯU TIÊN",
            "Revus (Mandipropamid)",
            "Curzate (Cymoxanil)",
            "Ranman (Cyazofamid)"
        ]
    },
    "Septoria Leaf Spot": {
        "name_vi": "Đốm Lá Septoria",
        "severity": "Trung bình",
        "description": "Bệnh do nấm Septoria lycopersici, gây đốm nhỏ có chấm đen giữa, thường ở lá già.",
        "symptoms": [
            "Đốm tròn nhỏ (2-3mm) màu xám/nâu",
            "Chấm đen nhỏ ở giữa đốm (bào tử nấm)",
            "Viền vàng quanh đốm",
            "Lá vàng và rụng từ dưới lên"
        ],
        "causes": [
            "Nhiệt độ ấm (20-25°C)",
            "Độ ẩm cao, mưa phùn",
            "Lá bị nước bắn từ đất",
            "Cây trồng quá dày"
        ],
        "treatment": {
            "immediate": [
                "✂️ Cắt bỏ lá bệnh (đặc biệt lá dưới gốc)",
                "🍄 Phun thuốc diệt nấm chlorothalonil",
                "🌾 Phủ rơm rạ dưới gốc, tránh bắn đất",
                "💨 Tỉa lá tăng thông gió"
            ],
            "shortterm": [
                "Phun thuốc 10-14 ngày/lần",
                "Tưới sáng sớm, tránh tối",
                "Bón phân cân đối NPK + micronutrients",
                "Dọn sạch lá rụng hàng tuần"
            ],
            "longterm": [
                "Phủ mulch plastic đen để tránh bắn đất",
                "Trồng giống kháng bệnh (Legend, Plum Regal)",
                "Luân canh 2-3 năm",
                "Giàn leo cao, tránh lá chạm đất",
                "Hệ thống tưới nhỏ giọt"
            ]
        },
        "prevention": [
            "Khoảng cách trồng 60-75cm",
            "Tỉa lá dưới gốc cao 30cm",
            "Phun phòng trước mùa mưa",
            "Không tưới phun, chỉ tưới gốc"
        ],
        "products": [
            "Bravo (chlorothalonil)",
            "Mancozeb",
            "Copper fungicide",
            "Azoxystrobin"
        ]
    },
    "Yellow Leaf Curl Virus": {
        "name_vi": "Virus Cuộn Lá Vàng",
        "severity": "Rất Cao",
        "description": "Bệnh virus do ruồi trắng (whitefly) truyền. KHÔNG CÓ THUỐC CHỮA, chỉ kiểm soát ruồi trắng.",
        "symptoms": [
            "Lá cuộn lại, vàng úa",
            "Cây còi cọc, không lớn",
            "Hoa rụng, không đậu quả",
            "Ruồi trắng bay rất nhiều khi lay cây"
        ],
        "causes": [
            "Ruồi trắng (Bemisia tabaci) truyền virus",
            "Thời tiết nóng khô (>30°C)",
            "Cây trồng liên tục, không luân canh",
            "Không có lưới chắn côn trùng"
        ],
        "treatment": {
            "immediate": [
                "🔴 KHÔNG CÓ THUỐC CHỮA - Nhổ bỏ cây bệnh NGAY!",
                "🪰 Diệt ruồi trắng khẩn cấp: Imidacloprid hoặc Thiamethoxam",
                "🟨 Treo bẫy dính màu vàng (yellow sticky traps)",
                "🧼 Xịt xà phòng gốc dầu neem để đuổi ruồi"
            ],
            "shortterm": [
                "Phun thuốc diệt ruồi trắng 5 ngày/lần trong 3 tuần",
                "Treo bẫy vàng mỗi 5-10m",
                "Xịt nước mạnh dưới lá để đánh rơi ruồi",
                "Che lưới chắn côn trùng (mesh 50)",
                "Loại bỏ cỏ dại (ổ chứa ruồi trắng)"
            ],
            "longterm": [
                "Trồng giống kháng virus (Tygress, SV7203)",
                "Nhà lưới/nhà kính với lưới chắn",
                "Không trồng cà chua gần dưa, bí",
                "Luân canh 6 tháng, nghỉ đất",
                "Sử dụng phản quang bạc (silver mulch) đuổi ruồi",
                "Trồng cây bẫy (hướng dương) xung quanh"
            ]
        },
        "prevention": [
            "Lưới chắn 40-50 mesh từ khi gieo hạt",
            "Giám sát ruồi trắng hàng tuần",
            "Phun dầu neem phòng bệnh",
            "Trồng giống kháng virus",
            "Tránh mua cây giống có ruồi trắng"
        ],
        "products": [
            "Imidacloprid (Confidor, Admire)",
            "Thiamethoxam (Actara)",
            "Spiromesifen (Oberon) - diệt trứng/nhộng",
            "Dầu Neem hữu cơ",
            "Bẫy dính vàng (Yellow sticky traps)"
        ]
    }
}

def get_disease_recommendation(disease_name: str, confidence: float) -> dict:
    """
    Lấy thông tin khuyến nghị và giải pháp cho bệnh
    
    Args:
        disease_name: Tên bệnh (tiếng Anh)
        confidence: Độ tin cậy (%)
        
    Returns:
        Dictionary chứa đầy đủ thông tin bệnh và giải pháp
    """
    if disease_name not in DISEASE_INFO:
        return {
            "name_vi": disease_name,
            "severity": "Không xác định",
            "description": "Không có thông tin chi tiết",
            "recommendations": []
        }
    
    info = DISEASE_INFO[disease_name]
    
    # Tạo khuyến nghị dựa trên độ tin cậy
    recommendations = []
    
    if confidence >= 90:
        certainty = "RẤT CAO"
        action_level = "Áp dụng ngay tất cả biện pháp điều trị"
    elif confidence >= 75:
        certainty = "CAO"
        action_level = "Áp dụng biện pháp điều trị khuyến nghị"
    elif confidence >= 60:
        certainty = "TRUNG BÌNH"
        action_level = "Theo dõi thêm và áp dụng biện pháp phòng ngừa"
    else:
        certainty = "THẤP"
        action_level = "Cần chụp ảnh rõ hơn để xác định chính xác"
    
    return {
        "name_vi": info["name_vi"],
        "severity": info["severity"],
        "certainty": certainty,
        "confidence": confidence,
        "description": info["description"],
        "symptoms": info["symptoms"],
        "causes": info["causes"],
        "treatment": info["treatment"],
        "prevention": info["prevention"],
        "products": info["products"],
        "action_level": action_level
    }

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
    
    # Tìm và load model (chỉ file .h5)
    model_paths = [
        "best_tomato_model.h5",  # Model .h5 mới nhất
        "Tomato_EfficientNetB0_Final.h5",  # Model .h5 backup
        "models/best_model.h5",
        "model.h5"
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
                
                # Load model .h5 với custom objects
                try:
                    loaded_model = keras.models.load_model(
                        model_path, 
                        custom_objects=custom_objects,
                        compile=False
                    )
                    print(f"✅ Đã load model .h5 thành công")
                except Exception as load_error:
                    # Fallback: dùng tf.keras
                    print(f"⚠️ Thử fallback với tf.keras...")
                    loaded_model = tf.keras.models.load_model(
                        model_path, 
                        custom_objects=custom_objects,
                        compile=False
                    )
                
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
    
    # Load class names - Ưu tiên từ file JSON, sau đó từ Train dataset
    class_names = None
    
    # Cách 1: Load từ file JSON (đã được lưu khi training)
    if os.path.exists('models/class_names.json'):
        with open('models/class_names.json', 'r', encoding='utf-8') as f:
            class_names = json.load(f)
        print(f"✅ Đã load class names từ file JSON: {class_names}")
    
    # Cách 2: Load từ Train dataset directory (thứ tự alphabet)
    if class_names is None:
        train_dirs = [
            "Tomato/Train",
            "../Hocmaynangcao/Tomato/Train",
            "H:/nam4ki1/Hocmaynangcao/Tomato/Train"
        ]
        
        for train_dir in train_dirs:
            if os.path.exists(train_dir):
                # Lấy tên folder và sort theo alphabet (giống TensorFlow)
                class_folders = sorted([d for d in os.listdir(train_dir) 
                                       if os.path.isdir(os.path.join(train_dir, d))])
                if class_folders:
                    class_names = class_folders
                    print(f"✅ Đã load class names từ Train dataset: {class_names}")
                    print(f"📂 Train directory: {train_dir}")
                    break
    
    # Cách 3: Fallback - hardcode nếu không tìm thấy
    if class_names is None:
        class_names = [
            "Bacterial Spot",
            "Early Blight", 
            "Healthy",
            "Late Blight",
            "Septoria Leaf Spot",
            "Yellow Leaf Curl Virus"
        ]
        print(f"⚠️ Class names FALLBACK (hardcoded): {class_names}")
    
    IMG_SIZE = model.input_shape[1]
    print(f"📏 Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"📝 Số lượng classes: {len(class_names)}")
    print(f"✅ Server đã sẵn sàng!\n")

# API endpoint chính
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/favicon.ico")
async def favicon():
    """Trả về empty response để tránh lỗi 404"""
    from fastapi.responses import Response
    return Response(status_code=204)

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
        
        # === BƯỚC 1: TIỀN XỬ LÝ ẢNH GIỐNG TRAINING ===
        # Model train với preprocessing đơn giản: resize + rescale (1./255)
        # KHÔNG DÙNG CLAHE/SHARPEN vì training không dùng!
        print("\n[Simple Preprocessing] Resize + Rescale only (matching training)")
        
        # Lưu ảnh gốc để hiển thị
        buffered_original = io.BytesIO()
        img.save(buffered_original, format="JPEG", quality=95)
        original_base64 = base64.b64encode(buffered_original.getvalue()).decode()
        
        # Resize ảnh
        img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
        preprocessed_img = img_resized
        
        # Lưu ảnh đã resize
        buffered_resized = io.BytesIO()
        img_resized.save(buffered_resized, format="JPEG", quality=95)
        resized_base64 = base64.b64encode(buffered_resized.getvalue()).decode()
        
        # Convert PIL Image sang bytes để phân tích
        buffered_temp = io.BytesIO()
        preprocessed_img.save(buffered_temp, format="JPEG", quality=95)
        preprocessed_contents = buffered_temp.getvalue()
        
        # === BƯỚC 2: PHÂN TÍCH ẢNH ĐÃ XỬ LÝ - SỬ DỤNG image_analysis.py ===
        # Phân tích ảnh ĐÃ được làm sạch: shape, color, texture
        print("\n[Image Analysis] Analyzing preprocessed image...")
        analysis_result = analyze_image(preprocessed_contents)
        
        # Kiểm tra xem có phải ảnh lá không
        final_score = analysis_result['finalScore']
        is_leaf = analysis_result['isLeaf']
        
        if not is_leaf:
            # Lấy đầy đủ thông tin để debug
            shape_data = analysis_result['shape']
            color_data = analysis_result['color']
            texture_data = analysis_result['texture']
            
            # Trích xuất các metrics quan trọng
            detailed_metrics = {
                "overall_score": round(final_score['score'] * 100, 1),
                "confidence_level": final_score['confidence'],
                
                # Shape metrics
                "shape": {
                    "score": final_score['shapeScore'],
                    "aspectRatio": shape_data.get('aspectRatio', 'N/A'),
                    "mainObjectRatio": shape_data.get('mainObjectRatio', 'N/A'),
                    "greenDensity": shape_data.get('greenDensity', 'N/A'),
                    "roundness": shape_data.get('roundness', 'N/A'),
                    "elongation": shape_data.get('elongation', 'N/A')
                },
                
                # Color metrics
                "color": {
                    "score": final_score['colorScore'],
                    "greenRatio": color_data.get('greenRatio', 'N/A'),
                    "yellowRatio": color_data.get('yellowRatio', 'N/A'),
                    "brownRatio": color_data.get('brownRatio', 'N/A'),
                    "avgSaturation": color_data.get('avgSaturation', 'N/A'),
                    "avgHue": color_data.get('avgHue', 'N/A')
                },
                
                # Texture/Vein metrics
                "texture": {
                    "score": final_score['textureScore'],
                    "veinScore": texture_data.get('veinScore', 'N/A'),
                    "edgeDensity": texture_data.get('edgeDensity', 'N/A'),
                    "contrast": texture_data.get('contrast', 'N/A')
                },
                
                # Thông tin trọng số (nếu có dynamic weighting)
                "weights_used": final_score.get('weights_used', {
                    "shape": 0.35,
                    "color": 0.50,
                    "texture": 0.15
                }),
                "situation": final_score.get('situation', 'normal')
            }
            
            # Tạo thông báo chi tiết
            criteria_check = {
                "green_ratio_check": {
                    "value": color_data.get('greenRatio', '0'),
                    "threshold": "≥ 0.20 (20%)",
                    "passed": float(color_data.get('greenRatio', 0)) >= 0.20
                },
                "vein_score_check": {
                    "value": texture_data.get('veinScore', '0'),
                    "threshold": "≥ 0.05",
                    "passed": float(texture_data.get('veinScore', 0)) >= 0.05
                },
                "green_density_check": {
                    "value": shape_data.get('greenDensity', '0'),
                    "threshold": "≥ 0.15 (15%)",
                    "passed": float(shape_data.get('greenDensity', 0)) >= 0.15
                },
                "overall_score_check": {
                    "value": f"{final_score['score'] * 100:.1f}%",
                    "threshold": "≥ 60%",
                    "passed": final_score['score'] >= 0.60
                }
            }
            
            return JSONResponse({
                "success": False,
                "error": "NOT_LEAF_IMAGE",
                "message": f"⚠️ {final_score['recommendation']}",
                "reason": final_score['confidence'],
                "recommendation": "Vui lòng upload ảnh lá cà chua rõ ràng để phát hiện bệnh",
                
                # Thông tin chi tiết
                "detailed_analysis": detailed_metrics,
                "criteria_check": criteria_check,
                
                # Backward compatibility
                "confidence": round(final_score['score'] * 100, 1),
                "analysis": {
                    "score": round(final_score['score'] * 100, 1),
                    "shapeScore": final_score['shapeScore'],
                    "colorScore": final_score['colorScore'],
                    "textureScore": final_score['textureScore'],
                    "greenRatio": analysis_result['color']['greenRatio'],
                    "veinScore": analysis_result['texture']['veinScore']
                }
            })
        
        # === BƯỚC 3: CHUẨN BỊ ẢNH CHO MODEL ===
        # QUAN TRỌNG: Model được train với input [0, 255] (KHÔNG rescale trước khi vào model)
        # Model có data_augmentation layer bên trong, nó sẽ tự xử lý
        # Chỉ cần resize về đúng kích thước và giữ nguyên range [0, 255]
        enhanced_img = preprocessed_img
        img_array = np.array(preprocessed_img, dtype=np.float32)  # Giữ nguyên [0, 255]
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"[Model Input] ✅ Array shape: {img_array.shape}, range: [{img_array.min():.1f}, {img_array.max():.1f}] (RAW [0-255])")
        
        # === BƯỚC 4: DỰ ĐOÁN BỆNH ===
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_idx] * 100)
        
        # DEBUG: In ra prediction values
        print(f"\n[DEBUG Prediction] All class probabilities:")
        for i, class_name in enumerate(class_names):
            print(f"  {i}. {class_name}: {predictions[0][i]*100:.2f}%")
        print(f"[DEBUG Prediction] Predicted: {class_names[predicted_class_idx]} ({confidence:.2f}%)\n")
        
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
        
        # === BƯỚC 5: PHÂN TÍCH CHẤT LƯỢNG ẢNH (từ analysis_result) ===
        image_analysis_data = {
            "score": round(final_score['score'] * 100, 1),
            "confidence": final_score['confidence'],
            "shapeScore": final_score['shapeScore'],
            "colorScore": final_score['colorScore'],
            "textureScore": final_score['textureScore'],
            "greenRatio": analysis_result['color']['greenRatio'],
            "veinScore": analysis_result['texture']['veinScore'],
            "edgeDensity": analysis_result['texture']['edgeDensity'],
            "recommendation": final_score['recommendation'],
            # Thêm metrics từ analysis_result
            "brightness": analysis_result.get('metrics', {}).get('brightness', 128),
            "contrast": analysis_result.get('metrics', {}).get('contrast', 50),
            "sharpness": analysis_result.get('metrics', {}).get('sharpness', 50),
            "noise": analysis_result.get('metrics', {}).get('noise', 1000)
        }
        
        # === BƯỚC 6: LƯU VÀO LỊCH SỬ ===
        # Convert ảnh sang base64 để lưu thumbnail
        img_thumbnail = img.copy()
        img_thumbnail.thumbnail((150, 150))
        buffered = io.BytesIO()
        img_thumbnail.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # === BƯỚC 7: LẤY THÔNG TIN BỆNH VÀ KHUYẾN NGHỊ ===
        disease_recommendation = get_disease_recommendation(
            class_names[predicted_class_idx], 
            confidence
        )
        
        history_entry = {
            "id": len(load_history()) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": file.filename,
            "predicted_class": class_names[predicted_class_idx],
            "confidence": round(confidence, 2),
            "vein_score": analysis_result['texture']['veinScore'],
            "thumbnail": f"data:image/jpeg;base64,{img_base64}",
            "top_predictions": top_predictions,
            "image_analysis": image_analysis_data,
            "disease_info": disease_recommendation,
            "preprocessing_summary": {"method": "simple", "steps": ["resize", "rescale"]}
        }
        add_to_history(history_entry)
        
        # Loại bỏ numpy arrays và convert numpy types sang Python types
        def clean_value(val):
            """Convert numpy types to Python native types"""
            if isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            elif isinstance(val, (np.floating, np.float64, np.float32)):
                return float(val)
            elif isinstance(val, (np.bool_, bool)):
                return bool(val)
            elif isinstance(val, dict):
                return {k: clean_value(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [clean_value(v) for v in val]
            else:
                return val
        
        # Tạo visualization preprocessing steps (6 bước đầy đủ)
        # Gọi preprocess_for_efficientnet để lấy steps visualization
        print("\n[Preprocessing Visualization] Generating 6-step visualization...")
        from efficientnet_preprocessor import preprocess_for_efficientnet
        preprocessing_result = preprocess_for_efficientnet(img, target_size=(IMG_SIZE, IMG_SIZE))
        
        preprocessing_steps_clean = []
        if preprocessing_result and 'steps' in preprocessing_result:
            for step in preprocessing_result['steps']:
                preprocessing_steps_clean.append({
                    'name': step.get('name', 'Unknown'),
                    'description': step.get('description', ''),
                    'image_base64': step.get('image_base64', None),
                    'metrics': clean_value(step.get('metrics', {}))
                })
        
        # Nếu không có steps, dùng fallback 2 steps
        if not preprocessing_steps_clean:
            preprocessing_steps_clean = [
                {
                    'name': 'resize',
                    'description': f'Resized to {IMG_SIZE}x{IMG_SIZE}',
                    'image_base64': f"data:image/jpeg;base64,{resized_base64}",
                    'metrics': {}
                },
                {
                    'name': 'normalize',
                    'description': 'Rescaled to [0,1] range',
                    'image_base64': f"data:image/jpeg;base64,{resized_base64}",
                    'metrics': {}
                }
            ]
        
        # Tạo summary và clean numpy types
        preprocessing_summary = preprocessing_result.get('summary', {}) if preprocessing_result else {}
        if not preprocessing_summary:
            preprocessing_summary = {
                "total_steps": len(preprocessing_steps_clean),
                "actions_taken": [step['name'] for step in preprocessing_steps_clean],
                "final_quality": {
                    "brightness": "Tốt",
                    "contrast": "Tốt",
                    "noise": "Sạch",
                    "sharpness": "Sắc nét"
                }
            }
        else:
            # Clean numpy types trong summary
            preprocessing_summary = clean_value(preprocessing_summary)
        
        response_data = {
            "success": True,
            "predicted_class": class_names[predicted_class_idx],
            "confidence": confidence,
            "top_predictions": top_predictions,
            "image_analysis": image_analysis_data,
            "preprocessing": {
                "steps": preprocessing_steps_clean,
                "summary": preprocessing_summary
            },
            "processedImages": {
                "original": f"data:image/jpeg;base64,{original_base64}",
                "resized": f"data:image/jpeg;base64,{resized_base64}"
            },
            "history_id": history_entry["id"],
            "disease_info": disease_recommendation  # Thông tin chi tiết về bệnh
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

@app.get("/history/{item_id}")
async def get_history_item(item_id: int):
    """Lấy chi tiết một item trong lịch sử"""
    try:
        history = load_history()
        item = next((h for h in history if h.get('id') == item_id), None)
        
        if item is None:
            return JSONResponse({
                "success": False,
                "error": "Không tìm thấy item"
            }, status_code=404)
        
        return JSONResponse({
            "success": True,
            "item": item
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

@app.post("/analyze")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    Phân tích ảnh chi tiết: shape, color, texture
    Preprocessing TRƯỚC khi phân tích để có kết quả chính xác hơn
    """
    try:
        # Đọc file
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Chuyển sang RGB nếu cần
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Phân tích ảnh GỐC (không preprocess)
        print("\n[Analyze Endpoint] Analyzing original image...")
        result = analyze_image(contents)
        
        return JSONResponse({
            "success": True,
            "analysis": result
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "message": "Lỗi khi phân tích ảnh"
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
