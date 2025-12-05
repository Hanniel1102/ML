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
        # Kiểm tra bổ sung để chặn động vật và đồ vật
        details = result.get('details', {})
        
        # Lấy các chỉ số quan trọng
        vein_score = float(details.get('vein_score', details.get('texture_score', 0)))
        main_obj_ratio = float(details.get('main_object_ratio', 0))
        green_ratio = float(details.get('green_ratio', 0))
        leaf_shape_score = float(details.get('leaf_shape_score', 0))
        
        # Configurable thresholds - TĂNG THRESHOLD để chặn động vật CHẮC CHẮN
        MIN_VEIN_SCORE = float(os.environ.get('MIN_VEIN_SCORE', '0.20'))
        MIN_GREEN_RATIO = float(os.environ.get('MIN_GREEN_RATIO', '0.15'))  # Tăng lên 15%
        MIN_LEAF_SHAPE = float(os.environ.get('MIN_LEAF_SHAPE', '0.15'))
        
        # CHIẾN LƯỢC CHẶT CHẼ NHẤT:
        # Phải có CẢ 3 điều kiện HOẶC có green_ratio rất cao (>30%):
        # 1. Có gân lá rõ (vein_score >= 0.20)
        # 2. Có màu xanh thực vật (green_ratio >= 15%)
        # 3. Có hình dạng lá (leaf_shape_score >= 0.15)
        
        has_vein_structure = vein_score >= MIN_VEIN_SCORE
        has_vegetation = green_ratio >= MIN_GREEN_RATIO
        has_reasonable_shape = leaf_shape_score >= MIN_LEAF_SHAPE
        has_high_green = green_ratio >= 0.30  # Lá thật thường có >30% màu xanh
        
        # Đếm số điều kiện thỏa mãn
        leaf_conditions_met = sum([has_vein_structure, has_vegetation, has_reasonable_shape])
        
        # Từ chối nếu:
        # - Có ít hơn 2 điều kiện HOẶC
        # - Không có màu xanh cao (động vật, người, đồ vật)
        is_likely_not_leaf = (leaf_conditions_met < 2) or (not has_vegetation and not has_high_green)
        
        # Allow override
        FORCE_PREDICT = os.environ.get('FORCE_PREDICT_ON_WEAK_LEAF', '0') == '1'
        
        if not FORCE_PREDICT and is_likely_not_leaf:
            # Return structured rejection with analysis
            rejection_reasons = []
            if not has_vein_structure:
                rejection_reasons.append("không phát hiện gân lá")
            if not has_vegetation:
                rejection_reasons.append("thiếu màu xanh thực vật (<15%)")
            if not has_reasonable_shape:
                rejection_reasons.append("không có hình dạng lá")
            
            # Thông báo cụ thể cho động vật
            if green_ratio < 0.10:
                message = "🚫 Đây không phải ảnh lá cây! Vui lòng chỉ upload ảnh lá cà chua."
            else:
                message = f"⚠️ Ảnh không đạt tiêu chuẩn lá cây ({', '.join(rejection_reasons)})"
            
            return JSONResponse({
                "success": False,
                "error": "LOW_LEAF_CONFIDENCE",
                "message": message,
                "recommendation": "Vui lòng chụp ảnh lá cà chua rõ nét, đủ ánh sáng, lấp đầy khung hình",
                "analysis": {
                    "vein_score": round(vein_score, 3),
                    "green_ratio": round(green_ratio * 100, 2),
                    "leaf_shape_score": round(leaf_shape_score, 3),
                    "main_object_ratio": round(main_obj_ratio, 4),
                    "has_vein_structure": has_vein_structure,
                    "has_vegetation": has_vegetation,
                    "has_reasonable_shape": has_reasonable_shape,
                    "has_high_green": has_high_green,
                    "conditions_met": leaf_conditions_met,
                    "minimum_required": 2
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
        
        # === BƯỚC 6: LẤY THÔNG TIN BỆNH VÀ KHUYẾN NGHỊ ===
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
            "image_type": image_analysis["type"],
            "vein_score": round(details.get('vein_score', 0), 2),
            "thumbnail": f"data:image/jpeg;base64,{img_base64}",
            "top_predictions": top_predictions,
            "image_analysis": image_analysis,
            "disease_info": disease_recommendation
        }
        add_to_history(history_entry)
        
        response_data = {
            "success": True,
            "predicted_class": class_names[predicted_class_idx],
            "confidence": confidence,
            "top_predictions": top_predictions,
            "image_analysis": image_analysis,
            "preprocessing": "enhanced" if details.get('is_dark_detected') else "standard",
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
