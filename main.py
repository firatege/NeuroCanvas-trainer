import base64
import io
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf # TensorFlow veya Keras için
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import logging
from typing import List, Optional

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()


model = None
# Model dosyasının yolu: main.py ile aynı dizinde olduğu için doğrudan dosya adı
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")

# Global olarak epoch değerini saklamak için
current_epoch_value: int = 1 # Varsayılan başlangıç değeri

@app.on_event("startup")
async def load_keras_model():
    """Uygulama başlatıldığında Keras (H5) modelini yükle."""
    global model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model dosyası bulunamadı: {MODEL_PATH}")
        raise RuntimeError(f"Model dosyası bulunamadı: {MODEL_PATH}")
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Keras modeli '{MODEL_PATH}' başarıyla yüklendi.")
        
        # Model mimarisini kontrol et
        logger.info("=== MODEL MİMARİSİ ===")
        model.summary(print_fn=logger.info)
        
        # Model ağırlıklarını kontrol et
        logger.info("=== MODEL AĞIRLIKLARI ===")
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'get_weights') and layer.get_weights():
                weights = layer.get_weights()
                logger.info(f"Katman {i} ({layer.name}): {len(weights)} ağırlık dizisi")
                for j, weight in enumerate(weights):
                    logger.info(f"  Ağırlık {j}: shape={weight.shape}, mean={np.mean(weight):.6f}, std={np.std(weight):.6f}")
        
        # Test verisi ile model performansını kontrol et
        logger.info("=== MODEL TEST ===")
        test_input = np.random.random((1, 28, 28, 1)).astype('float32')
        test_predictions = model.predict(test_input, verbose=0)
        test_probabilities = tf.nn.softmax(test_predictions, axis=1)
        test_probabilities = np.array(test_probabilities)[0]
        logger.info(f"Rastgele test girdisi için tahmin: {test_probabilities}")
        
        # Model eğitim geçmişini kontrol et (eğer varsa)
        if hasattr(model, 'history'):
            logger.info(f"Model eğitim geçmişi: {model.history}")
            
    except Exception as e:
        logger.error(f"Keras model yüklenirken hata oluştu: {e}")
        raise RuntimeError(f"Keras model yüklenemedi: {e}")

class ImageData(BaseModel):
    image: str # Base64 encoded image string (e.g., data:image/png;base64,...)

class EpochUpdate(BaseModel):
    epoch: int # Yeni epoch değeri

def preprocess_image_for_model(img_bytes: bytes):
    """
    Process image bytes for MNIST digit recognition model using OpenCV.
    """
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

    if img is None:
        print("⚠️ cv2.imdecode ile resim yüklenemedi.")
        return None

    if len(img.shape) == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        alpha = img[:, :, 3]
        gray[alpha == 0] = 255
        processed = gray
    elif len(img.shape) == 3:
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        processed = img

    image_size = (28, 28)
    processed = cv2.resize(processed, image_size)

    # Renkleri tersine çevir
    processed = 255 - processed

    processed = processed.astype('float32') / 255.0

    processed = np.expand_dims(processed, axis=0)
    processed = np.expand_dims(processed, axis=-1)

    # Debugging: işlenmiş görüntü hakkında bilgi ver
    logger.info(f"Processed image shape: {processed.shape}")
    logger.info(f"Processed image min/max: {processed.min():.4f}/{processed.max():.4f}")
    logger.info(f"Processed image mean: {processed.mean():.4f}")

    return processed

# Birinci POST Endpoint'i: Tahmin
@app.post("/predict")
async def predict_digit(data: ImageData):
    """Base64 kodlu görüntüyü alır, işler ve rakam tahmini döndürür."""
    if model is None:
        logger.error("Model henüz yüklenmedi veya yüklenirken hata oluştu.")
        raise HTTPException(status_code=503, detail="Model henüz hazır değil.")

    try:
        # Base64 kodlu görüntüyü çöz
        if not data.image.startswith("data:image/"):
            raise HTTPException(status_code=400, detail="Geçersiz görüntü formatı. Base64 kodlu bir resim bekleniyor.")
        header, encoded = data.image.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img_tensor = preprocess_image_for_model(img_bytes)
        if img_tensor is None:
            raise HTTPException(status_code=400, detail="Görüntü işlenemedi. Lütfen geçerli bir görüntü gönderin.")
        
        predictions = model.predict(img_tensor)    # zaten softmaxlı sonuç
        probabilities = predictions[0]             # doğrudan kullanın
        predicted_class = int(np.argmax(probabilities))  # ensure Python int
        
        probabilities_list = probabilities.tolist()

        logger.info(f"Tahmin: {predicted_class}, Olasılıklar: {probabilities_list}")

        return {
            "prediction": predicted_class,
            "probabilities": probabilities_list
        }

    except Exception as e:
        logger.error(f"Görüntü işleme veya tahmin sırasında hata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Görüntü işleme hatası: {e}")

# İkinci POST Endpoint'i: Epoch Değerini Ayarla / Eğitim Başlat
@app.post("/set_epoch")
async def set_epoch(data: EpochUpdate):
    """Frontend'den gelen epoch değerini günceller."""
    global current_epoch_value
    if data.epoch < 1:
        raise HTTPException(status_code=400, detail="Epoch değeri en az 1 olmalıdır.")
    
    current_epoch_value = data.epoch
    logger.info(f"Epoch değeri güncellendi: {current_epoch_value}")

    # Burada gerçek bir eğitim başlatma veya model güncelleme mantığı eklenebilir.
    # Örneğin:
    # if model is not None:
    #    start_model_training_async(current_epoch_value, model)
    # else:
    #    logger.warn("Model yüklenmediği için eğitim başlatılamıyor.")

    return {"status": "success", "message": f"Epoch değeri {current_epoch_value} olarak ayarlandı."}

# Epoch değerini almak için bir GET endpoint'i de ekleyebiliriz (isteğe bağlı)
@app.get("/get_epoch")
async def get_epoch():
    """Mevcut epoch değerini döndürür."""
    return {"current_epoch": current_epoch_value}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) # Rust'ın bağlanacağı adres