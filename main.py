# main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import base64
import cv2
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import json

# FastAPI uygulamasını başlatın
app = FastAPI()

# Modelinizi yükleyin
model = None
try:
    model = load_model('model.h5')
    print("Model başarıyla yüklendi.")
    model.summary()
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")

# Frontend dosyalarının bulunduğu dizini belirtin
frontend_dir = os.path.join(os.path.dirname(__file__), 'public')

# Statik dosyaları sunmak için StaticFiles'ı bağlayın
# Önemli: WebSocket endpoint'inden önce tanımlanmalı ve spesifik bir path'e mount edilmeli
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Ana route için HTML sayfasını döndür - UTF-8 encoding ile
@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open(os.path.join(frontend_dir, "index.html"), "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, media_type="text/html; charset=utf-8")

# Bağlı WebSocket istemcilerini takip etmek için
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"İstemci bağlandı: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"İstemci bağlantısı kesildi: {websocket.client}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# Resim ön işleme fonksiyonu
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

    return processed

# WebSocket endpoint'i
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Bağlantı kurulduğunda modelin durumu hakkında bilgi gönder
        await manager.send_personal_message(json.dumps({
            "type": "model_status",
            "status": "ready" if model is not None else "error",
            "message": "Model hazır" if model is not None else "Model yüklenirken hata oluştu"
        }), websocket)

        while True:
            # İstemciden metin mesajı al
            data = await websocket.receive_text()
            print(f"İstemciden mesaj alındı: {data}")

            try:
                message = json.loads(data)

                # Farklı mesaj tiplerini ele al
                if message.get("type") == "digit_image":
                    if model is None:
                        await manager.send_personal_message(json.dumps({
                            "type": "error",
                            "message": "Model henüz yüklenmedi veya yüklenirken hata oluştu."
                        }), websocket)
                        continue

                    imageData = message.get("image")

                    if not isinstance(imageData, str):
                         await manager.send_personal_message(json.dumps({
                            "type": "error",
                            "message": "Geçersiz resim verisi formatı."
                        }), websocket)
                         continue

                    # Base64 string'den byte verisine dönüştür
                    if "," in imageData:
                        header, base64_string = imageData.split(",", 1)
                    else:
                        base64_string = imageData

                    try:
                        img_bytes = base64.b64decode(base64_string)
                    except Exception as e:
                         await manager.send_personal_message(json.dumps({
                            "type": "error",
                            "message": f"Base64 çözme hatası: {e}"
                        }), websocket)
                         continue

                    # Resmi ön işle
                    processed_image = preprocess_image_for_model(img_bytes)

                    if processed_image is None:
                         await manager.send_personal_message(json.dumps({
                            "type": "error",
                            "message": "Resim verisi işlenemedi."
                        }), websocket)
                         continue

                    # Tahmin yap
                    predictions = model.predict(processed_image)
                    predicted_digit = int(np.argmax(predictions, axis=1)[0])
                    # Görselleştirme verisi
                    visualization_data = {
                        "prediction_scores": predictions.tolist()
                    }

                    # Tahmin sonucunu ve görselleştirme verisini istemciye geri gönder
                    await manager.send_personal_message(json.dumps({
                        "type": "prediction_result",
                        "prediction": predicted_digit,
                        "cnn_viz_data": visualization_data
                    }), websocket)

                elif message.get("type") == "epoch_update":
                    epochs = message.get("epochs")
                    print(f"İstemci {websocket.client} eğitim epoch sayısını güncelledi: {epochs}")
                    await manager.send_personal_message(json.dumps({
                        "type": "epoch_updated_ack",
                        "epochs": epochs,
                        "message": "Epoch değeri alındı."
                    }), websocket)

                else:
                    print(f"Bilinmeyen mesaj tipi: {message.get('type')}")
                    await manager.send_personal_message(json.dumps({
                        "type": "error",
                        "message": f"Bilinmeyen mesaj tipi: {message.get('type')}"
                    }), websocket)

            except json.JSONDecodeError:
                 await manager.send_personal_message(json.dumps({
                    "type": "error",
                    "message": "Geçersiz JSON formatı."
                }), websocket)
            except Exception as e:
                print(f"Mesaj işlenirken hata: {e}")
                await manager.send_personal_message(json.dumps({
                    "type": "error",
                    "message": f"Mesaj işlenirken hata oluştu: {e}"
                }), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f"İstemci bağlantısı kesildi: {websocket.client}")
    except Exception as e:
        print(f"WebSocket genel hata: {e}")
        manager.disconnect(websocket)
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Bir hata oluştu: {e}"
            }))
        except:
            pass