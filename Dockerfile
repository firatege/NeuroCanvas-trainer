# neurocanvas-trainer/Dockerfile

FROM python:3.10-slim-buster

WORKDIR /app

# Sistem bağımlılıkları (TensorFlow için faydalı olabilir)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir opencv-python-headless

COPY main.py .
COPY model.h5 .

EXPOSE 8000
ENV TF_ENABLE_ONEDNN_OPTS=0
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]