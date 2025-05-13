# Kullanılacak temel Python imajını belirtin
FROM python:3.12-slim

# Gerekli sistem paketlerini kurmak için ortam değişkenlerini ayarla
ENV DEBIAN_FRONTEND=noninteractive

# Sistem bağımlılıklarını ve pip'i güncelleyin
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libxrender1 \
    libxi6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini oluştur ve ayarla
WORKDIR /app

# Bağımlılık dosyasını kopyala ve yükle
COPY requirements.txt .

# Python bağımlılıklarını kur
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# entrypoint.sh dosyasını kopyala ve çalıştırılabilir yap
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Tüm proje dosyalarını kopyala
COPY . .

# (İsteğe bağlı) Belirtilecek port – belge amaçlı
EXPOSE 8080

# Başlatma komutu
ENTRYPOINT ["./entrypoint.sh"]
