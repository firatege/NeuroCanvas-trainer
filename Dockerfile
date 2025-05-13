# Kullanılacak temel Python imajını belirtin
FROM python:3.12-slim

# Çalışma dizini
WORKDIR /app

# Bağımlılık dosyasını kopyalayın
COPY requirements.txt ./

# Bağımlılıkları kurun
RUN pip install --no-cache-dir -r requirements.txt

# OpenCV gibi kütüphanelerin ihtiyaç duyduğu ek sistem bağımlılıklarını kurun
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libxrender1 libxi6 && rm -rf /var/lib/apt/lists/*



# entrypoint.sh dosyasını kopyalayın ve çalıştırılabilir yapın
COPY entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

# Diğer tüm proje dosyalarını kopyalayın
COPY . .

# Belirtilecek port (belge amaçlı)
EXPOSE 8080

# Uygulama başlangıç komutu
ENTRYPOINT ["/app/entrypoint.sh"]
