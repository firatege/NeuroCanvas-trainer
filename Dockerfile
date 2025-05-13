# Kullanılacak temel Python imajını belirtin
# slim versiyonu genellikle daha küçüktür
FROM python:3.12-slim

# Konteyner içinde çalışma dizinini belirleyin
WORKDIR /app

# Bağımlılıkları kurmak için önce requirements.txt dosyasını kopyalayın
# Bu, bağımlılıklar değişmediğinde Docker cache'inin kullanılmasını sağlar
COPY requirements.txt ./

# Belirtilen bağımlılıkları kurun
# --no-cache-dir: pip'in cache kullanmamasını sağlar, imaj boyutunu küçültür
RUN pip install --no-cache-dir -r requirements.txt

# entrypoint scriptini kopyalayın ve çalıştırılabilir yapın
# entrypoint.sh dosyasının projede WORKDIR /app dizinine (yani /app/entrypoint.sh yoluna) kopyalandığından emin olun
COPY entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

# requirements.txt dışındaki diğer proje dosyalarını konteynere kopyalayın
# Buna main.py, model.h5 ve public klasörü dahildir
COPY . .

# Uygulamanın dinleyeceği portu belirtin (dokümantasyon için)
# Fly.io, fly.toml'daki internal_port değerini kullanır, bu sadece dokümantasyon içindir.
EXPOSE 8080

# Konteyner başladığında çalıştırılacak scripti ENTRYPOINT olarak ayarlayın
# Köşeli parantez içindeki format (exec format) kullanılır.
# Scriptin konteyner içindeki tam yoluna işaret eder.
ENTRYPOINT ["/app/entrypoint.sh"]

# CMD satırı artık gerekmez (veya ENTRYPOINT için varsayılan argümanlar için kullanılabilir)
# CMD ["main:app"] # Örnek: Eğer entrypoint.sh sadece "uvicorn --host 0.0.0.0 --port $PORT" çalıştırsa