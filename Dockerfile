# Kullanılacak temel Python imajını belirtin
FROM python:3.12-slim

# Konteyner içinde çalışma dizinini belirleyin
WORKDIR /app

# Bağımlılıkları kurmak için önce requirements.txt dosyasını kopyalayın
COPY requirements.txt ./

# Belirtilen bağımlılıkları kurun
RUN pip install --no-cache-dir -r requirements.txt

# entrypoint scriptini kopyalayın ve çalıştırılabilir yapın
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# requirements.txt dışındaki diğer proje dosyalarını kopyalayın
COPY . .

# Uygulamanın dinleyeceği portu belirtin (dokümantasyon için)
EXPOSE 8080

# Konteyner başladığında çalıştırılacak scripti ENTRYPOINT olarak ayarlayın
# exec formatı değişken çözümlemesi için uygundur
ENTRYPOINT ["/app/entrypoint.sh"]

# CMD satırı artık gerekmez (veya ENTRYPOINT için varsayılan argümanlar için kullanılabilir)
# CMD ["main:app"] # Örnek: Eğer entrypoint.sh sadece "uvicorn --host 0.0.0.0 --port $PORT" çalıştırsa