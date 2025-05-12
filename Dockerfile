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

# requirements.txt dışındaki tüm proje dosyalarını konteynere kopyalayın
# Buna main.py, model.h5 ve public klasörü dahildir
COPY . .

# Uygulamanın dinleyeceği portu belirtin (isteğe bağlı ama iyi uygulama)
# Fly.io, fly.toml'daki internal_port değerini kullanır, bu sadece dokümantasyon içindir.
EXPOSE 8080

# Uygulamayı çalıştırmak için komutu tanımlayın
# Köşeli parantez içindeki format (exec format) ortam değişkenlerinin ($PORT gibi)
# doğru şekilde çözümlenmesini sağlar ve tercih edilir.
# Bu komut, Uvicorn'ın 0.0.0.0 adresinde (tüm arayüzler) ve
# Fly.io tarafından sağlanan $PORT değişkenindeki portta dinlemesini sağlar.
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
# Not: Eğer fly.toml'da internal_port'u 8080 dışında bir değer yaparsanız,
# buradaki EXPOSE değerini de güncellemeniz iyi olur (zorunlu değil ama iyi pratik).
# $PORT değişkeni, fly.toml'daki internal_port değerini otomatik alır.