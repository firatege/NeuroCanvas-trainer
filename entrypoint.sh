#!/bin/sh

# Hata ayıklama için port değerini yazdır (isteğe bağlı)
echo "Attempting to start uvicorn on port: $PORT"

# Ortam değişkenlerinin yüklenmesini beklemek veya emin olmak için ek adımlar eklenebilir,
# ancak genellikle $PORT ENTRYPOINT çalışırken kullanılabilir olmalıdır.

# Ana uygulama komutunu çalıştırın.
# 'exec' komutu, scriptin yerine uvicorn sürecini koyar, bu da daha iyi sinyal
# işleme ve kaynak kullanımı sağlar.
exec uvicorn main:app --host 0.0.0.0 --port $PORT