#!/bin/sh

# Fly.io veya yerel ortam için port ayarı
PORT=${PORT:-8080}

echo "🚀 Starting Uvicorn on port: $PORT"

# Ana uygulamayı başlat
exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
