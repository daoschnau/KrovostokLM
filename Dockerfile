# Лёгкий образ: на сервере нет локальных моделей (всё по API), поэтому
# не нужны ни torch, ни build-essential — все зависимости ставятся из wheel'ов.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Сначала зависимости — слой кешируется между сборками
COPY requirements.txt .
RUN pip install -r requirements.txt

# Исходники + вектор-база (data/vector_db). Тяжёлые data/raw, data/processed
# и логи отсекаются через .dockerignore.
COPY . .

CMD ["python", "src/bot.py"]
