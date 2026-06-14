"""
Пересборка вектор-базы ChromaDB.

ВАЖНО: эмбеддинг здесь ОБЯЗАН считаться тем же способом, что и запросы в рантайме
(core_hf.embed_query), иначе размерность/пространство не совпадут и бот упадёт с
"Collection expecting embedding with dimension of N, got M".

Два бэкенда (EMBED_BACKEND в .env):
  • "api" (дефолт) — DeepInfra, тот же e5-large, что отвечает на запросы. 1024-мер,
                     без torch. Гарантированно совместимо с рантаймом. Нужен
                     DEEPINFRA_API_KEY. ИМЕННО ЭТУ базу коммить в репозиторий.
  • "local"        — sentence-transformers локально (~2.5 ГБ). Тоже 1024-мер, но
                     если рантайм ходит в DeepInfra — оставит риск мелких расхождений.

Запуск:  python src/vectorize.py
"""
import os
import math
import time
from pathlib import Path

import pandas as pd
import chromadb
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "krovostok_quotes"
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "api").lower()
DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai/embeddings"


def _l2_normalize(vec: list) -> list:
    norm = math.sqrt(sum(x * x for x in vec))
    return vec if norm == 0 else [x / norm for x in vec]


def _embed_passages_api(texts: list) -> list:
    """Эмбеддинг документов через DeepInfra (prefix 'passage:'), батчами."""
    import requests

    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPINFRA_API_KEY не задан. Ключ: https://deepinfra.com/dash/api_keys"
        )
    headers = {"Authorization": f"Bearer {api_key}"}
    out = []
    batch = 64
    total = len(texts)
    for start in range(0, total, batch):
        chunk = ["passage: " + t for t in texts[start:start + batch]]
        for attempt in range(4):
            resp = requests.post(
                DEEPINFRA_URL,
                headers=headers,
                json={"model": EMBEDDING_MODEL, "input": chunk},
                timeout=60,
            )
            if resp.status_code == 429:  # троттлинг — подождём и повторим
                time.sleep(2 * (attempt + 1))
                continue
            resp.raise_for_status()
            break
        else:
            resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        out.extend(_l2_normalize(d["embedding"]) for d in data)
        print(f"  эмбеддинг {min(start + batch, total)}/{total}")
    return out


def _embed_passages_local(texts: list) -> list:
    from sentence_transformers import SentenceTransformer

    print(f"Загружаем локальную модель {EMBEDDING_MODEL} (~2.5 ГБ)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    emb = model.encode(
        ["passage: " + t for t in texts],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return emb.tolist()


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "dataset.parquet"
    db_path = base_dir / "data" / "vector_db"

    if not data_path.exists():
        print(f"[ОШИБКА] {data_path} не найден. Сначала запустите data_prep.py")
        return

    print("Читаем датасет...")
    df = pd.read_parquet(data_path)
    texts = df["text"].tolist()
    print(f"Чанков к векторизации: {len(texts)}")
    print(f"Бэкенд эмбеддинга: {EMBED_BACKEND}")

    if EMBED_BACKEND == "local":
        embeddings = _embed_passages_local(texts)
    else:
        embeddings = _embed_passages_api(texts)

    dim = len(embeddings[0])
    print(f"Размерность векторов: {dim}")

    print(f"Подключаемся к ChromaDB: {db_path}")
    client = chromadb.PersistentClient(path=str(db_path))

    # Чистая пересборка: иначе остаются «зомби» от старой модели/размерности
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Удалена старая коллекция")
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    total = len(df)
    batch_size = 100
    print(f"Загружаем {total} записей в базу...")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_df = df.iloc[start:end]
        collection.add(
            embeddings=embeddings[start:end],
            documents=batch_df["text"].tolist(),
            metadatas=[{"track_name": t} for t in batch_df["track_name"]],
            ids=batch_df["chunk_id"].tolist(),
        )
        print(f"  -> {end}/{total}")

    print(f"✅ Готово. Коллекция '{COLLECTION_NAME}', размерность {dim}, записей {total}.")
    print("   Закоммить data/vector_db и запушь — деплой подхватит.")


if __name__ == "__main__":
    main()
