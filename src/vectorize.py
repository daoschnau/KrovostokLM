import pandas as pd
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "krovostok_quotes"


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "dataset.parquet"
    db_path = base_dir / "data" / "vector_db"

    if not data_path.exists():
        print(f"[ОШИБКА] {data_path} не найден. Сначала запустите data_prep.py")
        return

    print("Читаем датасет...")
    df = pd.read_parquet(data_path)
    print(f"Чанков к векторизации: {len(df)}")

    print(f"Загружаем модель {EMBEDDING_MODEL}")
    print("(при первом запуске скачается ~560 MB)")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # e5 требует prefix 'passage:' для документов при индексации
    print("Векторизуем тексты...")
    texts = ["passage: " + t for t in df["text"]]
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"Подключаемся к ChromaDB: {db_path}")
    client = chromadb.PersistentClient(path=str(db_path))

    # Чистая пересборка: upsert не удаляет чанки-зомби при изменении модели
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Удалена старая коллекция")
    except Exception:
        pass

    # Без embedding_function — эмбеддинги уже готовы, ChromaDB хранит их как есть
    collection = client.create_collection(name=COLLECTION_NAME)

    total = len(df)
    batch_size = 100
    print(f"Загружаем {total} записей в базу...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_df = df.iloc[start:end]
        batch_emb = embeddings[start:end]

        collection.add(
            embeddings=batch_emb.tolist(),
            documents=batch_df["text"].tolist(),
            metadatas=[{"track_name": t} for t in batch_df["track_name"]],
            ids=batch_df["chunk_id"].tolist(),
        )
        print(f"  -> {end}/{total}")

    print("✅ Векторизация завершена. База ChromaDB готова.")


if __name__ == "__main__":
    main()
