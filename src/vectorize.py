import pandas as pd
import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions

def main():
    # Настраиваем пути
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "dataset.parquet"
    db_path = base_dir / "data" / "vector_db"

    if not data_path.exists():
        print(f"[ОШИБКА] Файл {data_path} не найден. Сначала запустите data_prep.py")
        return

    print("Читаем датасет...")
    df = pd.read_parquet(data_path)

    print("Инициализируем локальную модель эмбеддингов (при первом запуске она скачается ~400MB)...")
    # Используем мультиязычную модель без цензуры
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    print(f"Подключаемся к ChromaDB по пути: {db_path}")
    # Настраиваем персистентный клиент (данные сохранятся на диск)
    client = chromadb.PersistentClient(path=str(db_path))

    # Создаем или получаем коллекцию (таблицу)
    collection_name = "krovostok_quotes"
    collection = client.get_or_create_collection(
        name=collection_name, 
        embedding_function=sentence_transformer_ef
    )

    # ChromaDB не любит получать все 837 записей за один раз, разобьем на батчи
    batch_size = 100
    total_batches = (len(df) // batch_size) + 1

    print(f"Начинаем векторизацию и загрузку {len(df)} записей в базу...")
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_df = df.iloc[start_idx:end_idx]
        
        if batch_df.empty:
            break

        # Загружаем батч в базу
        collection.upsert(
            documents=batch_df["text"].tolist(),
            metadatas=[{"track_name": track} for track in batch_df["track_name"].tolist()],
            ids=batch_df["chunk_id"].tolist()
        )
        print(f"  -> Обработан батч {i+1}/{total_batches}")

    print("✅ Векторизация успешно завершена! База ChromaDB готова.")

if __name__ == "__main__":
    main()