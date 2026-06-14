"""
Версия пайплайна без Claude.

Вместо HyDE + LLM-реранкинга:
  1. Запрос пользователя векторизуется напрямую той же моделью, что использовалась при индексации
  2. ChromaDB возвращает топ-20 ближайших цитат
  3. Эвристический фильтр убирает обрывки и буквальные совпадения
  4. Возвращается лучший оставшийся результат
"""
import re
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "krovostok_quotes"
N_CANDIDATES = 20

_PREPOSITIONS = {
    "в", "на", "с", "по", "за", "к", "у", "из", "до", "от", "над",
    "под", "при", "про", "без", "через", "и", "а", "но", "что",
    "как", "или", "то", "же", "бы",
}

base_dir = Path(__file__).resolve().parent.parent
db_path = base_dir / "data" / "vector_db"

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        client = chromadb.PersistentClient(path=str(db_path))
        _collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    return _collection


def _is_valid(text: str, user_query: str) -> bool:
    text = text.strip()

    # Слишком короткий фрагмент
    if len(text) < 12:
        return False

    # Обрывается на предлоге или союзе
    last_word = re.sub(r"[.,!?…\"']+$", "", text).split()
    if last_word and last_word[-1].lower() in _PREPOSITIONS:
        return False

    # Буквальное пересечение слов с запросом (> 40% слов цитаты совпадают)
    query_words = {w.lower() for w in re.findall(r"\w+", user_query) if len(w) > 3}
    quote_words = [w.lower() for w in re.findall(r"\w+", text) if len(w) > 3]
    if query_words and quote_words:
        overlap = sum(1 for w in quote_words if w in query_words)
        if overlap / len(quote_words) > 0.4:
            return False

    return True


def find_quote(user_message: str) -> dict:
    """Находит цитату по запросу пользователя без использования LLM.

    Возвращает dict с ключами 'quote' и 'track'.
    """
    collection = _get_collection()

    results = collection.query(query_texts=[user_message], n_results=N_CANDIDATES)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\n[HF-DIRECT] Запрос: {user_message}")
    print(f"[HF-DIRECT] Топ-3 из ChromaDB (расстояния):")
    for i in range(min(3, len(docs))):
        print(f"  [{i}] dist={distances[i]:.4f} | {docs[i][:60]}...")

    # Берём первую цитату, прошедшую фильтр
    for doc, meta in zip(docs, metas):
        if _is_valid(doc, user_message):
            print(f"[HF-DIRECT] Выбрана: {doc[:80]}")
            return {"quote": doc, "track": meta["track_name"]}

    # Запасной вариант: топ-1 без фильтрации
    return {"quote": docs[0], "track": metas[0]["track_name"]}
