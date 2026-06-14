"""
Версия пайплайна без Claude (retrieval-ядро).

Вместо HyDE + LLM-реранкинга:
  1. Запрос пользователя векторизуется с префиксом 'query:' (требование e5)
  2. ChromaDB возвращает топ-30 ближайших цитат (cosine similarity)
  3. Эвристический фильтр убирает обрывки, запятые-хвосты и буквальные совпадения
  4. Возвращается лучший оставшийся результат

Эмбеддер: intfloat/multilingual-e5-large. Документы в базе проиндексированы с
префиксом 'passage:' (см. vectorize.py).

ДВА БЭКЕНДА ЭМБЕДДИНГА (EMBED_BACKEND в .env):
  • "api" (дефолт)  — DeepInfra по HTTP. На сервере НИЧЕГО локально не ставим
                      (без torch/sentence-transformers), образ ~400 МБ. Векторы
                      те же, что в базе, — перевекторизация не нужна.
  • "local"         — sentence-transformers грузит модель локально (~2.5 ГБ).
                      Только для пересборки базы и офлайн-тестов на дев-машине.
"""
import os
import re
import math
from pathlib import Path

import chromadb
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
COLLECTION_NAME = "krovostok_quotes"
N_CANDIDATES = 30

EMBED_BACKEND = os.getenv("EMBED_BACKEND", "api").lower()
DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai/embeddings"

_PREPOSITIONS = {
    "в", "на", "с", "по", "за", "к", "у", "из", "до", "от", "над",
    "под", "при", "про", "без", "через", "и", "а", "но", "что",
    "как", "или", "то", "же", "бы",
}

base_dir = Path(__file__).resolve().parent.parent
db_path = base_dir / "data" / "vector_db"

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        # Тяжёлый импорт — только при EMBED_BACKEND=local, на сервере не выполняется
        from sentence_transformers import SentenceTransformer
        print(f"[HF-DIRECT] Загрузка локальной модели {EMBEDDING_MODEL}...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(db_path))
        # Без embedding_function: эмбеддинги хранятся явно, мы сами их вычисляем
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def _is_valid(text: str, user_query: str) -> bool:
    text = text.strip()

    # Слишком короткий или малословный фрагмент
    if len(text) < 12 or len(text.split()) < 6:
        return False

    # Заканчивается запятой или двоеточием — фраза не завершена
    if text.rstrip()[-1] in {",", ":"}:
        return False

    # Артефакт скрапера: «слово ... слово» — обрыв оригинала
    if re.search(r"\s\.\.\.\s", text):
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


def _l2_normalize(vec: list) -> list:
    """Приводит вектор к единичной длине (база построена на нормированных e5)."""
    norm = math.sqrt(sum(x * x for x in vec))
    return vec if norm == 0 else [x / norm for x in vec]


def _embed_local(text: str) -> list:
    model = _get_model()
    return model.encode("query: " + text, normalize_embeddings=True).tolist()


def _embed_api(text: str) -> list:
    """Эмбеддинг запроса через DeepInfra — тот же e5-large, что и в базе."""
    import requests

    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPINFRA_API_KEY не задан. Получи ключ на "
            "https://deepinfra.com/dash/api_keys и положи в .env (см. .env.example). "
            "Либо переключись на локальную модель: EMBED_BACKEND=local"
        )
    resp = requests.post(
        DEEPINFRA_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": EMBEDDING_MODEL, "input": ["query: " + text]},
        timeout=30,
    )
    resp.raise_for_status()
    vec = resp.json()["data"][0]["embedding"]
    # DeepInfra и локальная модель — один e5, но нормируем сами, чтобы L2-расстояние
    # в ChromaDB совпадало с тем, как строилась база (normalize_embeddings=True).
    return _l2_normalize(vec)


def embed_query(text: str) -> list:
    """Векторизует текст как поисковый запрос e5 (prefix 'query:')."""
    if EMBED_BACKEND == "local":
        return _embed_local(text)
    return _embed_api(text)


def retrieve_candidates(
    embed_text: str,
    filter_against: str | None = None,
    n: int = N_CANDIDATES,
) -> list:
    """Достаёт топ-N кандидатов из ChromaDB по тексту embed_text.

    embed_text     — текст, который векторизуется для поиска (запрос или HyDE-гипотеза).
    filter_against — текст пользователя, против которого проверяется эвристика
                     валидности (если None — берётся embed_text).
    Возвращает список dict: {id, quote, track, distance, valid}.
    """
    if filter_against is None:
        filter_against = embed_text

    collection = _get_collection()
    query_embedding = embed_query(embed_text)
    results = collection.query(query_embeddings=[query_embedding], n_results=n)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        candidates.append({
            "id": i,
            "quote": doc,
            "track": meta["track_name"],
            "distance": dist,
            "valid": _is_valid(doc, filter_against),
        })
    return candidates


def find_quote(user_message: str) -> dict:
    """Находит цитату по запросу пользователя без использования LLM.

    Возвращает dict с ключами 'quote' и 'track'.
    """
    candidates = retrieve_candidates(user_message)

    print(f"\n[HF-DIRECT] Запрос: {user_message}")
    print("[HF-DIRECT] Топ-3 из ChromaDB (расстояния):")
    for c in candidates[:3]:
        print(f"  [{c['id']}] dist={c['distance']:.4f} | {c['quote'][:60]}...")

    for c in candidates:
        if c["valid"]:
            print(f"[HF-DIRECT] Выбрана: {c['quote'][:80]}")
            return {"quote": c["quote"], "track": c["track"]}

    # Запасной вариант: топ-1 без фильтрации
    return {"quote": candidates[0]["quote"], "track": candidates[0]["track"]}
