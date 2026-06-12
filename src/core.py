import os
import re
from pathlib import Path
from dotenv import load_dotenv
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from tenacity import retry, stop_after_attempt, wait_exponential
import requests

# Загружаем переменные окружения
load_dotenv()
llm_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Пути и инициализация векторной базы
base_dir = Path(__file__).resolve().parent.parent
db_path = base_dir / "data" / "vector_db"

# Используем удаленное API вместо загрузки модели в RAM
def get_embedding_robust(text: str) -> list:
    """Получает эмбеддинг через HF Inference API.

    Старый домен api-inference.huggingface.co отключён (ConnectionError),
    поэтому ходим через актуальный router.huggingface.co.
    """
    api_url = (
        "https://router.huggingface.co/hf-inference/models/"
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/pipeline/feature-extraction"
    )
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN не задан в переменных окружения")
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.post(api_url, headers=headers, json={"inputs": [text]}, timeout=60)
    if response.status_code != 200:
        # Кладём тело ответа в ошибку, чтобы в логах была реальная причина, а не голый RetryError
        raise RuntimeError(f"HF API вернул {response.status_code}: {response.text[:300]}")

    # Возвращаем сам вектор (он приходит как список списков)
    return response.json()

chroma_client = chromadb.PersistentClient(path=str(db_path))
# Убираем жесткую привязку функции, чтобы не бесить ChromaDB
collection = chroma_client.get_collection(name="krovostok_quotes")

def generate_psychologist_advice(user_query: str, history: list | None = None) -> str:
    """Шаг 1: HyDE. Чистая выжимка сути без пересказа ситуации пользователя."""
    system_prompt = (
        "Ты — старый, повидавший дерьма авторитетный текстовик группы «Кровосток». "
        "Твоя задача — выдать суровую философскую мораль или жесткое напутствие в ответ на боль пользователя. "
        "\n\n"
        "СТРОГОЕ КРИТИЧЕСКОЕ ПРАВИЛО:\n"
        "1. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО пересказывать, упоминать или комментировать саму ситуацию пользователя. Не используй слова из его запроса. Никаких вступлений вроде 'Слушай, брат...' или 'Бывает...'.\n"
        "2. НАЧИНАЙ СРАЗУ с главного тейка, вывода или сурового жизненного закона.\n"
        "3. Используй мрачный фатализм, уличную философию и метафоры Кровостока (безысходность, физиология, криминал, но с внутренним стержнем).\n"
        "4. Объем: строго 1-2 коротких предложения. Только концентрат смысла.\n"
        "\n"
        "Пример правильного ответа: 'Гниль съедает слабых, а сильные просто молча жуют стекло. Выплюнь кровь и иди дальше.'\n"
        "\n"
        "Если в диалоге есть предыдущие сообщения — используй их только как фон, чтобы понять ситуацию, "
        "но отвечай строго на последнее сообщение."
    )
    messages = list(history or []) + [{"role": "user", "content": user_query}]
    try:
        response = llm_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100, # Урезали лимит, чтобы он физически не мог лить воду
            temperature=0.8,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"[ОШИБКА HyDE] {e}")
        return "Всё тлен, брат. Просто терпи."

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=15), reraise=True)
def find_top_quotes(advice_text: str, n=20) -> list:
    """Шаг 2: Достаем ТОП-20 похожих цитат из ChromaDB для расширения выборки."""
    # Получаем вектор через нашу новую функцию
    query_vector = get_embedding_robust(advice_text)
    
    # Ищем по готовому вектору
    results = collection.query(query_embeddings=query_vector, n_results=n)
    
    quotes = []
    if results['documents']:
        for i in range(len(results['documents'][0])):
            quotes.append({
                "id": i,
                "quote": results['documents'][0][i],
                "track": results['metadatas'][0][i]['track_name']
            })
    return quotes

def rerank_quotes(user_query: str, quotes: list, history: list | None = None) -> dict:
    """Шаг 3: Reranking. Клод фильтрует мусор и выбирает ИДЕАЛЬНУЮ цитату из ТОП-20."""
    if not quotes:
        return {"quote": "База пуста.", "track": "Unknown"}

    quotes_text = "\n\n".join([f"[{q['id']}] {q['quote']} (Трек: {q['track']})" for q in quotes])

    # Короткий контекст беседы, чтобы реранкер понимал реплики вроде "и что мне теперь делать?"
    context_block = ""
    if history:
        lines = [
            f"{'Пользователь' if m['role'] == 'user' else 'Бот'}: {m['content']}"
            for m in history
        ]
        context_block = "Предыдущие реплики диалога (только для понимания контекста):\n" + "\n".join(lines) + "\n\n"

    prompt = (
        f"{context_block}"
        f"Ситуация пользователя: {user_query}\n\n"
        f"Кандидаты (цитаты группы Кровосток):\n{quotes_text}\n\n"
        "Твоя задача — выбрать ровно ОДНУ цитату, которая станет идеальным ироничным или суровым "
        "комментарием к ситуации пользователя.\n\n"
        "КРИТИЧЕСКИЕ ПРАВИЛА И ШТРАФЫ:\n"
        "1. ФИЛЬТР СИНТАКСИЧЕСКОГО МУСОРА (КРИТИЧНО): База нарезана механически. Беспощадно штрафуй и игнорируй цитаты, которые обрываются на предлогах, лишены логического начала/конца или не имеют законченной мысли. Выбирай только те, что звучат как цельный, хлесткий панчлайн.\n"
        "2. ЛОВУШКА ПРЯМЫХ СОВПАДЕНИЙ: Категорически избегай цитат, где есть буквальные повторения слов из ситуации пользователя (например, 'съел', 'директор', 'бег'). Тебе нужна смысловая метафора, а не тупое совпадение корней.\n"
        "3. ЭМОЦИОНАЛЬНЫЙ ВАЙБ: Ищи стоицизм, фатализм, черный юмор или абсурд, который тонко ложится на боль пользователя.\n\n"
        "В ответе напиши ТОЛЬКО номер лучшей цитаты в квадратных скобках (например: [14])."
    )

    try:
        response = llm_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=15, 
            temperature=0.0, 
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text.strip()
        print(f"[DEBUG RERANK] Ответ Клода: {answer}") 
        
        match = re.search(r'\d+', answer)
        if match:
            best_id = int(match.group(0))
            for q in quotes:
                if q['id'] == best_id:
                    return q
                    
        return quotes[0]
        
    except Exception as e:
        print(f"[ОШИБКА Rerank] {e}")
        return quotes[0]

def process_user_request(user_query: str, history: list | None = None) -> dict:
    """Главная функция пайплайна: HyDE -> Vector Search -> Rerank.

    history — последние реплики диалога в формате [{"role": "user"|"assistant", "content": str}, ...]
    """
    print(f"\n[ЗАПРОС ПОЛЬЗОВАТЕЛЯ]: {user_query}")

    advice = generate_psychologist_advice(user_query, history)
    print(f"\n[СОВЕТ ПСИХОЛОГА (CLAUDE)]: {advice}")

    # Ищем топ-10 кандидатов
    top_quotes = find_top_quotes(advice, n=10)

    # Выбираем лучшую цитату
    result = rerank_quotes(user_query, top_quotes, history)
    print(f"\n[ЦИТАТА КРОВОСТОКА]:\n{result['quote']}\n(Трек: {result['track']})\n")
    
    return result

if __name__ == "__main__":
    print("=== KrovostokLM Core запущен ===")
    while True:
        user_input = input("\nТвоя боль/ситуация (или 'exit'): ")
        if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
            break
        if not user_input.strip():
            continue
        process_user_request(user_input)