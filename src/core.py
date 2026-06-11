import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic
import chromadb
from chromadb.utils import embedding_functions

# Загружаем переменные окружения из .env
load_dotenv()

# Инициализация клиента Claude
llm_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Пути и инициализация векторной базы
base_dir = Path(__file__).resolve().parent.parent
db_path = base_dir / "data" / "vector_db"

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

chroma_client = chromadb.PersistentClient(path=str(db_path))
collection = chroma_client.get_collection(
    name="krovostok_quotes",
    embedding_function=sentence_transformer_ef
)

def generate_psychologist_advice(user_query: str) -> str:
    """Генерирует идеальный совет психолога (HyDE) с помощью Claude."""
    system_prompt = (
        "Ты — высококвалифицированный, эмпатичный и спокойный психолог. "
        "Твоя задача — дать короткий (2-3 предложения), мудрый и философский "
        "совет на жизненную ситуацию пользователя. Не используй клише, отвечай "
        "глубоко, метафорично и по-взрослому."
    )
    
    try:
        response = llm_client.messages.create(
            model="claude-haiku-4-5-20251001", # Установлена строго запрошенная модель
            max_tokens=150,
            temperature=0.7,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_query}
            ]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"[ОШИБКА LLM] {e}")
        return "Жизнь — сложная штука. Просто прими это и двигайся дальше."

def find_krovostok_quote(advice_text: str) -> dict:
    """Ищет в ChromaDB самую релевантную цитату на основе совета."""
    results = collection.query(
        query_texts=[advice_text],
        n_results=1
    )
    
    if results['documents'] and results['documents'][0]:
        quote = results['documents'][0][0]
        track_name = results['metadatas'][0][0]['track_name']
        return {"quote": quote, "track": track_name}
    
    return {"quote": "База молчит. На такое даже у Шило нет слов.", "track": "Unknown"}

def process_user_request(user_query: str) -> dict:
    """Главная функция: принимает запрос, генерирует совет и возвращает цитату."""
    print(f"\n[ЗАПРОС ПОЛЬЗОВАТЕЛЯ]: {user_query}")
    
    advice = generate_psychologist_advice(user_query)
    print(f"\n[СОВЕТ ПСИХОЛОГА (CLAUDE)]: {advice}")
    
    result = find_krovostok_quote(advice)
    print(f"\n[ЦИТАТА КРОВОСТОКА]:\n{result['quote']}\n(Трек: {result['track']})\n")
    
    return result

if __name__ == "__main__":
    print("=== KrovostokLM Core запущен ===")
    print("Введи свою ситуацию (или 'exit' для выхода).")
    
    while True:
        # Получаем текст из терминала
        user_input = input("\nТвоя боль/ситуация: ")
        
        # Условие выхода
        if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
            print("Завершение работы ядра.")
            break
            
        # Проверка на пустой ввод
        if not user_input.strip():
            continue
            
        # Вызов главной функции
        process_user_request(user_input)