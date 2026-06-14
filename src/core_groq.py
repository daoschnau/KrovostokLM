"""
Пайплайн с LLM-прокладкой на Groq (бесплатный tier, открытая модель).

Архитектура (как в старом core.py, но Claude → Groq, эмбеддинг → локальный e5):
  1. HyDE  — Groq генерирует короткую «мораль в духе Кровостока» по ситуации.
  2. Поиск — гипотеза векторизуется локальным e5, ChromaDB отдаёт топ-N кандидатов.
  3. Rerank — Groq выбирает из кандидатов одну цитату, чей вайб лучше ложится
              на ситуацию (отсекает синтаксический мусор и буквальные совпадения).

Зачем LLM-слой: цитаты Кровостока абсурдистские, чистый retrieval цепляется
за поверхностные совпадения слов. Реранкер оценивает уместность вайба, а сама
цитата остаётся настоящей (из базы) — модель ничего не сочиняет.

Модель: llama-3.3-70b-versatile (Groq free tier, сильна в русском).
Ключ:   GROQ_API_KEY в .env

HyDE можно отключить (USE_HYDE=false в .env) — тогда поиск идёт по сырому
запросу пользователя, остаётся только реранкинг.
"""
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Позволяет запускать/импортировать без установки пакета
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_hf import retrieve_candidates, EMBEDDING_MODEL  # noqa: E402

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
USE_HYDE = os.getenv("USE_HYDE", "true").lower() not in {"false", "0", "no"}
N_CANDIDATES = 20

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY не задан. Получи бесплатный ключ на https://console.groq.com "
                "и положи в .env (см. .env.example)"
            )
        _client = Groq(api_key=api_key)
    return _client


def generate_hyde(user_query: str) -> str:
    """Шаг 1 (HyDE): суровая мораль в духе Кровостока — концентрат смысла без
    пересказа ситуации. Её вектор ближе к нужным цитатам, чем сырой запрос."""
    system_prompt = (
        "Ты — старый, повидавший дерьма текстовик группы «Кровосток». "
        "Выдай суровую философскую мораль или жёсткое напутствие в ответ на боль пользователя.\n\n"
        "ПРАВИЛА:\n"
        "1. ЗАПРЕЩЕНО пересказывать или комментировать ситуацию пользователя. Не используй слова из его запроса. Никаких вступлений вроде 'Слушай, брат...'.\n"
        "2. НАЧИНАЙ СРАЗУ с главного тейка или сурового жизненного закона.\n"
        "3. Мрачный фатализм, уличная философия, метафоры Кровостока (безысходность, физиология, криминал, но с внутренним стержнем).\n"
        "4. Строго 1-2 коротких предложения. Только концентрат смысла.\n\n"
        "Пример: 'Гниль съедает слабых, а сильные просто молча жуют стекло. Выплюнь кровь и иди дальше.'"
    )
    try:
        response = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=100,
            temperature=0.8,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ОШИБКА HyDE] {e}")
        # Фолбэк: ищем по сырому запросу
        return user_query


def rerank_quotes(user_query: str, candidates: list) -> dict:
    """Шаг 3 (Rerank): Groq выбирает одну лучшую цитату из кандидатов."""
    if not candidates:
        return {"quote": "База пуста.", "track": "Unknown"}

    quotes_text = "\n\n".join(
        f"[{c['id']}] {c['quote']} (Трек: {c['track']})" for c in candidates
    )

    prompt = (
        f"Ситуация пользователя: {user_query}\n\n"
        f"Кандидаты (цитаты группы Кровосток):\n{quotes_text}\n\n"
        "Выбери ровно ОДНУ цитату, которая станет идеальным ироничным или суровым "
        "комментарием к ситуации пользователя.\n\n"
        "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
        "1. ФИЛЬТР МУСОРА: База нарезана механически. Беспощадно игнорируй цитаты, "
        "которые обрываются на предлогах, лишены логического начала/конца или не имеют "
        "законченной мысли. Выбирай только цельный, хлёсткий панчлайн.\n"
        "2. ЛОВУШКА ПРЯМЫХ СОВПАДЕНИЙ: избегай цитат с буквальным повтором слов из "
        "ситуации пользователя. Нужна смысловая метафора, а не совпадение корней.\n"
        "3. ЭМОЦИОНАЛЬНЫЙ ВАЙБ: стоицизм, фатализм, чёрный юмор или абсурд, который "
        "тонко ложится на боль пользователя.\n\n"
        "Ответь ТОЛЬКО номером лучшей цитаты в квадратных скобках, например: [14]"
    )

    try:
        response = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=15,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content.strip()
        print(f"[DEBUG RERANK] Ответ Groq: {answer}")

        match = re.search(r"\d+", answer)
        if match:
            best_id = int(match.group(0))
            for c in candidates:
                if c["id"] == best_id:
                    return {"quote": c["quote"], "track": c["track"]}
        return {"quote": candidates[0]["quote"], "track": candidates[0]["track"]}
    except Exception as e:
        print(f"[ОШИБКА Rerank] {e}")
        return {"quote": candidates[0]["quote"], "track": candidates[0]["track"]}


def find_quote(user_message: str) -> dict:
    """Главный пайплайн: HyDE -> Vector Search -> Rerank.

    Drop-in замена core_hf.find_quote: тот же возврат {quote, track}.
    """
    print(f"\n[GROQ] Запрос: {user_message}")

    if USE_HYDE:
        hyde = generate_hyde(user_message)
        print(f"[GROQ] HyDE-гипотеза: {hyde}")
        embed_text = hyde
    else:
        embed_text = user_message

    # Ищем по гипотезе, но мусор фильтруем против реального запроса пользователя
    candidates = retrieve_candidates(
        embed_text=embed_text,
        filter_against=user_message,
        n=N_CANDIDATES,
    )

    # На реранкинг подаём только валидные кандидаты (если есть)
    valid = [c for c in candidates if c["valid"]]
    pool = valid if valid else candidates
    # Перенумеруем для компактного промпта
    for new_id, c in enumerate(pool):
        c["id"] = new_id

    result = rerank_quotes(user_message, pool)
    print(f"[GROQ] Выбрана: {result['quote'][:80]}")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print(f"  КровостокLM — Groq Mode ({GROQ_MODEL})")
    print(f"  HyDE: {'вкл' if USE_HYDE else 'выкл'} | Эмбеддер: {EMBEDDING_MODEL}")
    print("=" * 60)
    while True:
        try:
            user_input = input("\nТвоя боль/ситуация (или 'exit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nПока.")
            break
        if user_input.lower() in {"exit", "quit", "выход", "q"}:
            break
        if not user_input:
            continue
        res = find_quote(user_input)
        print(f'\n"{res["quote"]}"\n— {res["track"]}\n')
