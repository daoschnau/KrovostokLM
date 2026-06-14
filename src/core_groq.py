"""
Пайплайн с LLM-прокладкой на Groq (бесплатный tier, открытая модель).

Архитектура (гибридный поиск):
  1. HyDE    — Groq генерирует короткую «мораль в духе Кровостока» по ситуации.
  2. Поиск   — два параллельных запроса к ChromaDB:
               • по HyDE-гипотезе (семантически далёкие, но меткие цитаты)
               • по сырому запросу пользователя (конкретные детали ситуации)
               Пулы объединяются и дедупируются → ~25 уникальных кандидатов.
  3. Rerank  — Groq выбирает одну цитату, следя за валентностью и избегая
               универсальных «магнит-цитат».

Зачем гибридный поиск: HyDE уходит семантически дальше и находит жемчужины,
которых raw не видит (#11, #14, #15, #20, #24 по тестовым прогонам). Raw держит
конкретные детали ситуации (#12 «начну с аптеки», #16 «качался/потел», #21
«малышка ходить не сможет»). Реранкер делает финальный выбор из объединённого пула.

Модель: llama-3.3-70b-versatile (Groq free tier, сильна в русском).
Ключ:   GROQ_API_KEY в .env

HyDE можно отключить (USE_HYDE=false в .env) — тогда поиск идёт только по сырому
запросу пользователя.
"""
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_hf import retrieve_candidates, EMBEDDING_MODEL  # noqa: E402

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
USE_HYDE = os.getenv("USE_HYDE", "true").lower() not in {"false", "0", "no"}
N_PER_ARM = 15  # кандидатов от каждого плеча поиска

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


def _classify_valence(user_query: str) -> str:
    """Быстрая лексическая эвристика: позитив или негатив.

    Не вызывает LLM — используется внутри generate_hyde, чтобы направить
    тон гипотезы и избежать мрачных HyDE для позитивных запросов.
    """
    positive_markers = {
        "горжусь", "победил", "добежал", "подтянулся", "купил", "накопил",
        "сделал", "достиг", "бросил пить", "держусь", "начал", "впервые",
        "рад", "счастлив", "удалось", "получилось", "наконец", "цель",
        "написал", "запустил", "заработал", "выиграл",
    }
    query_lower = user_query.lower()
    if any(m in query_lower for m in positive_markers):
        return "позитив"
    return "негатив"


def generate_hyde(user_query: str) -> str:
    """Шаг 1 (HyDE): суровая или дерзкая мораль в духе Кровостока.

    Тон регулируется валентностью запроса: позитивные запросы получают
    дерзкую гипотезу, чтобы не тащить мрачный кластер.
    """
    valence = _classify_valence(user_query)

    if valence == "позитив":
        tone_instruction = (
            "Запрос несёт ПОЗИТИВНЫЙ знак (достижение, гордость, стойкость, победа). "
            "Выдай дерзкое, кайфовое, мрачновато-торжествующее напутствие — кровосток-style. "
            "НЕ используй образы смерти, боли, безысходности, гниения."
        )
    else:
        tone_instruction = (
            "Выдай суровую философскую мораль — мрачный фатализм, уличная философия, "
            "метафоры Кровостока (безысходность, физиология, криминал, но с внутренним стержнем)."
        )

    system_prompt = (
        "Ты — старый, повидавший дерьма текстовик группы «Кровосток».\n\n"
        f"{tone_instruction}\n\n"
        "СТРОГИЕ ПРАВИЛА:\n"
        "1. ЗАПРЕЩЕНО пересказывать ситуацию. Не используй слова из запроса. "
        "Никаких вступлений.\n"
        "2. Начинай сразу с главного тейка.\n"
        "3. Строго 1-2 коротких предложения. Только концентрат.\n\n"
        "Пример негатив: 'Гниль съедает слабых, а сильные молча жуют стекло.'\n"
        "Пример позитив: 'Чемпион — это не тот, кто не падал, а тот, кто вставал быстрее всех.'"
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
        return user_query


def _merge_candidates(hyde_pool: list, raw_pool: list) -> list:
    """Объединяет два пула кандидатов, дедуплицирует по тексту цитаты,
    перенумеровывает. HyDE-кандидаты идут первыми."""
    seen = set()
    merged = []
    for c in hyde_pool + raw_pool:
        key = c["quote"].strip()
        if key not in seen:
            seen.add(key)
            merged.append(dict(c))
    for new_id, c in enumerate(merged):
        c["id"] = new_id
    return merged


def rerank_quotes(user_query: str, candidates: list) -> dict:
    """Шаг 3 (Rerank): Groq выбирает одну лучшую цитату из объединённого пула."""
    if not candidates:
        return {"quote": "База пуста.", "track": "Unknown"}

    quotes_text = "\n\n".join(
        f"[{c['id']}] {c['quote']} (Трек: {c['track']})" for c in candidates
    )

    prompt = (
        f"Ситуация пользователя: {user_query}\n\n"
        f"Кандидаты (цитаты группы Кровосток):\n{quotes_text}\n\n"
        "Выбери ровно ОДНУ цитату — идеальный ироничный или суровый комментарий "
        "именно к ЭТОЙ ситуации.\n\n"
        "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
        "1. ЗНАК ЭМОЦИИ (главное правило). Сначала определи знак ситуации:\n"
        "   - ПОЗИТИВ (достижение, гордость, радость, стойкость, выздоровление, победа) — "
        "выбирай цитату с совпадающим знаком: дерзость, кураж, мрачноватое торжество. "
        "КАТЕГОРИЧЕСКИ не выбирай строки про смерть, боль, наркоту, безысходность, суицид — "
        "это ломает тон.\n"
        "   - НЕГАТИВ (боль, потеря, тревога, пустота) — подойдёт фатализм, чёрный юмор, "
        "стоицизм.\n"
        "2. НЕТ УНИВЕРСАЛИЯМ. Избегай общефилософских строк, которые одинаково подходят к "
        "любой ситуации (абстрактно про «боль», «кровь», «раны», «одиночество вообще»). "
        "Выбирай цитату, цепляющую КОНКРЕТНУЮ деталь, образ или поворот этой ситуации.\n"
        "3. ФИЛЬТР МУСОРА. База нарезана механически. Игнорируй цитаты, которые обрываются "
        "на предлогах или лишены законченной мысли. Только цельный, хлёсткий панчлайн.\n"
        "4. НЕ В ЛОБ. Избегай буквального повтора слов из ситуации. Нужна смысловая "
        "метафора, а не совпадение корней.\n\n"
        "ФОРМАТ ОТВЕТА (строго):\n"
        "ЗНАК: <позитив/негатив>\n"
        "ПРИЧИНА: <одна короткая фраза, почему эта цитата>\n"
        "ОТВЕТ: [номер]"
    )

    try:
        response = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=160,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content.strip()
        print(f"[DEBUG RERANK] {answer.replace(chr(10), ' | ')}")

        ids = re.findall(r"\[(\d+)\]", answer) or re.findall(r"\d+", answer)
        if ids:
            best_id = int(ids[-1])
            for c in candidates:
                if c["id"] == best_id:
                    return {"quote": c["quote"], "track": c["track"]}
        return {"quote": candidates[0]["quote"], "track": candidates[0]["track"]}
    except Exception as e:
        print(f"[ОШИБКА Rerank] {e}")
        return {"quote": candidates[0]["quote"], "track": candidates[0]["track"]}


def find_quote(user_message: str) -> dict:
    """Главный пайплайн: HyDE + raw → объединённый пул → Rerank.

    Drop-in замена core_hf.find_quote: тот же возврат {quote, track}.
    """
    print(f"\n[GROQ] Запрос: {user_message}")

    raw_pool = retrieve_candidates(
        embed_text=user_message,
        filter_against=user_message,
        n=N_PER_ARM,
    )

    if USE_HYDE:
        hyde = generate_hyde(user_message)
        print(f"[GROQ] HyDE-гипотеза: {hyde}")
        hyde_pool = retrieve_candidates(
            embed_text=hyde,
            filter_against=user_message,
            n=N_PER_ARM,
        )
    else:
        hyde_pool = []

    merged = _merge_candidates(hyde_pool, raw_pool)
    valid = [c for c in merged if c["valid"]]
    pool = valid if valid else merged

    print(f"[GROQ] Пул: {len(pool)} уникальных кандидатов "
          f"(hyde={len(hyde_pool)}, raw={len(raw_pool)})")

    result = rerank_quotes(user_message, pool)
    print(f"[GROQ] Выбрана: {result['quote'][:80]}")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print(f"  КровостокLM — Groq Mode ({GROQ_MODEL})")
    print(f"  HyDE: {'вкл (гибрид)' if USE_HYDE else 'выкл (raw only)'}")
    print(f"  Эмбеддер: {EMBEDDING_MODEL}")
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
