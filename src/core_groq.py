"""
Пайплайн с LLM-прокладкой на Groq — одноплечий HyDE + реранкинг.

Архитектура (выбрана по результатам трёх тестовых прогонов):
  1. HyDE   — Groq пишет короткую «мораль в духе Кровостока» по ситуации.
              Тон подстраивается под валентность запроса (позитив/негатив),
              чтобы достижения не получали мрачных гипотез.
  2. Поиск  — гипотеза векторизуется локальным e5, ChromaDB отдаёт топ-N.
  3. Rerank — Groq выбирает одну цитату, следя за знаком эмоции, мягкостью на
              горе, трезвостью и избегая универсальных «магнит-цитат».
  +  Анти-магнит — на уровне сессии не повторяем недавно выданные треки
              (в Telegram сессия = chat_id), чтобы один трек не липнул ко всему.

УСТОЙЧИВОСТЬ (для публичного бота — главное требование):
  Groq free tier ограничен дневным бюджетом токенов (TPD, per-model). Когда
  бюджет кончается, бот НЕ падает, а плавно деградирует в чистый e5-retrieval
  (без API, без лимита). Механизмы:
    • любой сбой Groq → фоллбэк на лучший валидный кандидат e5;
    • после rate-limit включается cooldown: на время мы вообще не дёргаем
      Groq, отвечая мгновенно на сыром e5 (без лишней латентности под нагрузкой);
    • кэш одинаковых запросов экономит бюджет на повторах.

Модель: llama-3.3-70b-versatile (качество). Для публичного бота имеет смысл
        GROQ_MODEL=llama-3.1-8b-instant — слабее, но ~5x дневной бюджет.
Ключ:   GROQ_API_KEY в .env
"""
import os
import re
import sys
import time
import threading
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_hf import retrieve_candidates, EMBEDDING_MODEL  # noqa: E402

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
USE_HYDE = os.getenv("USE_HYDE", "true").lower() not in {"false", "0", "no"}
RATE_LIMIT_COOLDOWN = int(os.getenv("GROQ_COOLDOWN_SEC", "120"))

# Анти-магнит: сколько последних выданных треков помним на сессию, чтобы не
# повторять один и тот же трек подряд (в Telegram сессия = chat_id). 0 — выключить.
RECENT_TRACKS_MEMORY = int(os.getenv("GROQ_RECENT_TRACKS", "5"))

# Бюджет токенов — главный пожиратель это промпт реранкера, поэтому пул маленький.
N_CANDIDATES = 12

_client = None
_tokens_used = 0
_groq_disabled_until = 0.0  # пока time.time() < этого — Groq не дёргаем (cooldown)
_cache = {}                 # нормализованный запрос -> результат
_recent_by_session = {}     # session_id -> deque недавно выданных треков
_lock = threading.Lock()


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY не задан. Получи бесплатный ключ на https://console.groq.com "
                "и положи в .env (см. .env.example)"
            )
        # SDK сам ретраит транзиентные 429 (TPM-всплески) с backoff
        _client = Groq(api_key=api_key, max_retries=3)
    return _client


def tokens_used() -> int:
    return _tokens_used


def _track_usage(response) -> None:
    global _tokens_used
    usage = getattr(response, "usage", None)
    if usage is not None:
        _tokens_used += getattr(usage, "total_tokens", 0) or 0


def _groq_available() -> bool:
    """False, если мы на cooldown после недавнего rate-limit."""
    return time.time() >= _groq_disabled_until


def _trip_cooldown() -> None:
    """Включает cooldown после rate-limit: временно уводим бота на сырой e5."""
    global _groq_disabled_until
    _groq_disabled_until = time.time() + RATE_LIMIT_COOLDOWN
    print(f"[GROQ] Rate limit — cooldown {RATE_LIMIT_COOLDOWN}s, временно работаем на сыром e5")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _recent_tracks(session_id: str) -> set:
    """Треки, недавно выданные в этой сессии (для анти-магнит фильтра)."""
    with _lock:
        dq = _recent_by_session.get(session_id)
        return set(dq) if dq else set()


def _remember_track(session_id: str, track: str) -> None:
    """Запоминает выданный трек в скользящем окне сессии."""
    if RECENT_TRACKS_MEMORY <= 0:
        return
    with _lock:
        dq = _recent_by_session.get(session_id)
        if dq is None:
            dq = deque(maxlen=RECENT_TRACKS_MEMORY)
            _recent_by_session[session_id] = dq
        dq.append(track)


def _classify_valence(user_query: str) -> str:
    """Лексическая эвристика (без LLM): позитив или негатив."""
    positive_markers = {
        "горжусь", "победил", "добежал", "подтянулся", "купил", "накопил",
        "сделал", "достиг", "бросил пить", "держусь", "начал", "впервые",
        "рад", "счастлив", "удалось", "получилось", "наконец", "цель",
        "написал", "запустил", "заработал", "выиграл", "годовщина",
    }
    q = user_query.lower()
    return "позитив" if any(m in q for m in positive_markers) else "негатив"


def generate_hyde(user_query: str) -> str | None:
    """Шаг 1 (HyDE): мораль в духе Кровостока, тон по валентности.

    Возвращает None при сбое/cooldown — вызывающий код уходит на сырой запрос.
    """
    if not _groq_available():
        return None

    valence = _classify_valence(user_query)
    if valence == "позитив":
        tone = (
            "Запрос ПОЗИТИВНЫЙ (достижение, гордость, стойкость, победа). "
            "Гипотеза тоже про подъём — дерзость, кураж, мрачноватое торжество. "
            "НЕ используй образы смерти, боли, безысходности, гниения."
        )
    else:
        tone = (
            "Выдай суровую мораль — мрачный фатализм, уличная философия, "
            "метафоры Кровостока (безысходность, физиология, криминал, но со стержнем)."
        )

    system_prompt = (
        "Ты — старый, повидавший дерьма текстовик группы «Кровосток».\n\n"
        f"{tone}\n\n"
        "ПРАВИЛА:\n"
        "1. ЗАПРЕЩЕНО пересказывать ситуацию и использовать слова из запроса.\n"
        "2. Начинай сразу с тейка. Строго 1-2 коротких предложения.\n\n"
        "Пример негатив: 'Гниль съедает слабых, а сильные молча жуют стекло.'\n"
        "Пример позитив: 'Чемпион — не тот, кто не падал, а тот, кто вставал быстрее всех.'"
    )
    try:
        response = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=80,
            temperature=0.8,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )
        _track_usage(response)
        return response.choices[0].message.content.strip()
    except RateLimitError:
        _trip_cooldown()
        return None
    except Exception as e:
        print(f"[ОШИБКА HyDE] {e}")
        return None


def rerank_quotes(user_query: str, candidates: list) -> dict | None:
    """Шаг 3 (Rerank): Groq выбирает одну цитату.

    Возвращает None при сбое/cooldown — вызывающий код берёт лучший e5-кандидат.
    """
    if not candidates or not _groq_available():
        return None

    quotes_text = "\n\n".join(
        f"[{c['id']}] {c['quote']} (Трек: {c['track']})" for c in candidates
    )
    prompt = (
        f"Ситуация пользователя: {user_query}\n\n"
        f"Кандидаты (цитаты Кровостока):\n{quotes_text}\n\n"
        "Выбери ОДНУ цитату — меткий ироничный или суровый комментарий к ЭТОЙ ситуации.\n\n"
        "ПРАВИЛА:\n"
        "1. ЗНАК ЭМОЦИИ. Если ситуация позитивная (достижение, гордость, стойкость) — "
        "НЕ бери цитаты про смерть, боль, наркоту, безысходность. Тон цитаты должен "
        "совпадать со знаком ситуации.\n"
        "2. ТРЕЗВОСТЬ. Для ситуаций про отказ от вредного (бросил пить/курить, трезвость, "
        "воздержание) НЕ выбирай цитаты, прославляющие употребление алкоголя или наркотиков.\n"
        "3. ГОРЕ. Для запросов про смерть, утрату, похороны близких выбирай цитату скорее "
        "тихую и печальную, чем шок-комичную или абсурдную.\n"
        "4. НЕТ УНИВЕРСАЛИЯМ — избегай абстрактных строк, подходящих к чему угодно. "
        "Цепляй конкретную деталь ситуации.\n"
        "5. Игнорируй обрывки без законченной мысли.\n\n"
        "Ответь строго: ЗНАК: <позитив/негатив>  ОТВЕТ: [номер]"
    )
    try:
        response = _get_client().chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=40,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        _track_usage(response)
        answer = response.choices[0].message.content.strip()
        print(f"[DEBUG RERANK] {answer.replace(chr(10), ' ')}")

        ids = re.findall(r"\[(\d+)\]", answer) or re.findall(r"\d+", answer)
        if ids:
            best_id = int(ids[-1])
            for c in candidates:
                if c["id"] == best_id:
                    return {"quote": c["quote"], "track": c["track"]}
        return None
    except RateLimitError:
        _trip_cooldown()
        return None
    except Exception as e:
        print(f"[ОШИБКА Rerank] {e}")
        return None


def find_quote(user_message: str, session_id: str = "_global", min_score: int = 0) -> dict:
    """Главный пайплайн. Никогда не бросает исключений: при любом сбое Groq
    деградирует в чистый e5-retrieval.

    session_id — изолирует анти-магнит память (в Telegram передаём chat_id,
    чтобы цитаты одного юзера не влияли на других). По умолчанию общая сессия.

    Drop-in замена core_hf.find_quote: возврат {quote, track}.
    """
    key = _normalize(user_message)
    with _lock:
        if key in _cache:
            print(f"[GROQ] Кэш-хит: {user_message[:50]}")
            return _cache[key]

    print(f"\n[GROQ] Запрос: {user_message}")

    # HyDE-гипотеза (если доступна), иначе ищем по сырому запросу
    hyde = generate_hyde(user_message) if USE_HYDE else None
    if hyde:
        print(f"[GROQ] HyDE: {hyde}")
    embed_text = hyde or user_message

    candidates = retrieve_candidates(
        embed_text=embed_text,
        filter_against=user_message,
        n=N_CANDIDATES,
        min_score=min_score,
    )
    valid = [c for c in candidates if c["valid"]] or candidates

    # Анти-магнит: выкидываем недавно показанные треки, если остаётся из чего
    # выбирать (>= 3 кандидата), иначе оставляем как есть — лучше повтор, чем пусто.
    recent = _recent_tracks(session_id)
    if recent:
        fresh = [c for c in valid if c["track"] not in recent]
        if len(fresh) >= 3:
            valid = fresh
            print(f"[GROQ] Анти-магнит: исключены треки {recent}")

    for new_id, c in enumerate(valid):
        c["id"] = new_id

    result = rerank_quotes(user_message, valid)
    if result is None:
        # Деградация: лучший валидный кандидат e5 (поведение «сырого e5»)
        best = valid[0]
        result = {"quote": best["quote"], "track": best["track"]}
        print(f"[GROQ] Деградация на e5: {result['quote'][:60]}")
    else:
        print(f"[GROQ] Выбрана: {result['quote'][:60]}")

    _remember_track(session_id, result["track"])
    with _lock:
        _cache[key] = result
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
        print(f"[токенов израсходовано за сессию: {tokens_used()}]")
