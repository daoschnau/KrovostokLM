"""
Оценка качества цитат в базе по шкале 1-10.

Для каждой цитаты Claude выставляет оценку:
  — насколько она осмысленна и завершена как мысль
  — насколько звучит как афоризм, которым можно ответить на ситуацию

Дополнительно: выявляет соседей по треку, которых лучше слить в одну цитату.

Результат: data/scores.json
  {
    "chunk_id": {
      "score": 7,
      "reason": "...",
      "merge_with_next": false   # true если эта цитата + следующая по треку = лучше
    },
    ...
  }

Использует Anthropic Batches API (50% скидка, асинхронно).
Прогресс пишется в data/batch_state.json — при прерывании запуск с --resume
подхватит ожидающий батч или продолжит парсинг.

Запуск:
  python src/score_quotes.py
  python src/score_quotes.py --batch-size 8 --model claude-haiku-4-5
  python src/score_quotes.py --resume   # продолжить после прерывания
"""
import os
import re
import json
import time
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import anthropic

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "dataset.parquet"
SCORES_PATH = BASE_DIR / "data" / "scores.json"
STATE_PATH = BASE_DIR / "data" / "batch_state.json"

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
DEFAULT_BATCH = 8


SCORE_PROMPT = """\
Ты — редактор бота, который отвечает цитатами из русского рэпа на жизненные \
ситуации пользователей. Оцени каждую цитату группы «Кровосток» по шкале 1–10.

Главный критерий: насколько эта строка сработает как реплика-ответ, \
если пользователь рассказал тебе что-то из своей жизни?

Целевое распределение — калибруйся строго по нему, не сваливай всё в середину:
  1–3  (~30%) — мусор: обрывок, нарративный хвост, бессмыслица, незаконченная мысль.
  4–6  (~40%) — средняя: работает, но только на узкий запрос или слабовата.
  7–9  (~25%) — хорошая: ёмкая, образ понятен без знания песни, бьёт точно.
  10   (~5%)  — шедевр: горькая мудрость или парадокс — хочется переслать.

Правила:
- Мат и жёсткий контент НЕ снижают оценку — это стиль группы.
- Список действий или состояний («ревновал, боялся, влюблялся») — НЕ минус, \
если список создаёт узнаваемый образ ситуации.
- Снижай за: строку, понятную только внутри контекста песни; нарративный хвост; \
незаконченную мысль; строку настолько общую, что подходит ко всему и теряет удар.
- Повышай за: небанальность, иронию, парадокс, точный бытовой образ.

Тональность — укажи поле "valence" для каждой цитаты:
  "позитив"    — надежда, достижение, радость (пусть и горькая)
  "негатив"    — боль, потеря, упадок, безнадёга
  "нейтрал"    — описание без явной окраски
  "амбивалент" — одновременно и то и другое
  ВАЖНО: оценивай тон самой строки как реплики, а не тему, которой она касается. \
Сарказм про хорошее = амбивалент или негатив.

Дубли: если цитата идентична или почти идентична другой в этом же батче \
(кроме пробелов/пунктуации) — укажи "duplicate_of": "<id оригинала>". \
Иначе "duplicate_of": null.

Слияние: если в записи есть поле [следующая: ...] и вместе две строки образуют \
более сильную реплику — ставь "merge_with_next": true.

Ответь ТОЛЬКО валидным JSON-массивом. Первый символ «[», последний «]». \
Никакого текста до или после. Используй только id из входных данных — не выдумывай. \
Поле reason — не длиннее 8 слов.

[
  {"id": "...", "score": 7, "reason": "точный образ одиночества", \
"valence": "негатив", "duplicate_of": null, "merge_with_next": false},
  ...
]
"""


def build_prompt(batch: list[dict]) -> str:
    lines = []
    for item in batch:
        text = item["text"].replace("\n", " / ")
        has_next = item.get("has_next", False)
        next_text = ""
        if has_next:
            next_text = f'  [следующая: {item["next_text"].replace(chr(10), " / ")}]'
        lines.append(f'id="{item["chunk_id"]}" text="{text}"{next_text}')
    return SCORE_PROMPT + "\n\nЦитаты:\n" + "\n".join(lines)


def parse_response(raw: str) -> list[dict]:
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def submit_batch(client: anthropic.Anthropic, mini_batches: list[list[dict]], model: str) -> str:
    """Отправляет все мини-батчи одним Batch-запросом. Возвращает batch_id."""
    requests = []
    for i, mini in enumerate(mini_batches):
        requests.append(
            anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
                custom_id=f"mini_{i}",
                params={
                    "model": model,
                    "max_tokens": 1024,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": build_prompt(mini)}],
                },
            )
        )
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def poll_batch(client: anthropic.Anthropic, batch_id: str, poll_interval: int = 60) -> None:
    """Ждёт завершения батча, выводя статус каждые poll_interval секунд."""
    print(f"Ожидаем завершения батча {batch_id}...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        print(
            f"  [{time.strftime('%H:%M:%S')}] {status}: "
            f"processing={counts.processing}, succeeded={counts.succeeded}, "
            f"errored={counts.errored}"
        )
        if status == "ended":
            return
        time.sleep(poll_interval)


def collect_results(
    client: anthropic.Anthropic,
    batch_id: str,
    mini_batches: list[list[dict]],
    scores: dict,
) -> int:
    """Парсит результаты батча, пишет в scores. Возвращает число успешно оценённых цитат."""
    # Строим lookup: mini_{i} → список item-ов
    by_idx: dict[int, list[dict]] = {i: mb for i, mb in enumerate(mini_batches)}

    new_scored = 0
    for result in client.messages.batches.results(batch_id):
        idx = int(result.custom_id.split("_", 1)[1])
        mini = by_idx.get(idx, [])

        if result.result.type != "succeeded":
            err = getattr(result.result, "error", result.result.type)
            print(f"  [ошибка] mini_{idx}: {err}. Ставим нейтральную оценку 5.")
            for item in mini:
                if item["chunk_id"] not in scores:
                    scores[item["chunk_id"]] = {
                        "score": 5,
                        "reason": f"batch error: {err}",
                        "valence": "нейтрал",
                        "duplicate_of": None,
                        "merge_with_next": False,
                    }
            continue

        raw = result.result.message.content[0].text
        try:
            parsed = parse_response(raw)
        except json.JSONDecodeError as e:
            print(f"  [JSON ошибка] mini_{idx}: {e}. Ставим нейтральную оценку 5.")
            for item in mini:
                if item["chunk_id"] not in scores:
                    scores[item["chunk_id"]] = {
                        "score": 5,
                        "reason": "json parse error",
                        "valence": "нейтрал",
                        "duplicate_of": None,
                        "merge_with_next": False,
                    }
            continue

        scored_ids: set[str] = set()
        for r in parsed:
            cid = r.get("id") or r.get("chunk_id")
            if cid:
                scores[cid] = {
                    "score": int(r.get("score", 5)),
                    "reason": r.get("reason", ""),
                    "valence": r.get("valence", "нейтрал"),
                    "duplicate_of": r.get("duplicate_of") or None,
                    "merge_with_next": bool(r.get("merge_with_next", False)),
                }
                scored_ids.add(cid)
                new_scored += 1

        for item in mini:
            if item["chunk_id"] not in scored_ids:
                scores[item["chunk_id"]] = {
                    "score": 5,
                    "reason": "нет ответа от модели",
                    "valence": "нейтрал",
                    "duplicate_of": None,
                    "merge_with_next": False,
                }

    return new_scored


def print_distribution(scores: dict) -> None:
    dist: dict[int, int] = {}
    for v in scores.values():
        s = v["score"]
        dist[s] = dist.get(s, 0) + 1
    print("Распределение оценок:")
    for s in sorted(dist):
        bar = "█" * (dist[s] // 5)
        print(f"  {s:2d}: {dist[s]:4d} {bar}")
    merge_count = sum(1 for v in scores.values() if v.get("merge_with_next"))
    print(f"Кандидатов на слияние: {merge_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                        help="Цитат на один LLM-запрос внутри батча")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true",
                        help="Продолжить: переиспользовать батч из batch_state.json")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Интервал опроса статуса батча (секунды)")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY не задан в .env")

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Читаем датасет: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    print(f"Цитат: {len(df)}, треков: {df['track_name'].nunique()}")

    # Загружаем уже посчитанные оценки
    scores: dict = {}
    if SCORES_PATH.exists():
        scores = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
        if scores:
            print(f"Загружено существующих оценок: {len(scores)}")

    # Строим lookup «следующая цитата» по треку
    next_quote: dict[str, dict] = {}
    for _, group in df.groupby("track_name", sort=False):
        ids = group["chunk_id"].tolist()
        texts = group["text"].tolist()
        for i in range(len(ids) - 1):
            next_quote[ids[i]] = {"chunk_id": ids[i + 1], "text": texts[i + 1]}

    # Если resume — пробуем подхватить существующий батч
    state: dict = {}
    if args.resume and STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        print(f"Резьюм: найден batch_id={state.get('batch_id')}")

    if "batch_id" in state and "mini_batches" in state:
        batch_id = state["batch_id"]
        mini_batches = state["mini_batches"]
        print(f"Опрашиваем существующий батч {batch_id}...")
        poll_batch(client, batch_id, args.poll_interval)
    else:
        # Строим мини-батчи только для ещё не оценённых цитат
        pending = []
        for row in df.to_dict("records"):
            if row["chunk_id"] in scores:
                continue
            nxt = next_quote.get(row["chunk_id"])
            pending.append({
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "has_next": nxt is not None,
                "next_text": nxt["text"] if nxt else "",
            })

        total = len(pending)
        if total == 0:
            print("Все цитаты уже оценены.")
            print_distribution(scores)
            print(f"Результат: {SCORES_PATH}")
            return

        mini_batches = [
            pending[i:i + args.batch_size]
            for i in range(0, total, args.batch_size)
        ]
        print(
            f"К оценке: {total} цитат → {len(mini_batches)} мини-батчей "
            f"(batch_size={args.batch_size}), модель={args.model}"
        )
        print("Отправляем в Anthropic Batches API...")

        batch_id = submit_batch(client, mini_batches, args.model)
        print(f"Батч создан: {batch_id}")

        # Сохраняем состояние для resume
        STATE_PATH.write_text(
            json.dumps({"batch_id": batch_id, "mini_batches": mini_batches},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        poll_batch(client, batch_id, args.poll_interval)

    # Собираем результаты
    print("Собираем результаты...")
    new_count = collect_results(client, batch_id, mini_batches, scores)

    # Сохраняем оценки
    SCORES_PATH.write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Удаляем файл состояния — батч завершён
    if STATE_PATH.exists():
        STATE_PATH.unlink()

    print(f"\nГотово! Оценено {len(scores)} цитат (новых в этом запуске: {new_count}).")
    print_distribution(scores)
    print(f"Результат: {SCORES_PATH}")


if __name__ == "__main__":
    main()
