"""
Оценка качества цитат в базе по шкале 1-10.

Для каждой цитаты Groq выставляет оценку:
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

Запуск:
  python src/score_quotes.py
  python src/score_quotes.py --batch-size 8 --model llama-3.1-8b-instant
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
from groq import Groq, RateLimitError

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "dataset.parquet"
SCORES_PATH = BASE_DIR / "data" / "scores.json"

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_BATCH = 8


SCORE_PROMPT = """\
Ты — строгий редактор антологии русского рэпа. Оцени каждую цитату из песен \
группы «Кровосток» по шкале 1–10:

10 — острый, законченный афоризм: можно вставить в разговор как меткий ответ.
7–9 — сильная строфа: смысл есть, образ яркий, почти афоризм.
4–6 — проходная: что-то есть, но либо мысль не закончена, либо образ слабый.
1–3 — мусор: обрывок, бессмыслица, чисто описательная строка без удара.

Правила:
- Мат и жёсткий контент НЕ снижают оценку — это стиль группы.
- Снижай за: незаконченную мысль, слишком конкретное имя/топоним без отдачи,
  пустую дескрипцию без метафоры, явный обрывок фразы.
- Повышай за: универсальность, хлёсткость, философский или иронический удар.

Ещё оцени: если эта цитата И СЛЕДУЮЩАЯ за ней (из того же трека) вместе \
образуют более сильный афоризм — поставь merge_with_next: true.

Ответь ТОЛЬКО валидным JSON (без markdown-обёртки):
[
  {"id": "chunk_id_1", "score": 8, "reason": "...", "merge_with_next": false},
  ...
]
"""


def build_batch_prompt(batch: list[dict]) -> str:
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
    # снимаем markdown-обёртку если модель всё равно добавила
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def score_batch(client: Groq, batch: list[dict], model: str, retries: int = 3) -> list[dict]:
    prompt = build_batch_prompt(batch)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return parse_response(resp.choices[0].message.content)
        except RateLimitError:
            wait = 60 * (attempt + 1)
            print(f"  [rate limit] ждём {wait}с...")
            time.sleep(wait)
        except json.JSONDecodeError as e:
            print(f"  [JSON ошибка] попытка {attempt+1}: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(5)
        except Exception as e:
            print(f"  [ошибка] попытка {attempt+1}: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(10)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true", help="Продолжить с прерванного места")
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY не задан в .env")

    client = Groq(api_key=api_key, max_retries=0)

    print(f"Читаем датасет: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    print(f"Цитат: {len(df)}, треков: {df['track_name'].nunique()}")

    # Загружаем уже посчитанные оценки если resume
    scores: dict = {}
    if args.resume and SCORES_PATH.exists():
        scores = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
        print(f"Резьюм: уже оценено {len(scores)} цитат")

    # Строим lookup «следующая цитата» для merge-анализа
    next_quote: dict[str, dict] = {}
    by_track = df.groupby("track_name", sort=False)
    for track, group in by_track:
        ids = group["chunk_id"].tolist()
        texts = group["text"].tolist()
        for i in range(len(ids) - 1):
            next_quote[ids[i]] = {"chunk_id": ids[i + 1], "text": texts[i + 1]}

    # Готовим батчи (пропускаем уже оценённые)
    rows = df.to_dict("records")
    pending = []
    for row in rows:
        if row["chunk_id"] in scores:
            continue
        nxt = next_quote.get(row["chunk_id"])
        item = {
            "chunk_id": row["chunk_id"],
            "text": row["text"],
            "has_next": nxt is not None,
            "next_text": nxt["text"] if nxt else "",
        }
        pending.append(item)

    total = len(pending)
    print(f"К оценке: {total} цитат, батч={args.batch_size}, модель={args.model}")
    if total == 0:
        print("Всё уже оценено.")
        return

    done = 0
    for i in range(0, total, args.batch_size):
        batch = pending[i:i + args.batch_size]
        print(f"  [{done}/{total}] батч {i//args.batch_size + 1}...", end=" ", flush=True)

        try:
            results = score_batch(client, batch, args.model)
        except Exception as e:
            print(f"\n[КРИТИЧНО] батч упал: {e}. Сохраняем прогресс и выходим.")
            SCORES_PATH.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
            return

        # Записываем результаты
        scored_ids = set()
        for r in results:
            cid = r.get("id") or r.get("chunk_id")
            if cid:
                scores[cid] = {
                    "score": int(r.get("score", 5)),
                    "reason": r.get("reason", ""),
                    "merge_with_next": bool(r.get("merge_with_next", False)),
                }
                scored_ids.add(cid)

        # Если модель не вернула оценку для каких-то цитат — ставим нейтральную 5
        for item in batch:
            if item["chunk_id"] not in scored_ids:
                scores[item["chunk_id"]] = {"score": 5, "reason": "нет ответа от модели", "merge_with_next": False}

        done += len(batch)
        print(f"OK ({len(results)} оценок)")

        # Сохраняем после каждого батча — чтобы не терять прогресс
        SCORES_PATH.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nГотово! Оценено {len(scores)} цитат.")
    dist = {}
    for v in scores.values():
        s = v["score"]
        dist[s] = dist.get(s, 0) + 1
    print("Распределение оценок:")
    for s in sorted(dist):
        bar = "█" * (dist[s] // 5)
        print(f"  {s:2d}: {dist[s]:4d} {bar}")
    merge_count = sum(1 for v in scores.values() if v.get("merge_with_next"))
    print(f"Кандидатов на слияние: {merge_count}")
    print(f"Результат: {SCORES_PATH}")


if __name__ == "__main__":
    main()
