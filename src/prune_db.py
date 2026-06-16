"""
Причёсывание базы цитат по результатам оценки.

Читает data/scores.json, фильтрует dataset.parquet до топ-N цитат,
применяет слияния (merge), обновляет ChromaDB (удаляет выброшенные ID).

НЕ требует переиндексации — просто удаляет записи из ChromaDB, вектора
оставшихся цитат уже там есть.

Рекомендуемый порядок работы:
  1. python src/prune_db.py --update-metadata   # записать score+valence в ChromaDB (не удалять)
  2. python src/batch_test.py --min-score 5     # тестировать на разных порогах
  3. python src/prune_db.py --min-score 5       # физически удалить мусор, когда порог выбран

Остальные флаги:
  python src/prune_db.py --dry-run              # посмотреть что останется, не трогая файлы
  python src/prune_db.py --keep 400             # оставить топ-400
  python src/prune_db.py --dedup                # убрать текстовые дубли
  python src/prune_db.py --apply-merges         # применить слияния (требует переиндексации)
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "dataset.parquet"
SCORES_PATH = BASE_DIR / "data" / "scores.json"
DB_PATH = BASE_DIR / "data" / "vector_db"
COLLECTION_NAME = "krovostok_quotes"


def load_scores(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"scores.json не найден: {path}\nСначала запусти: python src/score_quotes.py")
    return json.loads(path.read_text(encoding="utf-8"))


def find_text_duplicates(df: pd.DataFrame) -> set[str]:
    """Возвращает set chunk_id записей-дублей (по нормализованному тексту).

    Первое вхождение сохраняется, повторы — в возвращаемый set.
    """
    seen: dict[str, str] = {}
    to_remove: set[str] = set()
    for _, row in df.iterrows():
        norm = re.sub(r"[^\w]", "", row["text"].lower())
        if norm in seen:
            to_remove.add(row["chunk_id"])
        else:
            seen[norm] = row["chunk_id"]
    return to_remove


def select_keepers(df: pd.DataFrame, scores: dict, keep: int | None, min_score: int | None) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df["chunk_id"].map(lambda cid: scores.get(cid, {}).get("score", 5))

    if min_score is not None:
        df = df[df["score"] >= min_score]
        print(f"Фильтр min_score>={min_score}: осталось {len(df)} цитат")

    # Сортируем по оценке убывая, внутри оценки — трек алфавитно (стабильность)
    df = df.sort_values(["score", "track_name"], ascending=[False, True])

    if keep is not None and len(df) > keep:
        df = df.head(keep)
        print(f"Обрезка до топ-{keep}: осталось {len(df)} цитат")

    return df


def apply_merges(df: pd.DataFrame, scores: dict) -> pd.DataFrame:
    """Сливает пары соседних цитат, помеченных merge_with_next=True.

    Merged-цитата получает id первой, текст = text1 + '\n' + text2,
    оценку = max(score1, score2). Вторая запись удаляется.
    ChromaDB при этом потребует полной переиндексации — предупреждаем.
    """
    to_merge: set[str] = set()
    for cid, info in scores.items():
        if info.get("merge_with_next"):
            to_merge.add(cid)

    if not to_merge:
        print("Кандидатов на слияние нет.")
        return df

    df = df.reset_index(drop=True)
    id_to_idx = {row["chunk_id"]: i for i, row in df.iterrows()}

    merged_away: set[str] = set()
    updates: list[dict] = []

    for cid in list(to_merge):
        if cid in merged_away:
            continue
        idx = id_to_idx.get(cid)
        if idx is None:
            continue

        track = df.at[idx, "track_name"]
        next_candidates = df[(df["track_name"] == track) & (df.index > idx)].head(1)
        if next_candidates.empty:
            continue

        next_idx = next_candidates.index[0]
        next_cid = df.at[next_idx, "chunk_id"]

        merged_text = df.at[idx, "text"] + " \n " + df.at[next_idx, "text"]
        merged_score = max(df.at[idx, "score"], df.at[next_idx, "score"])

        updates.append((idx, merged_text, merged_score))
        merged_away.add(next_cid)

    for idx, text, score in updates:
        df.at[idx, "text"] = text
        df.at[idx, "score"] = score

    df = df[~df["chunk_id"].isin(merged_away)].reset_index(drop=True)
    print(f"Слито пар: {len(updates)}, убрано дублей: {len(merged_away)}")
    print("  ⚠️  Применены слияния → нужна полная переиндексация: python src/vectorize.py")
    return df


def update_chroma_metadata(df: pd.DataFrame, scores: dict, dry_run: bool) -> None:
    """Записывает score+valence в метаданные ChromaDB без удаления документов.

    После этого ретривер может фильтровать по where={"score": {"$gte": N}}.
    """
    print(f"ChromaDB: обновляем метаданные для {len(df)} документов...")
    if dry_run:
        print("  [dry-run] Пропускаем обновление метаданных.")
        return

    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"  ⚠️  ChromaDB недоступна: {e}")
        return

    batch_size = 200
    rows = df.to_dict("records")
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        ids = [r["chunk_id"] for r in chunk]
        metadatas = [
            {
                "track_name": r["track_name"],
                "score": int(scores.get(r["chunk_id"], {}).get("score", 5)),
                "valence": scores.get(r["chunk_id"], {}).get("valence", "нейтрал"),
            }
            for r in chunk
        ]
        collection.update(ids=ids, metadatas=metadatas)
        print(f"  обновлено {min(i + batch_size, len(rows))}/{len(rows)}")

    print(f"  ✅ Метаданные обновлены. Теперь можно фильтровать по score в ретривере.")


def update_chroma(keepers_df: pd.DataFrame, all_ids: list[str], dry_run: bool) -> None:
    keeper_ids = set(keepers_df["chunk_id"].tolist())
    to_delete = [cid for cid in all_ids if cid not in keeper_ids]

    print(f"ChromaDB: удалить {len(to_delete)} записей, оставить {len(keeper_ids)}")
    if not to_delete:
        print("  Нечего удалять.")
        return
    if dry_run:
        print("  [dry-run] Пропускаем удаление из ChromaDB.")
        return

    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"  ⚠️  ChromaDB недоступна: {e}\n  Только parquet обновлён.")
        return

    batch = 200
    for i in range(0, len(to_delete), batch):
        chunk = to_delete[i:i + batch]
        collection.delete(ids=chunk)
        print(f"  удалено {min(i + batch, len(to_delete))}/{len(to_delete)}")

    count = collection.count()
    print(f"  ChromaDB после чистки: {count} записей")


def print_stats(df: pd.DataFrame, scores: dict) -> None:
    print("\n=== Топ-20 цитат по оценке ===")
    top = df.nlargest(20, "score")
    for _, row in top.iterrows():
        info = scores.get(row["chunk_id"], {})
        reason = info.get("reason", "")
        valence = info.get("valence", "")
        preview = row["text"].replace("\n", " / ")[:80]
        valence_tag = f" [{valence}]" if valence else ""
        print(f"  [{row['score']:2d}]{valence_tag} {preview}")
        if reason:
            print(f"       ↳ {reason[:100]}")

    print("\n=== Распределение тональности ===")
    valence_counts: dict[str, int] = {}
    for cid in df["chunk_id"]:
        v = scores.get(cid, {}).get("valence", "нет данных")
        valence_counts[v] = valence_counts.get(v, 0) + 1
    for v, cnt in sorted(valence_counts.items(), key=lambda x: -x[1]):
        print(f"  {v}: {cnt}")

    print("\n=== Треки после фильтрации ===")
    tc = df.groupby("track_name").size().sort_values(ascending=False)
    for track, cnt in tc.head(20).items():
        print(f"  {track}: {cnt}")
    if len(tc) > 20:
        print(f"  ... и ещё {len(tc) - 20} треков")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", type=int, default=500, help="Сколько лучших цитат оставить (0 = не ограничивать)")
    parser.add_argument("--min-score", type=int, default=None, help="Минимальная оценка для попадания в базу")
    parser.add_argument("--update-metadata", action="store_true",
                        help="Записать score+valence в ChromaDB без удаления — для тестирования порогов")
    parser.add_argument("--dedup", action="store_true", help="Убрать текстовые дубли")
    parser.add_argument("--apply-merges", action="store_true", help="Применить слияния (требует переиндексации)")
    parser.add_argument("--dry-run", action="store_true", help="Показать что получится, не изменяя файлы")
    args = parser.parse_args()

    print(f"Читаем датасет: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    all_ids = df["chunk_id"].tolist()
    print(f"Исходно: {len(df)} цитат из {df['track_name'].nunique()} треков")

    print(f"Читаем оценки: {SCORES_PATH}")
    scores = load_scores(SCORES_PATH)
    scored_count = sum(1 for cid in all_ids if cid in scores)
    print(f"Оценено: {scored_count}/{len(df)}")

    # Режим --update-metadata: просто пишем метаданные и выходим
    if args.update_metadata:
        df_with_scores = df.copy()
        df_with_scores["score"] = df_with_scores["chunk_id"].map(
            lambda cid: scores.get(cid, {}).get("score", 5)
        )
        update_chroma_metadata(df_with_scores, scores, dry_run=args.dry_run)
        return

    # Дедупликация по тексту
    if args.dedup:
        dupes = find_text_duplicates(df)
        if dupes:
            print(f"Текстовых дублей: {len(dupes)} — убираем")
            df = df[~df["chunk_id"].isin(dupes)].reset_index(drop=True)
            all_ids = df["chunk_id"].tolist()
        else:
            print("Текстовых дублей не найдено.")

    keep = args.keep if args.keep > 0 else None

    # Отбор
    keepers_df = select_keepers(df, scores, keep=keep, min_score=args.min_score)
    print(f"\nИтого отобрано: {len(keepers_df)} цитат из {keepers_df['track_name'].nunique()} треков")

    if args.apply_merges:
        keepers_df = apply_merges(keepers_df, scores)

    print_stats(keepers_df, scores)

    if args.dry_run:
        print("\n[dry-run] Файлы НЕ изменены.")
        return

    # Сохраняем отфильтрованный parquet
    out_df = keepers_df.drop(columns=["score"], errors="ignore")
    out_df.to_parquet(DATASET_PATH, index=False)
    print(f"\nParquet обновлён: {DATASET_PATH} ({len(out_df)} строк)")

    # Чистим ChromaDB
    update_chroma(keepers_df, all_ids, dry_run=False)

    print("\n✅ Готово.")
    if not args.apply_merges:
        print("   Переиндексация НЕ нужна — вектора оставшихся цитат уже в базе.")
    else:
        print("   Запусти vectorize.py для переиндексации merged-цитат.")


if __name__ == "__main__":
    main()
