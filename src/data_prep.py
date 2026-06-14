import re
import argparse
import pandas as pd
from pathlib import Path

# Кровосток пишет рифмованными двустишиями, поэтому базовая смысловая
# единица — куплет из двух строк. Берём НЕперекрывающиеся пары, чтобы не
# плодить обрывки на половине фразы (как делал старый sliding window).
LINES_PER_CHUNK = 2

# Минимальная длина нормализованного текста, чтобы отсечь пустышки/мусор.
MIN_CHUNK_LEN = 8


def clean_line(line: str) -> str:
    """Схлопывает лишние пробелы."""
    return " ".join(line.split())


def is_section_marker(line: str) -> bool:
    """Строки вида [Припев], (x2) и т.п. — это разметка, а не текст."""
    return bool(re.match(r"^\s*[\[\(].*[\]\)]\s*$", line))


def normalize(text: str) -> str:
    """Ключ для дедупликации: регистр, пунктуация и пробелы не важны."""
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def chunk_lines(lines: list[str], size: int = LINES_PER_CHUNK) -> list[str]:
    """Режет строки на НЕперекрывающиеся группы по `size` строк.

    Если в конце остаётся одинокая строка-«сирота» — приклеиваем её к
    предыдущему чанку, чтобы не было висящего огрызка.
    """
    groups: list[list[str]] = []
    i = 0
    while i < len(lines):
        group = lines[i:i + size]
        if len(group) < size and groups:
            groups[-1].extend(group)
            break
        groups.append(group)
        i += size
    return [" \n ".join(g) for g in groups]


def process_corpus(input_dir: Path) -> pd.DataFrame:
    """Обходит сырые тексты, режет на куплеты и собирает DataFrame.

    Попутно глобально дедуплицирует одинаковые куплеты: это убирает
    многократно повторяющиеся припевы и дубликаты одной песни, лежащие
    под разными именами файлов.
    """
    rows = []
    seen: set[str] = set()

    for filepath in sorted(input_dir.glob("*.txt")):
        track_name = filepath.stem
        raw_text = filepath.read_text(encoding="utf-8")

        lines = [
            clean_line(line)
            for line in raw_text.split("\n")
            if line.strip() and not is_section_marker(line)
        ]

        for idx, chunk in enumerate(chunk_lines(lines)):
            key = normalize(chunk)
            if len(key) < MIN_CHUNK_LEN or key in seen:
                continue
            seen.add(key)
            rows.append({
                "track_name": track_name,
                "chunk_id": f"{track_name}_part_{idx + 1}",
                "text": chunk,
            })

    return pd.DataFrame(rows)


def main():
    base_dir = Path(__file__).resolve().parent.parent
    default_input = base_dir / "data" / "raw"
    default_output = base_dir / "data" / "processed" / "dataset.parquet"

    parser = argparse.ArgumentParser(description="Подготовка текстов для KrovostokLM")
    parser.add_argument("--input", type=str, default=str(default_input), help="Директория с .txt файлами")
    parser.add_argument("--output", type=str, default=str(default_output), help="Путь для сохранения датасета")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ОШИБКА] Директория {input_path} не найдена. Положите сырые тексты в data/raw/")
        return

    print("Начинаю обработку корпуса текстов...")
    df = process_corpus(input_path)

    if df.empty:
        print("[ОШИБКА] Тексты не найдены или они пустые.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"Успешно! Уникальных куплетов: {len(df)} (из {df['track_name'].nunique()} треков)")
    print(f"Датасет сохранён в: {output_path}")


if __name__ == "__main__":
    main()
