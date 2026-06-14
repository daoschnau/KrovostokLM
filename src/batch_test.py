"""
Батч-прогон тестовых запросов через бота.

Читает файл с пронумерованными запросами (строки вида "12. текст запроса"),
прогоняет каждый через find_quote() и сохраняет результаты в лог.

Запуск из корня проекта:
    python src/batch_test.py путь/к/queries.txt
    python src/batch_test.py путь/к/queries.txt --out logs/my_run.txt

Если путь к файлу запросов не указан, берётся data/test_queries.txt.
Лог по умолчанию: logs/batch_<timestamp>.txt
"""
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Позволяет запускать скрипт напрямую без установки пакета
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_hf import find_quote, EMBEDDING_MODEL

base_dir = Path(__file__).resolve().parent.parent

# Строка-запрос: начинается с номера и точки, напр. "7. лучший друг занял денег"
QUERY_RE = re.compile(r"^\s*(\d+)\.\s*(.+)$")
# Заголовок секции: "# --- Работа и деньги ---"
SECTION_RE = re.compile(r"^#\s*-+\s*(.+?)\s*-+\s*$")


def parse_queries(path: Path):
    """Возвращает список (номер, секция, текст). Комментарии без номера пропускаются,
    но строки-заголовки секций запоминаются для группировки в логе."""
    items = []
    current_section = "—"
    for line in path.read_text(encoding="utf-8").splitlines():
        section_match = SECTION_RE.match(line)
        if section_match:
            current_section = section_match.group(1)
            continue
        if line.lstrip().startswith("#"):
            continue
        query_match = QUERY_RE.match(line)
        if query_match:
            num = int(query_match.group(1))
            text = query_match.group(2).strip()
            items.append((num, current_section, text))
    return items


def main():
    parser = argparse.ArgumentParser(description="Батч-прогон тестовых запросов через бота")
    parser.add_argument(
        "queries",
        nargs="?",
        default=str(base_dir / "data" / "test_queries.txt"),
        help="Путь к файлу с пронумерованными запросами",
    )
    parser.add_argument("--out", default=None, help="Путь к файлу лога")
    args = parser.parse_args()

    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"[ОШИБКА] Файл запросов не найден: {queries_path}")
        sys.exit(1)

    items = parse_queries(queries_path)
    if not items:
        print(f"[ОШИБКА] В файле {queries_path} не найдено пронумерованных запросов")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = base_dir / "logs" / f"batch_{timestamp}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Запросов к прогону: {len(items)}")
    print(f"Модель: {EMBEDDING_MODEL}")
    print(f"Лог: {out_path}\n")

    lines = []
    lines.append("=" * 70)
    lines.append("KrovostokLM — батч-прогон тестовых запросов")
    lines.append(f"Дата:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Модель:  {EMBEDDING_MODEL}")
    lines.append(f"Запросы: {queries_path}")
    lines.append(f"Всего:   {len(items)}")
    lines.append("=" * 70)
    lines.append("")

    last_section = None
    for num, section, text in items:
        if section != last_section:
            lines.append("")
            lines.append(f"### {section}")
            lines.append("")
            last_section = section

        print(f"[{num}/{len(items)}] {text[:50]}...")
        try:
            result = find_quote(text)
            quote = result["quote"]
            track = result["track"]
        except Exception as exc:
            quote = f"[ОШИБКА: {exc}]"
            track = "—"

        lines.append(f"[{num}] ЗАПРОС:  {text}")
        lines.append(f'    ЦИТАТА:  "{quote}"')
        lines.append(f"    ТРЕК:    {track}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Готово. Результаты сохранены: {out_path}")


if __name__ == "__main__":
    main()
