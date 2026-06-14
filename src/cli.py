"""
Локальный CLI для тестирования HuggingFace-direct пайплайна.

Запуск из корня проекта:
    python src/cli.py

Зависимости:
    pip install sentence-transformers chromadb
    (модель ~400 MB скачается при первом запуске и закешируется)
"""
import sys
from pathlib import Path

# Позволяет запускать скрипт напрямую без установки пакета
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core_hf import find_quote


def main():
    print("=" * 60)
    print("  КровостокLM — HuggingFace Direct Mode (без Claude)")
    print("=" * 60)
    print("Опиши ситуацию — получи цитату из Кровостока.")
    print("Для выхода введи 'exit' или нажми Ctrl+C.\n")

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nПока.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "выход", "q"}:
            print("Пока.")
            break

        result = find_quote(user_input)
        print(f'\n"{result["quote"]}"\n— {result["track"]}\n')


if __name__ == "__main__":
    main()
