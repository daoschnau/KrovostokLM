import os
import argparse
import pandas as pd
from pathlib import Path

def clean_line(line: str) -> str:
    """Базовая очистка строки от лишних пробелов и спецсимволов."""
    return " ".join(line.split())

def chunk_text(text: str, lines_per_chunk: int = 4) -> list[str]:
    """
    Разбивает сырой текст на смысловые блоки (по умолчанию - четверостишия).
    Игнорирует пустые строки.
    """
    # Очищаем строки и убираем пустые
    lines = [clean_line(line) for line in text.split('\n') if line.strip()]
    
    chunks = []
    # Итерируемся с шагом lines_per_chunk
    for i in range(0, len(lines), lines_per_chunk):
        chunk_lines = lines[i:i + lines_per_chunk]
        # Объединяем строки в один смысловой блок, сохраняя переносы для форматирования
        chunk_text = " \n ".join(chunk_lines)
        chunks.append(chunk_text)
        
    return chunks

def process_corpus(input_dir: Path) -> pd.DataFrame:
    """
    Обходит директорию с сырыми текстами, чанкует их и собирает в единый DataFrame.
    """
    data = []
    
    for filepath in input_dir.glob('*.txt'):
        song_title = filepath.stem # Имя файла используем как название трека
        with open(filepath, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        chunks = chunk_text(raw_text)
        
        for idx, chunk in enumerate(chunks):
            data.append({
                "track_name": song_title,
                "chunk_id": f"{song_title}_part_{idx+1}",
                "text": chunk
            })
            
    return pd.DataFrame(data)

import os
import argparse
import pandas as pd
from pathlib import Path

def clean_line(line: str) -> str:
    """Базовая очистка строки от лишних пробелов и спецсимволов."""
    return " ".join(line.split())

def chunk_text(text: str, lines_per_chunk: int = 4) -> list[str]:
    """Разбивает текст на смысловые блоки."""
    lines = [clean_line(line) for line in text.split('\n') if line.strip()]
    chunks = []
    for i in range(0, len(lines), lines_per_chunk):
        chunk_lines = lines[i:i + lines_per_chunk]
        chunk_text = " \n ".join(chunk_lines)
        chunks.append(chunk_text)
    return chunks

def process_corpus(input_dir: Path) -> pd.DataFrame:
    """Обходит директорию, чанкует тексты и собирает DataFrame."""
    data = []
    for filepath in input_dir.glob('*.txt'):
        song_title = filepath.stem
        with open(filepath, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
        chunks = chunk_text(raw_text)
        for idx, chunk in enumerate(chunks):
            data.append({
                "track_name": song_title,
                "chunk_id": f"{song_title}_part_{idx+1}",
                "text": chunk
            })
    return pd.DataFrame(data)

def main():
    # Надежно определяем пути относительно самого скрипта
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

    # Сохраняем в Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"Успешно! Обработано строк: {len(df)}")
    print(f"Датасет сохранен в: {output_path}")

if __name__ == "__main__":
    main()