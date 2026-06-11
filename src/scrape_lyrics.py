import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Конфигурация
BASE_URL = "https://lyricsworld.ru"
START_PAGES = [
    "https://lyricsworld.ru/Krovostok/",
    "https://lyricsworld.ru/Krovostok/P2.html"
]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw')

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

def clean_filename(title: str) -> str:
    """Очищает название от мусора, транслитерирует и форматирует для ФС."""
    # Словарь транслитерации
    mapping = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
        ' ': '_'
    }
    
    # Переводим в нижний регистр
    title = title.lower()
    
    # Вычищаем типовой мусор с сайта
    title = title.replace('кровосток - ', '').replace(' текст песни', '')
    title = title.replace('кровосток ', '')
    
    # Транслитерируем
    transliterated = ''.join(mapping.get(c, c) for c in title)
    
    # Оставляем только латиницу, цифры и подчеркивания
    clean_name = re.sub(r'[^a-z0-9_]', '', transliterated)
    
    # Убираем дублирующиеся подчеркивания (если были лишние пробелы)
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    
    return clean_name if clean_name else "unknown_track"

def get_song_links() -> list:
    """Собирает все ссылки на песни с указанных страниц."""
    song_links = set()
    
    for page_url in START_PAGES:
        print(f"Сбор ссылок со страницы: {page_url}")
        try:
            response = requests.get(page_url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if ('/Krovostok/' in href or '/lyrics/' in href) and href.endswith('.html') and 'P2' not in href:
                    full_url = urljoin(BASE_URL, href)
                    song_links.add(full_url)
                    
            time.sleep(1)
        except Exception as e:
            print(f"[ОШИБКА] Не удалось загрузить {page_url}: {e}")
            
    return list(song_links)

def parse_and_save_lyrics(url: str):
    """Парсит страницу песни и сохраняет текст в .txt файл."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Получаем сырое название
        title_tag = soup.find('h1')
        raw_title = title_tag.get_text(strip=True) if title_tag else url.split('/')[-1].replace('.html', '')
        
        # Форматируем название файла
        safe_title = clean_filename(raw_title)
        
        lyrics_div = soup.find('p', id='songLyricsDiv')
        if not lyrics_div:
            lyrics_div = soup.find('div', id='text')
            
        if lyrics_div:
            # Магия: separator='\n' заставит BeautifulSoup вставлять перенос строки 
            # вместо тегов <br>, <p> и других блочных элементов.
            lyrics_text = lyrics_div.get_text(separator='\n', strip=True)
            
            # На всякий случай схлопываем множественные пустые строки в одну
            lyrics_text = re.sub(r'\n{2,}', '\n', lyrics_text)
            
            file_path = os.path.join(OUTPUT_DIR, f"{safe_title}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(lyrics_text)
                
            print(f"[OK] Сохранено: {safe_title}.txt")
        else:
            print(f"[ПРОПУСК] Текст не найден: {url}")
            
    except Exception as e:
        print(f"[ОШИБКА] Сбой при парсинге {url}: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Начинаем процесс парсинга текстов «Кровосток»...")
    links = get_song_links()
    print(f"Найдено уникальных ссылок на треки: {len(links)}")
    
    if not links:
        return

    for idx, link in enumerate(links, 1):
        print(f"[{idx}/{len(links)}] Обработка: {link}")
        parse_and_save_lyrics(link)
        time.sleep(1)
        
    print("Парсинг успешно завершен. Тексты лежат в data/raw/")

if __name__ == "__main__":
    main()