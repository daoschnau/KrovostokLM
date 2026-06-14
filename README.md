# KrovostokLM

Telegram-бот «психологической поддержки»: отвечает на любую жизненную ситуацию идеально подобранной цитатой из текстов группы **«Кровосток»**.

Опиши боль, победу или экзистенциальный тупик — получишь хлёсткий панчлайн из дискографии Шило и компании с указанием трека.

```
Пользователь: сегодня уволился сам, хлопнул дверью, и теперь не знаю —
              это была смелость или глупость

KrovostokLM:  💬 Цитата:
              Ну ты чё бля мудак чтоле?
              И нахуя ты это сделал?
              🎵 Трек: autro
```

## Как это работает

RAG-пайплайн с одноплечим HyDE и LLM-реранкингом. Все «умные» операции — по API, на сервере нет ни одной локальной модели.

```
Сообщение пользователя
        │
        ▼
[1] HyDE — Groq (llama-3.3-70b) формулирует короткую «суровую мораль»
    в духе Кровостока. Тон зависит от валентности запроса:
    • негатив → фатализм, уличная философия
    • позитив (гордость, победа) → дерзость, мрачноватое торжество
        │
        ▼
[2] Векторный поиск — гипотеза векторизуется через DeepInfra (e5-large)
    → ChromaDB → топ-12 похожих цитат из ~1 200 чанков текстов
        │
        ▼
[3] Реранкинг — Groq выбирает одну цитату по правилам:
    • знак эмоции должен совпадать с ситуацией
    • для трезвости — не брать цитаты, прославляющие употребление
    • для горя — выбирать тихую/печальную, а не шок-комичную
    • никаких универсальных «магнит-цитат»
        │
        ▼
[4] Анти-магнит — из кандидатов убираются треки, которые уже
    звучали в этом чате (per-chat memory, скользящее окно 5 треков)
        │
        ▼
Ответ в Telegram (aiogram, long polling)
```

**Устойчивость к сбоям.** Groq free tier ограничен дневным бюджетом токенов. При любом сбое или rate-limit бот не падает: включается cooldown и до его истечения ответы идут напрямую с лучшего e5-кандидата (без LLM, мгновенно). Повторные одинаковые запросы кэшируются — бюджет расходуется только на уникальные ситуации.

## Стек

| Компонент | Технология |
|---|---|
| Telegram | [aiogram 3](https://docs.aiogram.dev/) (long polling) |
| LLM — HyDE и реранкинг | Groq API (`llama-3.3-70b-versatile`), бесплатно |
| Эмбеддинги | `intfloat/multilingual-e5-large` через [DeepInfra API](https://deepinfra.com/) |
| Векторная база | [ChromaDB](https://www.trychroma.com/) (persistent, 1024-мер, в репозитории) |
| Деплой | Docker → [Render](https://render.com/) (Background Worker) |

## Структура проекта

```
KrovostokLM/
├── data/
│   ├── raw/                 # Сырые тексты песен (.txt), один файл — один трек
│   ├── processed/           # dataset.parquet — нарезанные чанки (~1 200 двустиший)
│   └── vector_db/           # Готовая база ChromaDB (1024-мер, e5-large)
├── src/
│   ├── bot.py               # Telegram-бот, точка входа
│   ├── core_groq.py         # Главный пайплайн: HyDE → поиск → реранкинг → анти-магнит
│   ├── core_hf.py           # Retrieval-ядро: эмбеддинг (DeepInfra/local) + ChromaDB
│   ├── batch_test.py        # Батч-прогон тестовых запросов, --backend hf|groq
│   ├── vectorize.py         # Пересборка базы (DeepInfra API или локальная модель)
│   ├── data_prep.py         # Чанкование: sliding window по 2 строки
│   └── scrape_lyrics.py     # Скрейпер текстов
├── data/test_queries.txt    # 30 тестовых сценариев (8 эмоциональных категорий)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt         # Рантайм: aiogram, chromadb, groq, requests
├── requirements-local.txt   # Дев: + sentence-transformers, pandas, pyarrow
└── .env.example
```

## Запуск и деплой

### Переменные окружения

Скопируй `.env.example` в `.env` и заполни:

```env
TG_BOT_TOKEN=...           # токен бота от @BotFather
GROQ_API_KEY=...           # бесплатно на console.groq.com/keys
DEEPINFRA_API_KEY=...      # deepinfra.com/dash/api_keys (нужна карта, трата — центы)
```

Опциональные переменные описаны в `.env.example`.

### Локально

```bash
pip install -r requirements.txt
python src/bot.py
```

### Docker Compose (сервер)

```bash
docker compose up -d --build
docker compose logs -f
```

### Деплой на Render

Репозиторий подключается как **Background Worker** с типом **Docker**. Переменные `TG_BOT_TOKEN`, `GROQ_API_KEY`, `DEEPINFRA_API_KEY` задаются во вкладке *Environment*. Auto-Deploy с ветки `main` пересобирает сервис на каждый пуш.

Бот работает через long polling — порт и вебхуки не нужны.

## Пересборка базы цитат

Нужна только при изменении корпуса текстов. Требует `requirements-local.txt`.

```bash
python src/scrape_lyrics.py   # собрать тексты в data/raw/
python src/data_prep.py       # нарезать на двустишия → data/processed/dataset.parquet
python src/vectorize.py       # векторизовать через DeepInfra → data/vector_db/
```

После этого закоммить `data/vector_db` и запушить — деплой подхватит.

**Важно:** база и запросы должны векторизоваться одной и той же моделью через один и тот же бэкенд. По умолчанию оба используют DeepInfra (`EMBED_BACKEND=api`). Для офлайн-работы с локальной моделью — `EMBED_BACKEND=local` (нужен `sentence-transformers`, ~2.5 ГБ).

## Тестирование качества

```bash
# Чистый retrieval (без LLM)
python src/batch_test.py --backend hf

# Полный пайплайн с HyDE и реранкингом
python src/batch_test.py --backend groq
```

Результаты сохраняются в `logs/batch_<backend>_<timestamp>.txt`. Тестовый датасет — 30 сценариев в 8 эмоциональных категориях: работа, отношения, здоровье, спорт, тревога, потеря, скука, гордость.

## Дисклеймер

Проект сделан с любовью к творчеству группы «Кровосток» в образовательных и развлекательных целях. Права на тексты принадлежат их авторам. Бот не является заменой психологической помощи — хотя иногда работает лучше.
