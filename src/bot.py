import os
import asyncio
import traceback
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

# Импортируем наше ядро
from core import process_user_request

# Загружаем переменные окружения
load_dotenv()
TOKEN = os.getenv("TG_BOT_TOKEN")

# Инициализируем бота и диспетчер
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: types.Message) -> None:
    """Обработчик команды /start"""
    welcome_text = (
        "Привет. Это бот психологической поддержки KrovostokLM.\n\n"
        "Опиши свою ситуацию, боль или повод для гордости, а я подберу для тебя "
        "идеальную цитату из текстов Шило и компании.\n\n"
        "<i>Пиши как есть, без купюр.</i>"
    )
    await message.answer(welcome_text)

@dp.message()
async def text_handler(message: types.Message) -> None:
    """Обработчик всех входящих текстовых сообщений"""
    user_text = message.text
    
    # Отправляем сообщение о том, что бот "думает"
    processing_msg = await message.answer("<i>Анализирую ситуацию...</i>")
    
    try:
        # Поскольку наша функция из ядра синхронная, запускаем её в отдельном потоке,
        # чтобы не блокировать асинхронный цикл (event loop) Telegram-бота
        result = await asyncio.to_thread(process_user_request, user_text)
        
        # Формируем красивый ответ
        response_text = (
            f"💬 <b>Цитата:</b>\n{result['quote']}\n\n"
            f"🎵 <i>Трек: {result['track']}</i>"
        )
        
        # Удаляем сообщение "Анализирую..." и отправляем результат
        await processing_msg.delete()
        await message.answer(response_text)
        
    except Exception as e:
        print(f"[ОШИБКА БОТА] {e}")
        traceback.print_exc()
        await processing_msg.edit_text("Что-то пошло не так. База временно недоступна.")

async def main() -> None:
    print("=== Бот KrovostokLM запущен ===")
    # Запускаем поллинг (ожидание новых сообщений)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен.")