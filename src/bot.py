"""
Telegram-бот KrovostokLM (aiogram, long-polling).

Тонкий слой поверх core_groq.find_quote:
  • эмбеддинг запроса  — DeepInfra (API),
  • HyDE + реранкинг   — Groq (API),
  • вектор-база        — локальный ChromaDB (data/vector_db, едет в образе).
На сервере не крутится ни одна локальная модель — только сетевые вызовы.

session_id для анти-магнит памяти = chat.id, поэтому повторы треков
изолированы по пользователю (цитаты одного юзера не влияют на других).
"""
import os
import asyncio
import traceback

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

from core_groq import find_quote

load_dotenv()
TOKEN = os.getenv("TG_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TG_BOT_TOKEN не задан в .env (токен у @BotFather)")

bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: types.Message) -> None:
    welcome_text = (
        "Привет. Это бот психологической поддержки KrovostokLM.\n\n"
        "Опиши свою ситуацию, боль или повод для гордости, а я подберу для тебя "
        "идеальную цитату из текстов Шило и компании.\n\n"
        "<i>Пиши как есть, без купюр.</i>"
    )
    await message.answer(welcome_text)


@dp.message()
async def text_handler(message: types.Message) -> None:
    user_text = message.text
    if not user_text:
        return

    processing_msg = await message.answer("<i>Анализирую ситуацию...</i>")
    try:
        # find_quote синхронный (сетевые вызовы) — уводим в поток, чтобы не
        # блокировать event loop. session_id = chat.id для анти-магнита.
        result = await asyncio.to_thread(find_quote, user_text, str(message.chat.id))

        response_text = (
            f"💬 <b>Цитата:</b>\n{result['quote']}\n\n"
            f"🎵 <i>Трек: {result['track']}</i>"
        )
        await processing_msg.delete()
        await message.answer(response_text)

    except Exception as e:
        print(f"[ОШИБКА БОТА] {e}")
        traceback.print_exc()
        await processing_msg.edit_text("Что-то пошло не так. Попробуй ещё раз чуть позже.")


async def main() -> None:
    print("=== Бот KrovostokLM запущен ===")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Бот остановлен.")
