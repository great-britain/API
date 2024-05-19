import logging
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.types import ParseMode
from aiogram.utils import executor

API_TOKEN = '7122213704:AAHhI6ghsYSfVJGykLqxX0deRwGluO-uE-U'
API_URL = 'http://85.193.87.56:8000/classify'

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Включение логирования
logging.basicConfig(level=logging.INFO)

# Обработчик команды /start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Введите наименование строительного материала:")

# Обработчик текстовых сообщений
@dp.message_handler()
async def handle_message(message: types.Message):
    material_name = message.text
    response = requests.post(API_URL, json={"name": material_name})

    if response.status_code == 200:
        data = response.json()
        ksr_code = data.get("ksr_code")
        ksr_name = data.get("ksr_name")
        confidence = data.get("confidence")
        conversion_factor = data.get("conversion_factor")
        reply_text = (
            f"Код КСР: {ksr_code}\n"
            f"Наименование КСР: {ksr_name}\n"
            f"Уверенность: {confidence:.2f}\n"
            f"Коэффициент пересчета: {conversion_factor:.2f}"
        )
    else:
        reply_text = "Произошла ошибка при обращении к API."

    await message.reply(reply_text, parse_mode=ParseMode.HTML)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
