# API
Api
#Создаем директорию для проекта
mkdir ~/fastapi_project
cd ~/fastapi_project
#Создаем окружение
python3 -m venv venv
#Создаем структуру
/path/to/fastapi_project
├── datasets
│   ├── classification.xlsx
│   ├── train.xlsx
├── model_v1.py
├── main.py
├── requirements.txt
└── venv 
#Установим зависимости
pip install -r requirements.txt
Стартуем сервер: 
uvicorn main:app --host 0.0.0.0 --port 8000 --reload



# Bot
Устанавливаем aiogram==2.13, 
Устанавливаем Токен нужного бота ( по умолчанию бот @AmethystCapitalBot)
Запускаем файл bot.py с локальной машины.
