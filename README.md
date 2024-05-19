## Приведение номенклатуры участников рынка к официальному Классификатору Строительных Ресурсов
##### Организацтор: Цифровой прорыв 2024
https://hacks-ai.ru/events/1077375
##### Кейсодержатель: Аметист Кэпитал

***



### Структура репозитария:

1. model_v2.py - итоговый модуль (класс StroyModel) модели представленной как ансамбль intfloat/multilingual-e5-small и алгоритма BM25.  
2. datasets/classification.xlsx - классификатор КСР
3. datasets/train.xlsx - обучающая выборка
4. bot1.py - Телеграм-бот
5. main.py - API
6. requirements.txt - зависимости
7. Train_modal.ipynb - функция дообучения 'sentence-transformers/paraphrase-MiniLM-L6-v2'


###### API и Телеграм-бот используют модуль с моделью

#### Развертывание API:
- Создаем директорию для проекта mkdir ~/fastapi_project cd ~/fastapi_project 
- Создаем окружение python3 -m venv venv 
- Создаем структуру /path/to/fastapi_project ├── datasets │ ├── classification.xlsx │ ├── train.xlsx ├── model_v1.py ├── main.py ├── requirements.txt └── venv 
- Установим зависимости pip install -r requirements.txt 
- Стартуем сервер: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

#### Развертывание Телеграм-бота:
- Устанавливаем aiogram==2.13
- Устанавливаем Токен нужного бота ( по умолчанию бот @AmethystCapitalBot) 
- Запускаем файл bot.py
