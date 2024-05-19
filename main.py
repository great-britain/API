import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_v1 import StroyModel  # Модель Дммы и Ивана

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Инициализация приложения FastAPI
app = FastAPI()

# Класс для валидации входных данных
class Item(BaseModel):
    name: str

# Инициализация модели
PATH = '/root/fastapi_project'
DATASET_PATH = PATH + '/datasets'
ksr_file = DATASET_PATH + '/classification.xlsx'
train_file = DATASET_PATH + '/train.xlsx'

logging.info("Загружаем модель...")
stroy_model = StroyModel(
    ksr_file=ksr_file,
    train_file=train_file,
    model_name='intfloat/multilingual-e5-small',
    k_len=50
)
logging.info("Модель загружена!")

# Эндпоинт для классификации
@app.post("/classify")
async def classify(item: Item):
    try:
        logging.info(f"Получен запрос для классификации: {item.name}")
        ksr_code, ksr_name, cos_sim, final_score = stroy_model.predict_API(item.name)
        logging.info(f"Результат классификации: ksr_code={ksr_code}, ksr_name={ksr_name}, confidence={cos_sim}, conversion_factor={final_score}")
        return {
            "ksr_code": ksr_code,
            "ksr_name": ksr_name,
            "confidence": cos_sim,
            "conversion_factor": final_score
        }
    except Exception as e:
        logging.error(f"Ошибка при классификации: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Эндпоинт для проверки состояния сервиса
@app.get("/health")
async def health():
    return {"status": "OK"}

# Эндпоинт для корневого пути
@app.get("/")
async def read_root():
    return {"message": "Welcome to the UFO API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
