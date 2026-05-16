from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# 1. Инициализируем приложение FastAPI
app = FastAPI(title="ML Prediction API", description="API для нашей первой ML-модели")

# Глобальная переменная для хранения модели
model = None

# 2. Загружаем модель при старте сервера
@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("models/model.pkl")
        print("✅ Модель успешно загружена в память!")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")

# 3. Описываем схему входящих данных (Pydantic валидация)
# Наша синтетическая выборка генерировала 15 признаков
class PredictRequest(BaseModel):
    features: list[float]

# 4. Создаем endpoint для предсказаний
@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    
    # Проверяем количество признаков
    if len(request.features) != 15:
        raise HTTPException(status_code=400, detail="Ожидается массив из 15 чисел")
    
    input_data = np.array([request.features])
    prediction = model.predict(input_data)
    
    return {"prediction": int(prediction[0])}

