from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML prediction API", description="API for first ML model")

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("models/model.pkl")
        print("Model loaded")
    except Exception as e:
        print(f"Fail: {e}")

class PredictRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="The model has not been loaded")
    
    if len(request.features) != 15:
        raise HTTPException(status_code=400, detail="An array of 15 numbers is expected")
    
    input_data = np.array([request.features])
    prediction = model.predict(input_data)
    
    return {"prediction": int(prediction[0])}

