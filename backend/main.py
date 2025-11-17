from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import numpy as np

app = FastAPI(title="Churn Prediction API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    MODEL_PATH = os.getenv('MODEL_PATH', 'lrm.joblib')
    model = joblib.load(MODEL_PATH)
    print("✅ Модель успешно загружена")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    model = None

class CustomerData(BaseModel):
    City: str
    Gender: str
    Senior_Citizen: str
    Partner: str
    Dependents: str
    Tenure_Months: float
    Phone_Service: str
    Multiple_Lines: str
    Internet_Service: str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
    Monthly_Charges: float
    Total_Charges: float

class PredictionResult(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str

@app.get("/")
def read_root():
    return {"message": "Churn Prediction API"}

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResult)
def predict_churn(data: CustomerData):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        input_data = pd.DataFrame([data.dict()])
        
        print(f"📊 Получены данные: {list(input_data.columns)}")
        print(f"🔢 Размер данных: {input_data.shape}")
        
        # Предсказание
        probability = model.predict_proba(input_data)[0, 1]
        prediction = probability > 0.5
        
        # Уровень риска
        if probability < 0.3:
            risk = "Низкий"
        elif probability < 0.7:
            risk = "Средний" 
        else:
            risk = "Высокий"
            
        return {
            "churn_probability": round(probability, 3),
            "churn_prediction": bool(prediction),
            "risk_level": risk
        }
        
    except Exception as e:
        print(f"❌ Ошибка предсказания: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/debug")
def debug_data(data: CustomerData):
    """Эндпоинт для отладки данных"""
    input_data = pd.DataFrame([data.dict()])
    return {
        "columns_received": list(input_data.columns),
        "data_types": {col: str(dtype) for col, dtype in input_data.dtypes.items()},
        "sample_values": {col: input_data[col].iloc[0] for col in input_data.columns}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)