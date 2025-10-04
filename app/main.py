from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
from datetime import datetime
from app.ml import train_model, predict_point
from app.forecasting import AirQualityForecaster

# Путь к модели мгновенного предсказания
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

app = FastAPI(title="AirQualityAI MVP")

# CORS для frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация форкастера один раз
forecaster = AirQualityForecaster()

class PredictRequest(BaseModel):
    lat: float
    lon: float
    timestamp: str = None
    profile: dict = {}  # {"age":45, "asthma":True}

# ---------------------------
# Startup
# ---------------------------
@app.on_event("startup")
def load_or_train():
    if not os.path.exists(MODEL_PATH):
        print("No model found — training dummy model...")
        train_model(MODEL_PATH)
    else:
        print("Model exists at", MODEL_PATH)

# ---------------------------
# Health check
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------
# Predict current air quality
# ---------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    pred = predict_point(req.lat, req.lon)

    pm25 = pred["pm25"]
    aqi = forecaster._calculate_aqi(pm25)  # Используем AQI из форкастера
    conf = 0.9  # Пока фиксируем

    # Генерация персональных советов
    adv = []
    profile = req.profile
    if pm25 > 150:
        adv.append("Высокий уровень загрязнения — избегайте физических нагрузок на улице.")
        if profile.get("asthma"):
            adv.append("⚠️ У вас астма — оставайтесь в помещении и используйте ингалятор при необходимости.")
    elif pm25 > 100:
        adv.append("Умеренно повышенный уровень — люди с респираторными заболеваниями будьте осторожны.")
        if profile.get("asthma") or profile.get("age", 0) > 65:
            adv.append("⚠️ Ограничьте активность на открытом воздухе.")
    else:
        adv.append("Качество воздуха в норме. Можно гулять!")

    return {
        "pm25": round(pm25, 2),
        "no2": round(pred["no2"], 2),
        "so2": round(pred["so2"], 2),
        "o3": round(pred["o3"], 2),
        "aqi": aqi,
        "confidence": conf,
        "advice": adv,
        "location": {"lat": req.lat, "lon": req.lon},
        "timestamp": datetime.now().isoformat()
    }

# ---------------------------
# Train instant prediction model
# ---------------------------
@app.post("/train")
def train_endpoint():
    train_model(MODEL_PATH)
    return {"status": "trained", "model_path": MODEL_PATH}

# ---------------------------
# Forecast future air quality
# ---------------------------
@app.post("/forecast")
def forecast(req: PredictRequest):
    hours_ahead = [1, 3, 6, 12, 24, 48, 72]
    predictions = forecaster.predict_future(req.lat, req.lon, hours_ahead)

    # Добавляем персональные советы для каждого прогноза
    for p in predictions:
        adv = []
        pm25 = p["pm25"]
        profile = req.profile
        if pm25 > 150:
            adv.append("Высокий уровень загрязнения — избегайте физических нагрузок на улице.")
            if profile.get("asthma"):
                adv.append("⚠️ У вас астма — оставайтесь в помещении и используйте ингалятор при необходимости.")
        elif pm25 > 100:
            adv.append("Умеренно повышенный уровень — люди с респираторными заболеваниями будьте осторожны.")
            if profile.get("asthma") or profile.get("age", 0) > 65:
                adv.append("⚠️ Ограничьте активность на открытом воздухе.")
        else:
            adv.append("Качество воздуха в норме. Можно гулять!")
        p["advice"] = adv

    return {
        "location": {"lat": req.lat, "lon": req.lon},
        "current_time": datetime.now().isoformat(),
        "forecasts": predictions
    }

# ---------------------------
# Train forecasting model
# ---------------------------
@app.post("/train-forecast")
def train_forecast_model():
    forecaster.train_model(forecaster.generate_historical_data())
    return {"status": "forecast model trained"}

# ---------------------------
# Root
# ---------------------------
@app.get("/")
def root():
    return {"message": "AirQualityAI MVP API", "docs": "/docs"}
