# app/ml.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


def generate_dummy_dataset(n=1000):
    """
    Генерация синтетического датасета с загрязнителями и погодой.
    Features: sat (proxy), temp, wind, humidity
    Targets: pm25, no2, so2, o3
    """
    np.random.seed(42)
    sat = np.random.rand(n) * 20
    temp = 15 + np.random.randn(n) * 5
    wind = np.abs(np.random.randn(n)) * 3
    humidity = 50 + np.random.randn(n) * 10

    # Синтетика загрязнителей
    pm25 = 5 + 1.4 * sat + 0.5 * temp - 0.8 * wind + np.random.randn(n) * 5
    pm25 = np.maximum(pm25, 0)

    no2 = 20 + 0.8 * sat + 0.2 * temp + np.random.randn(n) * 3
    so2 = 5 + 0.3 * sat + 0.1 * humidity + np.random.randn(n) * 1
    o3 = 40 - 0.5 * pm25 + 0.3 * temp + np.random.randn(n) * 4

    df = pd.DataFrame({
        "sat": sat,
        "temp": temp,
        "wind": wind,
        "humidity": humidity,
        "pm25": pm25,
        "no2": no2,
        "so2": so2,
        "o3": o3,
    })
    return df


def train_model(path="model.joblib"):
    """
    Обучает RandomForestRegressor на множественных загрязнителях.
    """
    print("Training model with synthetic dataset...")
    df = generate_dummy_dataset(2000)

    X = df[["sat", "temp", "wind", "humidity"]]
    y = df[["pm25", "no2", "so2", "o3"]]  # Многомерный таргет

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Оценка качества
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
    r2 = r2_score(y_test, y_pred, multioutput="raw_values")

    print(f"Model trained!")
    print(f"PM2.5 MSE={mse[0]:.2f}, R²={r2[0]:.2f}")
    print(f"NO2   MSE={mse[1]:.2f}, R²={r2[1]:.2f}")
    print(f"SO2   MSE={mse[2]:.2f}, R²={r2[2]:.2f}")
    print(f"O3    MSE={mse[3]:.2f}, R²={r2[3]:.2f}")

    # Сохранение модели
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

    return model


def predict_point(lat, lon, model_path="model.joblib"):
    """
    Делает прогноз PM2.5, NO2, SO2, O3 для точки.
    """
    # Для MVP используем прокси (в реальности сюда подгрузим TEMPO + OpenMeteo)
    from app.utils.data_loader import get_satellite_proxy, get_meteo_proxy, fetch_tempo_no2

    sat = get_satellite_proxy(lat, lon)
    temp, wind, humidity = get_meteo_proxy(lat, lon)

    features = np.array([[sat, temp, wind, humidity]])

    model = joblib.load(model_path)
    preds = model.predict(features)[0]

    pm25, no2, so2, o3 = preds

    return {
        "pm25": round(max(0, pm25), 2),
        "no2": round(max(0, no2), 2),
        "so2": round(max(0, so2), 2),
        "o3": round(max(0, o3), 2),
    }


if __name__ == "__main__":
    # Запуск обучения при старте
    train_model("app/model.joblib")
