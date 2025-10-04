# app/forecasting.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime, timedelta


class AirQualityForecaster:
    """
    Модель для прогнозирования качества воздуха на будущее.
    Использует временные признаки, исторические данные и дополнительные загрязнители.
    """

    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.model_path = model_path or "air_quality_model.pkl"
        self.scaler_path = self.model_path.replace(".pkl", "_scaler.pkl")

        # Определяем используемые фичи
        self.feature_cols = [
            "hour", "day_of_week", "month", "is_weekend",
            "temp", "wind", "humidity",
            "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_6h",
            "no2_lag_1h", "so2_lag_1h", "o3_lag_1h",
            "pm25_rolling_3h", "no2_rolling_3h", "so2_rolling_3h", "o3_rolling_3h",
        ]

        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)

    # --------------------------
    # Генерация исторических данных
    # --------------------------
    def generate_historical_data(self, n_days=90):
        """
        Создаём синтетические данные: PM2.5, NO2, SO2, O3.
        """
        date_range = pd.date_range(
            end=datetime.now(), periods=n_days * 24, freq="H"
        )

        data = []
        for date in date_range:
            hour = date.hour
            day_of_week = date.weekday()
            month = date.month

            # Простые зависимости
            base_pollution = 30 + 15 * np.sin(2 * np.pi * month / 12)
            hourly_pattern = 10 * np.sin(hour / 24 * 2 * np.pi)
            weekly_pattern = 5 if day_of_week < 5 else 10
            noise = np.random.normal(0, 5)

            pm25 = max(5, base_pollution + hourly_pattern + weekly_pattern + noise)

            # Синтетика для других загрязнителей
            no2 = max(2, 20 + 5 * np.sin(hour / 3) + np.random.randn() * 2)
            so2 = max(1, 5 + np.random.rand() * 2)
            o3 = max(5, 40 + 10 * np.sin((hour - 12) * np.pi / 12) + np.random.randn() * 5)

            # Погода
            temp = 15 + 10 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.randn() * 2
            wind = np.random.uniform(0, 5)
            humidity = np.random.uniform(30, 80)

            data.append({
                "timestamp": date,
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "is_weekend": 1 if day_of_week >= 5 else 0,
                "temp": temp,
                "wind": wind,
                "humidity": humidity,
                "pm25": pm25,
                "no2": no2,
                "so2": so2,
                "o3": o3,
            })

        return pd.DataFrame(data)

    # --------------------------
    # Фичи
    # --------------------------
    def create_lag_features(self, df, lag_hours=[1, 3, 6, 12, 24]):
        for lag in lag_hours:
            for col in ["pm25", "no2", "so2", "o3"]:
                df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

        # Rolling averages
        for col in ["pm25", "no2", "so2", "o3"]:
            df[f"{col}_rolling_3h"] = df[col].rolling(window=3, min_periods=1).mean()
            df[f"{col}_rolling_12h"] = df[col].rolling(window=12, min_periods=1).mean()
            df[f"{col}_rolling_24h"] = df[col].rolling(window=24, min_periods=1).mean()

        return df.dropna()

    # --------------------------
    # Обучение модели
    # --------------------------
    def train_model(self, data):
        df = self.create_lag_features(data)

        X = df[self.feature_cols]
        y = df["pm25"]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1
        )
        self.model.fit(X_scaled, y)

        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

        return self.model

    # --------------------------
    # Прогноз на будущее
    # --------------------------
    def predict_future(self, lat, lon, hours_ahead=[1, 3, 6, 12, 24, 48, 72]):
        if self.model is None or self.scaler is None:
            raise ValueError("Модель не обучена. Сначала вызовите train_model().")

        now = datetime.now()

        # Инициализация "текущих" значений
        current_pm25 = 35.0
        current_no2 = 20.0
        current_so2 = 4.0
        current_o3 = 50.0

        historical_pm25 = [current_pm25] * 25
        historical_no2 = [current_no2] * 25
        historical_so2 = [current_so2] * 25
        historical_o3 = [current_o3] * 25

        predictions = []

        for hours in sorted(hours_ahead):
            future_time = now + timedelta(hours=hours)

            features = {
                "hour": future_time.hour,
                "day_of_week": future_time.weekday(),
                "month": future_time.month,
                "is_weekend": 1 if future_time.weekday() >= 5 else 0,
                "temp": 20 + np.random.randn() * 2,
                "wind": 2 + np.random.randn() * 0.5,
                "humidity": 60 + np.random.randn() * 10,
                "pm25_lag_1h": historical_pm25[-1],
                "no2_lag_1h": historical_no2[-1],
                "so2_lag_1h": historical_so2[-1],
                "o3_lag_1h": historical_o3[-1],
                "pm25_lag_3h": np.mean(historical_pm25[-3:]),
                "pm25_lag_6h": np.mean(historical_pm25[-6:]),
                "pm25_rolling_3h": np.mean(historical_pm25[-3:]),
                "no2_rolling_3h": np.mean(historical_no2[-3:]),
                "so2_rolling_3h": np.mean(historical_so2[-3:]),
                "o3_rolling_3h": np.mean(historical_o3[-3:]),
            }

            X = pd.DataFrame([features])[self.feature_cols]
            X_scaled = self.scaler.transform(X)

            pm25_pred = float(self.model.predict(X_scaled)[0])
            pm25_pred = max(5, pm25_pred)

            # Для NO2, SO2, O3 пока добавляем шум
            no2_pred = current_no2 + np.random.randn() * 1.5
            so2_pred = current_so2 + np.random.randn() * 0.3
            o3_pred = current_o3 + np.random.randn() * 4

            predictions.append({
                "hours_ahead": hours,
                "timestamp": future_time.isoformat(),
                "readable_time": future_time.strftime("%d.%m.%Y %H:%M"),
                "pm25": round(pm25_pred, 2),
                "no2": round(no2_pred, 2),
                "so2": round(so2_pred, 2),
                "o3": round(o3_pred, 2),
                "aqi": self._calculate_aqi(pm25_pred),
                "confidence": round(max(0.3, 0.9 - (hours / 100)), 2),
                "trend": self._get_trend(current_pm25, pm25_pred),
            })

            # Обновляем исторические данные
            historical_pm25.append(pm25_pred)
            historical_no2.append(no2_pred)
            historical_so2.append(so2_pred)
            historical_o3.append(o3_pred)

        return predictions

    # --------------------------
    # AQI и тренд
    # --------------------------
    def _calculate_aqi(self, pm25):
        if pm25 <= 12:
            return "Good"
        elif pm25 <= 35.4:
            return "Moderate"
        elif pm25 <= 55.4:
            return "Unhealthy for Sensitive"
        elif pm25 <= 150.4:
            return "Unhealthy"
        elif pm25 <= 250.4:
            return "Very Unhealthy"
        return "Hazardous"

    def _get_trend(self, current, future):
        if future > current * 1.1:
            return "rising"
        elif future < current * 0.9:
            return "falling"
        return "stable"
