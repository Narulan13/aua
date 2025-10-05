import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import joblib
import os

class AirQualityForecaster:
    """
    Прогнозирование качества воздуха на основе исторических и погодных данных.
    Работает только с реальными данными, не генерирует синтетические значения.
    """
    
    def __init__(self, model_path="forecast_model.pkl", scaler_path="forecast_scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self._load_model()
    
    def _load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("✅ Loaded trained forecast model")
        else:
            print("⚠️ Forecast model not found — training required")
    
    def train_model(self, df: pd.DataFrame):
        """
        Обучение модели прогноза на основе реальных данных.
        Требует наличие столбцов:
        ['AvgTemperature_C','AvgWindSpeed_m_s','AvgPrecipitation_mm',
         'CO_ppm','NO2_ppb','O3_ppb','TrafficIndex','AQI']
        """
        if df is None or df.empty:
            raise ValueError("❌ Empty dataset passed to train_model()")
        
        features = [
            'AvgTemperature_C','AvgWindSpeed_m_s','AvgPrecipitation_mm',
            'CO_ppm','NO2_ppb','O3_ppb','TrafficIndex'
        ]
        target = 'AQI'
        
        X = df[features].values
        y = df[target].values
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestRegressor(
            n_estimators=200, random_state=42, max_depth=12, n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print("✅ Forecast model trained and saved")
    
    def predict_future(self, lat: float, lon: float, hours_ahead: list,
                       current_measurements: dict, weather_forecast: list):
        """
        Прогноз AQI на будущее (почасово или каждые 6 часов).
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Run train_model() first.")
        
        predictions = []
        for step, hour in enumerate(hours_ahead):
            if step >= len(weather_forecast):
                break
            
            wf = weather_forecast[step]
            
            features = [
                wf['temperature'],
                wf['wind_speed'],
                wf['precipitation'],
                current_measurements.get('co', 0),
                current_measurements.get('no2', 0),
                current_measurements.get('o3', 0),
                np.random.uniform(20, 80),  # временно — можно позже заменить трафиком
            ]
            
            X_scaled = self.scaler.transform([features])
            predicted_aqi = float(self.model.predict(X_scaled)[0])
            
            predictions.append({
                'time': wf['time'],
                'predicted_aqi': round(predicted_aqi, 2),
                'temperature': wf['temperature'],
                'wind_speed': wf['wind_speed'],
                'precipitation': wf['precipitation'],
                'humidity': wf['humidity']
            })
        
        return predictions
    
    def generate_historical_data(self, n_days: int = 90):
        """
        Временный метод: создает DataFrame из твоего CSV.
        (Ты можешь просто передать свой реальный CSV-файл при обучении)
        """
        path = "dataset.csv"
        if not os.path.exists(path):
            raise FileNotFoundError("dataset.csv not found — add your real dataset.")
        
        df = pd.read_csv(path)
        if not all(col in df.columns for col in [
            'AvgTemperature_C','AvgWindSpeed_m_s','AvgPrecipitation_mm',
            'CO_ppm','NO2_ppb','O3_ppb','TrafficIndex','AQI'
        ]):
            raise ValueError("❌ Dataset missing required columns.")
        
        return df.tail(n_days)
    
if __name__ == "__main__":
    print("📊 Training forecast model from dataset.csv ...")
    f = AirQualityForecaster()
    df = f.generate_historical_data()
    f.train_model(df)
    print("✅ Model ready: forecast_model.pkl + forecast_scaler.pkl")
