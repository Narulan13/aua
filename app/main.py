# ============================================
# FIXED app/main.py - Полная интеграция
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import os
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json
from dotenv import load_dotenv

# ============================================
# ИСПРАВЛЕНИЕ 1: Правильные импорты
# ============================================
from models import GeoLocation, Measurement, DataSource, PollutantType
from fetchers.openaq import OpenAQFetcher
from fetchers.iqair import IQAirFetcher
from fetchers.tempo import TEMPOFetcher
from aggregator import AirQualityAggregator

load_dotenv()

# ============================================
# 2. GOOGLE MAPS TRAFFIC (без изменений)
# ============================================

class GoogleMapsTrafficFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api"
    
    def get_traffic_index(self, lat: float, lon: float, radius_km: float = 5) -> Dict:
        try:
            destinations = self._generate_nearby_points(lat, lon, radius_km)
            traffic_data = []
            
            for dest_lat, dest_lon in destinations:
                duration_traffic = self._get_travel_time(
                    lat, lon, dest_lat, dest_lon, departure_time="now"
                )
                duration_freeflow = self._get_travel_time(
                    lat, lon, dest_lat, dest_lon, departure_time=None
                )
                
                if duration_traffic and duration_freeflow:
                    delay_ratio = duration_traffic / duration_freeflow
                    traffic_data.append(delay_ratio)
            
            if not traffic_data:
                return self._get_fallback_traffic(lat, lon)
            
            avg_delay_ratio = np.mean(traffic_data)
            traffic_index = min(100, (avg_delay_ratio - 1.0) * 100)
            
            if traffic_index < 20:
                congestion_level = "low"
            elif traffic_index < 50:
                congestion_level = "moderate"
            elif traffic_index < 80:
                congestion_level = "high"
            else:
                congestion_level = "severe"
            
            return {
                'traffic_index': round(traffic_index, 2),
                'average_delay_ratio': round(avg_delay_ratio, 2),
                'congestion_level': congestion_level,
                'source': 'google_maps',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Google Maps error: {e}")
            return self._get_fallback_traffic(lat, lon)
    
    def _get_travel_time(self, origin_lat, origin_lon, dest_lat, dest_lon, 
                        departure_time=None):
        try:
            params = {
                'origins': f"{origin_lat},{origin_lon}",
                'destinations': f"{dest_lat},{dest_lon}",
                'mode': 'driving',
                'key': self.api_key
            }
            
            if departure_time:
                params['departure_time'] = departure_time
            
            response = requests.get(
                f"{self.base_url}/distancematrix/json",
                params=params,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['rows']:
                    element = data['rows'][0]['elements'][0]
                    if element['status'] == 'OK':
                        if departure_time:
                            return element.get('duration_in_traffic', {}).get('value')
                        else:
                            return element.get('duration', {}).get('value')
            return None
        except Exception as e:
            print(f"Travel time error: {e}")
            return None
    
    def _generate_nearby_points(self, lat, lon, radius_km, num_points=8):
        points = []
        radius_deg = radius_km / 111.0
        
        for i in range(num_points):
            angle = (2 * np.pi * i) / num_points
            point_lat = lat + radius_deg * np.cos(angle)
            point_lon = lon + radius_deg * np.sin(angle)
            points.append((point_lat, point_lon))
        
        return points
    
    def _get_fallback_traffic(self, lat, lon):
        hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        
        if day_of_week < 5:
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                traffic_index = 70
            elif 10 <= hour <= 16:
                traffic_index = 40
            else:
                traffic_index = 20
        else:
            traffic_index = 30
        
        return {
            'traffic_index': traffic_index,
            'congestion_level': 'estimated',
            'source': 'time_based_estimate',
            'timestamp': datetime.now().isoformat()
        }


# ============================================
# 3. POLLUTION INDEX CALCULATOR (без изменений)
# ============================================

class PollutionIndexCalculator:
    def __init__(self, weights_path: str = "aqi_weights.json"):
        self.weights = {}
        self.feature_ranges = {
            'CO_ppm': (0, 50),
            'NO2_ppb': (0, 400),
            'O3_ppb': (0, 400),
            'AvgTemperature_C': (-40, 50),
            'AvgWindSpeed_m_s': (0, 30),
            'TrafficIndex': (0, 100),
            'AvgPrecipitation_mm': (0, 100),
            'AQI': (0, 500),
        }
        
        try:
            with open(weights_path, 'r') as f:
                data = json.load(f)
                self.weights = data.get('weights', {})
                print(f"✅ Loaded ML weights from {weights_path}")
        except:
            print("⚠️ Using default weights")
            self.weights = {
                'CO_ppm': 0.20,
                'NO2_ppb': 0.18,
                'O3_ppb': 0.15,
                'TrafficIndex': 0.17,
                'AvgTemperature_C': 0.10,
                'AvgWindSpeed_m_s': 0.12,
                'AvgPrecipitation_mm': 0.08,
            }
    
    def calculate(self, factors: Dict[str, float]) -> Dict:
        normalized = {}
        contributions = {}
        
        for factor_name, value in factors.items():
            if factor_name in self.feature_ranges:
                xmin, xmax = self.feature_ranges[factor_name]
                normalized_value = (value - xmin) / (xmax - xmin)
                normalized_value = np.clip(normalized_value, 0, 1)
                normalized[factor_name] = normalized_value
        
        for factor_name, norm_value in normalized.items():
            weight = self.weights.get(factor_name, 0.0)
            contributions[factor_name] = weight * norm_value
        
        pollution_index = sum(contributions.values()) * 100
        
        top_contributors = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'pollution_index': round(pollution_index, 2),
            'normalized_factors': normalized,
            'contributions': contributions,
            'weights': self.weights,
            'top_contributors': top_contributors,
            'health_category': self._get_health_category(pollution_index)
        }
    
    def _get_health_category(self, index: float) -> str:
        if index <= 20:
            return "Excellent"
        elif index <= 40:
            return "Good"
        elif index <= 60:
            return "Moderate"
        elif index <= 80:
            return "Unhealthy"
        else:
            return "Very Unhealthy"


# ============================================
# 4. AQI PREDICTOR (без изменений)
# ============================================

class AQIPredictor:
    def __init__(self, model_path: str = "aqi_model.pkl", 
                 scaler_path: str = "aqi_scaler.pkl"):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.has_model = True
            print("✅ Loaded trained AQI model")
        except:
            self.model = None
            self.scaler = None
            self.has_model = False
            print("⚠️ No trained model found, using fallback")
    
    def predict(self, features: Dict[str, float]) -> float:
        if not self.has_model:
            pm25 = features.get('PM25', 35)
            return self._pm25_to_aqi(pm25)
        
        feature_vector = [
            features.get('AvgTemperature_C', 20),
            features.get('AvgWindSpeed_m_s', 3),
            features.get('AvgPrecipitation_mm', 0),
            features.get('CO_ppm', 1),
            features.get('NO2_ppb', 30),
            features.get('O3_ppb', 50),
            features.get('TrafficIndex', 50),
        ]
        
        X_scaled = self.scaler.transform([feature_vector])
        predicted_aqi = self.model.predict(X_scaled)[0]
        
        return round(predicted_aqi, 2)
    
    def _pm25_to_aqi(self, pm25: float) -> float:
        if pm25 <= 12.0:
            return (50 / 12.0) * pm25
        elif pm25 <= 35.4:
            return 50 + ((100 - 50) / (35.4 - 12.0)) * (pm25 - 12.0)
        elif pm25 <= 55.4:
            return 100 + ((150 - 100) / (55.4 - 35.4)) * (pm25 - 35.4)
        elif pm25 <= 150.4:
            return 150 + ((200 - 150) / (150.4 - 55.4)) * (pm25 - 55.4)
        else:
            return min(500, 200 + ((300 - 200) / (250.4 - 150.4)) * (pm25 - 150.4))


# ============================================
# 5. FASTAPI APP
# ============================================

app = FastAPI(title="AirQualityAI API v2.1 - FIXED")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# ИСПРАВЛЕНИЕ 2: Инициализация fetchers
# ============================================

google_maps_key = os.getenv("GOOGLE_MAPS_API_KEY")
openaq_key = os.getenv("OPENAQ_API_KEY")
iqair_key = os.getenv("IQAIR_API_KEY")
tempo_user = os.getenv("TEMPO_USERNAME")
tempo_pass = os.getenv("TEMPO_PASSWORD")

traffic_fetcher = GoogleMapsTrafficFetcher(google_maps_key) if google_maps_key else None
pollution_calculator = PollutionIndexCalculator()
aqi_predictor = AQIPredictor()

# Создаём агрегатор для сбора данных из всех источников
aggregator = AirQualityAggregator()

if openaq_key:
    aggregator.add_fetcher(OpenAQFetcher(openaq_key))
    print("✅ OpenAQ fetcher enabled")

if iqair_key:
    aggregator.add_fetcher(IQAirFetcher(iqair_key))
    print("✅ IQAir fetcher enabled")

if tempo_user and tempo_pass:
    aggregator.add_fetcher(TEMPOFetcher(tempo_user, tempo_pass))
    print("✅ TEMPO fetcher enabled")


# ============================================
# 6. API MODELS
# ============================================

class PredictRequest(BaseModel):
    lat: float
    lon: float
    profile: Optional[Dict] = {}


class PollutionIndexResponse(BaseModel):
    pollution_index: float
    predicted_aqi: float
    health_category: str
    traffic_data: Dict
    factors: Dict[str, float]
    contributions: Dict[str, float]
    top_contributors: List
    advice: List[str]
    location: Dict
    timestamp: str
    sources_used: List[str]


# ============================================
# 7. API ENDPOINTS
# ============================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "google_maps": traffic_fetcher is not None,
        "ml_model": aqi_predictor.has_model,
        "fetchers": len(aggregator.fetchers),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PollutionIndexResponse)
def predict(req: PredictRequest):
    """
    ИСПРАВЛЕНИЕ 3: Используем РЕАЛЬНЫЕ данные из fetchers
    """
    try:
        sources_used = []
        
        # 1. Traffic data
        if traffic_fetcher:
            traffic_data = traffic_fetcher.get_traffic_index(req.lat, req.lon)
            sources_used.append("google_maps")
        else:
            traffic_data = {
                'traffic_index': 50,
                'congestion_level': 'estimated',
                'source': 'fallback'
            }
        
        # 2. ИСПРАВЛЕНИЕ: Получаем РЕАЛЬНЫЕ данные о качестве воздуха
        location = GeoLocation(latitude=req.lat, longitude=req.lon)
        
        air_quality = {}
        
        if len(aggregator.fetchers) > 0:
            # Используем реальные fetchers
            snapshot = aggregator.get_snapshot(location)
            
            # Извлекаем данные из snapshot
            pm25_avg = snapshot.get_pollutant_avg('pm25')
            pm10_avg = snapshot.get_pollutant_avg('pm10')
            no2_avg = snapshot.get_pollutant_avg('no2')
            o3_avg = snapshot.get_pollutant_avg('o3')
            co_avg = snapshot.get_pollutant_avg('co')
            print(pm25_avg, pm10_avg, no2_avg, o3_avg, co_avg)
            air_quality = {
                'PM25': pm25_avg if pm25_avg else 35.0,
                'PM10': pm10_avg if pm10_avg else 50.0,
                'NO2_ppb': no2_avg if no2_avg else 30.0,
                'O3_ppb': o3_avg if o3_avg else 50.0,
                'CO_ppm': co_avg if co_avg else 1.0,
            }
            
            for measurement_list in snapshot.measurements.values():
                if measurement_list:
                    sources_used.append(measurement_list[0].source.value)
            
            sources_used = list(set(sources_used))  # Убираем дубликаты
        else:
            # Fallback к синтетическим данным ТОЛЬКО если нет fetchers
            print("⚠️ No fetchers configured, using synthetic data")
            air_quality = {
                'CO_ppm': 1.2 + np.random.randn() * 0.3,
                'NO2_ppb': 45.0 + np.random.randn() * 10,
                'O3_ppb': 55.0 + np.random.randn() * 10,
                'PM25': 35.0 + np.random.randn() * 5,
            }
            sources_used.append("synthetic")
        
        # 3. Weather data (можно добавить Open-Meteo API)
        weather = {
            'AvgTemperature_C': 25.0,
            'AvgWindSpeed_m_s': 3.5,
            'AvgPrecipitation_mm': 0.0,
        }
        
        # 4. Объединяем все факторы
        all_factors = {
            **air_quality,
            **weather,
            'TrafficIndex': traffic_data['traffic_index']
        }
        
        # 5. Calculate pollution index
        pollution_result = pollution_calculator.calculate(all_factors)
        
        # 6. Predict AQI
        predicted_aqi = aqi_predictor.predict(all_factors)
        
        # 7. Generate advice
        advice = _generate_advice(
            pollution_result['pollution_index'],
            predicted_aqi,
            req.profile
        )
        
        sources_used.append("ml_model")
        
        return PollutionIndexResponse(
            pollution_index=pollution_result['pollution_index'],
            predicted_aqi=predicted_aqi,
            health_category=pollution_result['health_category'],
            traffic_data=traffic_data,
            factors=all_factors,
            contributions=pollution_result['contributions'],
            top_contributors=pollution_result['top_contributors'],
            advice=advice,
            location={"lat": req.lat, "lon": req.lon},
            timestamp=datetime.now().isoformat(),
            sources_used=sources_used
        )
        
    except Exception as e:
        print(f"Error in predict(): {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/traffic/{lat}/{lon}")
def get_traffic(lat: float, lon: float):
    if not traffic_fetcher:
        raise HTTPException(status_code=503, detail="Google Maps API not configured")
    
    return traffic_fetcher.get_traffic_index(lat, lon)


def _generate_advice(pollution_index: float, aqi: float, profile: Dict) -> List[str]:
    advice = []
    
    if pollution_index <= 20:
        advice.append("✅ Excellent air quality! Perfect for all activities.")
    elif pollution_index <= 40:
        advice.append("🟢 Good air quality. Enjoy outdoor activities!")
    elif pollution_index <= 60:
        advice.append("🟡 Moderate pollution. Sensitive groups should limit prolonged outdoor activities.")
        if profile.get("asthma"):
            advice.append("⚠️ Monitor your symptoms and have your inhaler ready.")
    elif pollution_index <= 80:
        advice.append("🟠 Unhealthy air quality. Reduce outdoor activities.")
        advice.append("😷 Consider wearing an N95 mask outdoors.")
        if profile.get("asthma") or profile.get("age", 0) > 65:
            advice.append("🚨 High risk group: Stay indoors if possible.")
    else:
        advice.append("🔴 Very unhealthy! Stay indoors with air purification.")
        advice.append("🏥 Monitor health closely. Seek medical attention if needed.")
        if profile.get("asthma"):
            advice.append("🚨 CRITICAL: Use rescue inhaler and avoid all outdoor exposure.")
    
    return advice


@app.get("/")
def root():
    return {
        "message": "AirQualityAI API v2.1 - FIXED VERSION",
        "docs": "/docs",
        "fixes": [
            "✅ Real data fetchers integrated",
            "✅ Proper imports added",
            "✅ Traffic index properly used",
            "✅ Error handling improved",
            "✅ Sources tracking added"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("🚀 STARTING FIXED AIR QUALITY AI SERVER")
    print("="*60)
    print(f"\n📋 Configuration:")
    print(f"  • Google Maps API: {'✅ Active' if traffic_fetcher else '❌ Not configured'}")
    print(f"  • ML Model: {'✅ Loaded' if aqi_predictor.has_model else '❌ Not trained'}")
    print(f"  • Data Fetchers: {len(aggregator.fetchers)} configured")
    print(f"  • Pollution Calculator: ✅ Ready")
    print("\n🌐 Server: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)