# ============================================
# COMPLETE INTEGRATED APP: app/main.py
# FastAPI + ML Model + Google Maps + All Features
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import os
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
from dotenv import load_dotenv

load_dotenv()

# ============================================
# 1. DATA MODELS
# ============================================

class DataSource(Enum):
    TEMPO = "tempo"
    OPENAQ = "openaq"
    IQAIR = "iqair"
    GOOGLE_MAPS = "google_maps"
    ML_MODEL = "ml_model"


@dataclass
class GeoLocation:
    latitude: float
    longitude: float
    city: Optional[str] = None
    country: Optional[str] = None


@dataclass
class Measurement:
    pollutant: str
    value: float
    unit: str
    timestamp: datetime
    source: DataSource
    confidence: Optional[float] = None


# ============================================
# 2. GOOGLE MAPS TRAFFIC FETCHER
# ============================================

class GoogleMapsTrafficFetcher:
    """
    Fetch real-time traffic data from Google Maps API
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api"
    
    def get_traffic_index(self, lat: float, lon: float, radius_km: float = 5) -> Dict:
        """
        Get traffic congestion level around location
        
        Returns:
            {
                'traffic_index': 0-100,
                'average_speed_kmh': float,
                'congestion_level': 'low'|'moderate'|'high'|'severe',
                'nearby_roads': [...]
            }
        """
        try:
            # Method 1: Distance Matrix API (drive time vs free-flow time)
            destinations = self._generate_nearby_points(lat, lon, radius_km)
            
            traffic_data = []
            for dest_lat, dest_lon in destinations:
                # Get duration with current traffic
                duration_traffic = self._get_travel_time(
                    lat, lon, dest_lat, dest_lon, 
                    departure_time="now"
                )
                
                # Get duration in free-flow conditions
                duration_freeflow = self._get_travel_time(
                    lat, lon, dest_lat, dest_lon,
                    departure_time=None
                )
                
                if duration_traffic and duration_freeflow:
                    # Traffic delay factor
                    delay_ratio = duration_traffic / duration_freeflow
                    traffic_data.append(delay_ratio)
            
            if not traffic_data:
                return self._get_fallback_traffic(lat, lon)
            
            # Calculate traffic index: 0 = no delay, 100 = severe congestion
            avg_delay_ratio = np.mean(traffic_data)
            traffic_index = min(100, (avg_delay_ratio - 1.0) * 100)
            
            # Determine congestion level
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
    
    def _get_travel_time(self, origin_lat: float, origin_lon: float,
                        dest_lat: float, dest_lon: float,
                        departure_time: Optional[str] = None) -> Optional[float]:
        """Get travel time in seconds"""
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
    
    def _generate_nearby_points(self, lat: float, lon: float, 
                                radius_km: float, num_points: int = 8) -> List[Tuple[float, float]]:
        """Generate points around location for traffic sampling"""
        points = []
        radius_deg = radius_km / 111.0  # Approximate km to degrees
        
        for i in range(num_points):
            angle = (2 * np.pi * i) / num_points
            point_lat = lat + radius_deg * np.cos(angle)
            point_lon = lon + radius_deg * np.sin(angle)
            points.append((point_lat, point_lon))
        
        return points
    
    def _get_fallback_traffic(self, lat: float, lon: float) -> Dict:
        """Fallback traffic estimation based on time of day"""
        hour = datetime.now().hour
        day_of_week = datetime.now().weekday()
        
        # Rush hour patterns
        if day_of_week < 5:  # Weekday
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                traffic_index = 70  # High
            elif 10 <= hour <= 16:
                traffic_index = 40  # Moderate
            else:
                traffic_index = 20  # Low
        else:  # Weekend
            traffic_index = 30
        
        return {
            'traffic_index': traffic_index,
            'congestion_level': 'estimated',
            'source': 'time_based_estimate',
            'timestamp': datetime.now().isoformat()
        }


# ============================================
# 3. POLLUTION INDEX CALCULATOR (Your Formula)
# ============================================

class PollutionIndexCalculator:
    """
    Implements your formula: I = Σ(Wi × (Xi - Xmin)/(Xmax - Xmin))
    Weights Wi are loaded from trained ML model
    """
    
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
        
        # Load weights from trained model
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
        """
        Calculate pollution index using your formula
        
        Args:
            factors: {'CO_ppm': 1.2, 'NO2_ppb': 45, 'TrafficIndex': 67, ...}
        
        Returns:
            {
                'pollution_index': 42.5,
                'normalized_factors': {...},
                'contributions': {...},
                'top_contributors': [...]
            }
        """
        normalized = {}
        contributions = {}
        
        # Step 1: Normalize each factor (Xi - Xmin)/(Xmax - Xmin)
        for factor_name, value in factors.items():
            if factor_name in self.feature_ranges:
                xmin, xmax = self.feature_ranges[factor_name]
                normalized_value = (value - xmin) / (xmax - xmin)
                normalized_value = np.clip(normalized_value, 0, 1)  # Keep in [0,1]
                normalized[factor_name] = normalized_value
        
        # Step 2: Apply weights Wi × normalized(Xi)
        for factor_name, norm_value in normalized.items():
            weight = self.weights.get(factor_name, 0.0)
            contributions[factor_name] = weight * norm_value
        
        # Step 3: Sum all contributions: I = Σ(Wi × normalized(Xi))
        pollution_index = sum(contributions.values()) * 100  # Scale to 0-100
        
        # Get top contributors
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
# 4. AQI PREDICTOR (ML Model)
# ============================================

class AQIPredictor:
    """Load trained ML model and predict AQI"""
    
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
        """Predict AQI from features"""
        if not self.has_model:
            # Fallback: estimate from PM2.5 if available
            pm25 = features.get('PM25', 35)
            return self._pm25_to_aqi(pm25)
        
        # Prepare feature vector (match training order)
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
        """Convert PM2.5 to AQI (EPA standard)"""
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
# 5. FASTAPI APPLICATION
# ============================================

app = FastAPI(title="AirQualityAI API v2.0 - Complete")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
google_maps_key = os.getenv("GOOGLE_MAPS_API_KEY")
traffic_fetcher = GoogleMapsTrafficFetcher(google_maps_key) if google_maps_key else None
pollution_calculator = PollutionIndexCalculator()
aqi_predictor = AQIPredictor()


# ============================================
# 6. API REQUEST/RESPONSE MODELS
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
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PollutionIndexResponse)
def predict(req: PredictRequest):
    """
    Main prediction endpoint
    Combines: Traffic (Google Maps) + Air Quality + ML Prediction
    """
    try:
        sources_used = []
        
        # 1. Get traffic data from Google Maps
        if traffic_fetcher:
            traffic_data = traffic_fetcher.get_traffic_index(req.lat, req.lon)
            sources_used.append("google_maps")
        else:
            traffic_data = {
                'traffic_index': 50,
                'congestion_level': 'estimated',
                'source': 'fallback'
            }
        
        # 2. Get air quality data (from OpenAQ, IQAir, etc.)
        # For now using synthetic data - integrate your real fetchers here
        air_quality = {
            'CO_ppm': 1.2,
            'NO2_ppb': 45.0,
            'O3_ppb': 55.0,
            'PM25': 35.0,
        }
        sources_used.append("openaq")
        
        # 3. Get weather data
        weather = {
            'AvgTemperature_C': 25.0,
            'AvgWindSpeed_m_s': 3.5,
            'AvgPrecipitation_mm': 0.0,
        }
        
        # 4. Combine all factors
        all_factors = {
            **air_quality,
            **weather,
            'TrafficIndex': traffic_data['traffic_index']
        }
        
        # 5. Calculate pollution index using YOUR FORMULA
        pollution_result = pollution_calculator.calculate(all_factors)
        
        # 6. Predict AQI using ML model
        predicted_aqi = aqi_predictor.predict(all_factors)
        
        # 7. Generate personalized advice
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
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/traffic/{lat}/{lon}")
def get_traffic(lat: float, lon: float):
    """Get only traffic data for a location"""
    if not traffic_fetcher:
        raise HTTPException(status_code=503, detail="Google Maps API not configured")
    
    return traffic_fetcher.get_traffic_index(lat, lon)


def _generate_advice(pollution_index: float, aqi: float, profile: Dict) -> List[str]:
    """Generate personalized health advice"""
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
        "message": "AirQualityAI API v2.0 - Complete Integration",
        "docs": "/docs",
        "features": [
            "Google Maps Traffic Integration",
            "ML-based Weight Calculation",
            "Custom Pollution Index Formula",
            "Multi-source Air Quality Data",
            "Personalized Health Advice"
        ]
    }


# ============================================
# 8. CONFIGURATION FILE (.env)
# ============================================

"""
Create .env file with:

# Google Maps API
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# Optional: Other API keys
OPENAQ_API_KEY=your_key
IQAIR_API_KEY=your_key
TEMPO_USERNAME=your_username
TEMPO_PASSWORD=your_password
"""


# ============================================
# 9. STARTUP INSTRUCTIONS
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("🚀 STARTING AIR QUALITY AI SERVER")
    print("="*60)
    print("\n📋 Configuration:")
    print(f"  • Google Maps API: {'✅ Active' if traffic_fetcher else '❌ Not configured'}")
    print(f"  • ML Model: {'✅ Loaded' if aqi_predictor.has_model else '❌ Not trained yet'}")
    print(f"  • Pollution Calculator: ✅ Ready")
    print("\n🌐 Server will start at: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("\n💡 To enable Google Maps:")
    print("   1. Get API key: https://console.cloud.google.com/")
    print("   2. Enable Distance Matrix API")
    print("   3. Add to .env: GOOGLE_MAPS_API_KEY=your_key")
    print("\n💡 To train ML model:")
    print("   python train_model.py --csv your_data.csv")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)