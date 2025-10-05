import requests
from typing import List
from datetime import datetime
from app.fetchers.base import DataFetcher
from app.models import GeoLocation, Measurement, DataSource, PollutantType
from app.utils.unit_converter import convert_to_standard_units


class IQAirFetcher(DataFetcher):
    """
    Fetcher для IQAir API
    IQAir возвращает US AQI, нужно конвертировать в концентрацию
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.airvisual.com/v2"
    
    def fetch(self, location: GeoLocation) -> List[Measurement]:
        try:
            params = {
                'lat': location.latitude,
                'lon': location.longitude,
                'key': self.api_key
            }
            
            response = requests.get(
                f"{self.base_url}/nearest_city",
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"IQAir API error: {response.status_code}")
                return []
            
            data = response.json().get('data', {})
            current = data.get('current', {}).get('pollution', {})
            
            # IQAir возвращает US AQI, конвертируем в PM2.5
            us_aqi = current.get('aqius', 0)
            pm25_value = self._aqi_to_pm25(us_aqi)
            
            # Timestamp
            try:
                timestamp_str = current.get('ts', '')
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    timestamp = datetime.now()
            except:
                timestamp = datetime.now()
            
            measurements = [Measurement(
                pollutant=PollutantType.PM25,
                value=pm25_value,
                unit='μg/m³',
                timestamp=timestamp,
                source=DataSource.IQAIR,
                confidence=0.85
            )]
            
            print(f"IQAir: AQI {us_aqi} → PM2.5 {pm25_value:.1f} μg/m³")
            return measurements
            
        except Exception as e:
            print(f"IQAir fetcher error: {e}")
            return []
    
    def _aqi_to_pm25(self, aqi: int) -> float:
        """
        Обратная конвертация: US AQI → PM2.5 концентрация
        
        AQI breakpoints (EPA):
        0-50:    0-12.0 μg/m³
        51-100:  12.1-35.4 μg/m³
        101-150: 35.5-55.4 μg/m³
        151-200: 55.5-150.4 μg/m³
        201-300: 150.5-250.4 μg/m³
        301-500: 250.5-500.4 μg/m³
        """
        if aqi <= 50:
            return (aqi / 50) * 12.0
        elif aqi <= 100:
            return 12.0 + ((aqi - 50) / 50) * (35.4 - 12.0)
        elif aqi <= 150:
            return 35.4 + ((aqi - 100) / 50) * (55.4 - 35.4)
        elif aqi <= 200:
            return 55.4 + ((aqi - 150) / 50) * (150.4 - 55.4)
        elif aqi <= 300:
            return 150.4 + ((aqi - 200) / 100) * (250.4 - 150.4)
        else:
            return 250.4 + ((aqi - 300) / 200) * (500.4 - 250.4)