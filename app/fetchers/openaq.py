import requests
from typing import List, Optional
from datetime import datetime
from base import DataFetcher
from models import GeoLocation, Measurement, DataSource, PollutantType


class OpenAQFetcher(DataFetcher):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.openaq.org/v2"
    
    def fetch(self, location: GeoLocation) -> List[Measurement]:
        try:
            params = {
                'coordinates': f"{location.latitude},{location.longitude}",
                'radius': 25000,
                'limit': 100,
                'order_by': 'datetime',
                'sort': 'desc'
            }
            
            headers = {}
            if self.api_key:
                headers['X-API-Key'] = self.api_key
            
            response = requests.get(
                f"{self.base_url}/latest",
                params=params,
                headers=headers,
                timeout=10
            )
            
            if response.status_code != 200:
                return []
            
            data = response.json()
            measurements = []
            
            pollutant_map = {
                'pm25': PollutantType.PM25,
                'pm10': PollutantType.PM10,
                'no2': PollutantType.NO2,
                'so2': PollutantType.SO2,
                'o3': PollutantType.O3,
                'co': PollutantType.CO
            }
            
            for result in data.get('results', []):
                for m in result.get('measurements', []):
                    param = m.get('parameter', '').lower()
                    if param in pollutant_map:
                        try:
                            timestamp_str = m.get('lastUpdated', '')
                            if timestamp_str:
                                timestamp_str = timestamp_str.replace('Z', '+00:00')
                                timestamp = datetime.fromisoformat(timestamp_str)
                            else:
                                timestamp = datetime.now()
                            
                            measurements.append(Measurement(
                                pollutant=pollutant_map[param],
                                value=m.get('value', 0),
                                unit=m.get('unit', 'µg/m³'),
                                timestamp=timestamp,
                                source=DataSource.OPENAQ,
                                confidence=0.9
                            ))
                        except Exception as e:
                            print(f"Error parsing: {e}")
                            continue
            
            return measurements
            
        except Exception as e:
            print(f"OpenAQ error: {e}")
            return []