import requests
from typing import List, Optional
from datetime import datetime
from app.fetchers.base import DataFetcher
from app.models import GeoLocation, Measurement, DataSource, PollutantType
from app.utils.unit_converter import convert_to_standard_units


class OpenAQFetcher(DataFetcher):
    """
    Fetcher для OpenAQ API с правильной конвертацией единиц
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.openaq.org/v2"
    
    def fetch(self, location: GeoLocation) -> List[Measurement]:
        try:
            params = {
                'coordinates': f"{location.latitude},{location.longitude}",
                'radius': 25000,  # 25 km radius
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
                print(f"OpenAQ API error: {response.status_code}")
                return []
            
            data = response.json()
            measurements = []
            
            # Маппинг OpenAQ параметров на наши типы
            pollutant_map = {
                'pm25': PollutantType.PM25,
                'pm10': PollutantType.PM10,
                'no2': PollutantType.NO2,
                'so2': PollutantType.SO2,
                'o3': PollutantType.O3,
                'co': PollutantType.CO
            }
            
            # Стандартные единицы для нашей модели
            standard_units = {
                'pm25': 'μg/m³',
                'pm10': 'μg/m³',
                'no2': 'ppb',
                'so2': 'ppb',
                'o3': 'ppb',
                'co': 'ppm'
            }
            
            for result in data.get('results', []):
                for m in result.get('measurements', []):
                    param = m.get('parameter', '').lower()
                    
                    if param in pollutant_map:
                        try:
                            raw_value = m.get('value', 0)
                            raw_unit = m.get('unit', '')
                            
                            # КРИТИЧНО: Конвертируем в стандартные единицы
                            converted_value = convert_to_standard_units(
                                param, 
                                raw_value, 
                                raw_unit
                            )
                            
                            # Timestamp
                            timestamp_str = m.get('lastUpdated', '')
                            if timestamp_str:
                                timestamp_str = timestamp_str.replace('Z', '+00:00')
                                timestamp = datetime.fromisoformat(timestamp_str)
                            else:
                                timestamp = datetime.now()
                            
                            measurements.append(Measurement(
                                pollutant=pollutant_map[param],
                                value=converted_value,
                                unit=standard_units[param],  # Стандартная единица
                                timestamp=timestamp,
                                source=DataSource.OPENAQ,
                                confidence=0.9
                            ))
                            
                        except Exception as e:
                            print(f"Error parsing OpenAQ measurement: {e}")
                            continue
            
            print(f"OpenAQ: Fetched {len(measurements)} measurements")
            return measurements
            
        except Exception as e:
            print(f"OpenAQ fetcher error: {e}")
            return []