import requests


class IQAirFetcher(DataFetcher):
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
                return []
            
            data = response.json().get('data', {})
            current = data.get('current', {}).get('pollution', {})
            
            pm25_value = current.get('aqius', 0) * 0.4  # Convert AQI to µg/m³ (approximate)
            
            return [Measurement(
                pollutant=PollutantType.PM25,
                value=pm25_value,
                unit='µg/m³',
                timestamp=datetime.fromisoformat(current.get('ts', '').replace('Z', '+00:00')),
                source=DataSource.IQAIR,
                confidence=0.85
            )]
            
        except Exception as e:
            print(f"IQAir error: {e}")
            return []
