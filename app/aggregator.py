from typing import List
from datetime import datetime
from app.fetchers.base import DataFetcher
from app.models import GeoLocation, AirQualitySnapshot


class AirQualityAggregator:
    def __init__(self):
        self.fetchers: List[DataFetcher] = []
    
    def add_fetcher(self, fetcher: DataFetcher):
        self.fetchers.append(fetcher)
    
    def get_snapshot(self, location: GeoLocation) -> AirQualitySnapshot:
        all_measurements = {}
        
        for fetcher in self.fetchers:
            try:
                measurements = fetcher.fetch(location)
                for m in measurements:
                    key = m.pollutant.value
                    if key not in all_measurements:
                        all_measurements[key] = []
                    all_measurements[key].append(m)
            except Exception as e:
                print(f"Fetcher error: {e}")
        
        snapshot = AirQualitySnapshot(
            location=location,
            timestamp=datetime.now(),
            measurements=all_measurements
        )
        
        snapshot.aqi = self._calculate_aqi(snapshot)
        return snapshot
    
    def _calculate_aqi(self, snapshot: AirQualitySnapshot) -> int:
        pm25 = snapshot.get_pollutant_avg('pm25')
        if pm25 is None:
            return 50
        
        if pm25 <= 12.0:
            return int((50 / 12.0) * pm25)
        elif pm25 <= 35.4:
            return int(50 + ((100 - 50) / (35.4 - 12.0)) * (pm25 - 12.0))
        elif pm25 <= 55.4:
            return int(100 + ((150 - 100) / (55.4 - 35.4)) * (pm25 - 35.4))
        elif pm25 <= 150.4:
            return int(150 + ((200 - 150) / (150.4 - 55.4)) * (pm25 - 55.4))
        elif pm25 <= 250.4:
            return int(200 + ((300 - 200) / (250.4 - 150.4)) * (pm25 - 150.4))
        else:
            return int(300 + ((500 - 300) / (500.4 - 250.4)) * (pm25 - 250.4))