from typing import List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.fetchers.base import DataFetcher
from app.models import GeoLocation, AirQualitySnapshot
import time


class AirQualityAggregator:
    """
    Ускоренный агрегатор с параллельными запросами
    """
    def __init__(self, max_workers: int = 5):
        self.fetchers: List[DataFetcher] = []
        self.max_workers = max_workers
    
    def add_fetcher(self, fetcher: DataFetcher):
        self.fetchers.append(fetcher)
    
    def get_snapshot(self, location: GeoLocation) -> AirQualitySnapshot:
        """
        УСКОРЕННАЯ версия: параллельные запросы к API
        """
        all_measurements = {}
        start_time = time.time()
        
        print(f"\n🔍 Fetching data from {len(self.fetchers)} sources...")
        
        # Параллельный запуск всех fetchers
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Запускаем все fetchers одновременно
            future_to_fetcher = {
                executor.submit(fetcher.fetch, location): fetcher 
                for fetcher in self.fetchers
            }
            
            # Собираем результаты по мере готовности
            for future in as_completed(future_to_fetcher):
                fetcher = future_to_fetcher[future]
                fetcher_name = fetcher.__class__.__name__
                
                try:
                    measurements = future.result(timeout=10)
                    
                    if measurements:
                        print(f"  ✅ {fetcher_name}: {len(measurements)} measurements")
                        
                        for m in measurements:
                            key = m.pollutant.value
                            if key not in all_measurements:
                                all_measurements[key] = []
                            all_measurements[key].append(m)
                    else:
                        print(f"  ⚠️ {fetcher_name}: No data")
                        
                except Exception as e:
                    print(f"  ❌ {fetcher_name} error: {e}")
        
        elapsed = time.time() - start_time
        print(f"⏱️ Data fetching completed in {elapsed:.2f}s\n")
        
        # Создаем snapshot
        snapshot = AirQualitySnapshot(
            location=location,
            timestamp=datetime.now(),
            measurements=all_measurements
        )
        
        # Вычисляем AQI на основе PM2.5
        snapshot.aqi = self._calculate_aqi(snapshot)
        
        return snapshot
    
    def _calculate_aqi(self, snapshot: AirQualitySnapshot) -> int:
        """
        Вычисление AQI на основе PM2.5 (стандарт EPA)
        """
        pm25 = snapshot.get_pollutant_avg('pm25')
        
        if pm25 is None:
            # Если нет PM2.5, пробуем оценить по NO2
            no2 = snapshot.get_pollutant_avg('no2')
            if no2 is None:
                return 50  # Default moderate
            # Примерная конвертация NO2 -> AQI
            return min(200, int(no2 * 0.5))
        
        # EPA AQI breakpoints для PM2.5
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
            return min(500, int(300 + ((500 - 300) / (500.4 - 250.4)) * (pm25 - 250.4)))