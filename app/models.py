
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class DataSource(Enum):
    TEMPO = "tempo"
    OPENAQ = "openaq"
    IQAIR = "iqair"
    PM25_API = "pm25"


class PollutantType(Enum):
    PM25 = "pm25"
    PM10 = "pm10"
    NO2 = "no2"
    SO2 = "so2"
    O3 = "o3"
    CO = "co"


@dataclass
class GeoLocation:
    latitude: float
    longitude: float
    city: Optional[str] = None
    country: Optional[str] = None


@dataclass
class Measurement:
    pollutant: PollutantType
    value: float
    unit: str
    timestamp: datetime
    source: DataSource
    confidence: Optional[float] = None


@dataclass
class AirQualitySnapshot:
    location: GeoLocation
    timestamp: datetime
    measurements: Dict[str, List[Measurement]] = field(default_factory=dict)
    aqi: Optional[int] = None
    
    def get_pollutant_avg(self, pollutant: str) -> Optional[float]:
        if pollutant not in self.measurements:
            return None
        values = [m.value for m in self.measurements[pollutant]]
        return sum(values) / len(values) if values else None
