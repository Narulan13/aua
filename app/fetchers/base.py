from typing import List
from app.models import GeoLocation, Measurement

class DataFetcher:
    def fetch(self, location: GeoLocation) -> List[Measurement]:
        raise NotImplementedError