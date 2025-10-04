class DataFetcher:
    def fetch(self, location: GeoLocation) -> List[Measurement]:
        raise NotImplementedError
