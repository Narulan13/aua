from app.fetchers.openaq import OpenAQFetcher
from app.models import GeoLocation

if __name__ == "__main__":
    fetcher = OpenAQFetcher()
    location = GeoLocation(latitude=43.222, longitude=76.8512)  # Almaty

    measurements = fetcher.fetch(location)

    if not measurements:
        print("No data found or API key invalid.")
    else:
        for m in measurements:
            print(f"{m.pollutant.name}: {m.value} {m.unit} at {m.timestamp}")
