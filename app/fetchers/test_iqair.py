from app.fetchers.iqair import IQAirFetcher
from app.models import GeoLocation
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("IQAIR_API_KEY") or "91f0d257-1bdd-4630-894b-9c2bdf3498bf"
    location = GeoLocation(latitude=43.222, longitude=76.8512)  # Almaty

    fetcher = IQAirFetcher(api_key)
    measurements = fetcher.fetch(location)

    if not measurements:
        print("No data found or API key invalid.")
    else:
        for m in measurements:
            print(f"{m.pollutant.name}: {m.value} {m.unit} at {m.timestamp}")
