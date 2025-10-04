import requests
import netCDF4 as nc
import numpy as np
from io import BytesIO


class TEMPOFetcher(DataFetcher):
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.base_url = "https://data.gesdisc.earthdata.nasa.gov/data/TEMPO_L2"
    
    def fetch(self, location: GeoLocation) -> List[Measurement]:
        try:
            date_str = datetime.now().strftime("%Y%m%d")
            url = f"{self.base_url}/NO2/{date_str[:4]}/TEMPO_L2_NO2_{date_str}T1200Z.nc"
            
            response = requests.get(
                url, 
                auth=(self.username, self.password),
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"TEMPO fetch failed: {response.status_code}")
                return []
            
            ds = nc.Dataset("inmemory", memory=response.content)
            
            lats = ds.variables['latitude'][:]
            lons = ds.variables['longitude'][:]
            no2_data = ds.variables['nitrogendioxide_tropospheric_column'][:]
            
            dist = (lats - location.latitude)**2 + (lons - location.longitude)**2
            idx = np.unravel_index(dist.argmin(), dist.shape)
            no2_value = float(no2_data[idx])
            
            ds.close()
            
            return [Measurement(
                pollutant=PollutantType.NO2,
                value=no2_value,
                unit="mol/m²",
                timestamp=datetime.now(),
                source=DataSource.TEMPO,
                confidence=0.85
            )]
            
        except Exception as e:
            print(f"TEMPO error: {e}")
            return []