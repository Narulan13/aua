# app/utils/data_loader.py
import os
import requests
import netCDF4 as nc
import numpy as np
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("USER_NASA")
PASSWORD = os.getenv("PASS_NASA")


def fetch_tempo_no2(lat, lon, date="2023-07-01"):
    """
    Fetch TEMPO NO2 column for given coordinates from NASA Earthdata.
    
    Args:
        lat (float): latitude
        lon (float): longitude
        date (str): YYYY-MM-DD
    
    Returns:
        float or None: NO2 column in mol/m²
    """
    if not USERNAME or not PASSWORD:
        print(" NASA credentials not set in .env")
        return None

    # Пример URL — нужно подставлять правильный путь к файлу по дате
    url = f"https://data.gesdisc.earthdata.nasa.gov/data/TEMPO_L2/NO2/{date[:4]}/TEMPO_L2_NO2_{date.replace('-', '')}T1200Z.nc"

    try:
        # Скачиваем файл в память
        r = requests.get(url, auth=(USERNAME, PASSWORD))
        if r.status_code != 200:
            print("Download error:", r.status_code)
            return None

        # Открываем NetCDF из памяти
        ds = nc.Dataset("inmemory", memory=r.content)

        lats = ds.variables['latitude'][:]
        lons = ds.variables['longitude'][:]
        no2  = ds.variables['nitrogendioxide_tropospheric_column'][:]

        # Находим ближайший пиксель
        dist = (lats - lat)**2 + (lons - lon)**2
        idx = np.unravel_index(dist.argmin(), dist.shape)
        return float(no2[idx])

    except Exception as e:
        print("⚠️ Error fetching TEMPO data:", e)
        return None


def get_satellite_proxy(lat, lon):
    """
    Возвращает реальный NO2 с TEMPO. Если ошибка — возвращает заглушку.
    """
    no2_value = fetch_tempo_no2(lat, lon)
    if no2_value is None:
        # Заглушка, если не удалось скачать
        return np.random.rand() * 20
    return no2_value
