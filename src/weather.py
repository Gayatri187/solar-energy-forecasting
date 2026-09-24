"""Download hourly historical weather for the Stanford PV site from Open-Meteo.

Run from the project folder:
    python -m src.weather

Source: Open-Meteo Historical Weather API (https://open-meteo.com), which is based on
ECMWF reanalysis data (ERA5 / IFS). Free, no API key. Attribution: "Weather data by Open-Meteo.com".
"""
import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "data" / "raw" / "weather_openmeteo_2017_2019.csv"

# Huang Engineering Center, Stanford University (location of the SKIPP'D PV system)
LATITUDE = 37.427
LONGITUDE = -122.174

VARIABLES = [
    "temperature_2m",        # air temperature (°C)
    "relative_humidity_2m",  # humidity (%)
    "dew_point_2m",          # dew point (°C)
    "surface_pressure",      # air pressure (hPa)
    "precipitation",         # rain in the preceding hour (mm)
    "cloud_cover",           # total cloud cover (%)
    "cloud_cover_low",       # low clouds (%)
    "cloud_cover_mid",       # middle clouds (%)
    "cloud_cover_high",      # high clouds (%)
    "wind_speed_10m",        # wind speed (km/h)
    "shortwave_radiation",   # sunlight reaching the ground, mean of preceding hour (W/m²)
    "direct_radiation",      # direct beam part (W/m²)
    "diffuse_radiation",     # scattered part (W/m²)
]


def download(start: str = "2017-01-01", end: str = "2019-12-31") -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARIABLES),
        "timezone": "UTC",   # always UTC, like the PV data
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    # Use certifi's certificates if available (fixes SSL errors with python.org installs on macOS)
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()

    with urllib.request.urlopen(url, timeout=120, context=context) as response:
        payload = json.load(response)

    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")


if __name__ == "__main__":
    print("Downloading weather from Open-Meteo ...")
    weather = download()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(OUT_FILE)
    print(f"Saved {len(weather)} hours x {weather.shape[1]} variables to {OUT_FILE}")
    print(weather.describe().round(1).T)
