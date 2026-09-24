"""Build the model inputs (features) for next-hour solar forecasting.

Each row is one TARGET hour h. The goal is to predict the energy in hour h.
Features only use information that is available at the end of hour h-1, plus things
that are known in advance (clock time and sun position).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from src.data import load_hourly

ROOT = Path(__file__).resolve().parents[1]
WEATHER_FILE = ROOT / "data" / "raw" / "weather_openmeteo_2017_2019.csv"

LATITUDE, LONGITUDE, ALTITUDE_M = 37.427, -122.174, 30

# Open-Meteo gives these as the average/total of the PRECEDING hour, so they are
# moved back one hour to line up with our "start of hour" labels.
PRECEDING_HOUR_VARS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "precipitation"]

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "surface_pressure",
                "precipitation", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
                "cloud_cover_high", "wind_speed_10m", "shortwave_radiation",
                "direct_radiation", "diffuse_radiation"]


def load_weather() -> pd.DataFrame:
    """Hourly weather in UTC, aligned so each row describes the hour that STARTS at its label."""
    w = pd.read_csv(WEATHER_FILE, index_col="time", parse_dates=True)
    w[PRECEDING_HOUR_VARS] = w[PRECEDING_HOUR_VARS].shift(-1)
    return w


def sun_and_clearsky(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Sun elevation and clear-sky sunlight (GHI) at the MIDDLE of each hour, using pvlib."""
    location = pvlib.location.Location(LATITUDE, LONGITUDE, altitude=ALTITUDE_M)
    mid = index + pd.Timedelta(minutes=30)
    solpos = location.get_solarposition(mid)
    clearsky = location.get_clearsky(mid, model="ineichen")
    return pd.DataFrame({
        "sun_elevation": solpos["apparent_elevation"].to_numpy(),
        "clearsky_ghi": clearsky["ghi"].to_numpy(),
    }, index=index)


def build_features() -> pd.DataFrame:
    """One row per target hour h (UTC). Column 'target' is the energy (kWh) in hour h."""
    y = load_hourly().tz_convert("UTC")
    weather = load_weather().reindex(y.index)
    sun = sun_and_clearsky(y.index)
    local = y.index.tz_convert("America/Los_Angeles")

    X = pd.DataFrame(index=y.index)
    X["target"] = y

    # --- Known in advance for hour h: clock and sun ---------------------------
    X["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    X["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    X["sun_elevation"] = sun["sun_elevation"]
    X["clearsky_ghi"] = sun["clearsky_ghi"]

    # --- Past solar power (known at the end of hour h-1) ----------------------
    X["pv_lag1"] = y.shift(1)
    X["pv_lag2"] = y.shift(2)
    X["pv_lag24"] = y.shift(24)            # same hour yesterday
    csi = (y / sun["clearsky_ghi"]).where(sun["clearsky_ghi"] > 50)   # "clear-sky index" (kWh per W/m²)
    X["csi_lag1"] = csi.shift(1)
    X["pv_mean_last3"] = y.shift(1).rolling(3, min_periods=2).mean()

    # --- Weather observed in the previous hour (realistic) ---------------------
    for v in WEATHER_VARS:
        X[f"obs_{v}"] = weather[v].shift(1)

    # --- Weather for hour h itself = a PERFECT weather forecast ----------------
    # Only used in an "upper bound" experiment. A real system would use a weather
    # forecast here, which is less accurate than these values.
    for v in WEATHER_VARS:
        X[f"fc_{v}"] = weather[v]

    return X


FEATURE_SETS = {
    "time_sun": ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "sun_elevation", "clearsky_ghi"],
    "past_pv": ["pv_lag1", "pv_lag2", "pv_lag24", "csi_lag1", "pv_mean_last3"],
    "obs_weather": [f"obs_{v}" for v in WEATHER_VARS],
    "perfect_forecast_weather": [f"fc_{v}" for v in WEATHER_VARS],
}
