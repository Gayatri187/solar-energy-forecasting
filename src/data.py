"""Loading the cleaned data and splitting it by time.

Every notebook uses these functions, so all models see exactly the same data and splits.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
LOCAL_TZ = "America/Los_Angeles"

# Split by YEAR (local time). Never shuffle time-series data.
SPLITS = {
    "train": ("2017-01-01", "2017-12-31"),
    "val":   ("2018-01-01", "2018-12-31"),
    "test":  ("2019-01-01", "2019-12-31"),
}


def load_hourly() -> pd.Series:
    """Hourly energy (kWh), indexed by local California time."""
    df = pd.read_parquet(PROCESSED / "pv_hourly_clean.parquet")
    s = df["energy_kwh"]
    s.index = s.index.tz_convert(LOCAL_TZ).rename("time")
    return s


def split(obj, name: str):
    """Return the part of a Series/DataFrame that belongs to one split."""
    start, end = SPLITS[name]
    return obj.loc[start:end]
