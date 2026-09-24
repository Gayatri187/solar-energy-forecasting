"""Sky-image dataset for 15-minute-ahead solar forecasting (SKIPP'D benchmark).

Each sample is a time t (one minute). Inputs:
  - two sky images: now (t) and LAG minutes earlier (t - LAG), stacked as 6 channels
  - numbers: recent PV power, sun position and clear-sky sunlight
Target: PV power (kW) at t + HORIZON minutes.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SKIPPD_DIR = ROOT / "data" / "raw" / "skippd"
TZ = "America/Los_Angeles"
LAT, LON, ALT = 37.427, -122.174, 30

HORIZON = 15      # minutes ahead to predict
LAG = 6           # minutes between the two images
PV_LAGS = [0, 5, 10, 15]

NUMERIC_FEATURES = [f"pv_lag{l}" for l in PV_LAGS] + [
    "csi_now", "sun_elev_now", "sun_elev_target", "ghi_cs_now", "ghi_cs_target", "hour_sin", "hour_cos"]


def read_raw_times(files) -> pd.DataFrame:
    """Original timestamps and PV, with no corrections (used to show the time-offset check)."""
    return pd.concat([pq.read_table(f, columns=["time", "pv"]).to_pandas() for f in files], ignore_index=True)


def _read_time_pv(files):
    parts = []
    for f in files:
        t = pq.read_table(f, columns=["time", "pv"]).to_pandas()
        t["file"] = str(f)
        t["row"] = np.arange(len(t))
        parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    local = df["time"].dt.tz_convert(TZ)
    # Data-quality fix: during daylight-saving time the timestamps in this release are 1 hour late
    # (checked in notebook 06: the power peak only lines up with solar noon after this correction).
    in_dst = np.array([t.dst() != pd.Timedelta(0) for t in local])
    df["time"] = (local - pd.to_timedelta(in_dst.astype(int), unit="h")).dt.floor("min")
    return df.drop_duplicates("time").set_index("time").sort_index()


def _clearsky(times: pd.DatetimeIndex) -> pd.DataFrame:
    loc = pvlib.location.Location(LAT, LON, altitude=ALT)
    pos = loc.get_solarposition(times)
    cs = loc.get_clearsky(times, model="ineichen")
    return pd.DataFrame({"sun_elev": pos["apparent_elevation"].to_numpy(), "ghi_cs": cs["ghi"].to_numpy()}, index=times)


def build_samples(files, stride: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (samples, frame_index).

    samples: one row per usable time t, with numeric features, the target and the two image times.
    stride:  keep only every `stride`-th minute (saves memory when training).
    """
    frames = _read_time_pv(files)
    pv = frames["pv"]

    def at(offset_min):
        return pv.reindex(pv.index + pd.Timedelta(minutes=offset_min)).to_numpy()

    s = pd.DataFrame(index=pv.index)
    for l in PV_LAGS:
        s[f"pv_lag{l}"] = at(-l)
    s["target"] = at(HORIZON)
    s["img_now"] = s.index
    s["img_before"] = s.index - pd.Timedelta(minutes=LAG)
    s["has_before"] = s["img_before"].isin(frames.index)

    s = s[s["has_before"]].dropna()
    if stride > 1:
        s = s[(s.index.hour * 60 + s.index.minute) % stride == 0]

    now = _clearsky(s.index)
    tgt = _clearsky(s.index + pd.Timedelta(minutes=HORIZON))
    s["sun_elev_now"], s["ghi_cs_now"] = now["sun_elev"].to_numpy(), now["ghi_cs"].to_numpy()
    s["sun_elev_target"], s["ghi_cs_target"] = tgt["sun_elev"].to_numpy(), tgt["ghi_cs"].to_numpy()
    s["csi_now"] = (s["pv_lag0"] / s["ghi_cs_now"].clip(lower=50)).clip(0, 0.08)
    s["hour_sin"] = np.sin(2 * np.pi * (s.index.hour + s.index.minute / 60) / 24)
    s["hour_cos"] = np.cos(2 * np.pi * (s.index.hour + s.index.minute / 60) / 24)
    s["day"] = s.index.date
    return s.drop(columns="has_before"), frames


def smart_persistence(s: pd.DataFrame) -> np.ndarray:
    """Keep the current 'clear-sky index' and follow the sun's curve to t + HORIZON."""
    ratio = s["ghi_cs_target"] / s["ghi_cs_now"].clip(lower=50)
    return (s["pv_lag0"] * ratio.clip(0, 3)).to_numpy()


def load_images(frames: pd.DataFrame, times) -> tuple[np.ndarray, dict]:
    """Decode only the images needed. Returns (array N x 64 x 64 x 3 uint8, {time: position})."""
    need = frames.loc[pd.DatetimeIndex(sorted(set(times)))]
    images = np.empty((len(need), 64, 64, 3), dtype=np.uint8)
    position = {}
    k = 0
    for f, group in need.groupby("file", sort=False):
        table = pq.read_table(f, columns=["image"])
        col = table.column("image")
        for t, row in zip(group.index, group["row"]):
            png = col[int(row)].as_py()["bytes"]
            images[k] = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            position[t] = k
            k += 1
        del table, col
    return images, position
