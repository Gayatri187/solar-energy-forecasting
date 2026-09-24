"""The four baseline forecasts from notebook 02, as reusable functions."""
import pandas as pd

from src.data import split


def climatology(y: pd.Series) -> pd.Series:
    """Average energy for each (month, hour), learned from the training year only."""
    train = split(y, "train")
    table = train.groupby([train.index.month, train.index.hour]).mean()
    return pd.Series(table.reindex(list(zip(y.index.month, y.index.hour))).to_numpy(), index=y.index)


def make_baselines(y: pd.Series) -> tuple[dict, pd.Series]:
    """Return (forecasts, daytime_mask). `y` must be indexed by LOCAL time."""
    clim = climatology(y)
    ratio = (y / clim).where(clim >= 0.5).clip(0, 1.5)
    forecasts = {
        "Persistence": y.shift(1),
        "Yesterday": y.shift(24),
        "Climatology": clim,
        "Smart persistence": (ratio.shift(1) * clim).fillna(clim),
    }
    daytime = clim >= 0.5
    return forecasts, daytime
