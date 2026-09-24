"""Training helpers shared by the model notebooks."""
import numpy as np
import pandas as pd

from src.data import split


def fit_predict(model, X: pd.DataFrame, cols: list, train_splits: list) -> tuple[pd.Series, object]:
    """Train `model` on the given splits, then predict EVERY hour in X.

    - Trains only on hours when the sun is up (night is always 0, nothing to learn).
    - Predictions below 0 are set to 0 (a panel can't produce negative energy).
    - Night hours are set to exactly 0.
    """
    train = pd.concat([split(X, s) for s in train_splits])
    train = train[(train["sun_elevation"] > 0) & train["target"].notna()]
    model.fit(train[cols], train["target"])

    pred = pd.Series(np.clip(model.predict(X[cols]), 0, None), index=X.index)
    pred[X["sun_elevation"] <= 0] = 0.0
    return pred, model
