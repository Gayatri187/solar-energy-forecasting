"""Tools for prediction intervals (ranges) and how to score them."""
import numpy as np
import pandas as pd


def coverage(y, lower, upper) -> float:
    """Share of hours where the true value falls inside [lower, upper]."""
    y, lower, upper = map(np.asarray, (y, lower, upper))
    return float(np.mean((y >= lower) & (y <= upper)))


def mean_width(lower, upper) -> float:
    """Average width of the range (kWh). Narrower is more useful, IF coverage holds."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))


def interval_score(y, lower, upper, alpha: float) -> float:
    """Winkler interval score: width + a penalty for every miss. Lower is better.

    Combines sharpness (narrow) and reliability (few misses) in one number.
    alpha = 0.2 for an 80% range, 0.1 for a 90% range.
    """
    y, lower, upper = map(np.asarray, (y, lower, upper))
    width = upper - lower
    below = (2 / alpha) * (lower - y) * (y < lower)
    above = (2 / alpha) * (y - upper) * (y > upper)
    return float(np.mean(width + below + above))


def conformal_margin(y_cal, lower_cal, upper_cal, level: float) -> float:
    """Conformalized Quantile Regression (Romano et al., 2019).

    On calibration data, measure how far the true values fall OUTSIDE the predicted range
    (negative if inside). The `level` quantile of these scores is the margin to add
    on both sides so that the range contains the truth `level` of the time.
    """
    y_cal, lower_cal, upper_cal = map(np.asarray, (y_cal, lower_cal, upper_cal))
    scores = np.maximum(lower_cal - y_cal, y_cal - upper_cal)
    n = len(scores)
    q = min(1.0, np.ceil((n + 1) * level) / n)   # small-sample correction
    return float(np.quantile(scores, q))
