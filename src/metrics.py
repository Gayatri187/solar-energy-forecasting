"""Error metrics used for every model in this project."""
import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error: the average size of the mistake (kWh)."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error: like MAE, but punishes big mistakes more (kWh)."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def skill(rmse_model: float, rmse_reference: float) -> float:
    """Forecast skill vs a reference model (usually persistence).

    0 = no better than the reference, 1 = perfect, negative = worse than the reference.
    """
    return 1.0 - rmse_model / rmse_reference


def evaluate(y_true: pd.Series, forecasts: dict, mask: pd.Series, reference: str = "Persistence") -> pd.DataFrame:
    """Score several forecasts on the SAME hours (only where every forecast and the truth exist)."""
    frame = pd.DataFrame({"actual": y_true, **forecasts})
    frame = frame[mask.reindex(frame.index, fill_value=False)].dropna()

    rows = {}
    for name in forecasts:
        rows[name] = {"MAE (kWh)": mae(frame["actual"], frame[name]),
                      "RMSE (kWh)": rmse(frame["actual"], frame[name])}
    table = pd.DataFrame(rows).T
    table["nRMSE (%)"] = 100 * table["RMSE (kWh)"] / frame["actual"].mean()
    table["Skill vs persistence"] = [skill(r, table.loc[reference, "RMSE (kWh)"]) for r in table["RMSE (kWh)"]]
    table.attrs["n_hours"] = len(frame)
    return table.round(3)
