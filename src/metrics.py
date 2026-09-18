"""Metrics use explicit paired coverage; EPIAS comparisons share identical hours."""

import numpy as np
import pandas as pd


def scores(actual, prediction):
    y, p = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    if not len(y):
        return None
    error = y - p
    nonzero = y != 0
    return {
        "mae_mwh": float(np.abs(error).mean()),
        "rmse_mwh": float(np.sqrt((error**2).mean())),
        "mape_pct": float((np.abs(error[nonzero]) / np.abs(y[nonzero])).mean() * 100)
        if nonzero.any()
        else None,
        "hours": len(y),
        "mape_hours": int(nonzero.sum()),
    }


def evaluate_records(records, expected_hours=None):
    total = len(records)
    empty = {
        "forecast_hours": total,
        "expected_hours": expected_hours if expected_hours is not None else total,
        "observed_hours": 0,
        "comparison_hours": 0,
        "model": None,
        "paired_model": None,
        "epias": None,
    }
    if not records:
        return empty
    frame = pd.DataFrame(records)
    origins = {r.get("origin", "unspecified") for r in records}
    if len(origins) > 1:
        return {
            **empty,
            "by_origin": {
                origin: evaluate_records([r for r in records if r.get("origin", "unspecified") == origin])
                for origin in sorted(origins)
            },
            "note": "Different forecasting protocols are evaluated separately.",
        }
    empty["forecast_hours"] = int(frame.prediction.notna().sum())
    observed = frame.dropna(subset=["actual", "prediction"])
    paired = observed.dropna(subset=["epias_forecast"])
    return {
        **empty,
        "observed_hours": len(observed),
        "comparison_hours": len(paired),
        "model": scores(observed.actual, observed.prediction),
        "paired_model": scores(paired.actual, paired.prediction),
        "epias": scores(paired.actual, paired.epias_forecast),
    }
