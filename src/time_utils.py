"""Explicit timezone and hourly-data contracts shared by training and serving."""

from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from src.config import TIMEZONE


def now_local():
    return datetime.now(ZoneInfo(TIMEZONE))


def local_timestamp(value):
    ts = pd.Timestamp(value)
    return ts.tz_localize(TIMEZONE) if ts.tzinfo is None else ts.tz_convert(TIMEZONE)


def day_hours(value):
    return pd.date_range(local_timestamp(value).normalize(), periods=24, freq="h", name="date")


def utc_naive(value):
    """DB contract: naive UTC, including SQLite. Never store naive local time."""
    return local_timestamp(value).tz_convert("UTC").tz_localize(None).to_pydatetime()


def utc_iso(value):
    return value.replace(tzinfo=ZoneInfo("UTC")).isoformat() if value is not None else None


def hourly_frame(frame, column, *, allow_missing=False):
    if "date" not in frame or column not in frame:
        raise ValueError(f"Expected date and {column} columns")
    df = frame[["date", column]].copy()
    if df.empty:
        raise ValueError(f"Empty {column} data")
    df["date"] = pd.DatetimeIndex([local_timestamp(v) for v in df["date"]])
    if df["date"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {column}")
    if (df["date"] != df["date"].dt.floor("h")).any():
        raise ValueError(f"Non-hourly timestamps in {column}")
    df[column] = pd.to_numeric(df[column], errors="raise")
    values = df[column].to_numpy(dtype=float)
    if np.isinf(values).any() or (not allow_missing and np.isnan(values).any()):
        raise ValueError(f"Non-finite values in {column}")
    if (values < 0).any():
        raise ValueError(f"Negative values in {column}")
    return df.set_index("date").sort_index()
