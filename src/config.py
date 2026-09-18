"""Shared, environment-driven configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = "Europe/Istanbul"
FEATURE_VERSION = "consumption-calendar-v2"
FEATURE_COLUMNS = [
    "hour",
    "dayofweek",
    "dayofyear",
    "month",
    "quarter",
    "year",
    "is_holiday",
    "lag_48",
    "lag_72",
    "lag_168",
    "roll_mean_1d",
    "roll_std_1d",
    "roll_mean_1w",
    "roll_std_1w",
]
TARGET_COLUMN = "consumption"
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "model.json"))
DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL") or "sqlite:///monitoring.db"
EPIAS_USERNAME = os.getenv("EPIAS_USERNAME")
EPIAS_PASSWORD = os.getenv("EPIAS_PASSWORD")
FORECAST_CUTOFF_HOUR = 12
