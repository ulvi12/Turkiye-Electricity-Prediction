"""One feature definition for training, evaluation, and live forecasts."""

import holidays
import pandas as pd
from src.time_utils import hourly_frame


class FeatureEngineer:
    def process_data(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "date" not in frame.columns:
            frame = frame.reset_index()
        df = hourly_frame(frame, "consumption", allow_missing=True)
        grid = pd.date_range(df.index.min(), df.index.max(), freq="h", name="date")
        df = df.reindex(grid)
        for name in ("hour", "dayofweek", "dayofyear", "month", "quarter", "year"):
            df[name] = getattr(df.index, name)
        calendar = holidays.Turkey(years=df.index.year.unique().tolist())
        df["is_holiday"] = [int(ts.date() in calendar) for ts in df.index]
        # Reindex BEFORE shifting: missing hours remain gaps rather than shortening lags.
        for hours in (48, 72, 168):
            df[f"lag_{hours}"] = df["consumption"].shift(hours)
        for label, hours in (("1d", 24), ("1w", 168)):
            window = df["lag_48"].rolling(hours, min_periods=hours)
            df[f"roll_mean_{label}"] = window.mean()
            df[f"roll_std_{label}"] = window.std()
        return df
