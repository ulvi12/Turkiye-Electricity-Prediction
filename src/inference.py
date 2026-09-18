"""Generate complete forecasts and preserve the exact feature and input snapshot."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from src.config import FEATURE_COLUMNS, FEATURE_VERSION, MODEL_PATH
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.time_utils import day_hours, hourly_frame, local_timestamp


@dataclass
class ForecastResult:
    target_date: str
    model_version: str
    predictions: pd.DataFrame
    input_snapshot: dict


class InferencePipeline:
    def __init__(self, model_path=None, loader=None):
        path = Path(model_path or MODEL_PATH)
        self.metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        digest = sha256(path.read_bytes()).hexdigest()
        if self.metadata.get("model_sha256") != digest:
            raise ValueError("Model checksum does not match its metadata")
        if self.metadata.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Model feature version is incompatible; retrain the model")
        if not self.metadata.get("release_gate_passed"):
            raise ValueError("Model did not pass the chronological evaluation release gate")
        self.model = xgb.XGBRegressor()
        self.model.load_model(path)
        if self.model.get_booster().feature_names != FEATURE_COLUMNS:
            raise ValueError("Model feature columns do not match the feature pipeline")
        self.model_version = digest
        self.data_loader = loader or DataLoader()

    def predict(self, target_date):
        target = local_timestamp(target_date).normalize()
        history = self.data_loader.get_realtime_consumption(
            target - pd.Timedelta(days=10), target - pd.Timedelta(days=2)
        )
        return self.predict_from_history(target, history)

    def predict_from_history(self, target_date, history):
        target = local_timestamp(target_date).normalize()
        if target <= local_timestamp(self.metadata["training_end"]):
            raise ValueError("Forecast target must be after the model training period")
        past = hourly_frame(history, "consumption")
        past = past.loc[
            (past.index >= target - pd.Timedelta(days=10)) & (past.index < target - pd.Timedelta(days=1))
        ]
        frame = past.reindex(
            pd.date_range(
                target - pd.Timedelta(days=10), target + pd.Timedelta(hours=23), freq="h", name="date"
            )
        )
        features = FeatureEngineer().process_data(frame).loc[day_hours(target), FEATURE_COLUMNS]
        if not np.isfinite(features.to_numpy(dtype=float)).all():
            raise ValueError("Incomplete consumption history: cannot produce all 24 forecast hours")
        predictions = self.model.predict(features)
        if len(predictions) != 24 or not np.isfinite(predictions).all() or (predictions < 0).any():
            raise ValueError("Model returned invalid predictions")
        snapshot = {
            "feature_version": FEATURE_VERSION,
            "history": [
                {"date": ts.isoformat(), "consumption": float(v)} for ts, v in past["consumption"].items()
            ],
            "features": json.loads(features.reset_index().to_json(orient="records", date_format="iso")),
        }
        return ForecastResult(
            str(target.date()),
            self.model_version,
            pd.DataFrame({"date": features.index, "prediction": predictions}),
            snapshot,
        )
