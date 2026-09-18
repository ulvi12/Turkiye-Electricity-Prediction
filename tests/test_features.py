import numpy as np
import pandas as pd
import pytest
from src.config import FEATURE_COLUMNS
from src.features import FeatureEngineer
from src.inference import InferencePipeline
from src.time_utils import day_hours, hourly_frame


def test_gap_stays_a_gap():
    dates = pd.date_range("2025-01-01", periods=300, freq="h", tz="Europe/Istanbul")
    frame = pd.DataFrame({"date": dates, "consumption": np.arange(300)})
    processed = FeatureEngineer().process_data(frame.drop(index=10))
    assert np.isnan(processed.loc[dates[58], "lag_48"])
    assert processed.loc[dates[60], "lag_48"] == 12
    assert len(processed) == 300


def test_future_observations_cannot_change_past_features():
    dates = pd.date_range("2025-01-01", periods=300, freq="h", tz="Europe/Istanbul")
    frame = pd.DataFrame({"date": dates, "consumption": np.arange(300)})
    before = FeatureEngineer().process_data(frame)
    frame.loc[frame.index >= 253, "consumption"] = 1e9
    after = FeatureEngineer().process_data(frame)
    pd.testing.assert_series_equal(
        before.loc[dates[299], FEATURE_COLUMNS], after.loc[dates[299], FEATURE_COLUMNS]
    )


@pytest.mark.parametrize("problem", ["duplicate", "negative", "infinity", "off_hour"])
def test_invalid_history_rejected(history, problem):
    if problem == "duplicate":
        history = pd.concat([history, history.iloc[:1]])
    elif problem == "negative":
        history.loc[0, "consumption"] = -1
    elif problem == "infinity":
        history.loc[0, "consumption"] = np.inf
    else:
        history.loc[0, "date"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):
        hourly_frame(history, "consumption")


def test_timezone_equivalence(history):
    utc = history.copy()
    utc["date"] = utc.date.dt.tz_convert("UTC")
    pd.testing.assert_frame_equal(hourly_frame(history, "consumption"), hourly_frame(utc, "consumption"))


def test_real_model_produces_24_finite_forecasts(history):
    pipeline = InferencePipeline()
    result = pipeline.predict_from_history("2026-01-01", history)
    assert len(result.predictions) == 24
    assert np.isfinite(result.predictions.prediction).all()
    assert len(result.input_snapshot["features"]) == 24
    assert pd.DatetimeIndex(result.predictions.date).equals(day_hours("2026-01-01"))


def test_missing_required_history_fails(history):
    with pytest.raises(ValueError, match="Incomplete"):
        InferencePipeline().predict_from_history("2026-01-01", history.drop(index=50))


def test_inference_discards_future_inputs(history):
    pipeline = InferencePipeline()
    expected = pipeline.predict_from_history("2026-01-01", history)
    future = pd.DataFrame({"date": day_hours("2026-01-01"), "consumption": 1e9})
    actual = pipeline.predict_from_history("2026-01-01", pd.concat([history, future]))
    pd.testing.assert_frame_equal(expected.predictions, actual.predictions)


def test_training_overlap_rejected(history):
    with pytest.raises(ValueError, match="training period"):
        InferencePipeline().predict_from_history("2025-12-31", history)
