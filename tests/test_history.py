from datetime import date, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.main import create_app
from src.database import ForecastRun, MonitoringHistory
from src.metrics import evaluate_records
from src.time_utils import day_hours, local_timestamp


def test_old_database_is_visible_without_new_forecasts(db, populate_history):
    count = populate_history(db)
    with TestClient(create_app(db)) as client:
        status = client.get("/status").json()
        assert status["history_hours"] == count
        assert status["first_target_date"] == "2026-02-15"
        assert status["latest_target_date"] == "2026-05-31"
        assert status["latest_actual_at"] == "2026-05-31T20:00:00+00:00"
        assert status["latest_issued_date"] is None
        assert not status["tomorrow_ready"]
        data = client.get("/forecasts?start=2026-02-15&end=2026-05-31").json()["records"]
        assert len(data) == count
        assert data[0]["date"] == "2026-02-14T21:00:00+00:00"
        assert data[0]["issued_at"] is None
        metrics = client.get("/metrics?start=2026-02-15&end=2026-05-31&source=recorded").json()
        assert metrics["comparison_hours"] == count
        assert metrics["paired_model"]["mae_mwh"] == pytest.approx(500)
    with db.Session() as session:
        assert session.scalar(select(ForecastRun)) is None
        original = session.get(MonitoringHistory, datetime(2026, 2, 15))
        assert original.actual_consumption == 35000
        assert original.date == datetime(2026, 2, 15)


def test_overlapping_history_remains_accessible(db, populate_history, result):
    populate_history(db)
    result.target_date = "2026-02-15"
    result.predictions = pd.DataFrame({"date": day_hours(result.target_date), "prediction": 12345})
    db.save_forecast(result, local_timestamp("2026-02-14 10:00"))
    day = date(2026, 2, 15)
    recorded = db.series(day, day, "recorded")
    issued = db.series(day, day, "issued")
    combined = db.series(day, day)
    assert len(recorded) == len(issued) == len(combined) == 24
    assert recorded[0]["prediction"] == 35500
    assert issued[0]["prediction"] == combined[0]["prediction"] == 12345
    assert recorded[0]["origin"] == "historical_monitoring"
    assert issued[0]["origin"] == "live"


def test_different_protocols_are_not_pooled_into_one_score():
    metrics = evaluate_records(
        [
            {"actual": 10, "prediction": 11, "epias_forecast": 12, "origin": "historical_monitoring"},
            {"actual": 10, "prediction": 14, "epias_forecast": 13, "origin": "live"},
        ]
    )
    assert metrics["model"] is None
    assert metrics["by_origin"]["live"]["model"]["mae_mwh"] == 4
