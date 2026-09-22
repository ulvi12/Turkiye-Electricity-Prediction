from datetime import date, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select

from app.main import create_app
from src.database import ForecastRun, MonitoringHistory
from src.metrics import evaluate_records
from src.time_utils import day_hours, local_timestamp


def test_serving_queries_never_download_input_snapshots(db, result):
    result.input_snapshot["large_audit_payload"] = "x" * 32000
    run = db.save_forecast(result, local_timestamp("2025-12-31 10:00"))
    statements = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(db.engine, "before_cursor_execute", capture)
    try:
        rows = db.series(date(2026, 1, 1), date(2026, 1, 1))
        status = db.status()
    finally:
        event.remove(db.engine, "before_cursor_execute", capture)

    assert len(rows) == 24
    assert rows[0]["prediction"] == 35000
    assert rows[0]["run_id"] == run.id
    assert status["latest_issued_date"] == "2026-01-01"
    assert statements
    assert all("input_snapshot" not in statement.lower() for statement in statements)
    # The audit trail is preserved and still accessible through an explicit run lookup.
    assert db.get_run("2026-01-01").input_snapshot == run.input_snapshot


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


def test_complete_history_uses_all_matched_hours():
    metrics = evaluate_records(
        [
            {"actual": 10, "prediction": 11, "epias_forecast": 12, "origin": "historical_monitoring"},
            {"actual": 10, "prediction": 14, "epias_forecast": 13, "origin": "live"},
        ]
    )
    assert metrics["model"]["mae_mwh"] == pytest.approx(2.5)


def test_official_forecast_repairs_overlay_without_rewriting_history(db, populate_history):
    populate_history(db)
    stamp = datetime(2026, 3, 1, 8)
    with db.Session.begin() as session:
        session.get(MonitoringHistory, stamp).epias_forecast = None

    official = pd.DataFrame({"date": day_hours("2026-03-01"), "lep": 37000})
    db.save_operator_forecasts(official, local_timestamp("2026-03-02"))

    rows = db.series(date(2026, 3, 1), date(2026, 3, 1))
    assert rows[8]["epias_forecast"] == 37000
    with db.Session() as session:
        assert session.get(MonitoringHistory, stamp).epias_forecast is None
