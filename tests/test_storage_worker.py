from datetime import date
import json
import pandas as pd
import pytest
from scripts import daily_run
from src.inference import ForecastResult
from src.time_utils import day_hours, local_timestamp

ISSUE = local_timestamp("2025-12-31 10:00")


def test_forecast_retry_is_immutable(db, result):
    first = db.save_forecast(result, ISSUE)
    result.predictions["prediction"] += 999
    second = db.save_forecast(result, ISSUE + pd.Timedelta(minutes=30))
    assert first.id == second.id
    rows = db.series(date(2026, 1, 1), date(2026, 1, 1))
    assert len(rows) == 24
    assert rows[0]["prediction"] == 35000
    assert rows[0]["date"] == "2025-12-31T21:00:00+00:00"
    assert rows[0]["issued_at"] == "2025-12-31T07:00:00+00:00"


@pytest.mark.parametrize("issue", ["2025-12-31 12:00", "2026-01-01 01:00", "2025-12-30 10:00"])
def test_late_or_backdated_forecasts_rejected(db, result, issue):
    with pytest.raises(ValueError, match="previous day"):
        db.save_forecast(result, local_timestamp(issue))
    assert db.get_run(result.target_date) is None


def test_partial_forecast_transaction_rejected(db, result):
    result.predictions = result.predictions.iloc[:23]
    with pytest.raises(ValueError, match="24 hours"):
        db.save_forecast(result, ISSUE)
    assert db.get_run(result.target_date) is None


def test_actuals_update_does_not_change_forecast(db, result):
    saved = db.save_forecast(result, ISSUE)
    actual = pd.DataFrame({"date": day_hours(result.target_date), "consumption": 40000})
    db.save_actuals(actual, local_timestamp("2026-01-02"))
    actual["consumption"] = 41000
    db.save_actuals(actual, local_timestamp("2026-01-03"))
    assert db.get_run(result.target_date).input_sha256 == saved.input_sha256
    rows = db.series(date(2026, 1, 1), date(2026, 1, 1))
    assert rows[0]["actual"] == 41000 and rows[0]["prediction"] == 35000
    assert not db.missing_actual_dates(local_timestamp("2026-01-02"))


def test_snapshot_includes_benchmark_and_checksum(db, result):
    benchmark = pd.DataFrame({"date": day_hours(result.target_date), "lep": 36000})
    saved = db.save_forecast(result, ISSUE, benchmark, "complete")
    assert len(json.loads(saved.input_snapshot)["benchmark"]) == 24
    assert len(saved.input_sha256) == 64


def test_retry_skips_inference(db, result):
    db.save_forecast(result, ISSUE)

    def fail(**kwargs):
        raise AssertionError("Retry must not rerun inference")

    run = daily_run.issue_forecast(db, object(), lambda: ISSUE, fail)
    assert run.target_date == date(2026, 1, 1)


def test_completion_cutoff_checked_after_inference(db, result):
    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def predict(self, target):
            return result

    class Loader:
        def get_load_estimation_plan(self, *args):
            return pd.DataFrame()

    times = iter([ISSUE, local_timestamp("2025-12-31 12:01")])
    with pytest.raises(ValueError, match="previous day"):
        daily_run.issue_forecast(db, Loader(), lambda: next(times), Pipeline)


def test_benchmark_failure_does_not_prevent_forecast(db, result):
    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def predict(self, target):
            return result

    class Loader:
        def get_load_estimation_plan(self, *args):
            raise RuntimeError("Provider unavailable")

    saved = daily_run.issue_forecast(db, Loader(), lambda: ISSUE, Pipeline)
    assert saved.benchmark_status == "unavailable"


def test_actuals_recovery_respects_horizon(db, result):
    db.save_forecast(result, ISSUE)
    requested = []

    class Loader:
        def get_realtime_consumption(self, start, end):
            requested.append(start)
            return pd.DataFrame({"date": day_hours(start), "consumption": 40000})

    daily_run.reconcile_actuals(db, Loader(), 1, lambda: local_timestamp("2026-02-01 10:00"))
    assert date(2026, 1, 1) not in requested
    assert requested == [date(2026, 1, 31)]


def test_actuals_recovery_repairs_history_without_rewriting_it(db, populate_history):
    populate_history(db)
    missing_stamp = pd.Timestamp("2026-05-30 07:00").to_pydatetime()
    from src.database import MonitoringHistory

    with db.Session.begin() as session:
        session.get(MonitoringHistory, missing_stamp).actual_consumption = None

    requested = []

    class Loader:
        def get_realtime_consumption(self, start, end):
            requested.append(start)
            return pd.DataFrame({"date": day_hours(start), "consumption": 42000})

    daily_run.reconcile_actuals(db, Loader(), 7, lambda: local_timestamp("2026-06-01 10:00"))

    repaired = db.series(date(2026, 5, 30), date(2026, 5, 30), "recorded")
    assert date(2026, 5, 30) in requested
    assert repaired[7]["actual"] == 42000
    assert repaired[7]["actual_retrieved_at"] is not None
    with db.Session() as session:
        assert session.get(MonitoringHistory, missing_stamp).actual_consumption is None


def test_forecast_gap_backfill_is_simulated_and_not_issued(db):
    requested = []

    class Loader:
        def get_realtime_consumption(self, start, end):
            requested.append((local_timestamp(start), local_timestamp(end)))
            dates = pd.date_range(start, end + pd.Timedelta(hours=23), freq="h")
            return pd.DataFrame({"date": dates, "consumption": 40000})

        def get_load_estimation_plan(self, start, end):
            return pd.DataFrame(columns=["date", "lep"])

    class Pipeline:
        def __init__(self, **kwargs):
            self.metadata = {"training_end": "2025-12-31T23:00:00+03:00"}

        def predict_from_history(self, target, history):
            return ForecastResult(
                str(target),
                "b" * 64,
                pd.DataFrame({"date": day_hours(target), "prediction": 39000}),
                {"history_cutoff": str(local_timestamp(target) - pd.Timedelta(days=2))},
            )

    count = daily_run.backfill_forecast_gaps(
        db,
        Loader(),
        2,
        lambda: local_timestamp("2026-01-04 10:00"),
        Pipeline,
    )

    assert count == 2
    assert requested[0][1] == local_timestamp("2026-01-01")
    assert db.series(date(2026, 1, 2), date(2026, 1, 3), "issued") == []
    rows = db.series(date(2026, 1, 2), date(2026, 1, 3), "all")
    assert len(rows) == 48
    assert {row["origin"] for row in rows} == {"historical_simulation"}
    status = db.status()
    assert status["latest_issued_date"] is None
    assert status["first_simulated_date"] == "2026-01-02"
    assert status["latest_simulated_date"] == "2026-01-03"


def test_complete_recorded_days_are_not_backfilled(db, populate_history):
    populate_history(db)
    assert db.missing_forecast_dates(
        local_timestamp("2026-03-02"), local_timestamp("2026-03-01")
    ) == []


def test_worker_attempts_actuals_after_forecast_failure(db, monkeypatch):
    completed = []

    def fail(*args):
        raise RuntimeError("Forecast failure")

    monkeypatch.setattr(daily_run, "issue_forecast", fail)
    monkeypatch.setattr(daily_run, "reconcile_actuals", lambda *args: completed.append(True))
    with pytest.raises(RuntimeError, match="forecast"):
        daily_run.run(db=db, loader=object(), clock=lambda: ISSUE)
    assert completed == [True]
    assert {event["status"] for event in db.status()["jobs"]} == {"success", "failed"}
