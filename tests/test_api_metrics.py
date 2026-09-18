from fastapi.testclient import TestClient
from app.main import create_app
from src.metrics import evaluate_records
from src.time_utils import local_timestamp


def test_api_validation_and_empty_results(db):
    with TestClient(create_app(db)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json={"date": "garbage"}).status_code == 422
        assert client.post("/predict", json={"date": "2026-01-01", "extra": 1}).status_code == 422
        assert client.post("/predict", json={}).status_code == 404
        assert client.get("/forecasts/2026-01-01").status_code == 404
        assert client.get("/forecasts?start=2026-02-01&end=2026-01-01").status_code == 422
        assert client.get("/metrics?start=2020-01-01&end=2026-01-01").status_code == 422
        assert client.get("/status").json()["latest_target_date"] is None


def test_api_returns_saved_forecast_and_paired_metrics(db, result):
    db.save_forecast(result, local_timestamp("2025-12-31 10:00"))
    actual = result.predictions.rename(columns={"prediction": "consumption"})
    db.save_actuals(actual, local_timestamp("2026-01-02"))
    with TestClient(create_app(db)) as client:
        response = client.get("/forecasts/2026-01-01")
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 24
        assert (
            client.post("/predict", json={"date": "2026-01-01"}).json()["run_id"] == response.json()["run_id"]
        )
        metrics = client.get("/metrics?start=2026-01-01&end=2026-01-02").json()
        assert metrics["model"]["mae_mwh"] == 0
        assert metrics["observed_hours"] == 24 and metrics["expected_hours"] == 48
        assert metrics["comparison_hours"] == 0


def test_fair_comparison_and_zero_actual_handling():
    records = [
        {"actual": 100, "prediction": 110, "epias_forecast": 120},
        {"actual": 0, "prediction": 5, "epias_forecast": 7},
        {"actual": 100, "prediction": 150, "epias_forecast": None},
        {"actual": None, "prediction": 200, "epias_forecast": 210},
    ]
    metrics = evaluate_records(records)
    assert metrics["observed_hours"] == 3 and metrics["comparison_hours"] == 2
    assert metrics["model"]["mape_hours"] == 2
    assert metrics["paired_model"]["mae_mwh"] == 7.5
    assert metrics["paired_model"]["mape_pct"] == 10
    assert metrics["epias"]["mape_pct"] == 20


def test_empty_metrics_are_json_safe():
    assert evaluate_records([], 24)["model"] is None


def test_database_failure_is_503_and_does_not_leak_details(db, monkeypatch):
    from sqlalchemy.exc import OperationalError

    with TestClient(create_app(db)) as client:

        def fail():
            raise OperationalError("secret connection string", {}, Exception("password"))

        monkeypatch.setattr(db, "healthy", fail)
        response = client.get("/health")
        assert response.status_code == 503
        assert "secret" not in response.text and "password" not in response.text
