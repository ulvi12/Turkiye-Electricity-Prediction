from datetime import date
import requests
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
from dashboard.client import ForecastClient
from src.metrics import evaluate_records
from src.time_utils import day_hours, local_timestamp


def test_dashboard_populated_and_interactive(db, monkeypatch, populate_history):
    populate_history(db)
    calls = []

    def get(self, route, **params):
        calls.append((route, params))
        if route == "/status":
            return {**db.status(), "tomorrow_ready": False, "expected_target_date": "2026-01-01"}
        start, end = date.fromisoformat(params["start"]), date.fromisoformat(params["end"])
        records = db.series(start, end, params.get("source", "all"))
        if route == "/forecasts":
            return {"records": records}
        return evaluate_records(records, ((end - start).days + 1) * 24)

    monkeypatch.setattr(ForecastClient, "get", get)
    st.cache_data.clear()
    page = AppTest.from_file("dashboard/app.py").run(timeout=30)
    assert not page.exception
    assert len(page.metric) == 3
    assert not page.tabs
    assert not page.radio
    assert not page.error
    assert page.date_input[0].value == (date(2026, 2, 15), date(2026, 5, 31))
    page.date_input[0].set_value((date(2026, 3, 1), date(2026, 3, 26))).run()
    assert not page.exception
    assert not page.error
    assert len([route for route, _ in calls if route == "/forecasts"]) == 1
    assert "624 evaluated hours" in " ".join(caption.value for caption in page.caption)
    page.button[0].click().run()
    assert page.date_input[0].value == (date(2026, 2, 15), date(2026, 5, 31))
    assert len([route for route, _ in calls if route == "/forecasts"]) == 2
    st.cache_data.clear()


def test_dashboard_handles_unavailable_api(monkeypatch):
    def get(*args, **kwargs):
        raise requests.ConnectionError()

    monkeypatch.setattr(ForecastClient, "get", get)
    st.cache_data.clear()
    page = AppTest.from_file("dashboard/app.py").run()
    assert not page.exception
    assert "could not be reached" in page.error[0].value
    st.cache_data.clear()


def test_dashboard_handles_empty_database(db, monkeypatch):
    monkeypatch.setattr(ForecastClient, "get", lambda *a, **k: db.status())
    st.cache_data.clear()
    page = AppTest.from_file("dashboard/app.py").run()
    assert not page.exception
    assert "No monitoring records" in page.info[0].value
    st.cache_data.clear()


def test_complete_history_includes_new_forecasts(db, monkeypatch, populate_history, result):
    populate_history(db)
    result.target_date = "2026-06-01"
    result.predictions = pd.DataFrame({"date": day_hours(result.target_date), "prediction": 40000})
    db.save_forecast(result, local_timestamp("2026-05-31 10:00"))

    def get(self, route, **params):
        if route == "/status":
            return db.status()
        return {
            "records": db.series(
                date.fromisoformat(params["start"]), date.fromisoformat(params["end"]), params["source"]
            )
        }

    monkeypatch.setattr(ForecastClient, "get", get)
    st.cache_data.clear()
    page = AppTest.from_file("dashboard/app.py").run(timeout=30)
    assert not page.exception
    assert not page.radio and not page.tabs
    assert page.date_input[0].value == (date(2026, 2, 15), date(2026, 6, 1))
    st.cache_data.clear()
