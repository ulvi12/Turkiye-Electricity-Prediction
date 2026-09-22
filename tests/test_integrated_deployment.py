import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard.service import LocalAPI, local_api
from dashboard.service import connection_error_code
from scripts.publish_dashboard import publish
from src.database import Database


def test_private_api_serves_real_http_and_shuts_down():
    service = LocalAPI("sqlite:///:memory:")
    try:
        assert service.url.startswith("http://127.0.0.1:")
        response = requests.get(f"{service.url}/health", timeout=5)
        assert response.status_code == 200
        assert response.json()["service"] == "forecast-api"
    finally:
        service.close()
    assert not service.thread.is_alive()


def test_integrated_dashboard_uses_published_snapshot_without_database(tmp_path, monkeypatch, populate_history):
    db = Database("sqlite:///:memory:")
    db.initialize()
    populate_history(db)
    snapshot = tmp_path / "history.json.gz"
    publish(db, snapshot)
    db.engine.dispose()
    monkeypatch.setenv("DASHBOARD_SNAPSHOT_PATH", str(snapshot))
    monkeypatch.setenv("DATABASE_URL", "postgresql://invalid-host/invalid-database")
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:1")
    st.cache_data.clear()
    page = AppTest.from_file("dashboard/app.py").run(timeout=30)
    assert not page.exception
    assert not page.error
    assert len(page.metric) == 3
    assert not page.tabs and not page.radio
    st.cache_data.clear()


def test_concurrent_sessions_share_one_private_api():
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        services = list(pool.map(local_api, ["sqlite:///:memory:"] * 4))
    try:
        assert len({id(service) for service in services}) == 1
    finally:
        services[0].close()


def test_database_errors_are_classified_without_exposing_details():
    assert (
        connection_error_code(Exception("password authentication failed for user secret")) == "authentication"
    )
    assert connection_error_code(Exception("Network is unreachable")) == "network_route"
    assert connection_error_code(Exception("private unexpected detail")) == "connection"
