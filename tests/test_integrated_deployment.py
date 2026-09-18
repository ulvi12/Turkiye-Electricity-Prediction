import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from dashboard.client import ForecastClient
from dashboard.service import LocalAPI, local_api
from dashboard.service import connection_error_code
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


def test_integrated_dashboard_needs_only_existing_database_setting(tmp_path, monkeypatch, populate_history):
    url = f"sqlite:///{(tmp_path / 'existing.db').as_posix()}"
    db = Database(url)
    populate_history(db)
    db.engine.dispose()
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("SUPABASE_DB_URL", url)
    st.cache_data.clear()
    try:
        page = AppTest.from_file("dashboard/app.py").run(timeout=30)
        assert not page.exception
        assert not page.error
        assert len(page.metric) == 3
        assert not page.tabs and not page.radio
        first_service = local_api(url)
        assert ForecastClient(database_url=url).get("/health")["status"] == "ok"
        assert local_api(url) is first_service
    finally:
        local_api(url).close()
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
