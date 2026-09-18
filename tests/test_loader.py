from unittest.mock import Mock
import pytest
import requests
from src.data_loader import DataLoader
from src.time_utils import local_timestamp


def response(status=200, body=None):
    value = Mock()
    value.status_code = status
    value.json.return_value = body if body is not None else {"items": []}
    return value


def loader_with(responses):
    session = Mock()
    session.post.side_effect = responses
    loader = DataLoader(session)
    loader.tgt = "test"
    return loader, session


def test_monthly_requests_have_timeouts_and_correct_boundaries():
    loader, session = loader_with([response(), response()])
    loader.get_realtime_consumption(local_timestamp("2025-12-31"), local_timestamp("2026-01-01"))
    calls = session.post.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["timeout"] == (10, 45)
    assert calls[0].kwargs["json"]["endDate"].startswith("2025-12-31T23:59:59")
    assert calls[1].kwargs["json"]["startDate"].startswith("2026-01-01T00:00:00")


@pytest.mark.parametrize("value", [response(500), response(body={"unexpected": []})])
def test_provider_errors_raise(value):
    loader, _ = loader_with([value])
    with pytest.raises((RuntimeError, ValueError)):
        loader.get_realtime_consumption(local_timestamp("2026-01-01"), local_timestamp("2026-01-01"))


def test_network_failure_does_not_return_empty_success():
    loader, _ = loader_with([requests.Timeout()])
    with pytest.raises(requests.Timeout):
        loader.get_realtime_consumption(local_timestamp("2026-01-01"), local_timestamp("2026-01-01"))


def test_partial_official_forecast_response_preserves_available_hours():
    items = [
        {"date": "2026-01-01T00:00:00+03:00", "lep": 40000},
        {"date": "2026-01-01T01:00:00+03:00", "lep": None},
    ]
    loader, _ = loader_with([response(body={"items": items})])
    result = loader.get_load_estimation_plan(
        local_timestamp("2026-01-01"), local_timestamp("2026-01-01")
    )
    assert result.lep.notna().sum() == 1
