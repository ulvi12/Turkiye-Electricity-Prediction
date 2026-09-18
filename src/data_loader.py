"""Bounded EPIAS requests. Failed requests never become empty successes."""

import calendar
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.config import EPIAS_PASSWORD, EPIAS_USERNAME
from src.time_utils import hourly_frame, local_timestamp

BASE = "https://seffaflik.epias.com.tr/electricity-service/v1/consumption/data"


class DataLoader:
    def __init__(self, session=None):
        self.session = session or requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"]),
            respect_retry_after_header=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.tgt = None

    def _get_tgt(self):
        if not EPIAS_USERNAME or not EPIAS_PASSWORD:
            raise RuntimeError("Set EPIAS_USERNAME and EPIAS_PASSWORD before running the worker")
        response = self.session.post(
            "https://giris.epias.com.tr/cas/v1/tickets",
            data={"username": EPIAS_USERNAME, "password": EPIAS_PASSWORD},
            timeout=(10, 45),
        )
        # Never log response bodies, credentials, ticket URLs, or tokens.
        if response.status_code != 201 or "Location" not in response.headers:
            raise RuntimeError(f"EPIAS authentication failed (HTTP {response.status_code})")
        self.tgt = response.headers["Location"].rstrip("/").split("/")[-1]

    def _fetch_monthly(self, endpoint, start_date, end_date, column, *, allow_missing=False):
        start, end = local_timestamp(start_date).normalize(), local_timestamp(end_date).normalize()
        if end < start:
            raise ValueError("End date precedes start date")
        if not self.tgt:
            self._get_tgt()
        items = []
        while start <= end:
            month_end = start.replace(day=calendar.monthrange(start.year, start.month)[1])
            stop = min(month_end, end)
            payload = {
                "startDate": start.isoformat(),
                "endDate": (stop + pd.Timedelta(hours=23, minutes=59, seconds=59)).isoformat(),
            }
            for attempt in range(2):
                response = self.session.post(
                    f"{BASE}/{endpoint}",
                    json=payload,
                    headers={"TGT": self.tgt, "Content-Type": "application/json"},
                    timeout=(10, 45),
                )
                if response.status_code in (401, 403) and attempt == 0:
                    self._get_tgt()
                    continue
                break
            if response.status_code != 200:
                raise RuntimeError(f"EPIAS {endpoint} failed (HTTP {response.status_code})")
            body = response.json()
            if not isinstance(body.get("items"), list):
                raise ValueError(f"Unexpected EPIAS {endpoint} response schema")
            items.extend(body["items"])
            start = stop + pd.Timedelta(days=1)
        if not items:
            return pd.DataFrame(columns=["date", column])
        df = hourly_frame(pd.DataFrame(items), column, allow_missing=allow_missing)
        first, last = local_timestamp(start_date).normalize(), local_timestamp(end_date).normalize()
        return df.loc[(df.index >= first) & (df.index < last + pd.Timedelta(days=1))].reset_index()

    def get_realtime_consumption(self, start_date, end_date):
        return self._fetch_monthly("realtime-consumption", start_date, end_date, "consumption")

    def get_load_estimation_plan(self, start_date, end_date):
        return self._fetch_monthly(
            "load-estimation-plan", start_date, end_date, "lep", allow_missing=True
        )
