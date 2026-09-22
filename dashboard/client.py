"""Read a published dashboard snapshot, or call the private API elsewhere."""

from datetime import date, datetime
import gzip
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import requests


SNAPSHOT = Path(__file__).resolve().parent / "data" / "history.json.gz"


class ForecastClient:
    def __init__(self, base_url=None, database_url=None, *, snapshot_only=False, snapshot_path=None):
        self.base_url = base_url or os.getenv("API_BASE_URL")
        self.database_url = database_url
        self.snapshot_only = snapshot_only
        self.snapshot_path = Path(snapshot_path or os.getenv("DASHBOARD_SNAPSHOT_PATH") or SNAPSHOT)

    def get(self, route, **params):
        if self.snapshot_only:
            try:
                with gzip.open(self.snapshot_path, "rt", encoding="utf-8") as file:
                    snapshot = json.load(file)
            except (OSError, ValueError) as error:
                raise requests.ConnectionError("snapshot") from error
            if route == "/status":
                return snapshot["status"]
            if route == "/forecasts":
                start, end = date.fromisoformat(params["start"]), date.fromisoformat(params["end"])
                source = params.get("source", "all")
                if source not in ("all", "recorded", "issued"):
                    raise ValueError("Unknown data series")
                records = snapshot["records"]
                if source != "all" or (str(start), str(end)) != (
                    snapshot["status"]["first_target_date"],
                    snapshot["status"]["latest_target_date"],
                ):
                    zone = ZoneInfo("Europe/Istanbul")
                    records = [
                        row for row in records
                        if start <= datetime.fromisoformat(row["date"]).astimezone(zone).date() <= end
                        and (source == "all" or
                             source == "recorded" and row["origin"] == "historical_monitoring" or
                             source == "issued" and row["origin"] == "live")
                    ]
                return {"records": records}
            raise ValueError(f"Unsupported snapshot route: {route}")

        base_url = self.base_url
        if not base_url:
            from dashboard.service import local_api

            try:
                base_url = local_api(self.database_url).url
            except Exception as error:
                raise requests.ConnectionError(str(error) or "connection") from None
        response = requests.get(f"{base_url.rstrip('/')}{route}", params=params, timeout=(5, 20))
        response.raise_for_status()
        return response.json()
