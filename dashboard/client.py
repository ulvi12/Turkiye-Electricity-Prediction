"""Use a hosted API or a private API alongside Streamlit."""

import os
import requests


class ForecastClient:
    def __init__(self, base_url=None, database_url=None):
        self.base_url = base_url or os.getenv("API_BASE_URL")
        self.database_url = database_url

    def get(self, route, **params):
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
