import numpy as np
import pandas as pd
import pytest
from src.database import Database, HistoryBase, MonitoringHistory
from src.inference import ForecastResult
from src.time_utils import day_hours


@pytest.fixture
def db():
    database = Database("sqlite:///:memory:")
    database.initialize()
    yield database
    database.engine.dispose()


@pytest.fixture
def history():
    dates = pd.date_range("2025-12-22", "2025-12-30 23:00", freq="h", tz="Europe/Istanbul")
    return pd.DataFrame({"date": dates, "consumption": 35000.0 + np.arange(len(dates))})


@pytest.fixture
def result():
    return ForecastResult(
        "2026-01-01",
        "a" * 64,
        pd.DataFrame({"date": day_hours("2026-01-01"), "prediction": np.arange(24) + 35000}),
        {"features": [1, 2], "source": "test"},
    )


@pytest.fixture
def populate_history():
    def populate(database):
        HistoryBase.metadata.create_all(database.engine)
        dates = pd.date_range("2026-02-15", "2026-05-31 23:00", freq="h")
        with database.Session.begin() as session:
            session.add_all(
                [
                    MonitoringHistory(
                        date=stamp.to_pydatetime(),
                        actual_consumption=35000 + 5000 * np.sin(stamp.hour / 24 * 2 * np.pi),
                        model_prediction=35500 + 5000 * np.sin(stamp.hour / 24 * 2 * np.pi),
                        epias_forecast=36000 + 5000 * np.sin(stamp.hour / 24 * 2 * np.pi),
                    )
                    for stamp in dates
                ]
            )
        return len(dates)

    return populate
