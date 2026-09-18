"""Read-only serving API. External data collection belongs to the scheduled worker."""

from contextlib import asynccontextmanager
from datetime import date as Date, timedelta
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy.exc import SQLAlchemyError

from src.database import Database
from src.metrics import evaluate_records
from src.time_utils import now_local, utc_iso

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: Date | None = None


def create_app(database=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.db = database or Database()
        # Additive v2 tables only. No reads/writes to legacy monitoring records.
        app.state.db.initialize()
        yield
        if database is None:
            app.state.db.engine.dispose()

    application = FastAPI(title="Türkiye Grid Forecast API", version="2.0.0", lifespan=lifespan)

    @application.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, exc: SQLAlchemyError):
        logger.error("Database operation failed: %s", type(exc).__name__)
        return JSONResponse(
            status_code=503, content={"detail": "Forecast storage is temporarily unavailable"}
        )

    @application.get("/health")
    def health(request: Request):
        request.app.state.db.healthy()
        return {"status": "ok", "service": "forecast-api", "version": "2.0.0"}

    @application.get("/status")
    def status(request: Request):
        result = request.app.state.db.status()
        tomorrow = now_local().date() + timedelta(days=1)
        result["expected_target_date"] = tomorrow.isoformat()
        result["tomorrow_ready"] = (
            result["latest_issued_date"] == tomorrow.isoformat() and result["origin"] == "live"
        )
        return result

    def date_range(start, end):
        if end < start:
            raise HTTPException(422, "end must be on or after start")
        if (end - start).days > 365:
            raise HTTPException(422, "Choose at most 366 days per request")

    @application.get("/forecasts")
    def forecasts(
        request: Request,
        start: Date = Query(...),
        end: Date = Query(...),
        source: Literal["all", "recorded", "issued"] = "all",
    ):
        date_range(start, end)
        return {
            "timezone": "Europe/Istanbul",
            "unit": "MWh",
            "records": request.app.state.db.series(start, end, source),
        }

    @application.get("/metrics")
    def metrics(
        request: Request,
        start: Date = Query(...),
        end: Date = Query(...),
        source: Literal["all", "recorded", "issued"] = "all",
    ):
        date_range(start, end)
        return evaluate_records(
            request.app.state.db.series(start, end, source), ((end - start).days + 1) * 24
        )

    @application.get("/forecasts/{target_date}")
    def forecast(target_date: Date, request: Request):
        run = request.app.state.db.get_run(target_date)
        if run is None:
            raise HTTPException(404, "No stored forecast for this date; forecasts are issued by the worker")
        return {
            "run_id": run.id,
            "target_date": str(run.target_date),
            "issued_at": utc_iso(run.issued_at),
            "model_version": run.model_version,
            "input_sha256": run.input_sha256,
            "origin": run.origin,
            "benchmark_status": run.benchmark_status,
            "predictions": request.app.state.db.series(target_date, target_date, "issued"),
        }

    @application.post("/predict", deprecated=True)
    def predict(body: PredictionRequest, request: Request):
        # Compatibility route: typed validation, no external calls or expensive inference.
        return forecast(body.date or (now_local().date() + timedelta(days=1)), request)

    return application


app = create_app()
