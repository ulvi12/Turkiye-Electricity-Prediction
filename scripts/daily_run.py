"""Issue tomorrow's forecast, then reconcile actuals without rewriting forecasts."""

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FORECAST_CUTOFF_HOUR
from src.data_loader import DataLoader
from src.database import Database
from src.inference import InferencePipeline
from src.time_utils import day_hours, local_timestamp, now_local

logger = logging.getLogger(__name__)


def issue_forecast(db, loader, clock=now_local, pipeline_factory=InferencePipeline):
    now = local_timestamp(clock())
    target = (now.normalize() + pd.Timedelta(days=1)).date()
    existing = db.get_run(target)
    if existing:
        if existing.origin != "live":
            raise ValueError("Refusing to use a historical simulation as a live forecast")
        logger.info("Forecast for %s already exists; preserving run %s", target, existing.id)
        return existing
    if now.hour >= FORECAST_CUTOFF_HOUR:
        raise ValueError("Missed 12:00 Istanbul issuance cutoff; past forecasts are never backdated")
    pipeline = pipeline_factory(loader=loader)
    result = pipeline.predict(target)
    benchmark, benchmark_status = None, "unavailable"
    try:
        benchmark = loader.get_load_estimation_plan(target, target)
        benchmark_status = "complete" if len(benchmark) == 24 else "partial_or_unavailable"
    except Exception:
        logger.warning(
            "EPIAS comparison unavailable; publishing model forecast with explicit missing coverage"
        )
    # Check actual completion time, not job-start time, at the storage boundary.
    return db.save_forecast(result, clock(), benchmark, benchmark_status)


def reconcile_actuals(db, loader, lookback_days=365, clock=now_local):
    today = local_timestamp(clock()).normalize()
    since = today - pd.Timedelta(days=lookback_days)
    # Always attempt yesterday, then request only dates where stored hourly
    # records still have missing actuals. Repeated runs are idempotent.
    dates = {(today - pd.Timedelta(days=1)).date()}
    dates.update(db.missing_actual_dates(today, since))
    failures = []
    for target in sorted(dates):
        try:
            actuals = loader.get_realtime_consumption(target, target)
            if actuals.empty:
                failures.append(str(target))
                continue
            db.save_actuals(actuals, clock())
            received = pd.DatetimeIndex(actuals.date)
            if len(received.intersection(day_hours(target))) != 24:
                failures.append(str(target))
        except Exception:
            logger.warning("Actuals unavailable for %s; will retry on the next run", target)
            failures.append(str(target))
    if failures:
        raise RuntimeError(f"Actuals incomplete for {len(failures)} day(s): {', '.join(failures[:10])}")


def run(mode="all", lookback_days=365, db=None, loader=None, clock=now_local):
    db, loader = db or Database(), loader or DataLoader()
    db.initialize()
    failures = []
    tasks = []
    if mode in ("all", "forecast"):
        tasks.append(("forecast", lambda: issue_forecast(db, loader, clock)))
    if mode in ("all", "actuals"):
        tasks.append(("actuals", lambda: reconcile_actuals(db, loader, lookback_days, clock)))
    for name, task in tasks:
        try:
            task()
            db.record_job(name, "success", "Completed", clock())
        except Exception as exc:
            # Store a safe classification; secrets/provider response bodies stay out of DB/logs.
            detail = f"{type(exc).__name__}: {name} failed; inspect input coverage, credentials, and cutoff"
            logger.error(detail)
            db.record_job(name, "failed", detail, clock())
            failures.append(name)
    if failures:
        raise RuntimeError("Worker failed: " + ", ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["all", "forecast", "actuals"], default="all")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Past-day horizon scanned for missing actual consumption (default: 365)",
    )
    args = parser.parse_args()
    if not 1 <= args.lookback_days <= 366:
        parser.error("--lookback-days must be between 1 and 366")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(args.mode, args.lookback_days)
    except Exception as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
