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


def backfill_forecast_gaps(db, loader, lookback_days=365, clock=now_local, pipeline_factory=InferencePipeline):
    """Fill past forecast gaps with leakage-safe, explicitly simulated runs."""
    today = local_timestamp(clock()).normalize()
    pipeline = pipeline_factory(loader=loader)
    training_end = local_timestamp(pipeline.metadata["training_end"]).normalize()
    since = max(today - pd.Timedelta(days=lookback_days), training_end + pd.Timedelta(days=1))
    targets = db.missing_forecast_dates(today, since)
    if not targets:
        return 0

    first, last = local_timestamp(targets[0]).normalize(), local_timestamp(targets[-1]).normalize()
    # One bounded fetch is reused for every target. predict_from_history applies
    # its own per-target cutoff and discards all timestamps after target - 48h.
    history = loader.get_realtime_consumption(first - pd.Timedelta(days=10), last - pd.Timedelta(days=2))
    try:
        benchmarks = loader.get_load_estimation_plan(first, last)
        if not benchmarks.empty:
            db.save_operator_forecasts(benchmarks, clock())
    except Exception:
        logger.warning("Historical operator forecasts unavailable; simulations will omit the benchmark")
        benchmarks = pd.DataFrame(columns=["date", "lep"])

    completed, failures = 0, []
    for target in targets:
        try:
            result = pipeline.predict_from_history(target, history)
            target_hours = day_hours(target)
            benchmark = benchmarks.loc[pd.DatetimeIndex(benchmarks.date).isin(target_hours)]
            result.input_snapshot["simulation"] = {
                "generated_at": local_timestamp(clock()).isoformat(),
                "availability_lag_hours": 48,
                "label": "historical_simulation",
            }
            db.save_forecast(
                result,
                clock(),
                benchmark,
                "historical_record" if len(benchmark) == 24 else "partial_or_unavailable",
                origin="historical_simulation",
            )
            completed += 1
        except Exception:
            logger.warning("Historical simulation failed for %s", target)
            failures.append(str(target))
    if failures:
        raise RuntimeError(
            f"Historical simulations incomplete for {len(failures)} day(s): {', '.join(failures[:10])}"
        )
    return completed


def reconcile_operator_forecasts(db, loader, lookback_days=365, clock=now_local):
    today = local_timestamp(clock()).normalize()
    since = today - pd.Timedelta(days=lookback_days)
    targets = db.missing_operator_forecast_dates(today, since)
    if not targets:
        return 0
    forecasts = loader.get_load_estimation_plan(targets[0], targets[-1])
    if forecasts.empty:
        raise RuntimeError("Official historical forecasts are unavailable")
    db.save_operator_forecasts(forecasts, clock())
    remaining = db.missing_operator_forecast_dates(today, since)
    if remaining:
        raise RuntimeError(
            f"Official forecasts incomplete for {len(remaining)} day(s): "
            f"{', '.join(str(day) for day in remaining[:10])}"
        )
    return len(targets)


def run(mode="all", lookback_days=365, db=None, loader=None, clock=now_local):
    db, loader = db or Database(), loader or DataLoader()
    db.initialize()
    failures = []
    tasks = []
    if mode in ("all", "forecast"):
        tasks.append(("forecast", lambda: issue_forecast(db, loader, clock)))
    if mode in ("all", "actuals"):
        tasks.append(
            ("forecast_backfill", lambda: backfill_forecast_gaps(db, loader, lookback_days, clock))
        )
        tasks.append(("actuals", lambda: reconcile_actuals(db, loader, lookback_days, clock)))
        tasks.append(
            (
                "operator_forecasts",
                lambda: reconcile_operator_forecasts(db, loader, lookback_days, clock),
            )
        )
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
