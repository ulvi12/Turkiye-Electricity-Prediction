"""Versioned schema: immutable forecast runs, separately reconciled observations.

Existing daily_monitoring tables are deliberately left untouched: they cannot
establish issuance time and must not be relabeled as day-ahead forecasts.
"""

from datetime import date, datetime, time, timedelta
from hashlib import sha256
import json
from uuid import uuid4

import pandas as pd
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from src.config import DATABASE_URL, FORECAST_CUTOFF_HOUR
from src.time_utils import day_hours, hourly_frame, local_timestamp, utc_iso, utc_naive

Base = declarative_base()
HistoryBase = declarative_base()


class MonitoringHistory(HistoryBase):
    """Original records use naive Istanbul timestamps; read without rewriting."""

    __tablename__ = "daily_monitoring"
    date = Column(DateTime, primary_key=True)
    actual_consumption = Column(Float)
    epias_forecast = Column(Float)
    model_prediction = Column(Float)


class ForecastRun(Base):
    __tablename__ = "forecast_runs_v2"
    id = Column(String(36), primary_key=True)
    target_date = Column(Date, nullable=False, unique=True, index=True)
    issued_at = Column(DateTime, nullable=False)
    model_version = Column(String(64), nullable=False)
    input_sha256 = Column(String(64), nullable=False)
    input_snapshot = Column(Text, nullable=False)
    origin = Column(String(32), nullable=False, default="live")
    benchmark_status = Column(String(100), nullable=False)


class ForecastHour(Base):
    __tablename__ = "forecast_hours_v2"
    run_id = Column(String(36), ForeignKey("forecast_runs_v2.id"), primary_key=True)
    date = Column(DateTime, primary_key=True, index=True)
    prediction = Column(Float, nullable=False)
    epias_forecast = Column(Float)


class Observation(Base):
    __tablename__ = "observations_v2"
    date = Column(DateTime, primary_key=True)
    consumption = Column(Float, nullable=False)
    retrieved_at = Column(DateTime, nullable=False)


class OperatorForecast(Base):
    __tablename__ = "operator_forecasts_v2"
    date = Column(DateTime, primary_key=True)
    forecast = Column(Float, nullable=False)
    retrieved_at = Column(DateTime, nullable=False)


class JobEvent(Base):
    __tablename__ = "job_events_v2"
    id = Column(Integer, primary_key=True, autoincrement=True)
    finished_at = Column(DateTime, nullable=False)
    task = Column(String(32), nullable=False)
    status = Column(String(16), nullable=False)
    detail = Column(String(300), nullable=False)


class Database:
    def __init__(self, db_url=None):
        url = db_url or DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        options = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if ":memory:" in url:
                options["poolclass"] = StaticPool
        else:
            options["connect_args"] = {"connect_timeout": 15}
        self.engine = create_engine(url, **options)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def initialize(self):
        Base.metadata.create_all(self.engine)

    def has_history(self):
        return inspect(self.engine).has_table(MonitoringHistory.__tablename__)

    def healthy(self):
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            conn.execute(select(ForecastRun.id).limit(1))

    def get_run(self, target):
        with self.Session() as session:
            return session.scalar(
                select(ForecastRun).where(ForecastRun.target_date == date.fromisoformat(str(target)))
            )

    def save_forecast(
        self, result, issued_at, benchmark=None, benchmark_status="unavailable", *, origin="live"
    ):
        target = local_timestamp(result.target_date)
        issue = local_timestamp(issued_at)
        if origin not in ("live", "historical_simulation"):
            raise ValueError("Unknown forecast origin")
        if origin == "live":
            cutoff = target - pd.Timedelta(days=1) + pd.Timedelta(hours=FORECAST_CUTOFF_HOUR)
            if issue.date() != (target - pd.Timedelta(days=1)).date() or issue >= cutoff:
                raise ValueError("Forecast must be issued the previous day before 12:00 Europe/Istanbul")
        predictions = hourly_frame(result.predictions, "prediction")
        if not predictions.index.equals(day_hours(target)):
            raise ValueError("A forecast must contain exactly the target day's 24 hours")
        comparison = pd.Series(dtype=float)
        if benchmark is not None and not benchmark.empty:
            comparison = hourly_frame(benchmark, "lep")["lep"].reindex(day_hours(target))
        snapshot = dict(result.input_snapshot)
        snapshot["benchmark"] = {ts.isoformat(): float(value) for ts, value in comparison.dropna().items()}
        payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
        run = ForecastRun(
            id=str(uuid4()),
            target_date=target.date(),
            issued_at=utc_naive(issue),
            model_version=result.model_version,
            input_sha256=sha256(payload.encode()).hexdigest(),
            input_snapshot=payload,
            origin=origin,
            benchmark_status=benchmark_status,
        )
        try:
            with self.Session.begin() as session:
                session.add(run)
                session.flush()
                for ts, row in predictions.iterrows():
                    value = comparison.get(ts)
                    session.add(
                        ForecastHour(
                            run_id=run.id,
                            date=utc_naive(ts),
                            prediction=float(row.prediction),
                            epias_forecast=float(value) if pd.notna(value) else None,
                        )
                    )
        except IntegrityError:
            # A concurrent retry can only read the winning run, never replace it.
            existing = self.get_run(result.target_date)
            if existing is None:
                raise
            return existing
        return run

    def save_actuals(self, frame, retrieved_at):
        values = hourly_frame(frame, "consumption")
        return self._upsert_hourly(Observation, "consumption", values, retrieved_at)

    def _upsert_hourly(self, model, value_column, values, retrieved_at):
        retrieved = utc_naive(retrieved_at)
        payload = [
            {
                "date": utc_naive(ts),
                value_column: float(getattr(row, value_column if value_column != "forecast" else "lep")),
                "retrieved_at": retrieved,
            }
            for ts, row in values.iterrows()
        ]
        insert_factory = {
            "postgresql": postgresql_insert,
            "sqlite": sqlite_insert,
        }.get(self.engine.dialect.name)
        if insert_factory is None:
            raise RuntimeError(f"Unsupported database dialect: {self.engine.dialect.name}")
        with self.Session.begin() as session:
            for offset in range(0, len(payload), 500):
                statement = insert_factory(model).values(payload[offset : offset + 500])
                statement = statement.on_conflict_do_update(
                    index_elements=[model.date],
                    set_={
                        value_column: getattr(statement.excluded, value_column),
                        "retrieved_at": statement.excluded.retrieved_at,
                    },
                )
                session.execute(statement)
        return len(payload)

    def save_operator_forecasts(self, frame, retrieved_at):
        values = hourly_frame(frame, "lep", allow_missing=True).dropna(subset=["lep"])
        if values.empty:
            return 0
        return self._upsert_hourly(OperatorForecast, "forecast", values, retrieved_at)

    def missing_actual_dates(self, before, since=None):
        """Return days with missing actuals inside an explicit local-date horizon.

        New observations are stored in observations_v2 even when they repair gaps
        in the read-only daily_monitoring history.
        """
        before_local = local_timestamp(before).normalize()
        since_local = local_timestamp(since).normalize() if since is not None else None
        before_utc = utc_naive(before_local)
        since_utc = utc_naive(since_local) if since_local is not None else None
        with self.Session() as session:
            issued_query = (
                select(ForecastHour.date)
                .outerjoin(Observation, ForecastHour.date == Observation.date)
                .where(Observation.date.is_(None), ForecastHour.date < before_utc)
            )
            if since_utc is not None:
                issued_query = issued_query.where(ForecastHour.date >= since_utc)
            missing_stamps = set(session.scalars(issued_query).all())

            if self.has_history():
                # daily_monitoring timestamps are naive Europe/Istanbul wall time.
                history_query = select(MonitoringHistory.date).where(
                    MonitoringHistory.actual_consumption.is_(None),
                    MonitoringHistory.date < before_local.tz_localize(None).to_pydatetime(),
                )
                if since_local is not None:
                    history_query = history_query.where(
                        MonitoringHistory.date >= since_local.tz_localize(None).to_pydatetime()
                    )
                history_stamps = session.scalars(history_query).all()
                observation_stamps = set(
                    session.scalars(
                        select(Observation.date).where(
                            Observation.date < before_utc,
                            *([Observation.date >= since_utc] if since_utc is not None else []),
                        )
                    ).all()
                )
                missing_stamps.update(
                    utc_naive(local_timestamp(stamp))
                    for stamp in history_stamps
                    if utc_naive(local_timestamp(stamp)) not in observation_stamps
                )

        return sorted(
            {pd.Timestamp(ts, tz="UTC").tz_convert("Europe/Istanbul").date() for ts in missing_stamps}
        )

    def missing_forecast_dates(self, before, since):
        """Return past days without a complete recorded or versioned forecast."""
        before_date = local_timestamp(before).date()
        since_date = local_timestamp(since).date()
        if since_date >= before_date:
            return []
        candidates = set(pd.date_range(since_date, before_date - timedelta(days=1), freq="D").date)
        with self.Session() as session:
            stored = set(
                session.scalars(
                    select(ForecastRun.target_date).where(
                        ForecastRun.target_date >= since_date,
                        ForecastRun.target_date < before_date,
                    )
                ).all()
            )
            candidates.difference_update(stored)
            if self.has_history() and candidates:
                rows = session.execute(
                    select(MonitoringHistory.date, MonitoringHistory.model_prediction).where(
                        MonitoringHistory.date >= datetime.combine(since_date, time.min),
                        MonitoringHistory.date < datetime.combine(before_date, time.min),
                    )
                ).all()
                coverage = {}
                for stamp, prediction in rows:
                    if prediction is not None:
                        coverage.setdefault(stamp.date(), set()).add(stamp.hour)
                candidates = {day for day in candidates if len(coverage.get(day, set())) < 24}
        return sorted(candidates)

    def missing_operator_forecast_dates(self, before, since):
        """Return forecast-bearing days without 24 official hourly forecasts."""
        before_date = local_timestamp(before).date()
        since_date = local_timestamp(since).date()
        if since_date >= before_date:
            return []
        relevant, covered = set(), {}
        with self.Session() as session:
            relevant.update(
                session.scalars(
                    select(ForecastRun.target_date).where(
                        ForecastRun.target_date >= since_date,
                        ForecastRun.target_date < before_date,
                    )
                ).all()
            )
            versioned = session.execute(
                select(ForecastHour.date, ForecastHour.epias_forecast)
                .join(ForecastRun, ForecastHour.run_id == ForecastRun.id)
                .where(ForecastRun.target_date >= since_date, ForecastRun.target_date < before_date)
            ).all()
            for stamp, value in versioned:
                if value is not None:
                    local = pd.Timestamp(stamp, tz="UTC").tz_convert("Europe/Istanbul")
                    covered.setdefault(local.date(), set()).add(local.hour)

            if self.has_history():
                history = session.execute(
                    select(MonitoringHistory.date, MonitoringHistory.epias_forecast).where(
                        MonitoringHistory.date >= datetime.combine(since_date, time.min),
                        MonitoringHistory.date < datetime.combine(before_date, time.min),
                    )
                ).all()
                for stamp, value in history:
                    relevant.add(stamp.date())
                    if value is not None:
                        covered.setdefault(stamp.date(), set()).add(stamp.hour)

            official = session.scalars(
                select(OperatorForecast.date).where(
                    OperatorForecast.date >= utc_naive(datetime.combine(since_date, time.min)),
                    OperatorForecast.date < utc_naive(datetime.combine(before_date, time.min)),
                )
            ).all()
            for stamp in official:
                local = pd.Timestamp(stamp, tz="UTC").tz_convert("Europe/Istanbul")
                covered.setdefault(local.date(), set()).add(local.hour)
        return sorted(day for day in relevant if len(covered.get(day, set())) < 24)

    def record_job(self, task, status, detail, at):
        with self.Session.begin() as session:
            session.add(JobEvent(task=task, status=status, detail=detail[:300], finished_at=utc_naive(at)))

    def history_series(self, start, end):
        if not self.has_history():
            return []
        with self.Session() as session:
            rows = session.scalars(
                select(MonitoringHistory)
                .where(
                    MonitoringHistory.date >= datetime.combine(start, time.min),
                    MonitoringHistory.date < datetime.combine(end + timedelta(days=1), time.min),
                )
                .order_by(MonitoringHistory.date)
            ).all()
            observations = session.scalars(
                select(Observation).where(
                    Observation.date >= utc_naive(datetime.combine(start, time.min)),
                    Observation.date < utc_naive(datetime.combine(end + timedelta(days=1), time.min)),
                )
            ).all()
            observed = {row.date: row for row in observations}
            official = session.scalars(
                select(OperatorForecast).where(
                    OperatorForecast.date >= utc_naive(datetime.combine(start, time.min)),
                    OperatorForecast.date < utc_naive(datetime.combine(end + timedelta(days=1), time.min)),
                )
            ).all()
            operator = {row.date: row for row in official}
            return [
                {
                    "date": utc_iso(utc_naive(row.date)),
                    "prediction": row.model_prediction,
                    "epias_forecast": (
                        operator[utc_naive(row.date)].forecast
                        if utc_naive(row.date) in operator
                        else row.epias_forecast
                    ),
                    "actual": (
                        observed[utc_naive(row.date)].consumption
                        if utc_naive(row.date) in observed
                        else row.actual_consumption
                    ),
                    "actual_retrieved_at": (
                        utc_iso(observed[utc_naive(row.date)].retrieved_at)
                        if utc_naive(row.date) in observed
                        else None
                    ),
                    "run_id": None,
                    "issued_at": None,
                    "model_version": None,
                    "origin": "historical_monitoring",
                    "benchmark_status": "recorded",
                }
                for row in rows
            ]

    def series(self, start, end, source="all"):
        if source not in ("all", "recorded", "issued"):
            raise ValueError("Unknown data series")
        history = self.history_series(start, end) if source != "issued" else []
        if source == "recorded":
            return history
        with self.Session() as session:
            issued_query = (
                select(ForecastHour, ForecastRun, Observation, OperatorForecast)
                .join(ForecastRun, ForecastHour.run_id == ForecastRun.id)
                .outerjoin(Observation, ForecastHour.date == Observation.date)
                .outerjoin(OperatorForecast, ForecastHour.date == OperatorForecast.date)
                .where(ForecastRun.target_date >= start, ForecastRun.target_date <= end)
                .order_by(ForecastHour.date)
            )
            if source == "issued":
                issued_query = issued_query.where(ForecastRun.origin == "live")
            records = session.execute(issued_query).all()
            issued = [
                {
                    "date": utc_iso(hour.date),
                    "prediction": hour.prediction,
                    "epias_forecast": official.forecast if official else hour.epias_forecast,
                    "actual": actual.consumption if actual else None,
                    "actual_retrieved_at": utc_iso(actual.retrieved_at) if actual else None,
                    "run_id": run.id,
                    "issued_at": utc_iso(run.issued_at),
                    "model_version": run.model_version,
                    "origin": run.origin,
                    "benchmark_status": run.benchmark_status,
                }
                for hour, run, actual, official in records
            ]
        combined = {row["date"]: row for row in history}
        for row in issued:
            previous = combined.get(row["date"])
            if previous:
                if row["actual"] is None:
                    row["actual"] = previous["actual"]
                    row["actual_retrieved_at"] = previous["actual_retrieved_at"]
                if row["epias_forecast"] is None:
                    row["epias_forecast"] = previous["epias_forecast"]
            combined[row["date"]] = row
        return sorted(combined.values(), key=lambda row: row["date"])

    def status(self):
        history_first = history_last = history_actual = None
        history_hours = 0
        if self.has_history():
            with self.Session() as session:
                history_first, history_last, history_hours = session.execute(
                    select(
                        func.min(MonitoringHistory.date),
                        func.max(MonitoringHistory.date),
                        func.count(MonitoringHistory.date),
                    )
                ).one()
                history_actual = session.scalar(
                    select(func.max(MonitoringHistory.date)).where(
                        MonitoringHistory.actual_consumption.is_not(None)
                    )
                )
        with self.Session() as session:
            latest_any = session.scalar(
                select(ForecastRun).order_by(ForecastRun.target_date.desc()).limit(1)
            )
            latest = session.scalar(
                select(ForecastRun)
                .where(ForecastRun.origin == "live")
                .order_by(ForecastRun.target_date.desc())
                .limit(1)
            )
            first_any = session.scalar(select(func.min(ForecastRun.target_date)))
            first = session.scalar(
                select(func.min(ForecastRun.target_date)).where(ForecastRun.origin == "live")
            )
            first_simulated, latest_simulated = session.execute(
                select(func.min(ForecastRun.target_date), func.max(ForecastRun.target_date)).where(
                    ForecastRun.origin == "historical_simulation"
                )
            ).one()
            last_actual = session.scalar(select(func.max(Observation.date)))
            refreshed = session.scalar(select(func.max(Observation.retrieved_at)))
            events = session.scalars(select(JobEvent).order_by(JobEvent.id.desc()).limit(10)).all()
            starts = [d for d in (first_any, history_first.date() if history_first else None) if d]
            ends = [
                d
                for d in (
                    latest_any.target_date if latest_any else None,
                    history_last.date() if history_last else None,
                )
                if d
            ]
            actual_dates = [
                d for d in (last_actual, utc_naive(history_actual) if history_actual else None) if d
            ]
            return {
                "first_target_date": str(min(starts)) if starts else None,
                "latest_target_date": str(max(ends)) if ends else None,
                "first_issued_date": str(first) if first else None,
                "latest_issued_date": str(latest.target_date) if latest else None,
                "first_simulated_date": str(first_simulated) if first_simulated else None,
                "latest_simulated_date": str(latest_simulated) if latest_simulated else None,
                "history_first_date": str(history_first.date()) if history_first else None,
                "history_latest_date": str(history_last.date()) if history_last else None,
                "history_hours": history_hours,
                "latest_issued_at": utc_iso(latest.issued_at) if latest else None,
                "model_version": latest.model_version if latest else None,
                "origin": latest.origin if latest else "historical_monitoring" if history_hours else None,
                "latest_actual_at": utc_iso(max(actual_dates)) if actual_dates else None,
                "actuals_refreshed_at": utc_iso(refreshed),
                "jobs": [
                    {
                        "task": e.task,
                        "status": e.status,
                        "detail": e.detail,
                        "finished_at": utc_iso(e.finished_at),
                    }
                    for e in events
                ],
            }
