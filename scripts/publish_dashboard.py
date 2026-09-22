"""Publish a compact, read-only copy of the dashboard's historical series."""

import gzip
import json
import os
from pathlib import Path

from src.database import Database


OUTPUT = Path(__file__).resolve().parents[1] / "dashboard" / "data" / "history.json.gz"


def publish(db, output=OUTPUT):
    status = db.status()
    first, last = status["first_target_date"], status["latest_target_date"]
    if not first or not last:
        raise RuntimeError("No monitoring records are available to publish")

    from datetime import date

    records = db.series(date.fromisoformat(first), date.fromisoformat(last))
    if not records:
        raise RuntimeError("Dashboard publication would contain no hourly records")
    data = {"status": status, "records": records}
    compressed = gzip.compress(
        json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"),
        mtime=0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(compressed)
    temporary.replace(output)
    return len(records), len(compressed)


if __name__ == "__main__":
    if not (os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")):
        raise SystemExit("Missing persistent database secret")
    count, size = publish(Database())
    print(f"Published {count} hourly records ({size:,} compressed bytes)")
