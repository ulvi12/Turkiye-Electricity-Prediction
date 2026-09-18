from hashlib import sha256
import json
import pytest
from src.config import ROOT


def test_evaluation_has_complete_holdout_and_consistent_monthly_scores():
    report = json.loads((ROOT / "reports/evaluation.json").read_text())
    assert report["test"]["model"]["hours"] == 8760
    monthly = [values["prediction"] for values in report["monthly"].values()]
    assert sum(row["hours"] for row in monthly) == 8760
    weighted_mae = sum(row["mae_mwh"] * row["hours"] for row in monthly) / 8760
    assert report["test"]["model"]["mae_mwh"] == pytest.approx(weighted_mae)
    assert report["release_gate_passed"]
    assert (
        report["data_sha256"] == sha256((ROOT / "notebook/epias_data_2022-2025.csv").read_bytes()).hexdigest()
    )


def test_model_manifest_matches_artifact():
    metadata = json.loads((ROOT / "model.metadata.json").read_text())
    assert metadata["model_sha256"] == sha256((ROOT / "model.json").read_bytes()).hexdigest()
    assert metadata["training_end"].startswith("2025-12-31")
