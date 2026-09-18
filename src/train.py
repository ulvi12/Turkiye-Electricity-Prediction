"""Reproducible chronological evaluation followed by a separately identified refit.

2022-23: fit; 2024: select tree count; 2025: untouched final test.
Historical results assume consumption was available with a 48-hour lag.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import FEATURE_COLUMNS, FEATURE_VERSION, ROOT
from src.features import FeatureEngineer
from src.metrics import scores
from src.time_utils import hourly_frame, now_local

PARAMS = {
    "learning_rate": 0.03,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": 2,
    "tree_method": "hist",
}


def train(data_path, output_path, report_dir, artifact_dir=None):
    data_path, output_path, report_dir = Path(data_path), Path(output_path), Path(report_dir)
    artifact_dir = Path(artifact_dir or ROOT / "artifacts/training")
    raw = pd.read_csv(data_path)
    original = hourly_frame(raw, "consumption")
    if original.index.min().year != 2022 or original.index.max().year != 2025:
        raise ValueError("This evaluation protocol requires the 2022-2025 dataset")
    featured = FeatureEngineer().process_data(raw)
    usable = featured.dropna(subset=FEATURE_COLUMNS + ["consumption"])
    missing_after_warmup = len(featured.iloc[215:]) - len(usable)
    if missing_after_warmup:
        raise ValueError(
            f"Training data has {missing_after_warmup} incomplete rows after the 215-hour warmup"
        )
    if not np.isfinite(usable[FEATURE_COLUMNS + ["consumption"]].to_numpy()).all():
        raise ValueError("Training inputs must be finite")
    fit = usable.loc[usable.index.year < 2024]
    valid = usable.loc[usable.index.year == 2024]
    test = usable.loc[usable.index.year == 2025]
    if min(len(fit), len(valid), len(test)) < 24 * 300:
        raise ValueError("Insufficient coverage for chronological evaluation")
    selector = xgb.XGBRegressor(**PARAMS, n_estimators=1500, early_stopping_rounds=60)
    selector.fit(
        fit[FEATURE_COLUMNS],
        fit.consumption,
        eval_set=[(valid[FEATURE_COLUMNS], valid.consumption)],
        verbose=False,
    )
    trees = selector.best_iteration + 1
    development = usable.loc[usable.index.year < 2025]
    evaluator = xgb.XGBRegressor(**PARAMS, n_estimators=trees)
    evaluator.fit(development[FEATURE_COLUMNS], development.consumption)
    predictions = evaluator.predict(test[FEATURE_COLUMNS])
    metrics = {
        "model": scores(test.consumption, predictions),
        "lag_48": scores(test.consumption, test.lag_48),
        "lag_168": scores(test.consumption, test.lag_168),
    }
    passed = metrics["model"]["mae_mwh"] < min(metrics[k]["mae_mwh"] for k in ("lag_48", "lag_168"))
    test_output = pd.DataFrame(
        {
            "date": test.index,
            "actual": test.consumption.to_numpy(),
            "prediction": predictions,
            "lag_48": test.lag_48.to_numpy(),
            "lag_168": test.lag_168.to_numpy(),
        }
    )
    monthly = {}
    for month, rows in test_output.groupby(test_output.date.dt.strftime("%Y-%m")):
        monthly[month] = {key: scores(rows.actual, rows[key]) for key in ("prediction", "lag_48", "lag_168")}
    report = {
        "protocol": "2022-23 fit; 2024 tree-count selection; refit through 2024; held-out 2025",
        "feature_version": FEATURE_VERSION,
        "features": FEATURE_COLUMNS,
        "data_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "rows": len(original),
        "usable_rows": len(usable),
        "warmup_hours": 215,
        "missing_after_warmup": missing_after_warmup,
        "selected_trees": trees,
        "parameters": PARAMS,
        "test_year": 2025,
        "test": metrics,
        "monthly": monthly,
        "release_gate_passed": passed,
        "assumptions": [
            "Historical consumption is treated as available with a 48-hour lag; vintage revisions are not archived.",
            "Historical simulation is not evidence of forecasts actually issued during 2025.",
            "Weather excluded: original weather data had no verifiable issuance cutoff.",
            "No EPIAS superiority claim: the historical comparison CSV has no issuance provenance.",
            "Final production refit includes 2025; reported test scores belong to the model fit through 2024.",
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evaluator_path = artifact_dir / "evaluation_model.json"
    evaluator.save_model(evaluator_path)
    report["evaluation_model_sha256"] = sha256(evaluator_path.read_bytes()).hexdigest()
    (report_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    test_output.to_csv(artifact_dir / "holdout_predictions.csv", index=False)
    if not passed:
        raise RuntimeError(
            "Model failed the holdout MAE gate against both naive baselines; production artifact unchanged"
        )
    final = xgb.XGBRegressor(**PARAMS, n_estimators=trees)
    final.fit(usable[FEATURE_COLUMNS], usable.consumption)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".pending.json")
    final.save_model(temporary)
    digest = sha256(temporary.read_bytes()).hexdigest()
    metadata = {
        "model_sha256": digest,
        "feature_version": FEATURE_VERSION,
        "features": FEATURE_COLUMNS,
        "training_start": usable.index.min().isoformat(),
        "training_end": usable.index.max().isoformat(),
        "created_at": now_local().isoformat(),
        "release_gate_passed": passed,
        "data_sha256": report["data_sha256"],
        "evaluation_model_sha256": report["evaluation_model_sha256"],
        "selected_trees": trees,
        "parameters": PARAMS,
        "runtime": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
        },
    }
    temporary.replace(output_path)
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"selected_trees": trees, "test": metrics, "release_gate_passed": passed}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "notebook/epias_data_2022-2025.csv"))
    parser.add_argument("--output", default=str(ROOT / "model.json"))
    parser.add_argument("--reports", default=str(ROOT / "reports"))
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts/training"))
    args = parser.parse_args()
    train(args.data, args.output, args.reports, args.artifacts)


if __name__ == "__main__":
    main()
