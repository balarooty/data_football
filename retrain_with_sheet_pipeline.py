#!/usr/bin/env python3
"""
End-to-end pipeline:
1) Pull + clean Google Sheet prediction data
2) Train football prediction model bundle
3) Predict upcoming fixtures using cleaned sheet fixtures

This guarantees the cleaned sheet is used on every retraining run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from clean_google_sheet_predictions import clean_sheet, get_csv_url
from prediction_system import (
    build_feature_columns,
    build_leak_free_features,
    load_all_matches,
    save_artifacts,
    train_prediction_system,
)

DEFAULT_SHEET_ID = "1feRsH0y7xfLtv8_zA6Iqsf9Fh0g4J3_qXURkG0htJYI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain model and use cleaned sheet on every run")
    parser.add_argument("--sheet-id", default=DEFAULT_SHEET_ID, help="Google Sheet ID")
    parser.add_argument("--sheet-url", default=None, help="Optional direct CSV URL")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="historical data directory")
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("artifacts/football_prediction_bundle.joblib"),
        help="trained model artifact path",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/training_metrics.json"),
        help="metrics JSON path",
    )
    parser.add_argument(
        "--raw-sheet-csv",
        type=Path,
        default=Path("artifacts/sheet_predictions_raw.csv"),
        help="raw sheet CSV snapshot (latest run)",
    )
    parser.add_argument(
        "--cleaned-sheet-csv",
        type=Path,
        default=Path("artifacts/sheet_predictions_cleaned.csv"),
        help="cleaned sheet CSV output",
    )
    parser.add_argument(
        "--cleaned-sheet-json",
        type=Path,
        default=Path("artifacts/sheet_predictions_cleaned.json"),
        help="cleaned sheet JSON output",
    )
    parser.add_argument(
        "--sheet-diagnostics-json",
        type=Path,
        default=Path("artifacts/sheet_cleaning_diagnostics.json"),
        help="sheet diagnostics output",
    )
    parser.add_argument(
        "--history-csv",
        type=Path,
        default=Path("artifacts/prediction_history.csv"),
        help="append-only prediction history ledger",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="directory for immutable per-run snapshots",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="optional run id (default: UTC timestamp id)",
    )
    parser.add_argument(
        "--sheet-predictions-csv",
        type=Path,
        default=Path("artifacts/upcoming_predictions_from_sheet.csv"),
        help="predictions output CSV for cleaned sheet fixtures",
    )
    parser.add_argument(
        "--sheet-predictions-json",
        type=Path,
        default=Path("artifacts/upcoming_predictions_from_sheet.json"),
        help="predictions output JSON for cleaned sheet fixtures",
    )
    parser.add_argument("--form-window", type=int, default=5)
    parser.add_argument("--elo-k", type=float, default=20.0)
    parser.add_argument("--elo-home-advantage", type=float, default=75.0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-score-goals", type=int, default=7)
    return parser.parse_args()


def run_clean_sheet(args: argparse.Namespace) -> dict:
    csv_url = get_csv_url(args.sheet_id, args.sheet_url)
    raw = pd.read_csv(csv_url)
    cleaned, diagnostics = clean_sheet(raw)

    args.raw_sheet_csv.parent.mkdir(parents=True, exist_ok=True)
    args.cleaned_sheet_csv.parent.mkdir(parents=True, exist_ok=True)
    args.cleaned_sheet_json.parent.mkdir(parents=True, exist_ok=True)
    args.sheet_diagnostics_json.parent.mkdir(parents=True, exist_ok=True)

    raw.to_csv(args.raw_sheet_csv, index=False)
    cleaned.to_csv(args.cleaned_sheet_csv, index=False)
    cleaned.to_json(args.cleaned_sheet_json, orient="records", indent=2, date_format="iso")

    with open(args.sheet_diagnostics_json, "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2)

    return diagnostics


def run_training(args: argparse.Namespace) -> dict:
    matches = load_all_matches(args.data_dir)
    features = build_leak_free_features(
        matches,
        form_window=args.form_window,
        elo_k=args.elo_k,
        elo_home_advantage=args.elo_home_advantage,
    )

    bundle = train_prediction_system(
        features,
        feature_columns=build_feature_columns(),
        test_fraction=args.test_fraction,
        max_score_goals=args.max_score_goals,
        form_window=args.form_window,
        elo_k=args.elo_k,
        elo_home_advantage=args.elo_home_advantage,
    )

    save_artifacts(bundle, args.artifact_path, args.metrics_path)
    return bundle.metrics


def run_sheet_predictions(args: argparse.Namespace, run_id: str) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "predict_fixtures.py"),
        "--data-dir",
        str(args.data_dir),
        "--artifact-path",
        str(args.artifact_path),
        "--fixtures-csv",
        str(args.cleaned_sheet_csv),
        "--output-path",
        str(args.sheet_predictions_csv),
        "--json-output-path",
        str(args.sheet_predictions_json),
        "--run-id",
        run_id,
        "--history-csv",
        str(args.history_csv),
    ]
    subprocess.run(cmd, check=True)


def snapshot_run_outputs(args: argparse.Namespace, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    files_to_copy = {
        args.raw_sheet_csv: run_dir / "sheet_predictions_raw.csv",
        args.cleaned_sheet_csv: run_dir / "sheet_predictions_cleaned.csv",
        args.cleaned_sheet_json: run_dir / "sheet_predictions_cleaned.json",
        args.sheet_diagnostics_json: run_dir / "sheet_cleaning_diagnostics.json",
        args.artifact_path: run_dir / "football_prediction_bundle.joblib",
        args.metrics_path: run_dir / "training_metrics.json",
        args.sheet_predictions_csv: run_dir / "upcoming_predictions_from_sheet.csv",
        args.sheet_predictions_json: run_dir / "upcoming_predictions_from_sheet.json",
    }

    for src, dst in files_to_copy.items():
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    run_id = args.run_id.strip() if args.run_id else datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.runs_dir / run_id

    print("[1/3] Cleaning sheet...")
    diagnostics = run_clean_sheet(args)
    print(f"  cleaned rows: {diagnostics.get('rows')}")

    print("[2/3] Training model...")
    metrics = run_training(args)
    print(f"  winner_accuracy_blended: {metrics.get('winner_accuracy_blended')}")
    print(f"  over_under_accuracy_blended: {metrics.get('over_under_accuracy_blended')}")

    print("[3/3] Predicting sheet fixtures...")
    run_sheet_predictions(args, run_id=run_id)

    snapshot_run_outputs(args, run_dir=run_dir)

    history_rows = 0
    if args.history_csv.exists():
        history_rows = len(pd.read_csv(args.history_csv))

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "raw_sheet_csv": str(args.raw_sheet_csv),
        "cleaned_sheet_csv": str(args.cleaned_sheet_csv),
        "artifact_path": str(args.artifact_path),
        "sheet_predictions_csv": str(args.sheet_predictions_csv),
        "history_csv": str(args.history_csv),
        "history_rows": history_rows,
        "rows_cleaned": diagnostics.get("rows"),
        "winner_accuracy_blended": metrics.get("winner_accuracy_blended"),
        "over_under_accuracy_blended": metrics.get("over_under_accuracy_blended"),
    }

    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Pipeline complete.")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
