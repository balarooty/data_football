#!/usr/bin/env python3
"""
Generate predictions for upcoming fixtures using trained artifact bundle.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prediction_system import (
    build_feature_columns,
    build_leak_free_features,
    load_all_matches,
    predict_scoreline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict upcoming football fixtures")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="root data directory")
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("artifacts/football_prediction_bundle.joblib"),
        help="path to trained model bundle",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/upcoming_predictions.csv"),
        help="path to save upcoming predictions CSV",
    )
    parser.add_argument(
        "--json-output-path",
        type=Path,
        default=Path("artifacts/upcoming_predictions.json"),
        help="path to save upcoming predictions JSON",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=3,
        help="include unplayed fixtures only if match date >= (today - lookback_days)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.artifact_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {args.artifact_path}")

    bundle = joblib.load(args.artifact_path)
    matches = load_all_matches(args.data_dir)

    config = bundle.get("config", {})
    features = build_leak_free_features(
        matches,
        form_window=int(config.get("form_window", 5)),
        elo_k=float(config.get("elo_k", 20.0)),
        elo_home_advantage=float(config.get("elo_home_advantage", 75.0)),
    )

    feature_columns = bundle.get("feature_columns") or build_feature_columns()

    upcoming = features[~features["is_played"]].copy()
    recency_cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(days=max(0, args.lookback_days))
    parsed_dt = pd.to_datetime(upcoming["match_datetime"], errors="coerce")
    upcoming = upcoming[parsed_dt >= recency_cutoff].copy()
    if upcoming.empty:
        print("No upcoming fixtures found (all rows appear to be played).")
        return 0

    upcoming = upcoming.sort_values("match_datetime", kind="mergesort").reset_index(drop=True)
    X = upcoming[feature_columns].astype(float)

    over_under_model = bundle["over_under_model"]
    winner_model = bundle["winner_model"]
    winner_encoder = bundle["winner_encoder"]
    bookings_model = bundle["bookings_model"]
    home_goals_model = bundle["home_goals_model"]
    away_goals_model = bundle["away_goals_model"]

    p_over = over_under_model.predict_proba(X)[:, 1]
    winner_proba = winner_model.predict_proba(X)
    winner_pred_idx = winner_model.predict(X)
    winner_pred = winner_encoder.inverse_transform(winner_pred_idx)
    bookings_pred = np.clip(bookings_model.predict(X), 0, None)

    home_lambda = np.clip(home_goals_model.predict(X), 0.01, None)
    away_lambda = np.clip(away_goals_model.predict(X), 0.01, None)

    score_preds = []
    score_top3 = []
    max_score_goals = int(config.get("max_score_goals", 7))

    for lh, la in zip(home_lambda, away_lambda):
        h, a, top3 = predict_scoreline(lh, la, max_goals=max_score_goals)
        score_preds.append(f"{h}-{a}")
        score_top3.append(json.dumps(top3))

    class_positions = {label: pos for pos, label in enumerate(winner_encoder.classes_)}

    out = upcoming[["Date", "country", "league", "HomeTeam", "AwayTeam"]].copy()
    out["generated_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    out["p_over_2_5"] = p_over
    out["p_under_2_5"] = 1.0 - p_over
    out["predicted_winner"] = winner_pred
    out["p_home_win"] = winner_proba[:, class_positions.get("H", 0)] if "H" in class_positions else np.nan
    out["p_draw"] = winner_proba[:, class_positions.get("D", 0)] if "D" in class_positions else np.nan
    out["p_away_win"] = winner_proba[:, class_positions.get("A", 0)] if "A" in class_positions else np.nan
    out["pred_total_bookings"] = bookings_pred
    out["expected_home_goals"] = home_lambda
    out["expected_away_goals"] = away_lambda
    out["pred_correct_score"] = score_preds
    out["pred_correct_score_top3"] = score_top3

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)
    out.to_json(args.json_output_path, orient="records", indent=2)

    print(f"Upcoming fixtures predicted: {len(out)}")
    print(f"CSV:  {args.output_path}")
    print(f"JSON: {args.json_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
