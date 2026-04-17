#!/usr/bin/env python3
"""
Generate predictions for upcoming fixtures using trained artifact bundle.

Supports:
- Local upcoming rows from scraped CSV files
- External fixtures CSV (for API-fetched upcoming fixtures)
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
        "--fixtures-csv",
        type=Path,
        default=None,
        help="optional external fixtures CSV (e.g. from fetch_upcoming_fixtures_api.py)",
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
        help="when using local unplayed fixtures, keep only rows with datetime >= (now - lookback_days)",
    )
    return parser.parse_args()


def _normalize_prob_vector(values: np.ndarray) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = np.full_like(arr, np.nan, dtype=float)
    valid = np.isfinite(arr).all(axis=1) & (arr > 0).all(axis=1)
    if np.any(valid):
        sums = arr[valid].sum(axis=1, keepdims=True)
        keep = np.isfinite(sums).ravel() & (sums.ravel() > 0)
        if np.any(keep):
            idx = np.where(valid)[0][keep]
            out[idx] = arr[idx] / sums[keep]
    return out


def _blend_multiclass(model_proba: np.ndarray, implied_proba: np.ndarray, alpha: float) -> np.ndarray:
    blended = model_proba.copy()
    if implied_proba.shape != model_proba.shape:
        return blended
    valid = np.isfinite(implied_proba).all(axis=1)
    if np.any(valid):
        blended[valid] = alpha * model_proba[valid] + (1.0 - alpha) * implied_proba[valid]
    return blended


def _blend_binary(model_p1: np.ndarray, implied_p1: np.ndarray, alpha: float) -> np.ndarray:
    blended = model_p1.copy()
    valid = np.isfinite(implied_p1)
    if np.any(valid):
        blended[valid] = alpha * model_p1[valid] + (1.0 - alpha) * implied_p1[valid]
    return np.clip(blended, 1e-6, 1.0 - 1e-6)


def _prepare_external_fixtures(fixtures_csv: Path) -> pd.DataFrame:
    if not fixtures_csv.exists():
        raise FileNotFoundError(f"Fixtures CSV not found: {fixtures_csv}")

    ext = pd.read_csv(fixtures_csv)
    if ext.empty:
        return ext

    ext = ext.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"}).copy()

    required = ["Date", "HomeTeam", "AwayTeam"]
    missing = [col for col in required if col not in ext.columns]
    if missing:
        raise ValueError(f"Fixtures CSV missing required columns: {missing}")

    for col in [
        "Time",
        "country",
        "league",
        "FTHG",
        "FTAG",
        "FTR",
        "HY",
        "AY",
        "HS",
        "AS",
        "HST",
        "AST",
        "HC",
        "AC",
        "HF",
        "AF",
        "odds_home",
        "odds_draw",
        "odds_away",
        "odds_over25",
        "odds_under25",
        "api_match_id",
        "match_datetime",
    ]:
        if col not in ext.columns:
            ext[col] = np.nan

    ext["country"] = ext["country"].fillna("api")
    ext["league"] = ext["league"].fillna("api")
    ext["FTR"] = ext["FTR"].fillna("")

    dt = pd.to_datetime(ext["match_datetime"], errors="coerce", utc=True)
    missing_dt = dt.isna()
    if np.any(missing_dt):
        date_part = pd.to_datetime(ext["Date"], dayfirst=True, errors="coerce", utc=True)
        time_part = pd.to_timedelta(ext["Time"].fillna("00:00").astype(str) + ":00", errors="coerce")
        fallback = date_part + time_part.fillna(pd.Timedelta(0))
        dt = dt.where(~missing_dt, fallback)

    ext["match_datetime"] = dt.dt.tz_convert(None)
    ext["league_type"] = "api_fixtures"
    ext["source_file"] = str(fixtures_csv)
    ext["is_external_fixture"] = True

    ext = ext.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
    return ext


def _build_upcoming_feature_frame(matches: pd.DataFrame, fixtures_csv: Path | None, config: dict, lookback_days: int) -> pd.DataFrame:
    if fixtures_csv is not None:
        played_history = matches[matches[["FTHG", "FTAG"]].notna().all(axis=1)].copy()
        external = _prepare_external_fixtures(fixtures_csv)
        if external.empty:
            return external

        merged = pd.concat([played_history, external], ignore_index=True, sort=False)
        merged["_sort_dt"] = pd.to_datetime(merged["match_datetime"], errors="coerce")
        merged["_sort_dt"] = merged["_sort_dt"].fillna(pd.to_datetime(merged["Date"], dayfirst=True, errors="coerce"))
        merged["_sort_dt"] = merged["_sort_dt"].fillna(pd.Timestamp("1900-01-01"))
        merged = merged.sort_values(["_sort_dt", "country", "league", "HomeTeam", "AwayTeam"], kind="mergesort").reset_index(drop=True)
        merged["match_id"] = np.arange(len(merged))

        features = build_leak_free_features(
            merged,
            form_window=int(config.get("form_window", 5)),
            elo_k=float(config.get("elo_k", 20.0)),
            elo_home_advantage=float(config.get("elo_home_advantage", 75.0)),
        )
        upcoming = features[features["is_external_fixture"] == True].copy()
        return upcoming

    features = build_leak_free_features(
        matches,
        form_window=int(config.get("form_window", 5)),
        elo_k=float(config.get("elo_k", 20.0)),
        elo_home_advantage=float(config.get("elo_home_advantage", 75.0)),
    )

    upcoming = features[~features["is_played"]].copy()
    recency_cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(days=max(0, lookback_days))
    parsed_dt = pd.to_datetime(upcoming["match_datetime"], errors="coerce")
    upcoming = upcoming[parsed_dt >= recency_cutoff].copy()
    return upcoming


def main() -> int:
    args = parse_args()

    if not args.artifact_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {args.artifact_path}")

    bundle = joblib.load(args.artifact_path)
    matches = load_all_matches(args.data_dir)
    config = bundle.get("config", {})

    upcoming = _build_upcoming_feature_frame(
        matches=matches,
        fixtures_csv=args.fixtures_csv,
        config=config,
        lookback_days=args.lookback_days,
    )

    if upcoming.empty:
        print("No upcoming fixtures found for prediction.")
        return 0

    upcoming = upcoming.sort_values("match_datetime", kind="mergesort").reset_index(drop=True)
    feature_columns = bundle.get("feature_columns") or build_feature_columns()
    X = upcoming[feature_columns].astype(float)

    over_under_model = bundle["over_under_model"]
    winner_model = bundle["winner_model"]
    winner_encoder = bundle["winner_encoder"]
    bookings_model = bundle["bookings_model"]
    home_goals_model = bundle["home_goals_model"]
    away_goals_model = bundle["away_goals_model"]

    p_over_model = over_under_model.predict_proba(X)[:, 1]
    winner_proba_model = winner_model.predict_proba(X)
    bookings_pred = np.clip(bookings_model.predict(X), 0, None)

    # Blend with implied probabilities (if present in fixture odds).
    winner_alpha = float(config.get("winner_blend_alpha", 1.0))
    ou_alpha = float(config.get("ou_blend_alpha", 1.0))

    class_pos = {label: idx for idx, label in enumerate(winner_encoder.classes_)}
    implied_winner = np.full_like(winner_proba_model, np.nan, dtype=float)
    source_by_label = {
        "H": pd.to_numeric(upcoming.get("implied_home"), errors="coerce").to_numpy(dtype=float),
        "D": pd.to_numeric(upcoming.get("implied_draw"), errors="coerce").to_numpy(dtype=float),
        "A": pd.to_numeric(upcoming.get("implied_away"), errors="coerce").to_numpy(dtype=float),
    }
    for label, idx in class_pos.items():
        src = source_by_label.get(label)
        if src is not None:
            implied_winner[:, idx] = src
    implied_winner = _normalize_prob_vector(implied_winner)
    winner_proba = _blend_multiclass(winner_proba_model, implied_winner, alpha=winner_alpha)

    implied_over = pd.to_numeric(upcoming.get("implied_over25"), errors="coerce").to_numpy(dtype=float)
    implied_under = pd.to_numeric(upcoming.get("implied_under25"), errors="coerce").to_numpy(dtype=float)
    implied_pair = _normalize_prob_vector(np.column_stack([implied_under, implied_over]))
    p_over = _blend_binary(p_over_model, implied_pair[:, 1], alpha=ou_alpha)

    winner_pred_idx = np.argmax(winner_proba, axis=1)
    winner_pred = winner_encoder.inverse_transform(winner_pred_idx)

    home_lambda = np.clip(home_goals_model.predict(X), 0.01, None)
    away_lambda = np.clip(away_goals_model.predict(X), 0.01, None)

    score_preds = []
    score_top3 = []
    max_score_goals = int(config.get("max_score_goals", 7))

    for lh, la in zip(home_lambda, away_lambda):
        h, a, top3 = predict_scoreline(lh, la, max_goals=max_score_goals)
        score_preds.append(f"{h}-{a}")
        score_top3.append(json.dumps(top3))

    out = upcoming[["Date", "country", "league", "HomeTeam", "AwayTeam"]].copy()
    if "api_match_id" in upcoming.columns:
        out["api_match_id"] = upcoming["api_match_id"]
    out["generated_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")

    out["p_over_2_5"] = p_over
    out["p_under_2_5"] = 1.0 - p_over
    out["p_over_2_5_model"] = p_over_model

    out["predicted_winner"] = winner_pred
    out["p_home_win"] = winner_proba[:, class_pos.get("H", 0)] if "H" in class_pos else np.nan
    out["p_draw"] = winner_proba[:, class_pos.get("D", 0)] if "D" in class_pos else np.nan
    out["p_away_win"] = winner_proba[:, class_pos.get("A", 0)] if "A" in class_pos else np.nan

    out["pred_total_bookings"] = bookings_pred
    out["expected_home_goals"] = home_lambda
    out["expected_away_goals"] = away_lambda
    out["pred_correct_score"] = score_preds
    out["pred_correct_score_top3"] = score_top3

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, index=False)
    out.to_json(args.json_output_path, orient="records", indent=2)

    print(f"Upcoming fixtures predicted: {len(out)}")
    print(f"CSV:  {args.output_path}")
    print(f"JSON: {args.json_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
