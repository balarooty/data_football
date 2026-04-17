#!/usr/bin/env python3
"""
Football prediction system trained only on local scraped CSV data.

Targets:
1) Over/Under 2.5 goals
2) Match winner (H/D/A)
3) Total bookings (HY + AY)
4) Correct score (via Poisson scoreline from expected goals models)

Outputs:
- Model bundle (joblib)
- Metrics JSON
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder

COLUMN_ALIASES = {
    "ï»¿Div": "Div",
    "ï»¿Country": "Country",
    "Home": "HomeTeam",
    "Away": "AwayTeam",
    "HG": "FTHG",
    "AG": "FTAG",
    "Res": "FTR",
}

ESSENTIAL_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

ODDS_HOME_COLS = ["B365H", "PSH", "MaxH", "AvgH", "B365CH", "PSCH", "MaxCH", "AvgCH"]
ODDS_DRAW_COLS = ["B365D", "PSD", "MaxD", "AvgD", "B365CD", "PSCD", "MaxCD", "AvgCD"]
ODDS_AWAY_COLS = ["B365A", "PSA", "MaxA", "AvgA", "B365CA", "PSCA", "MaxCA", "AvgCA"]
ODDS_OVER25_COLS = ["B365>2.5", "P>2.5", "Max>2.5", "Avg>2.5", "B365C>2.5", "PC>2.5"]
ODDS_UNDER25_COLS = ["B365<2.5", "P<2.5", "Max<2.5", "Avg<2.5", "B365C<2.5", "PC<2.5"]


@dataclass
class TrainingArtifacts:
    feature_columns: List[str]
    winner_encoder: LabelEncoder
    over_under_model: HistGradientBoostingClassifier
    winner_model: HistGradientBoostingClassifier
    bookings_model: HistGradientBoostingRegressor
    home_goals_model: HistGradientBoostingRegressor
    away_goals_model: HistGradientBoostingRegressor
    config: Dict[str, float]
    metrics: Dict[str, float]


def _first_valid(row: pd.Series, columns: List[str]) -> float:
    for column in columns:
        if column in row and pd.notna(row[column]):
            try:
                value = float(row[column])
                if value > 0:
                    return value
            except (TypeError, ValueError):
                continue
    return np.nan


def _safe_num(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_match_result(row: pd.Series) -> str:
    if pd.notna(row.get("FTR")) and str(row["FTR"]).strip() in {"H", "D", "A"}:
        return str(row["FTR"]).strip()

    hg = row.get("FTHG")
    ag = row.get("FTAG")
    if pd.isna(hg) or pd.isna(ag):
        return ""
    if hg > ag:
        return "H"
    if hg < ag:
        return "A"
    return "D"


def _extract_metadata_from_path(file_path: Path, data_dir: Path) -> Tuple[str, str, str]:
    relative = file_path.relative_to(data_dir)
    parts = list(relative.parts)

    league_type = parts[0] if parts else "unknown"
    country = parts[1] if len(parts) > 1 else "unknown"
    league = parts[2] if len(parts) > 2 else file_path.stem

    if league_type == "extra_leagues" and len(parts) > 2:
        league = Path(parts[-1]).stem

    return league_type, country, league


def load_all_matches(data_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue

        if df.empty:
            continue

        df = df.rename(columns=COLUMN_ALIASES).copy()
        league_type, country, league = _extract_metadata_from_path(csv_file, data_dir)

        for column in ["Div", "Country", "League", "Season", "Time", "HY", "AY", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF"]:
            if column not in df.columns:
                df[column] = np.nan

        df["league_type"] = league_type
        df["country"] = country
        df["league"] = league
        df["source_file"] = str(csv_file)

        # Prefer source metadata when CSV does not include identifiers.
        df["Country"] = df["Country"].fillna(country)
        df["League"] = df["League"].fillna(league)

        # Parse datetime.
        dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        if "Time" in df.columns:
            time_text = df["Time"].fillna("00:00").astype(str)
            td = pd.to_timedelta(time_text + ":00", errors="coerce")
            dt = dt + td.fillna(pd.Timedelta(0))
        df["match_datetime"] = dt

        for col in ["FTHG", "FTAG", "HY", "AY", "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["FTR"] = df.apply(_normalize_match_result, axis=1)
        frames.append(df)

    if not frames:
        raise RuntimeError("CSV files were found but none could be parsed.")

    all_matches = pd.concat(frames, ignore_index=True, sort=False).copy()
    all_matches = all_matches.dropna(subset=["Date", "HomeTeam", "AwayTeam"], how="any")
    all_matches = all_matches.reset_index(drop=True)

    # Stable order for chronological feature generation.
    all_matches["_fallback_dt"] = pd.to_datetime(all_matches["Date"], dayfirst=True, errors="coerce")
    all_matches["_sort_dt"] = all_matches["match_datetime"].fillna(all_matches["_fallback_dt"])
    all_matches["_sort_dt"] = all_matches["_sort_dt"].fillna(pd.Timestamp("1900-01-01"))

    all_matches = all_matches.sort_values(["_sort_dt", "country", "league", "HomeTeam", "AwayTeam"], kind="mergesort")
    all_matches = all_matches.reset_index(drop=True)
    all_matches["match_id"] = np.arange(len(all_matches))
    return all_matches


def build_leak_free_features(matches: pd.DataFrame, form_window: int, elo_k: float, elo_home_advantage: float) -> pd.DataFrame:
    stat_keys = [
        "gf",
        "ga",
        "pts",
        "cards",
        "shots",
        "sot",
        "corners",
        "fouls",
        "total_goals",
    ]

    history = defaultdict(lambda: {key: deque(maxlen=form_window) for key in stat_keys})
    elo = defaultdict(lambda: 1500.0)
    played_count = defaultdict(int)

    feature_rows: List[Dict[str, float]] = []

    def avg(team: str, key: str) -> float:
        values = history[team][key]
        if not values:
            return np.nan
        arr = np.array(values, dtype=float)
        if not np.isfinite(arr).any():
            return np.nan
        return float(np.nanmean(arr))

    for _, row in matches.iterrows():
        home = str(row["HomeTeam"])
        away = str(row["AwayTeam"])

        odds_home = _first_valid(row, ODDS_HOME_COLS)
        odds_draw = _first_valid(row, ODDS_DRAW_COLS)
        odds_away = _first_valid(row, ODDS_AWAY_COLS)
        odds_over25 = _first_valid(row, ODDS_OVER25_COLS)
        odds_under25 = _first_valid(row, ODDS_UNDER25_COLS)

        implied_home = 1.0 / odds_home if pd.notna(odds_home) and odds_home > 0 else np.nan
        implied_draw = 1.0 / odds_draw if pd.notna(odds_draw) and odds_draw > 0 else np.nan
        implied_away = 1.0 / odds_away if pd.notna(odds_away) and odds_away > 0 else np.nan
        implied_over25 = 1.0 / odds_over25 if pd.notna(odds_over25) and odds_over25 > 0 else np.nan

        home_elo = elo[home]
        away_elo = elo[away]
        elo_diff = home_elo - away_elo

        feat = {
            "match_id": int(row["match_id"]),
            "match_datetime": row["match_datetime"],
            "Date": row["Date"],
            "country": row["country"],
            "league": row["league"],
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": row["FTHG"],
            "FTAG": row["FTAG"],
            "FTR": row["FTR"],
            "HY": row.get("HY", np.nan),
            "AY": row.get("AY", np.nan),
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": elo_diff,
            "home_matches_played": float(played_count[home]),
            "away_matches_played": float(played_count[away]),
            "home_avg_gf": avg(home, "gf"),
            "home_avg_ga": avg(home, "ga"),
            "home_avg_pts": avg(home, "pts"),
            "home_avg_cards": avg(home, "cards"),
            "home_avg_shots": avg(home, "shots"),
            "home_avg_sot": avg(home, "sot"),
            "home_avg_corners": avg(home, "corners"),
            "home_avg_fouls": avg(home, "fouls"),
            "home_avg_total_goals": avg(home, "total_goals"),
            "away_avg_gf": avg(away, "gf"),
            "away_avg_ga": avg(away, "ga"),
            "away_avg_pts": avg(away, "pts"),
            "away_avg_cards": avg(away, "cards"),
            "away_avg_shots": avg(away, "shots"),
            "away_avg_sot": avg(away, "sot"),
            "away_avg_corners": avg(away, "corners"),
            "away_avg_fouls": avg(away, "fouls"),
            "away_avg_total_goals": avg(away, "total_goals"),
            "odds_home": odds_home,
            "odds_draw": odds_draw,
            "odds_away": odds_away,
            "odds_over25": odds_over25,
            "odds_under25": odds_under25,
            "implied_home": implied_home,
            "implied_draw": implied_draw,
            "implied_away": implied_away,
            "implied_over25": implied_over25,
        }

        feat["diff_avg_gf"] = feat["home_avg_gf"] - feat["away_avg_gf"]
        feat["diff_avg_ga"] = feat["home_avg_ga"] - feat["away_avg_ga"]
        feat["diff_avg_pts"] = feat["home_avg_pts"] - feat["away_avg_pts"]
        feat["diff_avg_cards"] = feat["home_avg_cards"] - feat["away_avg_cards"]
        feat["diff_avg_shots"] = feat["home_avg_shots"] - feat["away_avg_shots"]
        feat["diff_avg_sot"] = feat["home_avg_sot"] - feat["away_avg_sot"]
        feat["diff_avg_corners"] = feat["home_avg_corners"] - feat["away_avg_corners"]
        feature_rows.append(feat)

        # Update state only for played matches.
        if pd.notna(row["FTHG"]) and pd.notna(row["FTAG"]):
            hg = _safe_num(row["FTHG"])
            ag = _safe_num(row["FTAG"])

            if hg > ag:
                home_pts, away_pts = 3.0, 0.0
                actual_home_result = 1.0
            elif hg < ag:
                home_pts, away_pts = 0.0, 3.0
                actual_home_result = 0.0
            else:
                home_pts, away_pts = 1.0, 1.0
                actual_home_result = 0.5

            # Home row updates.
            history[home]["gf"].append(hg)
            history[home]["ga"].append(ag)
            history[home]["pts"].append(home_pts)
            history[home]["cards"].append(_safe_num(row.get("HY", np.nan), default=np.nan))
            history[home]["shots"].append(_safe_num(row.get("HS", np.nan), default=np.nan))
            history[home]["sot"].append(_safe_num(row.get("HST", np.nan), default=np.nan))
            history[home]["corners"].append(_safe_num(row.get("HC", np.nan), default=np.nan))
            history[home]["fouls"].append(_safe_num(row.get("HF", np.nan), default=np.nan))
            history[home]["total_goals"].append(hg + ag)

            # Away row updates.
            history[away]["gf"].append(ag)
            history[away]["ga"].append(hg)
            history[away]["pts"].append(away_pts)
            history[away]["cards"].append(_safe_num(row.get("AY", np.nan), default=np.nan))
            history[away]["shots"].append(_safe_num(row.get("AS", np.nan), default=np.nan))
            history[away]["sot"].append(_safe_num(row.get("AST", np.nan), default=np.nan))
            history[away]["corners"].append(_safe_num(row.get("AC", np.nan), default=np.nan))
            history[away]["fouls"].append(_safe_num(row.get("AF", np.nan), default=np.nan))
            history[away]["total_goals"].append(hg + ag)

            played_count[home] += 1
            played_count[away] += 1

            # Elo update.
            expected_home = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + elo_home_advantage)) / 400.0))
            expected_away = 1.0 - expected_home
            elo[home] = home_elo + elo_k * (actual_home_result - expected_home)
            elo[away] = away_elo + elo_k * ((1.0 - actual_home_result) - expected_away)

    features = pd.DataFrame(feature_rows)

    features["total_goals"] = pd.to_numeric(features["FTHG"], errors="coerce") + pd.to_numeric(features["FTAG"], errors="coerce")
    features["over_2_5"] = (features["total_goals"] > 2.5).astype(float)
    features["bookings_total"] = pd.to_numeric(features["HY"], errors="coerce") + pd.to_numeric(features["AY"], errors="coerce")
    features["is_played"] = features[["FTHG", "FTAG"]].notna().all(axis=1)
    return features


def build_feature_columns() -> List[str]:
    return [
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_matches_played",
        "away_matches_played",
        "home_avg_gf",
        "home_avg_ga",
        "home_avg_pts",
        "home_avg_cards",
        "home_avg_shots",
        "home_avg_sot",
        "home_avg_corners",
        "home_avg_fouls",
        "home_avg_total_goals",
        "away_avg_gf",
        "away_avg_ga",
        "away_avg_pts",
        "away_avg_cards",
        "away_avg_shots",
        "away_avg_sot",
        "away_avg_corners",
        "away_avg_fouls",
        "away_avg_total_goals",
        "diff_avg_gf",
        "diff_avg_ga",
        "diff_avg_pts",
        "diff_avg_cards",
        "diff_avg_shots",
        "diff_avg_sot",
        "diff_avg_corners",
        "odds_home",
        "odds_draw",
        "odds_away",
        "odds_over25",
        "odds_under25",
        "implied_home",
        "implied_draw",
        "implied_away",
        "implied_over25",
    ]


def _winner_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=350,
        min_samples_leaf=25,
        random_state=42,
    )


def _over_under_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=20,
        random_state=42,
    )


def _bookings_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson",
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=20,
        random_state=42,
    )


def _goals_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson",
        max_depth=4,
        learning_rate=0.05,
        max_iter=320,
        min_samples_leaf=20,
        random_state=42,
    )


def _poisson_pmf(k: int, lam: float) -> float:
    lam = max(0.01, float(lam))
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def predict_scoreline(lambda_home: float, lambda_away: float, max_goals: int = 7) -> Tuple[int, int, List[Tuple[str, float]]]:
    matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for h in range(max_goals + 1):
        p_h = _poisson_pmf(h, lambda_home)
        for a in range(max_goals + 1):
            matrix[h, a] = p_h * _poisson_pmf(a, lambda_away)

    best_h, best_a = np.unravel_index(np.argmax(matrix), matrix.shape)

    flat = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            flat.append((f"{h}-{a}", float(matrix[h, a])))
    flat.sort(key=lambda x: x[1], reverse=True)
    return int(best_h), int(best_a), flat[:3]


def chronological_split(df: pd.DataFrame, test_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("_sort_dt", kind="mergesort").reset_index(drop=True)
    split_index = int(len(df_sorted) * (1.0 - test_fraction))
    split_index = min(max(split_index, 1), len(df_sorted) - 1)
    return df_sorted.iloc[:split_index].copy(), df_sorted.iloc[split_index:].copy()


def train_prediction_system(
    features: pd.DataFrame,
    feature_columns: List[str],
    test_fraction: float,
    max_score_goals: int,
    form_window: int,
    elo_k: float,
    elo_home_advantage: float,
) -> TrainingArtifacts:
    played = features[features["is_played"]].copy()
    if len(played) < 1000:
        raise RuntimeError(f"Not enough played matches to train robust models. Found: {len(played)}")

    played["_sort_dt"] = pd.to_datetime(played["match_datetime"], errors="coerce")
    played["_sort_dt"] = played["_sort_dt"].fillna(pd.to_datetime(played["Date"], dayfirst=True, errors="coerce"))
    played["_sort_dt"] = played["_sort_dt"].fillna(pd.Timestamp("1900-01-01"))

    train_df, test_df = chronological_split(played, test_fraction=test_fraction)

    X_train = train_df[feature_columns].astype(float)
    X_test = test_df[feature_columns].astype(float)

    # Winner model.
    winner_encoder = LabelEncoder()
    y_train_winner = winner_encoder.fit_transform(train_df["FTR"].astype(str))
    y_test_winner = winner_encoder.transform(test_df["FTR"].astype(str))

    winner_model = _winner_model()
    winner_model.fit(X_train, y_train_winner)
    winner_proba = winner_model.predict_proba(X_test)
    winner_pred = winner_model.predict(X_test)

    # Over/Under model.
    y_train_ou = train_df["over_2_5"].astype(int)
    y_test_ou = test_df["over_2_5"].astype(int)

    over_under_model = _over_under_model()
    over_under_model.fit(X_train, y_train_ou)
    ou_proba = over_under_model.predict_proba(X_test)[:, 1]
    ou_pred = (ou_proba >= 0.5).astype(int)

    # Bookings model (only rows with card data).
    book_train = train_df.dropna(subset=["bookings_total"]).copy()
    book_test = test_df.dropna(subset=["bookings_total"]).copy()
    bookings_model = _bookings_model()
    bookings_model.fit(book_train[feature_columns].astype(float), np.clip(book_train["bookings_total"].astype(float), 0, None))
    pred_bookings = bookings_model.predict(book_test[feature_columns].astype(float))

    # Expected goals models for correct score.
    home_goals_model = _goals_model()
    away_goals_model = _goals_model()

    home_goals_model.fit(X_train, np.clip(train_df["FTHG"].astype(float), 0, None))
    away_goals_model.fit(X_train, np.clip(train_df["FTAG"].astype(float), 0, None))

    pred_home_lambda = np.clip(home_goals_model.predict(X_test), 0.01, None)
    pred_away_lambda = np.clip(away_goals_model.predict(X_test), 0.01, None)

    cs_exact_hits = 0
    cs_top3_hits = 0
    for i, row in test_df.reset_index(drop=True).iterrows():
        ph, pa, top3 = predict_scoreline(pred_home_lambda[i], pred_away_lambda[i], max_goals=max_score_goals)
        actual = f"{int(row['FTHG'])}-{int(row['FTAG'])}"
        pred = f"{ph}-{pa}"
        if pred == actual:
            cs_exact_hits += 1
        if actual in {score for score, _ in top3}:
            cs_top3_hits += 1

    metrics: Dict[str, float] = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "bookings_test_rows": int(len(book_test)),
        "winner_accuracy": float(accuracy_score(y_test_winner, winner_pred)),
        "winner_log_loss": float(log_loss(y_test_winner, winner_proba, labels=list(range(len(winner_encoder.classes_))))),
        "over_under_accuracy": float(accuracy_score(y_test_ou, ou_pred)),
        "over_under_roc_auc": float(roc_auc_score(y_test_ou, ou_proba)) if len(np.unique(y_test_ou)) > 1 else float("nan"),
        "bookings_mae": float(mean_absolute_error(book_test["bookings_total"].astype(float), pred_bookings)) if len(book_test) else float("nan"),
        "bookings_rmse": float(np.sqrt(mean_squared_error(book_test["bookings_total"].astype(float), pred_bookings))) if len(book_test) else float("nan"),
        "correct_score_exact_accuracy": float(cs_exact_hits / len(test_df)),
        "correct_score_top3_accuracy": float(cs_top3_hits / len(test_df)),
    }

    # Refit final models on all played data for production inference.
    X_full = played[feature_columns].astype(float)
    y_full_winner = winner_encoder.fit_transform(played["FTR"].astype(str))
    y_full_ou = played["over_2_5"].astype(int)

    final_winner = _winner_model()
    final_winner.fit(X_full, y_full_winner)

    final_over_under = _over_under_model()
    final_over_under.fit(X_full, y_full_ou)

    final_bookings = _bookings_model()
    played_bookings = played.dropna(subset=["bookings_total"]).copy()
    final_bookings.fit(
        played_bookings[feature_columns].astype(float),
        np.clip(played_bookings["bookings_total"].astype(float), 0, None),
    )

    final_home_goals = _goals_model()
    final_away_goals = _goals_model()
    final_home_goals.fit(X_full, np.clip(played["FTHG"].astype(float), 0, None))
    final_away_goals.fit(X_full, np.clip(played["FTAG"].astype(float), 0, None))

    return TrainingArtifacts(
        feature_columns=feature_columns,
        winner_encoder=winner_encoder,
        over_under_model=final_over_under,
        winner_model=final_winner,
        bookings_model=final_bookings,
        home_goals_model=final_home_goals,
        away_goals_model=final_away_goals,
        config={
            "form_window": float(form_window),
            "elo_k": float(elo_k),
            "elo_home_advantage": float(elo_home_advantage),
            "max_score_goals": float(max_score_goals),
            "test_fraction": float(test_fraction),
        },
        metrics=metrics,
    )


def save_artifacts(bundle: TrainingArtifacts, artifact_path: Path, metrics_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "feature_columns": bundle.feature_columns,
        "winner_encoder": bundle.winner_encoder,
        "over_under_model": bundle.over_under_model,
        "winner_model": bundle.winner_model,
        "bookings_model": bundle.bookings_model,
        "home_goals_model": bundle.home_goals_model,
        "away_goals_model": bundle.away_goals_model,
        "config": bundle.config,
        "metrics": bundle.metrics,
    }

    joblib.dump(payload, artifact_path)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(payload["metrics"], handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train football prediction system")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="root data directory")
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("artifacts/football_prediction_bundle.joblib"),
        help="path to save trained model bundle",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/training_metrics.json"),
        help="path to save evaluation metrics JSON",
    )
    parser.add_argument("--form-window", type=int, default=5, help="rolling window size per team")
    parser.add_argument("--elo-k", type=float, default=20.0, help="elo k-factor")
    parser.add_argument("--elo-home-advantage", type=float, default=75.0, help="elo home advantage offset")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="chronological holdout fraction")
    parser.add_argument("--max-score-goals", type=int, default=7, help="max goals grid for scoreline probability")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    matches = load_all_matches(args.data_dir)
    features = build_leak_free_features(
        matches,
        form_window=args.form_window,
        elo_k=args.elo_k,
        elo_home_advantage=args.elo_home_advantage,
    )

    feature_columns = build_feature_columns()
    bundle = train_prediction_system(
        features,
        feature_columns=feature_columns,
        test_fraction=args.test_fraction,
        max_score_goals=args.max_score_goals,
        form_window=args.form_window,
        elo_k=args.elo_k,
        elo_home_advantage=args.elo_home_advantage,
    )

    save_artifacts(bundle, args.artifact_path, args.metrics_path)

    print("Training completed.")
    print(f"Artifact: {args.artifact_path}")
    print(f"Metrics:  {args.metrics_path}")
    print(json.dumps(bundle.metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
