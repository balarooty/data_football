#!/usr/bin/env python3
"""
Clean and normalize prediction-sheet data from Google Sheets.

Fixes:
- Converts percentage probabilities (0-100) to decimal (0-1)
- Repairs corrupted 'Expected Away Goals' values like '12-31' -> 1.31
- Normalizes categorical outputs (winner/best pick/ou lean)
- Adds data quality diagnostics

Usage:
  python3 clean_google_sheet_predictions.py
  python3 clean_google_sheet_predictions.py --sheet-id <id>
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DEFAULT_SHEET_ID = "1feRsH0y7xfLtv8_zA6Iqsf9Fh0g4J3_qXURkG0htJYI"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean prediction data from Google Sheet")
    parser.add_argument("--sheet-id", default=DEFAULT_SHEET_ID, help="Google Sheet ID")
    parser.add_argument(
        "--input-url",
        default=None,
        help="Optional direct CSV URL (overrides --sheet-id)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/sheet_predictions_cleaned.csv"),
        help="Output cleaned CSV path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/sheet_predictions_cleaned.json"),
        help="Output cleaned JSON path",
    )
    parser.add_argument(
        "--diagnostics-json",
        type=Path,
        default=Path("artifacts/sheet_cleaning_diagnostics.json"),
        help="Output diagnostics JSON path",
    )
    return parser.parse_args()


def get_csv_url(sheet_id: str, input_url: str | None) -> str:
    if input_url:
        return input_url
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"


def to_numeric_clean(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    # Keep only digits, minus and dot for robust conversion.
    text = text.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def repair_expected_away_goals(value) -> Tuple[float, bool, str]:
    """
    Returns: (numeric_value, repaired_flag, repair_note)

    Known corruptions in sheet:
    - 12-31 => 1.31
    - 12-30 => 1.30
    - 1-1   => 1.10
    """
    if pd.isna(value):
        return np.nan, False, "nan"

    text = str(value).strip()

    direct = pd.to_numeric(text, errors="coerce")
    if pd.notna(direct):
        return float(direct), False, "ok"

    m = re.match(r"^(\d+)-(\d+)$", text)
    if not m:
        return np.nan, False, "unparsed"

    left, right = m.group(1), m.group(2)

    # Primary fixes observed in this sheet.
    if left == "12" and len(right) == 2:
        return float(f"1.{right}"), True, "repair_12_xx_to_1_xx"

    if len(left) == 1 and len(right) == 1:
        return float(f"{left}.{right}0"), True, "repair_x_y_to_x_y0"

    if len(left) == 1 and len(right) == 2:
        return float(f"{left}.{right}"), True, "repair_x_yy_to_x_yy"

    if len(left) == 2 and left.startswith("1") and len(right) == 1:
        return float(f"1.{left[1]}{right}"), True, "repair_1x_y_to_1_xy"

    return np.nan, False, "unparsed_pattern"


def normalize_winner(row: pd.Series) -> str:
    pred = str(row.get("Predicted Winner", "")).strip()
    home = str(row.get("Home Team", "")).strip()
    away = str(row.get("Away Team", "")).strip()

    p_lower = pred.lower()
    if "draw" in p_lower:
        return "D"

    if home and (pred.startswith(home) or pred == "H" or p_lower == "home"):
        return "H"

    if away and (pred.startswith(away) or pred == "A" or p_lower == "away"):
        return "A"

    if pred.upper() in {"H", "D", "A"}:
        return pred.upper()

    return ""


def normalize_ou_lean(value) -> str:
    text = str(value).strip().lower()
    if text.startswith("over"):
        return "OVER"
    if text.startswith("under"):
        return "UNDER"
    return ""


def build_best_pick_standard(row: pd.Series) -> str:
    probs = {
        "HOME_WIN": row.get("p_home_win", np.nan),
        "DRAW": row.get("p_draw", np.nan),
        "AWAY_WIN": row.get("p_away_win", np.nan),
        "OVER_2_5": row.get("p_over_2_5", np.nan),
        "UNDER_2_5": row.get("p_under_2_5", np.nan),
    }
    valid = {k: v for k, v in probs.items() if pd.notna(v)}
    if not valid:
        return ""
    return max(valid, key=valid.get)


def clean_sheet(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    out = df.copy()

    required = [
        "Date",
        "Country",
        "League",
        "Home Team",
        "Away Team",
        "P(Home Win)",
        "P(Draw)",
        "P(Away Win)",
        "P(Over 2.5)",
        "P(Under 2.5)",
        "Expected Home Goals",
        "Expected Away Goals",
        "Expected Bookings",
    ]
    missing_cols = [c for c in required if c not in out.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Parse base fields.
    out["date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=True)
    out["country"] = out["Country"].astype(str).str.strip()
    out["league"] = out["League"].astype(str).str.strip()
    out["home_team"] = out["Home Team"].astype(str).str.strip()
    out["away_team"] = out["Away Team"].astype(str).str.strip()

    # Probabilities.
    p_home_raw = to_numeric_clean(out["P(Home Win)"])
    p_draw_raw = to_numeric_clean(out["P(Draw)"])
    p_away_raw = to_numeric_clean(out["P(Away Win)"])
    p_over_raw = to_numeric_clean(out["P(Over 2.5)"])
    p_under_raw = to_numeric_clean(out["P(Under 2.5)"])

    # Auto-detect % scale.
    winner_scale = 100.0 if max(p_home_raw.max(), p_draw_raw.max(), p_away_raw.max()) > 1.5 else 1.0
    ou_scale = 100.0 if max(p_over_raw.max(), p_under_raw.max()) > 1.5 else 1.0

    out["p_home_win"] = p_home_raw / winner_scale
    out["p_draw"] = p_draw_raw / winner_scale
    out["p_away_win"] = p_away_raw / winner_scale
    out["p_over_2_5"] = p_over_raw / ou_scale
    out["p_under_2_5"] = p_under_raw / ou_scale

    # Confidence.
    out["confidence_text"] = out.get("Confidence", "").astype(str).str.strip().str.upper()
    out["confidence_score"] = out[["p_home_win", "p_draw", "p_away_win"]].max(axis=1)

    # Goals / bookings.
    out["expected_home_goals"] = to_numeric_clean(out["Expected Home Goals"])

    repaired_values = out["Expected Away Goals"].apply(repair_expected_away_goals)
    out["expected_away_goals"] = repaired_values.map(lambda x: x[0])
    out["expected_away_goals_repaired"] = repaired_values.map(lambda x: bool(x[1]))
    out["expected_away_goals_repair_note"] = repaired_values.map(lambda x: x[2])

    out["expected_bookings"] = to_numeric_clean(out["Expected Bookings"])

    # Normalized labels.
    out["predicted_winner_label"] = out.apply(normalize_winner, axis=1)
    out["ou_lean_label"] = out.get("O/U Lean", "").apply(normalize_ou_lean)
    out["best_pick_standard"] = out.apply(build_best_pick_standard, axis=1)

    # Parse expected score.
    score = out.get("Expected Score", "").astype(str).str.extract(r"^(\d+)-(\d+)$")
    out["expected_score_home"] = pd.to_numeric(score[0], errors="coerce")
    out["expected_score_away"] = pd.to_numeric(score[1], errors="coerce")

    # Quality diagnostics.
    winner_sum = out["p_home_win"] + out["p_draw"] + out["p_away_win"]
    ou_sum = out["p_over_2_5"] + out["p_under_2_5"]

    diagnostics: Dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": int(len(out)),
        "winner_probability_scale_detected": int(winner_scale),
        "ou_probability_scale_detected": int(ou_scale),
        "expected_away_goals_repaired_rows": int(out["expected_away_goals_repaired"].sum()),
        "unparsed_expected_away_goals_rows": int(out["expected_away_goals"].isna().sum()),
        "winner_prob_sum_bad_gt_0_03": int(((winner_sum - 1.0).abs() > 0.03).sum()),
        "ou_prob_sum_bad_gt_0_03": int(((ou_sum - 1.0).abs() > 0.03).sum()),
        "winner_label_missing_rows": int((out["predicted_winner_label"] == "").sum()),
        "date_unparsed_rows": int(out["date"].isna().sum()),
        "duplicate_key_rows": int(
            out.duplicated(subset=["date", "country", "league", "home_team", "away_team"], keep=False).sum()
        ),
    }

    # Reorder key columns for training use.
    ordered_cols = [
        "date",
        "country",
        "league",
        "home_team",
        "away_team",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "p_over_2_5",
        "p_under_2_5",
        "confidence_score",
        "confidence_text",
        "expected_home_goals",
        "expected_away_goals",
        "expected_away_goals_repaired",
        "expected_away_goals_repair_note",
        "expected_bookings",
        "predicted_winner_label",
        "ou_lean_label",
        "best_pick_standard",
        "Expected Score",
        "Top 3 Scores",
        "Best Pick",
        "Predicted Winner",
        "Date",
        "Country",
        "League",
        "Home Team",
        "Away Team",
    ]

    existing = [c for c in ordered_cols if c in out.columns]
    remaining = [c for c in out.columns if c not in existing]
    out = out[existing + remaining]

    return out, diagnostics


def main() -> int:
    args = parse_args()
    csv_url = get_csv_url(args.sheet_id, args.input_url)

    raw = pd.read_csv(csv_url)
    cleaned, diagnostics = clean_sheet(raw)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_json.parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_csv(args.output_csv, index=False)
    cleaned.to_json(args.output_json, orient="records", indent=2, date_format="iso")

    with open(args.diagnostics_json, "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2)

    print(f"Cleaned rows: {len(cleaned)}")
    print(f"Clean CSV: {args.output_csv}")
    print(f"Clean JSON: {args.output_json}")
    print(f"Diagnostics: {args.diagnostics_json}")
    print(json.dumps(diagnostics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
