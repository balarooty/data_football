#!/usr/bin/env python3
"""
Fetch upcoming fixtures from TheStatsAPI and export as CSV for prediction.

Auth:
  Authorization: Bearer <api_key>

Example:
  export THESTATSAPI_API_KEY='your_key'
  python3 fetch_upcoming_fixtures_api.py --days 10 --include-odds
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.thestatsapi.com/api/football"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch upcoming fixtures from TheStatsAPI")
    parser.add_argument("--api-key", default=None, help="TheStatsAPI key (or use THESTATSAPI_API_KEY env var)")
    parser.add_argument("--date-from", default=None, help="start date YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--date-to", default=None, help="end date YYYY-MM-DD (default: date_from + days)")
    parser.add_argument("--days", type=int, default=7, help="window size in days when date-to not provided")
    parser.add_argument("--status", default="scheduled", help="match status filter (default: scheduled)")
    parser.add_argument("--per-page", type=int, default=100, help="page size")
    parser.add_argument("--max-pages", type=int, default=50, help="max pages to request")
    parser.add_argument("--include-odds", dest="include_odds", action="store_true", help="fetch match odds per fixture (default: on)")
    parser.add_argument("--no-odds", dest="include_odds", action="store_false", help="skip odds calls for faster fixture fetch")
    parser.set_defaults(include_odds=True)
    parser.add_argument("--odds-sleep-ms", type=int, default=120, help="sleep between odds calls")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/upcoming_fixtures_api.csv"),
        help="output CSV path",
    )
    parser.add_argument(
        "--json-output-path",
        type=Path,
        default=Path("artifacts/upcoming_fixtures_api.json"),
        help="output JSON path",
    )
    return parser.parse_args()


def _resolve_api_key(explicit_key: Optional[str]) -> str:
    if explicit_key:
        return explicit_key.strip()
    import os

    env_key = os.getenv("THESTATSAPI_API_KEY", "").strip()
    if env_key:
        return env_key
    raise ValueError("No API key found. Pass --api-key or set THESTATSAPI_API_KEY")


def _request_json(session: requests.Session, url: str, params: Optional[dict] = None) -> dict:
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_competitions_map(session: requests.Session) -> Dict[str, dict]:
    page = 1
    per_page = 100
    max_pages = 20
    mapping: Dict[str, dict] = {}

    while page <= max_pages:
        try:
            payload = _request_json(
                session,
                f"{BASE_URL}/competitions",
                params={"page": page, "per_page": per_page},
            )
        except requests.HTTPError:
            # Some plans/endpoints can be strict on pagination params.
            payload = _request_json(session, f"{BASE_URL}/competitions", params=None)
        data = payload.get("data", [])
        meta = payload.get("meta", {})

        for comp in data:
            comp_id = comp.get("id")
            if comp_id:
                mapping[comp_id] = comp

        total_pages = int(meta.get("total_pages", page))
        if page >= total_pages:
            break
        page += 1

    return mapping


def _extract_market_price(item: dict) -> float:
    if not isinstance(item, dict):
        return np.nan
    for key in ("last_seen", "opening"):
        val = item.get(key)
        if val is None:
            continue
        try:
            v = float(val)
            if v > 1.0:
                return v
        except (TypeError, ValueError):
            continue
    return np.nan


def fetch_match_odds_snapshot(session: requests.Session, match_id: str) -> dict:
    try:
        payload = _request_json(session, f"{BASE_URL}/matches/{match_id}/odds")
    except requests.HTTPError:
        return {}

    bookmakers = payload.get("data", {}).get("bookmakers", [])
    if not bookmakers:
        return {}

    homes: List[float] = []
    draws: List[float] = []
    aways: List[float] = []
    overs: List[float] = []
    unders: List[float] = []

    for bookmaker in bookmakers:
        markets = bookmaker.get("markets", {}) if isinstance(bookmaker, dict) else {}

        mo = markets.get("match_odds", {})
        homes.append(_extract_market_price(mo.get("home", {})))
        draws.append(_extract_market_price(mo.get("draw", {})))
        aways.append(_extract_market_price(mo.get("away", {})))

        tg = markets.get("total_goals", {})
        tg25 = tg.get("2.5", {}) if isinstance(tg, dict) else {}
        overs.append(_extract_market_price(tg25.get("over", {})))
        unders.append(_extract_market_price(tg25.get("under", {})))

    def _mean(xs: List[float]) -> float:
        arr = np.array(xs, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).any():
            return np.nan
        return float(np.nanmean(arr))

    return {
        "odds_home": _mean(homes),
        "odds_draw": _mean(draws),
        "odds_away": _mean(aways),
        "odds_over25": _mean(overs),
        "odds_under25": _mean(unders),
    }


def fetch_upcoming_matches(
    session: requests.Session,
    date_from: str,
    date_to: str,
    status: str,
    per_page: int,
    max_pages: int,
) -> List[dict]:
    page = 1
    rows: List[dict] = []

    while page <= max_pages:
        payload = _request_json(
            session,
            f"{BASE_URL}/matches",
            params={
                "status": status,
                "date_from": date_from,
                "date_to": date_to,
                "page": page,
                "per_page": per_page,
            },
        )

        data = payload.get("data", [])
        meta = payload.get("meta", {})
        total_pages = int(meta.get("total_pages", page))

        rows.extend(data)
        if page >= total_pages:
            break
        page += 1

    return rows


def normalize_rows(raw_rows: List[dict], competitions_map: Dict[str, dict], include_odds: bool, session: requests.Session, odds_sleep_ms: int) -> pd.DataFrame:
    out_rows: List[dict] = []

    for row in raw_rows:
        match_id = row.get("id")
        comp_id = row.get("competition_id")
        comp_meta = competitions_map.get(comp_id, {})

        utc_date = row.get("utc_date")
        dt = pd.to_datetime(utc_date, errors="coerce", utc=True)

        home_team = (row.get("home_team") or {}).get("name")
        away_team = (row.get("away_team") or {}).get("name")

        clean = {
            "api_match_id": match_id,
            "competition_id": comp_id,
            "country": comp_meta.get("country") or "unknown",
            "league": comp_meta.get("name") or "unknown",
            "Date": dt.strftime("%d/%m/%Y") if pd.notna(dt) else "",
            "Time": dt.strftime("%H:%M") if pd.notna(dt) else "",
            "utc_date": utc_date,
            "match_datetime": dt.isoformat() if pd.notna(dt) else "",
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "status": row.get("status"),
            "odds_available": bool(row.get("odds_available", False)),
            "xg_available": bool(row.get("xg_available", False)),
            "FTHG": np.nan,
            "FTAG": np.nan,
            "FTR": "",
        }

        if include_odds:
            odds = fetch_match_odds_snapshot(session, match_id)
            clean.update(
                {
                    "odds_home": odds.get("odds_home", np.nan),
                    "odds_draw": odds.get("odds_draw", np.nan),
                    "odds_away": odds.get("odds_away", np.nan),
                    "odds_over25": odds.get("odds_over25", np.nan),
                    "odds_under25": odds.get("odds_under25", np.nan),
                }
            )
            if odds_sleep_ms > 0:
                time.sleep(odds_sleep_ms / 1000.0)
        else:
            clean.update(
                {
                    "odds_home": np.nan,
                    "odds_draw": np.nan,
                    "odds_away": np.nan,
                    "odds_over25": np.nan,
                    "odds_under25": np.nan,
                }
            )

        out_rows.append(clean)

    df = pd.DataFrame(out_rows)
    if not df.empty:
        df = df.sort_values("match_datetime", kind="mergesort").reset_index(drop=True)
    return df


def main() -> int:
    args = parse_args()
    api_key = _resolve_api_key(args.api_key)

    now_utc = datetime.now(UTC)
    if args.date_from:
        date_from = args.date_from
    else:
        date_from = now_utc.date().isoformat()

    if args.date_to:
        date_to = args.date_to
    else:
        date_to = (pd.to_datetime(date_from).date() + timedelta(days=max(0, args.days))).isoformat()

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}"})

    competitions_map = fetch_competitions_map(session)
    raw_rows = fetch_upcoming_matches(
        session=session,
        date_from=date_from,
        date_to=date_to,
        status=args.status,
        per_page=args.per_page,
        max_pages=args.max_pages,
    )

    fixtures_df = normalize_rows(
        raw_rows=raw_rows,
        competitions_map=competitions_map,
        include_odds=args.include_odds,
        session=session,
        odds_sleep_ms=args.odds_sleep_ms,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_output_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures_df.to_csv(args.output_path, index=False)

    with open(args.json_output_path, "w", encoding="utf-8") as handle:
        json.dump(fixtures_df.to_dict(orient="records"), handle, indent=2)

    print(f"Fetched fixtures: {len(fixtures_df)}")
    print(f"Date window: {date_from} -> {date_to}")
    print(f"CSV:  {args.output_path}")
    print(f"JSON: {args.json_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
