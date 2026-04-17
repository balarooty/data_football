#!/usr/bin/env python3
"""
Football Data Scraper
=====================
Downloads CSV files from https://www.football-data.co.uk for main and extra
leagues, with automatic season rollover and update modes.

Usage examples:
    python scrape_football_data.py
    python scrape_football_data.py --mode replace --seasons 6
    python scrape_football_data.py --mode append --loop-hours 24

Update modes:
    replace: overwrite existing files with freshly scraped content (recommended)
    append : merge existing + new rows, deduplicate, and save
    skip   : only download files that do not exist locally
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASE_URL = "https://www.football-data.co.uk"
OUTPUT_DIR = Path(__file__).parent / "data"
DEFAULT_SEASONS = 5
DEFAULT_SEASON_START_MONTH = 7  # July rollover for EU leagues

MAIN_LEAGUES = {
    "england": {
        "country_name": "England",
        "divisions": {
            "E0": "premier_league",
            "E1": "championship",
            "E2": "league_1",
            "E3": "league_2",
            "EC": "conference",
        },
    },
    "scotland": {
        "country_name": "Scotland",
        "divisions": {
            "SC0": "premiership",
            "SC1": "division_1",
            "SC2": "division_2",
            "SC3": "division_3",
        },
    },
    "germany": {
        "country_name": "Germany",
        "divisions": {
            "D1": "bundesliga_1",
            "D2": "bundesliga_2",
        },
    },
    "italy": {
        "country_name": "Italy",
        "divisions": {
            "I1": "serie_a",
            "I2": "serie_b",
        },
    },
    "spain": {
        "country_name": "Spain",
        "divisions": {
            "SP1": "la_liga_primera",
            "SP2": "la_liga_segunda",
        },
    },
    "france": {
        "country_name": "France",
        "divisions": {
            "F1": "le_championnat",
            "F2": "division_2",
        },
    },
    "netherlands": {
        "country_name": "Netherlands",
        "divisions": {
            "N1": "eredivisie",
        },
    },
    "belgium": {
        "country_name": "Belgium",
        "divisions": {
            "B1": "jupiler_league",
        },
    },
    "portugal": {
        "country_name": "Portugal",
        "divisions": {
            "P1": "liga_i",
        },
    },
    "turkey": {
        "country_name": "Turkey",
        "divisions": {
            "T1": "ligi_1",
        },
    },
    "greece": {
        "country_name": "Greece",
        "divisions": {
            "G1": "ethniki_katigoria",
        },
    },
}

EXTRA_LEAGUES = {
    "argentina": {"code": "ARG", "division": "primera_division"},
    "austria": {"code": "AUT", "division": "bundesliga"},
    "brazil": {"code": "BRA", "division": "serie_a"},
    "china": {"code": "CHN", "division": "super_league"},
    "denmark": {"code": "DNK", "division": "superliga"},
    "finland": {"code": "FIN", "division": "veikkausliiga"},
    "ireland": {"code": "IRL", "division": "premier_division"},
    "japan": {"code": "JPN", "division": "j_league"},
    "mexico": {"code": "MEX", "division": "liga_mx"},
    "norway": {"code": "NOR", "division": "eliteserien"},
    "poland": {"code": "POL", "division": "ekstraklasa"},
    "romania": {"code": "ROU", "division": "liga_1"},
    "russia": {"code": "RUS", "division": "premier_league"},
    "sweden": {"code": "SWE", "division": "allsvenskan"},
    "switzerland": {"code": "SWZ", "division": "super_league"},
    "usa": {"code": "USA", "division": "mls"},
}


class Colors:
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_header() -> None:
    print(
        f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
║  Football Data Scraper — football-data.co.uk               ║
║  Dynamic seasons + replace/append update modes             ║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    )


def print_section(title: str) -> None:
    print(f"\n{Colors.MAGENTA}{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}{Colors.RESET}")


def print_success(msg: str) -> None:
    print(f"  {Colors.GREEN}✓{Colors.RESET} {msg}")


def print_skip(msg: str) -> None:
    print(f"  {Colors.YELLOW}⊘{Colors.RESET} {Colors.DIM}{msg}{Colors.RESET}")


def print_fail(msg: str) -> None:
    print(f"  {Colors.RED}✗{Colors.RESET} {msg}")


def print_summary(stats: Dict[str, int], output_dir: Path) -> None:
    total = stats["success"] + stats["skipped"] + stats["failed"]
    print(
        f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗
║  Summary                                                    ║
╠══════════════════════════════════════════════════════════════╣
║  Total files attempted:  {total:<35}║
║  {Colors.GREEN}Updated/downloaded:      {stats['success']:<35}{Colors.CYAN}║
║  {Colors.YELLOW}Skipped:                 {stats['skipped']:<35}{Colors.CYAN}║
║  {Colors.RED}Failed:                  {stats['failed']:<35}{Colors.CYAN}║
╠══════════════════════════════════════════════════════════════╣
║  Output directory: {str(output_dir):<41}║
╚══════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    )


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/csv,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": BASE_URL,
        }
    )

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def build_seasons(count: int, season_start_month: int = DEFAULT_SEASON_START_MONTH) -> Dict[str, str]:
    now = datetime.now()
    start_year = now.year if now.month >= season_start_month else now.year - 1

    seasons: Dict[str, str] = {}
    for offset in range(count):
        season_start = start_year - offset
        season_end = season_start + 1
        code = f"{season_start % 100:02d}{season_end % 100:02d}"
        label = f"{season_start}-{season_end}"
        seasons[code] = label
    return seasons


def fetch_csv(session: requests.Session, url: str, delay_seconds: float) -> Tuple[bool, int | None, str, str]:
    """Return (success, status_code, error_message, content)."""
    try:
        time.sleep(max(0.0, delay_seconds))
        response = session.get(url, timeout=30)
        status = response.status_code

        if status == 200:
            content = response.text
            stripped = content.strip()
            if not stripped:
                return False, status, "Empty response", ""
            if stripped.startswith("<!DOCTYPE") or stripped.lower().startswith("<html"):
                return False, status, "Response is HTML, not CSV", ""
            return True, status, "", content

        if status == 404:
            return False, status, "Not found", ""
        return False, status, f"HTTP {status}", ""

    except requests.exceptions.RequestException as exc:
        return False, None, str(exc), ""


def choose_dedupe_columns(df: pd.DataFrame) -> Iterable[str]:
    candidates = [
        ["Date", "Time", "HomeTeam", "AwayTeam"],
        ["Date", "HomeTeam", "AwayTeam"],
        ["Date", "Time", "Home", "Away"],
        ["Date", "Home", "Away"],
    ]
    columns = set(df.columns)
    for candidate in candidates:
        if set(candidate).issubset(columns):
            return candidate
    return [c for c in df.columns if c]


def safe_append_merge(dest_path: Path, new_csv_content: str) -> Tuple[bool, str]:
    """Append new content into existing CSV with dedupe. Falls back to replace on parse issues."""
    try:
        existing_df = pd.read_csv(dest_path)
        new_df = pd.read_csv(io.StringIO(new_csv_content))

        if existing_df.empty:
            merged = new_df
        elif new_df.empty:
            merged = existing_df
        else:
            merged = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
            dedupe_cols = list(choose_dedupe_columns(merged))
            merged = merged.drop_duplicates(subset=dedupe_cols, keep="last")

            if "Date" in merged.columns:
                parsed = pd.to_datetime(merged["Date"], dayfirst=True, errors="coerce")
                if "Time" in merged.columns:
                    parsed_time = pd.to_timedelta(merged["Time"].fillna("00:00") + ":00", errors="coerce")
                    sort_key = parsed + parsed_time.fillna(pd.Timedelta(0))
                else:
                    sort_key = parsed
                merged = merged.assign(_sort_key=sort_key).sort_values("_sort_key", kind="mergesort")
                merged = merged.drop(columns=["_sort_key"])

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(dest_path, index=False)
        return True, ""
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def write_csv(dest_path: Path, csv_content: str, mode: str) -> Tuple[bool, str]:
    """Write CSV according to update mode."""
    if mode == "skip" and dest_path.exists():
        return False, "SKIPPED_EXISTS"

    if mode == "append" and dest_path.exists():
        ok, error = safe_append_merge(dest_path, csv_content)
        if ok:
            return True, ""
        # Fallback to replace if append merge fails
        print_skip(f"append merge failed for {dest_path.name}, falling back to replace ({error})")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(csv_content)
    return True, ""


def scrape_main_leagues(
    session: requests.Session,
    stats: Dict[str, int],
    seasons: Dict[str, str],
    mode: str,
    delay_seconds: float,
) -> None:
    print_section("MAIN LEAGUES")

    for country_key, league_info in MAIN_LEAGUES.items():
        country_name = league_info["country_name"]
        print(f"\n  {Colors.BOLD}{country_name}{Colors.RESET}")

        for league_code, division_name in league_info["divisions"].items():
            for season_code, season_display in seasons.items():
                url = f"{BASE_URL}/mmz4281/{season_code}/{league_code}.csv"
                dest = OUTPUT_DIR / "main_leagues" / country_key / division_name / f"{season_display}.csv"
                label = f"{division_name} {season_display}"

                if mode == "skip" and dest.exists():
                    print_skip(f"{label} — already exists (skip mode)")
                    stats["skipped"] += 1
                    continue

                success, status, error, content = fetch_csv(session, url, delay_seconds)
                if not success:
                    if status == 404 or error == "Response is HTML, not CSV":
                        print_skip(f"{label} — not available yet")
                        stats["skipped"] += 1
                    else:
                        print_fail(f"{label} — {error}")
                        stats["failed"] += 1
                    continue

                updated, write_error = write_csv(dest, content, mode)
                if not updated and write_error == "SKIPPED_EXISTS":
                    print_skip(f"{label} — already exists (skip mode)")
                    stats["skipped"] += 1
                elif not updated:
                    print_fail(f"{label} — write failed: {write_error}")
                    stats["failed"] += 1
                else:
                    size_kb = dest.stat().st_size / 1024
                    print_success(f"{label} ({size_kb:.1f} KB)")
                    stats["success"] += 1


def scrape_extra_leagues(
    session: requests.Session,
    stats: Dict[str, int],
    mode: str,
    delay_seconds: float,
) -> None:
    print_section("EXTRA LEAGUES")

    for country_key, info in EXTRA_LEAGUES.items():
        country_code = info["code"]
        division_name = info["division"]
        label = f"{country_key.capitalize()} ({division_name})"

        url = f"{BASE_URL}/new/{country_code}.csv"
        dest = OUTPUT_DIR / "extra_leagues" / country_key / f"{division_name}.csv"

        if mode == "skip" and dest.exists():
            print_skip(f"{label} — already exists (skip mode)")
            stats["skipped"] += 1
            continue

        success, status, error, content = fetch_csv(session, url, delay_seconds)
        if not success:
            if status == 404:
                print_skip(f"{label} — not available")
                stats["skipped"] += 1
            else:
                print_fail(f"{label} — {error}")
                stats["failed"] += 1
            continue

        updated, write_error = write_csv(dest, content, mode)
        if not updated and write_error == "SKIPPED_EXISTS":
            print_skip(f"{label} — already exists (skip mode)")
            stats["skipped"] += 1
        elif not updated:
            print_fail(f"{label} — write failed: {write_error}")
            stats["failed"] += 1
        else:
            size_kb = dest.stat().st_size / 1024
            print_success(f"{label} ({size_kb:.1f} KB)")
            stats["success"] += 1


def run_once(args: argparse.Namespace) -> int:
    print_header()
    seasons = build_seasons(args.seasons, args.season_start_month)

    print(f"  {Colors.DIM}Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {args.mode}")
    print(f"  Seasons: {', '.join(seasons.values())}")
    print(f"  Output dir: {OUTPUT_DIR}{Colors.RESET}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = create_session()
    stats = {"success": 0, "skipped": 0, "failed": 0}

    try:
        scrape_main_leagues(session, stats, seasons, args.mode, args.delay_seconds)
        if args.include_extra:
            scrape_extra_leagues(session, stats, args.mode, args.delay_seconds)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}  Interrupted by user{Colors.RESET}")

    print_summary(stats, OUTPUT_DIR)
    return 0 if stats["failed"] == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape football-data.co.uk CSV files")
    parser.add_argument(
        "--mode",
        choices=["replace", "append", "skip"],
        default="replace",
        help="update mode for existing files (default: replace)",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        default=DEFAULT_SEASONS,
        help=f"number of rolling seasons for main leagues (default: {DEFAULT_SEASONS})",
    )
    parser.add_argument(
        "--season-start-month",
        type=int,
        default=DEFAULT_SEASON_START_MONTH,
        help=f"month when a new season starts (default: {DEFAULT_SEASON_START_MONTH})",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.5,
        help="delay between requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--no-extra",
        dest="include_extra",
        action="store_false",
        help="skip extra leagues scraping",
    )
    parser.set_defaults(include_extra=True)
    parser.add_argument(
        "--loop-hours",
        type=float,
        default=0.0,
        help="run continuously; if > 0, rerun scrape every N hours",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.loop_hours <= 0:
        return run_once(args)

    interval_seconds = max(1.0, args.loop_hours * 3600)
    exit_code = 0

    while True:
        exit_code = run_once(args)
        print(f"\n{Colors.DIM}Sleeping for {args.loop_hours} hours...{Colors.RESET}\n")
        try:
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Loop stopped by user.{Colors.RESET}")
            break

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
