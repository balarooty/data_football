# Football Prediction System

This project now includes:

1. A **regular scraper** with dynamic season rollover (`scrape_football_data.py`)
2. A **4-target prediction trainer** (`prediction_system.py`)
3. A **fixture prediction generator** (`predict_fixtures.py`)
4. A **TheStatsAPI upcoming fixture fetcher** (`fetch_upcoming_fixtures_api.py`)

All models are trained **only from your local `data/` CSV files**.

## 1) Update Data Regularly

### Recommended mode (replace)
`replace` overwrites local files with freshly scraped versions from football-data.co.uk.

```bash
python3 scrape_football_data.py --mode replace --seasons 5
```

### Append mode
`append` merges new rows with existing rows and deduplicates by match keys.

```bash
python3 scrape_football_data.py --mode append --seasons 5
```

### Run continuously every 24 hours

```bash
python3 scrape_football_data.py --mode replace --seasons 5 --loop-hours 24
```

## 2) Train Prediction Models

```bash
python3 prediction_system.py \
  --data-dir data \
  --artifact-path artifacts/football_prediction_bundle.joblib \
  --metrics-path artifacts/training_metrics.json
```

Targets trained:

1. Over/Under 2.5 goals
2. Match winner (H/D/A)
3. Total bookings
4. Correct score (Poisson scoreline from expected goals models)

## 3) Predict Upcoming Fixtures

### Option A: predict from local unplayed rows

```bash
python3 predict_fixtures.py \
  --data-dir data \
  --artifact-path artifacts/football_prediction_bundle.joblib \
  --output-path artifacts/upcoming_predictions.csv \
  --json-output-path artifacts/upcoming_predictions.json
```

### Option B: predict from TheStatsAPI upcoming fixtures

```bash
export THESTATSAPI_API_KEY='YOUR_KEY'

python3 fetch_upcoming_fixtures_api.py \
  --days 14 \
  --status scheduled \
  --output-path artifacts/upcoming_fixtures_api.csv \
  --json-output-path artifacts/upcoming_fixtures_api.json

python3 predict_fixtures.py \
  --data-dir data \
  --artifact-path artifacts/football_prediction_bundle.joblib \
  --fixtures-csv artifacts/upcoming_fixtures_api.csv \
  --output-path artifacts/upcoming_predictions_api.csv \
  --json-output-path artifacts/upcoming_predictions_api.json
```

### Optional: pull API odds too (for stronger blended probabilities)

Odds are now fetched by default.  
Use `--no-odds` only when you want faster/cheaper fixture pulls.

```bash
python3 fetch_upcoming_fixtures_api.py \
  --days 7 \
  --output-path artifacts/upcoming_fixtures_api_with_odds.csv \
  --json-output-path artifacts/upcoming_fixtures_api_with_odds.json
```

## 4) Improved Modeling Approach Used

The system combines proven football methods:

- **Rolling form features** (team goals, points, shots, cards, corners, fouls)
- **Elo strength ratings** with home advantage
- **Odds-implied probabilities** from available bookmaker columns
- **Poisson goal modeling** for correct-score probabilities
- **Market blending calibration**:
  - winner probabilities blend model + implied market probabilities using tuned alpha
  - over/under probabilities blend model + implied 2.5 market probabilities using tuned alpha

## Current Baseline Metrics (from your dataset)

From the latest run in this workspace:

- `winner_accuracy`: `0.4993`
- `winner_accuracy_blended`: `0.5018`
- `over_under_accuracy`: `0.5663`
- `over_under_accuracy_blended`: `0.5715`
- `bookings_mae`: `1.6904`
- `correct_score_exact_accuracy`: `0.1318`
- `correct_score_top3_accuracy`: `0.3433`

## Suggested Daily Workflow

```bash
python3 scrape_football_data.py --mode replace --seasons 5
python3 prediction_system.py
python3 fetch_upcoming_fixtures_api.py --days 7
python3 predict_fixtures.py --fixtures-csv artifacts/upcoming_fixtures_api.csv
```

This keeps your files updated, retrains the models, and outputs fresh predictions for your agent.
