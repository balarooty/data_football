# Football Prediction System

This project now includes:

1. A **regular scraper** with dynamic season rollover (`scrape_football_data.py`)
2. A **4-target prediction trainer** (`prediction_system.py`)
3. A **fixture prediction generator** (`predict_fixtures.py`)

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

```bash
python3 predict_fixtures.py \
  --data-dir data \
  --artifact-path artifacts/football_prediction_bundle.joblib \
  --output-path artifacts/upcoming_predictions.csv \
  --json-output-path artifacts/upcoming_predictions.json
```

## 4) Proven Modeling Approach Used

The system combines proven football methods:

- **Rolling form features** (team goals, points, shots, cards, corners, fouls)
- **Elo strength ratings** with home advantage
- **Odds-implied probabilities** from available bookmaker columns
- **Poisson goal modeling** for correct-score probabilities

## Current Baseline Metrics (from your dataset)

From the latest run in this workspace:

- `winner_accuracy`: `0.4993`
- `over_under_accuracy`: `0.5663`
- `bookings_mae`: `1.6904`
- `correct_score_exact_accuracy`: `0.1318`
- `correct_score_top3_accuracy`: `0.3433`

## Suggested Daily Workflow

```bash
python3 scrape_football_data.py --mode replace --seasons 5
python3 prediction_system.py
python3 predict_fixtures.py
```

This keeps your files updated, retrains the models, and outputs fresh predictions for your agent.
