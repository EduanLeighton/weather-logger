# Daily Weather Logger

A small, fully-automated pipeline that fetches live weather stats every day
and commits them straight into this repository — no server, no database,
no manual steps.

![Daily weather log](https://github.com/<your-username>/<your-repo>/actions/workflows/daily-weather.yml/badge.svg)

## How it works

```
GitHub Actions (cron, daily)
        │
        ▼
weather_logger.py  ──►  Open-Meteo API (free, no key needed)
        │
        ▼
data/weather_history.csv   (one row appended per day)
        │
        ▼
predict_temperature.py  ──►  scores yesterday's forecast,
        │                    trains a fresh model, predicts tomorrow
        ▼
data/predictions.csv
        │
        ▼
generate_chart.py  ──►  chart.png + predictions.png (regenerated each run)
        │
        ▼
git commit + push  (back into this repo)
```

Everything runs on GitHub's own infrastructure via a scheduled
[GitHub Actions](.github/workflows/daily-weather.yml) workflow — there's
nothing to host or keep running locally.

## Temperature trend

![Temperature trend](chart.png)

*(Updates automatically once the workflow has run a few times.)*

## Next-day forecast

Alongside logging, the project trains a small scikit-learn model
(`predict_temperature.py`) on its own accumulated history — seasonality
(day of year) plus recent-day lag features — to predict tomorrow's
temperature. Each day it also scores *yesterday's* prediction against
what actually happened, so `data/predictions.csv` becomes a running,
self-graded forecast log.

![Predicted vs actual temperature](predictions.png)

It needs at least 10 days of logged history per location before it starts
predicting, so this chart appears empty at first and fills in as the
dataset grows — accuracy (mean absolute error) is printed by
`generate_chart.py` each run and improves as the training set does.

## Data collected

Each day's row in `data/weather_history.csv` includes:

| Field | Description |
|---|---|
| `temperature_c` | Current temperature |
| `feels_like_c` | Apparent temperature |
| `humidity_pct` | Relative humidity |
| `wind_speed_kmh` | Wind speed |
| `precipitation_mm` | Precipitation |
| `day_min_c` / `day_max_c` | Day's forecast min/max |
| `conditions` | Human-readable weather description |

## Running it yourself

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
python weather_logger.py       # fetch today's stats
python predict_temperature.py  # score yesterday's forecast, predict tomorrow
python generate_chart.py       # regenerate chart.png + predictions.png
```

To track your own location(s), edit the `LOCATIONS` dictionary at the top
of `weather_logger.py`.

## Automation

The workflow in `.github/workflows/daily-weather.yml` runs on a daily cron
schedule and can also be triggered manually from the **Actions** tab. It
needs no secrets — it authenticates with the automatically-provided
`GITHUB_TOKEN`, scoped to `contents: write` for this repo only.

## Tech stack

Python · Requests · Pandas · Matplotlib · scikit-learn · GitHub Actions · Open-Meteo API
