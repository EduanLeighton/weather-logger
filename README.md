# Daily Weather Logger

An automated pipeline that logs daily weather data and predicts next-day
temperatures, running entirely on GitHub Actions.

## What it does

- Fetches current weather stats from a public API each day and appends
  them to a historical log.
- Trains a small machine learning model on that history to forecast the
  next day's temperature, then scores its own past predictions for
  accuracy.
- Regenerates summary charts and commits everything back to the
  repository automatically — no server or manual steps required.

## Tech stack

Python · Pandas · Matplotlib · scikit-learn · GitHub Actions

## License

MIT
