#!/usr/bin/env python3
"""
A small self-scoring temperature forecaster.

Each day, per location, this script:
  1. Scores any earlier prediction whose target date has now arrived,
     by comparing it against the actual temperature just logged.
  2. Trains a fresh scikit-learn model on all history collected so
     far and predicts tomorrow's temperature.

Predictions live in data/predictions.csv, which grows a little more
meaningful every day as the training set behind it grows — early on
there won't be enough history to predict from, and that's expected.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_DIR = Path(__file__).parent / "data"
HISTORY_PATH = DATA_DIR / "weather_history.csv"
PREDICTIONS_PATH = DATA_DIR / "predictions.csv"

# Minimum days of history required per location before a prediction is
# attempted — below this the model has too little signal to be useful.
MIN_HISTORY_DAYS = 10

PREDICTION_FIELDS = [
    "location", "predicted_on", "target_date",
    "predicted_temp_c", "actual_temp_c", "abs_error_c",
]

FEATURE_COLUMNS = ["doy_sin", "doy_cos", "lag_1", "lag_2", "rolling_mean_3"]


def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_PATH, parse_dates=["date"])
    # keep one row per location per day, in case the workflow ever runs twice
    return df.sort_values("date").drop_duplicates(["location", "date"], keep="first")


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame(columns=PREDICTION_FIELDS)
    return pd.read_csv(PREDICTIONS_PATH, parse_dates=["predicted_on", "target_date"])


def save_predictions(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(PREDICTIONS_PATH, index=False)


def score_pending_predictions(history: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Fill in actual_temp_c / abs_error_c once a prediction's target date has data."""
    if predictions.empty:
        return predictions

    pending = predictions["actual_temp_c"].isna()
    for idx in predictions[pending].index:
        loc = predictions.at[idx, "location"]
        target_date = predictions.at[idx, "target_date"]
        match = history[(history["location"] == loc) & (history["date"] == target_date)]
        if not match.empty:
            actual = match.iloc[0]["temperature_c"]
            predicted = predictions.at[idx, "predicted_temp_c"]
            predictions.at[idx, "actual_temp_c"] = actual
            predictions.at[idx, "abs_error_c"] = round(abs(predicted - actual), 2)
    return predictions


def build_features(location_history: pd.DataFrame) -> pd.DataFrame:
    """Turn a per-location time series into a feature table: seasonality + lag temps."""
    df = location_history.sort_values("date").reset_index(drop=True)
    day_of_year = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["lag_1"] = df["temperature_c"].shift(1)
    df["lag_2"] = df["temperature_c"].shift(2)
    df["rolling_mean_3"] = df["temperature_c"].shift(1).rolling(3).mean()
    return df


def predict_next_day(location: str, location_history: pd.DataFrame) -> Optional[float]:
    if len(location_history) < MIN_HISTORY_DAYS:
        print(f"{location}: only {len(location_history)} day(s) logged, "
              f"need {MIN_HISTORY_DAYS} before predicting.")
        return None

    features = build_features(location_history)
    train = features.dropna(subset=FEATURE_COLUMNS)
    if len(train) < 5:
        print(f"{location}: not enough complete feature rows yet.")
        return None

    model = LinearRegression()
    model.fit(train[FEATURE_COLUMNS], train["temperature_c"])

    latest = features.iloc[[-1]][FEATURE_COLUMNS]
    if latest.isna().any(axis=None):
        print(f"{location}: latest row missing lag features, skipping.")
        return None

    return float(model.predict(latest)[0])


def main() -> None:
    history = load_history()
    if history.empty:
        print("No weather history yet — run weather_logger.py first.")
        return

    predictions = load_predictions()
    predictions = score_pending_predictions(history, predictions)

    today = datetime.now(timezone.utc).date()
    tomorrow = today + timedelta(days=1)
    new_rows = []

    for location, group in history.groupby("location"):
        already_predicted = (
            (predictions["location"] == location)
            & (predictions["target_date"] == pd.Timestamp(tomorrow))
        ).any() if not predictions.empty else False
        if already_predicted:
            continue

        predicted_temp = predict_next_day(location, group)
        if predicted_temp is None:
            continue

        new_rows.append({
            "location": location,
            "predicted_on": pd.Timestamp(today),
            "target_date": pd.Timestamp(tomorrow),
            "predicted_temp_c": round(predicted_temp, 2),
            "actual_temp_c": np.nan,
            "abs_error_c": np.nan,
        })
        print(f"{location}: predicted {predicted_temp:.1f}°C for {tomorrow}")

    if new_rows:
        predictions = pd.concat([predictions, pd.DataFrame(new_rows)], ignore_index=True)

    save_predictions(predictions)


if __name__ == "__main__":
    main()
