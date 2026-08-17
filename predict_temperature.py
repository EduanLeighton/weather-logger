#!/usr/bin/env python3
"""
A self-scoring temperature forecaster using Random Forest.

Each day, per location, this script:
  1. Scores any earlier prediction whose target date has now arrived,
     by comparing it against the actual temperature just logged.
  2. Evaluates and prints the cumulative Mean Absolute Error (MAE).
  3. Trains a fresh RandomForestRegressor on all history collected so
     far to predict tomorrow's temperature.

Predictions live in data/predictions.csv, which grows a little more
meaningful every day as the training set behind it grows.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = Path(__file__).parent / "data"
HISTORY_PATH = DATA_DIR / "weather_history.csv"
PREDICTIONS_PATH = DATA_DIR / "predictions.csv"

MIN_HISTORY_DAYS = 10

PREDICTION_FIELDS = [
    "location", "predicted_on", "target_date",
    "predicted_temp_c", "actual_temp_c", "abs_error_c",
]

BASE_FEATURE_COLUMNS = [
    "doy_sin",
    "doy_cos",
    "temperature_c",
    "feels_like_c",
    "humidity_pct",
    "wind_speed_kmh",
    "precipitation_mm",
    "day_min_c",
    "day_max_c",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_5",
    "lag_7",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_std_7",
]


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

    # Report performance (MAE) per location
    scored = predictions.dropna(subset=["abs_error_c"])
    if not scored.empty:
        print("\n--- Model Performance ---")
        for loc, group in scored.groupby("location"):
            mae = group["abs_error_c"].mean()
            print(f"{loc} Average Error (MAE): {mae:.2f}°C (over {len(group)} predictions)")
        print("-------------------------\n")

    return predictions


def build_features(location_history: pd.DataFrame) -> pd.DataFrame:
    """Turn a per-location time series into a rich feature table."""
    df = location_history.sort_values("date").reset_index(drop=True)

    # Day of year sine/cosine encoding
    day_of_year = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Lags on temperature
    df["lag_1"] = df["temperature_c"].shift(1)
    df["lag_2"] = df["temperature_c"].shift(2)
    df["lag_3"] = df["temperature_c"].shift(3)
    df["lag_5"] = df["temperature_c"].shift(5)
    df["lag_7"] = df["temperature_c"].shift(7)

    # Rolling statistics (using shift(1) so today's values are not polluted by future information)
    df["rolling_mean_3"] = df["temperature_c"].shift(1).rolling(3).mean()
    df["rolling_mean_7"] = df["temperature_c"].shift(1).rolling(7).mean()
    df["rolling_std_7"] = df["temperature_c"].shift(1).rolling(7).std()

    # Target variable: Tomorrow's temperature
    df["target"] = df["temperature_c"].shift(-1)

    # One-hot encode weather conditions if the column exists
    if "conditions" in df.columns:
        encoded_conditions = pd.get_dummies(df["conditions"], prefix="cond", dtype=float)
        df = pd.concat([df, encoded_conditions], axis=1)

    return df


def predict_next_day(location: str, location_history: pd.DataFrame) -> Optional[float]:
    if len(location_history) < MIN_HISTORY_DAYS:
        print(f"{location}: only {len(location_history)} day(s) logged, "
              f"need {MIN_HISTORY_DAYS} before predicting.")
        return None

    features = build_features(location_history)

    # Identify all feature columns (base features + any dynamic dummy variables from conditions)
    condition_cols = [c for c in features.columns if c.startswith("cond_")]
    feature_cols = BASE_FEATURE_COLUMNS + condition_cols

    # Ensure required base columns are available in the input DataFrame
    missing_cols = [col for col in BASE_FEATURE_COLUMNS if col not in features.columns]
    if missing_cols:
        print(f"{location}: history missing required columns: {missing_cols}")
        return None

    # Separate training data (rows where we know both features and tomorrow's target)
    train = features.dropna(subset=feature_cols + ["target"])
    if len(train) < 5:
        print(f"{location}: not enough complete feature/target training rows yet.")
        return None

    # Latest row represents TODAY — used to predict TOMORROW
    latest = features.iloc[[-1]][feature_cols]
    if latest.isna().any(axis=None):
        print(f"{location}: latest row missing required features/lags, skipping.")
        return None

    # Fit Random Forest model
    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(train[feature_cols], train["target"])

    prediction = model.predict(latest)[0]
    return float(prediction)


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
