#!/usr/bin/env python3
"""
Reads data/weather_history.csv and regenerates a temperature trend
chart, saved to chart.png so it can be embedded in the README and
stay up to date automatically each day.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path(__file__).parent / "data" / "weather_history.csv"
PREDICTIONS_PATH = Path(__file__).parent / "data" / "predictions.csv"
CHART_PATH = Path(__file__).parent / "chart.png"
PREDICTIONS_CHART_PATH = Path(__file__).parent / "predictions.png"


def generate_temperature_chart() -> None:
    if not CSV_PATH.exists():
        print("No data yet — run weather_logger.py first.")
        return

    df = pd.read_csv(CSV_PATH, parse_dates=["date"])

    fig, ax = plt.subplots(figsize=(10, 4))
    for location, group in df.groupby("location"):
        ax.plot(group["date"], group["temperature_c"], marker="o", label=location)

    ax.set_title("Temperature over time")
    ax.set_ylabel("°C")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=120)
    print(f"Chart saved to {CHART_PATH}")


def generate_predictions_chart() -> None:
    if not PREDICTIONS_PATH.exists():
        return

    df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["target_date"])
    scored = df.dropna(subset=["actual_temp_c"])
    if scored.empty:
        print("No scored predictions yet — nothing to chart.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for location, group in scored.groupby("location"):
        group = group.sort_values("target_date")
        ax.plot(group["target_date"], group["predicted_temp_c"], marker="o",
                 linestyle="--", label=f"{location} (predicted)")
        ax.plot(group["target_date"], group["actual_temp_c"], marker="o",
                 label=f"{location} (actual)")

    ax.set_title("Predicted vs. actual temperature")
    ax.set_ylabel("°C")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PREDICTIONS_CHART_PATH, dpi=120)

    mae = scored["abs_error_c"].mean()
    print(f"Predictions chart saved to {PREDICTIONS_CHART_PATH} (MAE so far: {mae:.2f}°C)")


def main() -> None:
    generate_temperature_chart()
    generate_predictions_chart()


if __name__ == "__main__":
    main()
