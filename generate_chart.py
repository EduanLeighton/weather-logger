#!/usr/bin/env python3
"""
Reads data/weather_history.csv (and data/predictions.csv) and
regenerates chart.png / predictions.png so they stay embedded and
up to date in the README.

The x-axis automatically adapts to how much data exists: daily
points while the log is young, then weekly and eventually monthly
averages as it grows, so the chart stays readable over months of
history instead of becoming a wall of daily points.
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = Path(__file__).parent / "data" / "weather_history.csv"
PREDICTIONS_PATH = Path(__file__).parent / "data" / "predictions.csv"
CHART_PATH = Path(__file__).parent / "chart.png"
PREDICTIONS_CHART_PATH = Path(__file__).parent / "predictions.png"

# Thresholds (in days of covered date range) for switching aggregation level.
WEEKLY_THRESHOLD_DAYS = 60    # beyond ~2 months, switch daily points to weekly averages
MONTHLY_THRESHOLD_DAYS = 400  # beyond ~13 months, switch to monthly averages


def resample_rule_for_span(span_days: int) -> Tuple[Optional[str], str]:
    """Pick a pandas resample rule and a label describing it, based on date range."""
    if span_days <= WEEKLY_THRESHOLD_DAYS:
        return None, "daily"
    if span_days <= MONTHLY_THRESHOLD_DAYS:
        return "W", "weekly average"
    return "MS", "monthly average"


def style_date_axis(ax) -> None:
    """Let matplotlib pick sensible, non-overlapping date tick spacing automatically."""
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def generate_temperature_chart() -> None:
    if not CSV_PATH.exists():
        print("No data yet — run weather_logger.py first.")
        return

    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    span_days = (df["date"].max() - df["date"].min()).days
    rule, label = resample_rule_for_span(span_days)

    fig, ax = plt.subplots(figsize=(10, 4))
    for location, group in df.groupby("location"):
        group = group.sort_values("date")
        if rule:
            series = group.set_index("date")["temperature_c"].resample(rule).mean().dropna()
            x, y = series.index, series.values
        else:
            x, y = group["date"], group["temperature_c"]
        ax.plot(x, y, marker="o", label=location)

    ax.set_title(f"Temperature over time ({label})")
    ax.set_ylabel("°C")
    ax.legend()
    ax.grid(alpha=0.3)
    style_date_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=120)
    print(f"Chart saved to {CHART_PATH} ({label}, {span_days} day span)")


def generate_predictions_chart() -> None:
    if not PREDICTIONS_PATH.exists():
        return

    df = pd.read_csv(PREDICTIONS_PATH, parse_dates=["target_date"])
    scored = df.dropna(subset=["actual_temp_c"])
    if scored.empty:
        print("No scored predictions yet — nothing to chart.")
        return

    span_days = (scored["target_date"].max() - scored["target_date"].min()).days
    rule, label = resample_rule_for_span(span_days)

    fig, ax = plt.subplots(figsize=(10, 4))
    for location, group in scored.groupby("location"):
        group = group.sort_values("target_date")
        if rule:
            indexed = group.set_index("target_date")
            predicted = indexed["predicted_temp_c"].resample(rule).mean().dropna()
            actual = indexed["actual_temp_c"].resample(rule).mean().dropna()
            ax.plot(predicted.index, predicted.values, marker="o", linestyle="--",
                     label=f"{location} (predicted)")
            ax.plot(actual.index, actual.values, marker="o", label=f"{location} (actual)")
        else:
            ax.plot(group["target_date"], group["predicted_temp_c"], marker="o",
                     linestyle="--", label=f"{location} (predicted)")
            ax.plot(group["target_date"], group["actual_temp_c"], marker="o",
                     label=f"{location} (actual)")

    ax.set_title(f"Predicted vs. actual temperature ({label})")
    ax.set_ylabel("°C")
    ax.legend()
    ax.grid(alpha=0.3)
    style_date_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PREDICTIONS_CHART_PATH, dpi=120)

    mae = scored["abs_error_c"].mean()
    print(f"Predictions chart saved to {PREDICTIONS_CHART_PATH} "
          f"({label}, MAE so far: {mae:.2f}°C)")


def main() -> None:
    generate_temperature_chart()
    generate_predictions_chart()


if __name__ == "__main__":
    main()
