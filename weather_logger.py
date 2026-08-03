#!/usr/bin/env python3
"""
Daily weather logger.

Fetches current + daily weather stats from the free Open-Meteo API
(no API key required) and appends them as a new row to a CSV file.

Meant to be run once a day by a GitHub Actions workflow, which then
commits the updated CSV (and chart) back to the repository. See
.github/workflows/daily-weather.yml.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOCATIONS = {
    "Lichtenburg, North West": {"latitude": -26.152, "longitude": 26.160}, 
}

DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "weather_history.csv"

# WMO weather codes -> human-readable descriptions
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm",
}

FIELDNAMES = [
    "date", "time_utc", "location", "temperature_c", "feels_like_c",
    "humidity_pct", "wind_speed_kmh", "precipitation_mm",
    "day_min_c", "day_max_c", "conditions",
]


def fetch_weather(lat: float, lon: float) -> dict:
    """Call the Open-Meteo API and return the parsed JSON response."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "precipitation,weather_code,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,weather_code",
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def build_row(location_name: str, payload: dict) -> dict:
    current = payload["current"]
    daily = payload["daily"]
    now = datetime.now(timezone.utc)

    return {
        "date": now.strftime("%Y-%m-%d"),
        "time_utc": now.strftime("%H:%M:%S"),
        "location": location_name,
        "temperature_c": current["temperature_2m"],
        "feels_like_c": current["apparent_temperature"],
        "humidity_pct": current["relative_humidity_2m"],
        "wind_speed_kmh": current["wind_speed_10m"],
        "precipitation_mm": current["precipitation"],
        "day_min_c": daily["temperature_2m_min"][0],
        "day_max_c": daily["temperature_2m_max"][0],
        "conditions": WEATHER_CODES.get(current["weather_code"], "Unknown"),
    }


def append_row(row: dict) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    file_exists = CSV_PATH.exists()

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    for location_name, coords in LOCATIONS.items():
        payload = fetch_weather(coords["latitude"], coords["longitude"])
        row = build_row(location_name, payload)
        append_row(row)
        print(f"Logged {location_name}: {row['temperature_c']}°C, {row['conditions']}")


if __name__ == "__main__":
    main()
