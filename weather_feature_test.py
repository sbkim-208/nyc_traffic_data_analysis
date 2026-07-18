"""Check whether adding weather features (temperature, precipitation, wind, humidity,
cloud cover) to XGBoost improves prediction performance — especially the fall
(September) test performance, which has been the hardest case so far.

Weather data: Open-Meteo historical weather API (free, no API key, reproducible)
- Coordinates: a representative Manhattan point (40.7128, -74.0060), timezone: America/New_York
- Time resolution: hourly (merged into the 5-minute traffic data via floor-hour)

Reuses the stratified season split established in season_investigation.py (so the three
seasons are evenly mixed across train/val/test) to keep methodology consistent.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from soobin.cleaning.preprocess import load_and_clean
from season_feature_test import build_training_frame_season, CACHE_DIR, DATA_PATHS
from season_investigation import stratified_season_split, season_label_of, XGB_KWARGS
from soobin.analysis.ml_forecast import _metrics

WEATHER_PATH = "backend/data/raw/weather_2024-04-01_2024-10-01.csv"

BASE_COLS = [
    "id", "month", "hour", "dow", "is_weekend", "hour_sin", "hour_cos", "speed",
    "lag_5min", "lag_10min", "lag_30min", "lag_1d", "lag_30d", "roll_1d_mean", "roll_1d_std",
]
WEATHER_COLS = [
    "temperature_2m", "precipitation", "rain", "windspeed_10m",
    "relative_humidity_2m", "cloudcover",
]


def attach_weather(full: pd.DataFrame) -> pd.DataFrame:
    weather = pd.read_csv(WEATHER_PATH)
    weather["time"] = pd.to_datetime(weather["time"])
    if weather["snowfall"].sum() == 0:
        weather = weather.drop(columns=["snowfall"])  # constant column — no snow Apr-Sep

    full = full.copy()
    full["hour_ts"] = full["timestamp"].dt.floor("h")
    merged = full.merge(weather, left_on="hour_ts", right_on="time", how="left")
    missing = merged["temperature_2m"].isna().sum()
    if missing:
        print(f"Rows with failed weather match: {missing} (dropped)")
    return merged.dropna(subset=["temperature_2m"]).drop(columns=["hour_ts", "time"])


def main():
    df = load_and_clean(DATA_PATHS, f"{CACHE_DIR}/cleaned_6mo.parquet")
    full = build_training_frame_season(df)
    season_cols = sorted(c for c in full.columns if c.startswith("season_"))

    full = attach_weather(full)
    print(f"Frame after weather merge: {full.shape}")
    print(full[WEATHER_COLS].describe().round(2).to_string())

    train, val, test = stratified_season_split(full, season_cols)
    label_test = season_label_of(test, season_cols)
    print(f"train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

    base_feat = BASE_COLS + season_cols
    weather_feat = base_feat + WEATHER_COLS

    model_base = XGBRegressor(**XGB_KWARGS)
    model_base.fit(train[base_feat], train["target_60min"])
    pred_base = model_base.predict(test[base_feat])

    model_weather = XGBRegressor(**XGB_KWARGS)
    model_weather.fit(train[weather_feat], train["target_60min"])
    pred_weather = model_weather.predict(test[weather_feat])

    print("\n=== Full test: base vs +weather ===")
    print("base   :", _metrics(test["target_60min"], pred_base))
    print("weather:", _metrics(test["target_60min"], pred_weather))

    print("\n=== Test RMSE by season: base vs +weather ===")
    rows = []
    for season in sorted(label_test.unique()):
        mask = (label_test == season).values
        m_base = _metrics(test.loc[mask, "target_60min"], pred_base[mask])
        m_weather = _metrics(test.loc[mask, "target_60min"], pred_weather[mask])
        rows.append({
            "season": season, "n": int(mask.sum()),
            "rmse_base": m_base["rmse"], "rmse_weather": m_weather["rmse"],
            "improvement_pct": 100 * (m_base["rmse"] - m_weather["rmse"]) / m_base["rmse"],
        })
    result_df = pd.DataFrame(rows)
    print(result_df.round(4).to_string(index=False))

    importance = pd.Series(model_weather.feature_importances_, index=weather_feat).sort_values(ascending=False)
    print("\n=== XGB feature importance (incl. +weather) ===")
    print(importance.to_string())

    print("\n=== Sum of importance for weather features only ===")
    print(f"{importance[WEATHER_COLS].sum():.4f} (share of total)")


if __name__ == "__main__":
    main()
