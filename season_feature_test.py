"""Extend the data from 4 months (Apr-Jul) to 6 months (Apr-Sep) and check whether
grouping month into spring/summer/fall/winter season features affects XGBoost/RF
performance and feature importance.

Note: since the data starts in April, even 6 months only covers Apr-Sep (2 months of
spring, 3 of summer, 1 of fall) — winter (Dec-Feb) is never included. So only 3 season
categories can be compared, and even then fall (a single month, September) is a small,
imbalanced sample that needs to be interpreted with that caveat.

Feature construction reuses the same calendar+lag+rolling spec as build_training_frame
in spatial_feature_test.py, but adds season one-hot columns instead of neighbor_avg_speed.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from soobin.cleaning.preprocess import load_and_clean
from soobin.analysis.stgnn import build_speed_matrix
from soobin.analysis.ml_forecast import mask_stuck, time_split_3way, _metrics, evaluate

FREQ = "5min"
STEPS_PER_DAY = 288
RAW_DIR = "backend/data/raw"
CACHE_DIR = "backend/data/analysis_cache"
DATA_PATHS = [
    f"{RAW_DIR}/speeds_2024-04-01_2024-08-01.csv",
    f"{RAW_DIR}/speeds_2024-08-01_2024-10-01.csv",
]

_SEASON_MAP = {
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall",
    12: "Winter", 1: "Winter", 2: "Winter",
}


def build_training_frame_season(df: pd.DataFrame) -> pd.DataFrame:
    wide = build_speed_matrix(df, freq=FREQ)
    node_ids = list(wide.columns)

    lag_5min = wide.shift(1)
    lag_10min = wide.shift(2)
    lag_30min = wide.shift(6)
    lag_1d = wide.shift(STEPS_PER_DAY)
    lag_30d = wide.shift(30 * STEPS_PER_DAY)
    roll_1d_mean = wide.shift(1).rolling(STEPS_PER_DAY).mean()
    roll_1d_std = wide.shift(1).rolling(STEPS_PER_DAY).std()
    target_60min = wide.shift(-12)
    stuck = mask_stuck(wide)

    idx = wide.index
    hour_sin = np.sin(2 * np.pi * (idx.hour + idx.minute / 60) / 24)
    hour_cos = np.cos(2 * np.pi * (idx.hour + idx.minute / 60) / 24)
    season = idx.month.map(_SEASON_MAP)

    frames = []
    for node_id in node_ids:
        block = pd.DataFrame({
            "timestamp": idx,
            "id": node_id,
            "month": idx.month, "hour": idx.hour, "dow": idx.dayofweek,
            "is_weekend": (idx.dayofweek >= 5).astype(int),
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "season": season,
            "speed": wide[node_id].values,
            "lag_5min": lag_5min[node_id].values,
            "lag_10min": lag_10min[node_id].values,
            "lag_30min": lag_30min[node_id].values,
            "lag_1d": lag_1d[node_id].values,
            "lag_30d": lag_30d[node_id].values,
            "roll_1d_mean": roll_1d_mean[node_id].values,
            "roll_1d_std": roll_1d_std[node_id].values,
            "target_60min": target_60min[node_id].values,
            "stuck": stuck[node_id].values,
        })
        frames.append(block)

    full = pd.concat(frames, ignore_index=True)
    full = full[~full["stuck"]].drop(columns="stuck")
    full = pd.get_dummies(full, columns=["season"], prefix="season")
    return full.dropna().reset_index(drop=True)


def main():
    df = load_and_clean(DATA_PATHS, f"{CACHE_DIR}/cleaned_6mo.parquet")
    print(f"Cleaned data: {df.shape}, range: {df['data_as_of'].min()} ~ {df['data_as_of'].max()}")

    full = build_training_frame_season(df)
    season_cols = sorted(c for c in full.columns if c.startswith("season_"))
    print(f"Training frame: {full.shape}, season categories: {season_cols}")
    print(full[season_cols].sum().to_string())

    train, val, test = time_split_3way(full)
    print(f"train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

    base_cols = [
        "id", "month", "hour", "dow", "is_weekend", "hour_sin", "hour_cos", "speed",
        "lag_5min", "lag_10min", "lag_30min", "lag_1d", "lag_30d", "roll_1d_mean", "roll_1d_std",
    ]
    season_feat_cols = base_cols + season_cols

    rf_kwargs = dict(n_estimators=50, max_depth=12, min_samples_leaf=50, random_state=42, n_jobs=-1)
    xgb_kwargs = dict(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)

    results = {}
    results["RF (base 15, incl. month)"] = evaluate(RandomForestRegressor(**rf_kwargs), train, test, base_cols)
    results["RF (+season)"] = evaluate(RandomForestRegressor(**rf_kwargs), train, test, season_feat_cols)
    results["XGB (base 15, incl. month)"] = evaluate(XGBRegressor(**xgb_kwargs), train, test, base_cols)
    results["XGB (+season)"] = evaluate(XGBRegressor(**xgb_kwargs), train, test, season_feat_cols)

    print("\n=== base (incl. month) vs +season one-hot (test, horizon=60min, 6-month data) ===")
    print(pd.DataFrame(results).T.round(4).to_string())

    xgb_season = XGBRegressor(**xgb_kwargs)
    xgb_season.fit(train[season_feat_cols], train["target_60min"])
    importance = pd.Series(xgb_season.feature_importances_, index=season_feat_cols).sort_values(ascending=False)
    print("\n=== XGB feature importance (incl. +season) ===")
    print(importance.to_string())


if __name__ == "__main__":
    main()
