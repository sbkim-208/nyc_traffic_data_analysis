"""Year-crossing validation: train on all of 2024 (Jan-Dec) and predict Jan-Mar 2025,
to check whether the model holds up on a completely unseen year.

Uses all of 2024 as train (no separate val split — reuses the already-settled XGBoost
config rather than re-tuning), test=Jan-Mar 2025. Compared against the earlier 6-month
(Apr-Sep) results to see whether performance holds up across a much longer time gap.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from soobin.cleaning.preprocess import load_and_clean
from soobin.analysis.stgnn import build_speed_matrix
from soobin.analysis.ml_forecast import mask_stuck, _metrics

RAW_DIR = "backend/data/raw"
CACHE_DIR = "backend/data/analysis_cache"
FREQ = "5min"
STEPS_PER_DAY = 288

DATA_PATHS = [
    f"{RAW_DIR}/speeds_2024-01-01_2024-04-01.csv",
    f"{RAW_DIR}/speeds_2024-04-01_2024-08-01.csv",
    f"{RAW_DIR}/speeds_2024-08-01_2024-10-01.csv",
    f"{RAW_DIR}/speeds_2024-10-01_2024-12-01.csv",
    f"{RAW_DIR}/speeds_2024-12-01_2025-02-01.csv",
    f"{RAW_DIR}/speeds_2025-02-01_2025-04-01.csv",
]

BASE_COLS = [
    "id", "month", "hour", "dow", "is_weekend", "hour_sin", "hour_cos", "speed",
    "lag_5min", "lag_10min", "lag_30min", "lag_1d", "lag_30d", "roll_1d_mean", "roll_1d_std",
]

XGB_KWARGS = dict(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
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

    frames = []
    for node_id in node_ids:
        block = pd.DataFrame({
            "timestamp": idx,
            "id": node_id,
            "month": idx.month, "hour": idx.hour, "dow": idx.dayofweek,
            "is_weekend": (idx.dayofweek >= 5).astype(int),
            "hour_sin": hour_sin, "hour_cos": hour_cos,
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
    return full.dropna().reset_index(drop=True)


def main():
    df = load_and_clean(DATA_PATHS, f"{CACHE_DIR}/cleaned_full_2024_2025q1.parquet")
    print(f"Cleaned data: {df.shape}, range: {df['data_as_of'].min()} ~ {df['data_as_of'].max()}")

    full = build_training_frame(df)
    print(f"Training frame: {full.shape}")

    train = full[full["timestamp"] < "2025-01-01"]
    test = full[full["timestamp"] >= "2025-01-01"]
    print(f"train (all of 2024): {len(train):,} rows ({train['timestamp'].min()} ~ {train['timestamp'].max()})")
    print(f"test (2025): {len(test):,} rows ({test['timestamp'].min()} ~ {test['timestamp'].max()})")

    model = XGBRegressor(**XGB_KWARGS)
    model.fit(train[BASE_COLS], train["target_60min"])
    pred = model.predict(test[BASE_COLS])

    print("\n=== 2025 prediction performance (train=all of 2024) ===")
    print(_metrics(test["target_60min"], pred))

    print("\n=== Monthly breakdown (2025) ===")
    test = test.copy()
    test["pred"] = pred
    for month, g in test.groupby("month"):
        print(f"month={month}  n={len(g):>8,}  {_metrics(g['target_60min'], g['pred'])}")

    print("\n=== For reference: 6-month (Apr-Sep) experiment test RMSE = 7.841 (stratified split) ===")


if __name__ == "__main__":
    main()
