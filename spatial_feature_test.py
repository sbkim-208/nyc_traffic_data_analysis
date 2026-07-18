"""A/B test checking whether adding neighboring-segment average speed
(neighbor_avg_speed) to XGBoost/RF improves prediction performance. STGNN (graph
neural network, README section 5) gained nothing due to the sparse graph, so instead
we hand-craft spatial information into a single feature and add it on top of the
tree models that already perform well.

Training frame construction (build_training_frame, etc.) lives in
`soobin/analysis/ml_forecast.py` — the actual implementation of the spec documented
in README sections 4/5 (calendar hour_sin/cos, dow, is_weekend, lag_5/10/30min,
lag_1d/30d, roll_1d_mean/std, neighbor_avg_speed, horizon=60min).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from soobin.cleaning.preprocess import load_and_clean
from soobin.analysis.geo import build_segment_geodataframe, build_adjacency
from soobin.analysis.stgnn import build_speed_matrix
from soobin.analysis.ml_forecast import build_training_frame, time_split_3way, evaluate

RAW_DIR = "backend/data/raw"
CACHE_DIR = "backend/data/analysis_cache"
FREQ = "5min"


def main():
    df = load_and_clean(
        [f"{RAW_DIR}/speeds_2024-04-01_2024-08-01.csv"],
        f"{CACHE_DIR}/cleaned_4mo.parquet",
    )

    gdf = build_segment_geodataframe(df)
    adj_full, adj_ids = build_adjacency(gdf, threshold_m=60.0)

    speed_ids = list(build_speed_matrix(df, freq=FREQ).columns)
    node_ids = [i for i in adj_ids if i in speed_ids]
    pos = {seg_id: p for p, seg_id in enumerate(adj_ids)}
    keep = [pos[i] for i in node_ids]
    adj = adj_full[np.ix_(keep, keep)]
    print(f"Node count: {len(node_ids)}, edge count: {int(adj.sum()/2)}, isolated nodes: {int((adj.sum(axis=1)==0).sum())}")

    full = build_training_frame(df, adj, node_ids)
    print(f"Training frame: {full.shape}")

    train, val, test = time_split_3way(full)
    print(f"train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

    base_cols = [
        "id", "month", "hour", "dow", "is_weekend", "hour_sin", "hour_cos", "speed",
        "lag_5min", "lag_10min", "lag_30min", "lag_1d", "lag_30d", "roll_1d_mean", "roll_1d_std",
    ]
    spatial_cols = base_cols + ["neighbor_avg_speed"]

    rf_kwargs = dict(n_estimators=50, max_depth=12, min_samples_leaf=50, random_state=42, n_jobs=-1)
    xgb_kwargs = dict(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)

    results = {}
    results["RF (base 15)"] = evaluate(RandomForestRegressor(**rf_kwargs), train, test, base_cols)
    results["RF (+neighbor_avg)"] = evaluate(RandomForestRegressor(**rf_kwargs), train, test, spatial_cols)
    results["XGB (base 15)"] = evaluate(XGBRegressor(**xgb_kwargs), train, test, base_cols)
    results["XGB (+neighbor_avg)"] = evaluate(XGBRegressor(**xgb_kwargs), train, test, spatial_cols)

    print("\n=== base 15 features vs +neighbor_avg_speed (test, horizon=60min) ===")
    print(pd.DataFrame(results).T.round(4).to_string())

    # Check neighbor_avg_speed feature importance (XGB, gain-based) — if it's near zero
    # despite being added, the model effectively ignored it
    xgb_spatial = XGBRegressor(**xgb_kwargs)
    xgb_spatial.fit(train[spatial_cols], train["target_60min"])
    importance = pd.Series(xgb_spatial.feature_importances_, index=spatial_cols).sort_values(ascending=False)
    print("\n=== XGB feature importance (incl. +neighbor_avg) ===")
    print(importance.to_string())


if __name__ == "__main__":
    main()
