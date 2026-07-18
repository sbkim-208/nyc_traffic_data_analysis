"""Two follow-up investigations into the season_feature_test.py results:

1) Diagnose why the 6-month result was worse than the 4-month one. The existing
   time_split_3way just cuts the timeline into the first 60%/20%/20%, so with 6 months
   of data the entire test window ends up being September — a month the model never saw
   during training. We separate whether the degradation comes from "more data made it
   worse" versus "predicting an unseen month (September) is inherently harder."

2) Whether prediction performance (RMSE) actually differs by season (spring/summer/fall).
   The season one-hot feature itself had zero importance, but that's a separate question
   from "does per-season prediction difficulty differ." We re-split so each season is
   evenly represented in train/val/test, then compute test RMSE per season separately.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from soobin.cleaning.preprocess import load_and_clean
from season_feature_test import build_training_frame_season, RAW_DIR, CACHE_DIR, DATA_PATHS
from soobin.analysis.ml_forecast import _metrics

XGB_KWARGS = dict(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)


def season_label_of(df: pd.DataFrame, season_cols: list) -> pd.Series:
    return df[season_cols].idxmax(axis=1).str.replace("season_", "")


def stratified_season_split(full: pd.DataFrame, season_cols: list, train_frac=0.6, val_frac=0.2):
    """Split each season into its own 60/20/20 and concatenate — this makes train/val/test
    each contain a similar mix of the three seasons, removing the bias where one season
    ends up concentrated only in the test set."""
    label = season_label_of(full, season_cols)
    trains, vals, tests = [], [], []
    for season in label.unique():
        sub = full[label == season]
        times = np.sort(sub["timestamp"].unique())
        train_cut = times[int(len(times) * train_frac)]
        val_cut = times[int(len(times) * (train_frac + val_frac))]
        trains.append(sub[sub["timestamp"] <= train_cut])
        vals.append(sub[(sub["timestamp"] > train_cut) & (sub["timestamp"] <= val_cut)])
        tests.append(sub[sub["timestamp"] > val_cut])
    return pd.concat(trains), pd.concat(vals), pd.concat(tests)


def main():
    df = load_and_clean(DATA_PATHS, f"{CACHE_DIR}/cleaned_6mo.parquet")
    full = build_training_frame_season(df)
    season_cols = sorted(c for c in full.columns if c.startswith("season_"))
    base_cols = [
        "id", "month", "hour", "dow", "is_weekend", "hour_sin", "hour_cos", "speed",
        "lag_5min", "lag_10min", "lag_30min", "lag_1d", "lag_30d", "roll_1d_mean", "roll_1d_std",
    ]
    feat_cols = base_cols + season_cols

    # ===== 1) Re-evaluate with the season-stratified split =====
    train, val, test = stratified_season_split(full, season_cols)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {len(part):,} rows, {part['timestamp'].min()} ~ {part['timestamp'].max()}")

    model = XGBRegressor(**XGB_KWARGS)
    model.fit(train[feat_cols], train["target_60min"])
    pred = model.predict(test[feat_cols])
    print("\n=== Full test with season-stratified split ===")
    print(_metrics(test["target_60min"], pred))

    print("\n=== Test RMSE breakdown by season ===")
    label_test = season_label_of(test, season_cols)
    for season in sorted(label_test.unique()):
        mask = (label_test == season).values
        m = _metrics(test.loc[mask, "target_60min"], pred[mask])
        print(f"{season:8s} n={mask.sum():>7,}  {m}")

    # ===== 2) Regime-shift diagnostic: does September exposure matter =====
    sept_mask = full["month"] == 9
    non_sept = full[~sept_mask]
    sept = full[sept_mask]
    sept_times = np.sort(sept["timestamp"].unique())
    half_cut = sept_times[len(sept_times) // 2]
    sept_early = sept[sept["timestamp"] <= half_cut]
    sept_late = sept[sept["timestamp"] > half_cut]

    model_no_sept = XGBRegressor(**XGB_KWARGS)
    model_no_sept.fit(non_sept[feat_cols], non_sept["target_60min"])

    model_with_sept = XGBRegressor(**XGB_KWARGS)
    train_with_sept = pd.concat([non_sept, sept_early])
    model_with_sept.fit(train_with_sept[feat_cols], train_with_sept["target_60min"])

    print("\n=== September exposure comparison (same eval window: late September) ===")
    pred_a = model_no_sept.predict(sept_late[feat_cols])
    print("Trained on Apr-Aug only (no September exposure) ->", _metrics(sept_late["target_60min"], pred_a))
    pred_b = model_with_sept.predict(sept_late[feat_cols])
    print("Trained on Apr-Aug + early September (partial September exposure) ->", _metrics(sept_late["target_60min"], pred_b))

    print("\n=== For reference: trained on Apr-Aug only -> evaluated on all of September (early+late) ===")
    pred_full_sept = model_no_sept.predict(sept[feat_cols])
    print(_metrics(sept["target_60min"], pred_full_sept))


if __name__ == "__main__":
    main()
