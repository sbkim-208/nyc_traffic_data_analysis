import pandas as pd
from soobin.cleaning.preprocess import (
    drop_duplicate_ids,
    remove_dead_segments,
    fill_missing_speed,
    fill_missing_travel_time,
)
from soobin.analysis.ml_forecast import (
    build_ml_feature_df,
    _mask_stuck_runs,
    add_forecast_target,
    evaluate_naive_baseline,
)

DATA_PATH = "backend/data/raw/speeds_2024-04-01_2024-08-01.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    df = drop_duplicate_ids(df)
    df = remove_dead_segments(df)
    df = fill_missing_speed(df)
    df = fill_missing_travel_time(df)
    df["data_as_of"] = pd.to_datetime(df["data_as_of"])

    # Build lag columns (per segment/id, for ML/DL)
    # The original shift(1,2,3) only lagged by "N rows back" — since raw readings
    # land at irregular ~1-minute intervals per segment, that didn't guarantee an
    # exact number of minutes. Instead, build_ml_feature_df reindexes each segment's
    # timestamps onto a regular grid, then builds 5/10/30-min and 1/30-day lags.
    # Since all requested lags are multiples of 5 minutes, freq="5min" represents them
    # exactly while cutting row count to 1/5 of a 1-minute grid, keeping memory
    # manageable even running all 125 segments x 4 months.
    # day_lags=7 (1 week, matching the template's lag_168h) was also tested but
    # excluded — corr(lag_7d)=0.767 was actually lower than corr(lag_1d)=0.778
    # (i.e. "7 days ago, same day of week" is less similar to now than just
    # yesterday), meaning there's no extra signal from day-of-week alignment beyond
    # simple time-distance decay (1d < 7d < 30d). The dow/is_weekend calendar
    # features already encode day-of-week, so this would have been redundant.
    lag_df = build_ml_feature_df(
        df,
        value_col="speed",
        freq="5min",
        min_lags=(5, 10, 30),
        day_lags=(1, 30),
    )
    lag_cols = ["speed", "lag_5min", "lag_10min", "lag_30min", "lag_1d", "lag_30d"]
    print("\n=== After adding lag columns (head) ===")
    print(lag_df[["id", *lag_cols]].head(10).to_string())
    print(f"\nLag feature df shape: {lag_df.shape}")

    # Lag correlation analysis — how linearly similar the current speed is to each lag
    # (past value). Closer to 1 means "the value back then resembles now" = useful for prediction.
    print("\n=== Lag correlation (corr(speed, lag_X), raw) ===")
    print(lag_df[lag_cols].corr()["speed"].drop("speed").to_string())

    # Stuck-run diagnostic (segments where a dead sensor's interpolation fallback froze flat) —
    # a perfectly constant value has an artificially near-1 correlation with its own lag,
    # which can inflate the correlation above. Flag runs stuck for min_run_steps=12 * 5min
    # = 1 hour or more, exclude them, and recompute for comparison.
    stuck = lag_df.groupby("id")["speed"].transform(
        lambda s: _mask_stuck_runs(s, min_run_steps=12).isna()
    )
    print(f"\n=== Stuck-run diagnostic (values frozen for 1+ hour) ===")
    print(f"Stuck share of all rows: {stuck.mean():.4f} ({stuck.sum():,}/{len(lag_df):,})")
    print("\nTop 10 segments (id) by stuck share:")
    print(lag_df.assign(stuck=stuck).groupby("id")["stuck"].mean().sort_values(ascending=False).head(10).to_string())

    print("\n=== Lag correlation (stuck-run excluded) ===")
    print(lag_df.loc[~stuck, lag_cols].corr()["speed"].drop("speed").to_string())


if __name__ == "__main__":
    main()
