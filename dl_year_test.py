"""Re-validate LSTM vs Transformer under a year-crossing condition (train on all of
2024, predict Jan-Mar 2025).

On the 4-month dataset, the Transformer overfit so badly it did worse than a naive
baseline (consistent with the conventional wisdom that "Transformers need more data").
This checks whether that reproduces now that the data is ~4-5x larger (15 months,
tens of millions of rows once unrolled per node without a graph), and directly compares
whether LSTM does better under the same condition. XGBoost's year-crossing result
(RMSE 7.945, R2 0.769, from year_over_year_test.py) is the comparison baseline.
"""

import numpy as np
import pandas as pd

from soobin.cleaning.preprocess import load_and_clean
from soobin.analysis.stgnn import build_node_feature_tensor, make_windows
from soobin.analysis.transformer_ts import train_transformer
from year_over_year_test import DATA_PATHS, CACHE_DIR


def main():
    df = load_and_clean(DATA_PATHS, f"{CACHE_DIR}/cleaned_full_2024_2025q1.parquet")
    print(f"Cleaned data: {df.shape}, range: {df['data_as_of'].min()} ~ {df['data_as_of'].max()}")

    node_features, node_ids, timestamps = build_node_feature_tensor(df, freq="5min")
    print(f"Node feature tensor: {node_features.shape} (timestamps x segments x features)")

    X, y, sample_idx = make_windows(node_features, seq_len=12, horizon=12)
    # make_windows: y[i] = node_features[sample_idx[i] + horizon], so the target time uses the same offset
    target_times = timestamps[sample_idx + 12]
    print(f"Window sample count: {X.shape[0]}  ({X.shape[2]} segments x samples = {X.shape[0]*X.shape[2]:,} rows unrolled)")

    train_mask = target_times < pd.Timestamp("2025-01-01")
    test_mask = ~train_mask
    train = (X[train_mask], y[train_mask])
    test = (X[test_mask], y[test_mask])
    # val: carve off the last 10% of train for best-checkpoint selection
    n_train = len(train[0])
    val_cut = int(n_train * 0.9)
    val = (train[0][val_cut:], train[1][val_cut:])
    train = (train[0][:val_cut], train[1][:val_cut])

    print(f"train: {train[0].shape[0]:,}  val: {val[0].shape[0]:,}  test: {test[0].shape[0]:,}  (window sample counts)")

    results = {}
    for model_type in ["lstm", "transformer"]:
        print(f"\n{'='*20} {model_type.upper()} {'='*20}")
        result = train_transformer(
            train, val, test, num_nodes=X.shape[2],
            d_model=32, nhead=4, num_layers=2, epochs=15, batch_size=1024,
            model_type=model_type,
        )
        print(f"{model_type} test:", result["metrics_test"])
        results[model_type] = result["metrics_test"]

    print("\n=== Final comparison (test=Jan-Mar 2025) ===")
    comparison = pd.DataFrame({
        "naive (persistence)": {"mae": 5.994, "rmse": 9.881, "r2": 0.658},
        "LSTM (year-crossing)": results["lstm"],
        "Transformer (year-crossing)": results["transformer"],
        "XGBoost (year-crossing)": {"mae": 5.119, "rmse": 7.945, "r2": 0.769},
    }).T
    print(comparison.round(3).sort_values("rmse").to_string())


if __name__ == "__main__":
    main()
