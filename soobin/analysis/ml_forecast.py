"""Granger-significant leader road -> follower road SI forecast model

Only uses (leader, follower) pairs that lag_correlation.get_granger_causality() judged
significant (significant=True) as features. Each leader is shifted back in time by its
own causal lag, aligned to predict the follower road's "current" SI — pairs with
lag_min=0 are excluded since they're only a contemporaneous correlation, not leading
information usable for prediction.

With only 5 follower roads, per-road individual models would be sample-starved, so all
follower roads are pooled into a single model, with follower_id passed in as a
categorical feature so the model can absorb road-to-road differences.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .lag_correlation import _build_road_series, _LEADER_PREFIXES, _FOLLOWER_PREFIXES
from .stgnn import build_speed_matrix



_AR_LAGS_MIN = [5, 10, 15]  # the follower road's own past values (autoregressive features)


def get_significant_pairs(granger_df: pd.DataFrame) -> pd.DataFrame:
    """Return only pairs that passed the Granger test and have lag > 0 (i.e. carry real leading information)"""
    return granger_df[
        granger_df["significant"] & (granger_df["best_lag_min"] > 0)
    ].copy()


def _to_regular_5min(series: pd.Series) -> pd.Series:
    """Reindex an irregular timestamp index onto a regular 5-minute-interval index

    For shift(steps) to match an actual time difference in minutes, the index needs to
    run at unbroken 5-minute intervals. Only gaps of 30 minutes or less are interpolated;
    longer gaps are left as NaN so they're naturally filtered out by a later dropna.
    """
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="5min")
    return series.reindex(full_idx).interpolate("time", limit=6)


def build_feature_matrix(
    resampled: pd.DataFrame,
    granger_df: pd.DataFrame,
    value_col: str = "si",
) -> pd.DataFrame:
    """Build the feature matrix for pooled training

    Return columns:
      timestamp, follower_id, target,
      lead__<leader>  (per Granger-significant leader road, its value at its own best_lag back)
      ar_lag<N>       (the follower road's own value N minutes ago)
      hour, dow, is_weekend
    A follower road with no significant leader at all is excluded.
    """
    sig = get_significant_pairs(granger_df)
    leader_ts = _build_road_series(resampled, _LEADER_PREFIXES, value_col).pipe(
        lambda d: d.apply(_to_regular_5min)
    )
    follower_ts = _build_road_series(resampled, _FOLLOWER_PREFIXES, value_col).pipe(
        lambda d: d.apply(_to_regular_5min)
    )

    rows = []
    for follower in follower_ts.columns:
        pairs = sig[sig["follower"] == follower]
        if pairs.empty:
            continue

        block = pd.DataFrame(index=follower_ts.index)
        block["target"] = follower_ts[follower]

        for _, pair in pairs.iterrows():
            leader = pair["leader"]
            steps = int(pair["best_lag_min"]) // 5
            block[f"lead__{leader}"] = leader_ts[leader].shift(steps)

        for lag_min in _AR_LAGS_MIN:
            block[f"ar_lag{lag_min}"] = follower_ts[follower].shift(lag_min // 5)

        block["follower_id"] = follower
        block["hour"] = block.index.hour
        block["dow"] = block.index.dayofweek
        block["is_weekend"] = (block["dow"] >= 5).astype(int)
        block = block.reset_index(names="timestamp")
        rows.append(block)

    matrix = pd.concat(rows, ignore_index=True)
    matrix["follower_id"] = matrix["follower_id"].astype("category")
    return matrix.dropna().reset_index(drop=True)


def time_based_split(matrix: pd.DataFrame, test_frac: float = 0.2):
    """Use the last test_frac by timestamp as the test set (prevents future -> past leakage)"""
    cutoff = matrix["timestamp"].quantile(1 - test_frac)
    train = matrix[matrix["timestamp"] <= cutoff]
    test = matrix[matrix["timestamp"] > cutoff]
    return train, test
    


def train_si_forecast_model(matrix: pd.DataFrame, test_frac: float = 0.2):
    """Train and evaluate the pooled SI forecast model with HistGradientBoostingRegressor

    Compares against a baseline (persistence: uses ar_lag5, the value 5 minutes ago,
    as-is) to check whether the model gets a real improvement from the causal lag features.

    Returns: dict(model, feature_cols, metrics, feature_importance, test_df)
    """
    feature_cols = [c for c in matrix.columns if c not in ("timestamp", "target")]
    train, test = time_based_split(matrix, test_frac)

    model = HistGradientBoostingRegressor(
        categorical_features=["follower_id"],
        random_state=42,
    )
    model.fit(train[feature_cols], train["target"])
    pred = model.predict(test[feature_cols])

    baseline_pred = test["ar_lag5"]  # use the value 5 minutes ago as-is for the prediction

    def _metrics(y_true, y_pred):
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
            "r2": r2_score(y_true, y_pred),
        }

    metrics = {
        "model": _metrics(test["target"], pred),
        "baseline_persistence": _metrics(test["target"], baseline_pred),
    }

    # HistGradientBoostingRegressor doesn't provide feature_importances_,
    # so we use permutation importance instead
    from sklearn.inspection import permutation_importance

    perm = permutation_importance(
        model, test[feature_cols], test["target"], n_repeats=5, random_state=42
    )
    importance = pd.Series(perm.importances_mean, index=feature_cols)
    feature_importance = importance.sort_values(ascending=False).reset_index()
    feature_importance.columns = ["feature", "importance"]

    test_df = test[["timestamp", "follower_id", "target"]].copy()
    test_df["pred"] = pred
    test_df["pred_baseline"] = baseline_pred.values

    return {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "test_df": test_df,
    }


# ============================================================
# Temporal (+lightweight-spatial) ML pipeline (README sections 4/5)
#
# Training-frame construction functions for a pooled model that directly predicts
# per-segment speed(t+60min) from calendar+lag+rolling features. Also includes
# neighbor_avg_speed (the current average speed of physically adjacent segments) as a
# lightweight spatial feature used instead of STGNN (graph neural network).
# ============================================================


def mask_stuck(wide: pd.DataFrame, min_run_steps: int = 12) -> pd.DataFrame:
    """Flag points where a column's value stayed unchanged for min_run_steps or more consecutive steps as True

    A value frozen for 1 hour (=12*5min, at freq="5min") or more is treated as a
    stuck-run suspected of being a dead sensor.
    """
    result = {}
    for col in wide.columns:
        same_as_prev = wide[col].diff().eq(0)
        run_id = (~same_as_prev).cumsum()
        run_len = same_as_prev.groupby(run_id.values).cumsum()
        result[col] = run_len >= (min_run_steps - 1)
    return pd.DataFrame(result, index=wide.index)


def build_training_frame(
    df: pd.DataFrame, adj: np.ndarray, node_ids: list, freq: str = "5min",
) -> pd.DataFrame:
    """Build a training frame with calendar+lag+rolling+neighbor_avg_speed (spatial)+target_60min

    Stacks long-format rows per segment (id). neighbor_avg_speed is the current average
    speed of neighboring segments, normalized by adj (the adjacency matrix) — isolated
    segments with no neighbors fall back to their own speed. target_60min is the speed
    60 minutes ahead (a genuine future value, preventing nowcasting). stuck-runs
    (segments suspected of being dead sensors) are excluded here.
    """
    wide = build_speed_matrix(df, freq=freq)[node_ids]
    steps_per_day = pd.Timedelta("1D") // pd.Timedelta(freq)

    lag_5min = wide.shift(1)
    lag_10min = wide.shift(2)
    lag_30min = wide.shift(6)
    lag_1d = wide.shift(steps_per_day)
    lag_30d = wide.shift(30 * steps_per_day)
    roll_1d_mean = wide.shift(1).rolling(steps_per_day).mean()
    roll_1d_std = wide.shift(1).rolling(steps_per_day).std()
    target_60min = wide.shift(-12)

    row_sum = adj.sum(axis=1, keepdims=True)
    adj_norm = np.divide(adj, row_sum, out=np.zeros_like(adj), where=row_sum > 0)
    neighbor_avg = pd.DataFrame(wide.values @ adj_norm.T, index=wide.index, columns=wide.columns)
    isolated_ids = [nid for nid, s in zip(node_ids, row_sum.flatten()) if s == 0]
    if isolated_ids:
        # For segments with no neighbors, substitute their own current speed (a neutral value) for the "no information" state
        neighbor_avg[isolated_ids] = wide[isolated_ids]

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
            "neighbor_avg_speed": neighbor_avg[node_id].values,
            "target_60min": target_60min[node_id].values,
            "stuck": stuck[node_id].values,
        })
        frames.append(block)

    full = pd.concat(frames, ignore_index=True)
    full = full[~full["stuck"]].drop(columns="stuck")
    return full.dropna().reset_index(drop=True)


def time_split_3way(full: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2):
    """Chronological 60/20/20 split by timestamp (prevents future->past leakage, no shuffling)"""
    times = np.sort(full["timestamp"].unique())
    train_cut = times[int(len(times) * train_frac)]
    val_cut = times[int(len(times) * (train_frac + val_frac))]
    train = full[full["timestamp"] <= train_cut]
    val = full[(full["timestamp"] > train_cut) & (full["timestamp"] <= val_cut)]
    test = full[full["timestamp"] > val_cut]
    return train, val, test


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def evaluate(model, train, test, feature_cols):
    """Fit model on train and return its target_60min prediction performance on test (mae/rmse/r2)"""
    model.fit(train[feature_cols], train["target_60min"])
    pred = model.predict(test[feature_cols])
    return _metrics(test["target_60min"], pred)
