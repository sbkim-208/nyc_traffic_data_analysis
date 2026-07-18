"""STGNN (graph + time series) based speed prediction — for performance comparison
against the existing temporal-only model (ml_feature_analysis.py). Each segment (id) is
treated as a graph node; neighbor-segment information is mixed in via the physical
adjacency matrix built by geo.py::build_adjacency (GCN), and the time axis is handled
by a GRU.

Node features include not just raw speed but the same calendar features (hour_sin/cos,
dow, is_weekend, shared across all nodes) and lag_1d/roll_1d_mean/std (per-node, computed
after shift(1) then a 24-hour rolling window — to prevent leaking the value into its own
rolling stats) as the ML pipeline (build_ml_feature_df, documented in README section 4).
5/10/30-minute lags aren't added separately since the input sequence (past 60 minutes)
already serves that role.

Matched to the same condition as the existing pipeline (past 60 min -> predict speed 60
min ahead) so it can be directly compared against naive/LR/RF/XGB.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn.dense import DenseGCNConv


def build_speed_matrix(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Speed wide matrix reindexed per id onto a regular time grid (freq) (index=timestamp, columns=id)

    Every node needs a value at each graph snapshot (a GCN input requirement), so all
    segments are aligned to a shared 5-minute grid spanning the full df's min~max. Since
    raw observations land at irregular ~1-minute intervals, reindexing straight onto the
    grid would leave almost every slot NaN — instead we first time-interpolate on the
    union of the original timestamps and the target grid, then pull out just the grid.
    Remaining gaps (edges outside a segment's observed range) are filled with ffill/bfill,
    the same as fill_missing_speed.
    """
    full_idx = pd.date_range(df["data_as_of"].min(), df["data_as_of"].max(), freq=freq)
    wide = {}
    for seg_id, g in df.groupby("id"):
        s = g.set_index("data_as_of")["speed"].sort_index()
        s = s[~s.index.duplicated(keep="first")]
        combined_idx = s.index.union(full_idx)
        s_dense = s.reindex(combined_idx).interpolate("time")
        s_grid = s_dense.reindex(full_idx)
        if s_grid.isna().any():
            s_grid = s_grid.ffill().bfill()
        wide[seg_id] = s_grid
    return pd.DataFrame(wide, index=full_idx)


FEATURE_NAMES = [
    "speed", "lag_1d", "roll_1d_mean", "roll_1d_std", "hour_sin", "hour_cos", "dow", "is_weekend",
    "lag_5min", "lag_10min", "lag_30min",
]
SPEED_CHANNEL = 0
N_BASE_FEATURES = 8  # base feature count excluding the short-lag channels (last 3) — toggled on/off via trailing slice


def build_node_feature_tensor(
    df: pd.DataFrame, freq: str = "5min", include_short_lags: bool = False
) -> tuple[np.ndarray, list, pd.DatetimeIndex]:
    """Add calendar + lag_1d/roll_1d (+ optional 5/10/30-min lag) features to the speed wide matrix, returned as a [T, N, F] tensor

    lag_1d, roll_1d_mean/std, and lag_5/10/30min differ per node (segment), while
    hour_sin/cos, dow, and is_weekend are broadcast identically to all nodes for a given
    timestamp. The first day's worth of rows becomes NaN due to lag_1d/rolling
    requirements, so we trim to the first fully-valid timestamp.
    include_short_lags=True appends lag_5min/10min/30min as extra channels (8->11) — off
    by default since the past-60-minute input sequence already carries this information
    (avoiding redundancy); this is an ablation-study option.
    Returns: (node_features [T, N, F], node_ids, timestamps) — F order matches FEATURE_NAMES
    """
    speed_wide = build_speed_matrix(df, freq=freq)
    steps_per_day = pd.Timedelta("1D") // pd.Timedelta(freq)
    step_min = pd.Timedelta(freq) // pd.Timedelta("1min")

    lag_1d = speed_wide.shift(steps_per_day)
    roll_1d_mean = speed_wide.shift(1).rolling(steps_per_day).mean()
    roll_1d_std = speed_wide.shift(1).rolling(steps_per_day).std()
    short_lags = {
        m: speed_wide.shift(m // step_min) for m in (5, 10, 30)
    } if include_short_lags else {}

    valid = lag_1d.notna().all(axis=1) & roll_1d_mean.notna().all(axis=1) & roll_1d_std.notna().all(axis=1)
    for s in short_lags.values():
        valid &= s.notna().all(axis=1)

    speed_wide, lag_1d, roll_1d_mean, roll_1d_std = (
        speed_wide[valid], lag_1d[valid], roll_1d_mean[valid], roll_1d_std[valid]
    )
    short_lags = {m: s[valid] for m, s in short_lags.items()}
    idx = speed_wide.index
    n_nodes = speed_wide.shape[1]

    hour = idx.hour + idx.minute / 60
    hour_sin = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    dow = idx.dayofweek.values.astype(np.float32)
    is_weekend = (idx.dayofweek >= 5).astype(np.float32)
    calendar = np.stack([hour_sin, hour_cos, dow, is_weekend], axis=1)  # [T, 4]
    calendar_b = np.repeat(calendar[:, None, :], n_nodes, axis=1)  # [T, N, 4]

    per_node = [speed_wide.values, lag_1d.values, roll_1d_mean.values, roll_1d_std.values]
    per_node = np.stack(per_node, axis=-1)  # [T, N, 4]

    channels = [per_node, calendar_b]
    if include_short_lags:
        channels.append(np.stack([short_lags[5].values, short_lags[10].values, short_lags[30].values], axis=-1))

    node_features = np.concatenate(channels, axis=-1).astype(np.float32)  # [T, N, 8] or [T, N, 11]
    return node_features, list(speed_wide.columns), idx


def make_windows(
    node_features: np.ndarray, seq_len: int = 12, horizon: int = 12, speed_channel: int = SPEED_CHANNEL
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows that predict speed `horizon` steps ahead from the past `seq_len` steps of features

    Default seq_len=12, horizon=12, freq=5min = predict speed 60 minutes ahead from the
    past 60 minutes of features (same condition as target_60min in ml_feature_analysis.py
    — for direct comparability).
    Returns: X [n_samples, seq_len, num_nodes, num_features], y [n_samples, num_nodes] (speed only), sample_idx
    """
    T = node_features.shape[0]
    X = np.stack([node_features[t - seq_len:t] for t in range(seq_len, T - horizon)])
    y = np.stack([node_features[t + horizon, :, speed_channel] for t in range(seq_len, T - horizon)])
    sample_idx = np.arange(seq_len, T - horizon)
    return X, y, sample_idx


def time_based_split_3way(
    X: np.ndarray, y: np.ndarray, sample_idx: np.ndarray,
    train_frac: float = 0.6, val_frac: float = 0.2,
):
    """Chronological 60/20/20 split by target time (prevents future->past leakage, no shuffling)"""
    n = len(sample_idx)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return (
        (X[:i_train], y[:i_train]),
        (X[i_train:i_val], y[i_train:i_val]),
        (X[i_val:], y[i_val:]),
    )


class STGNNForecaster(nn.Module):
    """Simple STGCN variant combining DenseGCNConv (spatial) + GRU (temporal)

    At every timestep, a graph convolution mixes in neighbor-segment information (plus
    the node's own calendar/lag_1d/rolling features) to form an embedding; the resulting
    embedding sequence is passed through a GRU per node as if it were an independent time
    series (weights shared across nodes), and the final hidden state predicts each node's
    normalized speed change (delta). Restoring the absolute value (last raw speed + delta)
    is handled in train_stgnn.
    """

    def __init__(self, in_channels: int, hidden: int = 32, num_gcn_layers: int = 1):
        super().__init__()
        self.gcn_layers = nn.ModuleList([
            DenseGCNConv(in_channels if i == 0 else hidden, hidden) for i in range(num_gcn_layers)
        ])
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, F], adj: [N, N]
        B, T, N, F = x.shape
        adj_b = adj.unsqueeze(0).expand(B, -1, -1)
        embeds = []
        for t in range(T):
            h = x[:, t]
            for gcn in self.gcn_layers:
                h = torch.relu(gcn(h, adj_b))  # [B, N, hidden] — 2+ layers extends to 2-hop neighbors
            embeds.append(h)
        seq = torch.stack(embeds, dim=1)  # [B, T, N, hidden]
        seq = seq.permute(0, 2, 1, 3).reshape(B * N, T, -1)  # [B*N, T, hidden]
        _, h_n = self.gru(seq)
        delta = self.out(h_n.squeeze(0)).view(B, N)  # [B, N] — normalized speed change
        return delta


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1 - ss_res / ss_tot),
    }


def train_stgnn(
    train: tuple, val: tuple, test: tuple, adj: np.ndarray,
    speed_channel: int = SPEED_CHANNEL,
    hidden: int = 32, epochs: int = 15, batch_size: int = 64, lr: float = 1e-3,
    num_gcn_layers: int = 1,
    device: str | None = None,
) -> dict:
    """Train STGNNForecaster and return train/val/test predictions as an evaluate_model-style dict

    Since features have different scales (speed in the 0-60s, dow 0-6, hour_sin/cos
    -1 to 1, etc.), each channel is standardized independently. The model only predicts
    the normalized speed change (delta); metrics are computed after restoring the
    absolute value in mph by adding delta*speed_std to the last raw speed (a scale
    directly comparable to naive/RF/XGB and the README results).
    """
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    X_train_raw, y_train_raw = train
    mean = X_train_raw.mean(axis=(0, 1, 2))  # [F]
    std = X_train_raw.std(axis=(0, 1, 2))
    std[std == 0] = 1.0
    speed_mean, speed_std = mean[speed_channel], std[speed_channel]

    def _norm_x(a):
        return torch.tensor((a - mean) / std, dtype=torch.float32)

    def _last_raw_speed(a):
        return torch.tensor(a[:, -1, :, speed_channel], dtype=torch.float32)

    def _target_delta_norm(y_raw, last_raw):
        return torch.tensor((y_raw - last_raw.numpy()) / speed_std, dtype=torch.float32)

    X_train, X_val, X_test = _norm_x(train[0]), _norm_x(val[0]), _norm_x(test[0])
    last_train, last_val, last_test = _last_raw_speed(train[0]), _last_raw_speed(val[0]), _last_raw_speed(test[0])
    y_train_delta = _target_delta_norm(train[1], last_train)
    y_val_raw, y_test_raw = val[1], test[1]
    adj_t = torch.tensor(adj, device=device)

    in_channels = X_train.shape[-1]
    model = STGNNForecaster(in_channels=in_channels, hidden=hidden, num_gcn_layers=num_gcn_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_train = len(X_train)
    history = []
    best_val_rmse = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train[idx].to(device)
            yb = y_train_delta[idx].to(device)
            optimizer.zero_grad()
            pred = model(xb, adj_t)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_delta = model(X_val.to(device), adj_t).cpu().numpy()
        val_pred = last_val.numpy() + val_delta * speed_std
        val_rmse = _metrics(y_val_raw, val_pred)["rmse"]
        history.append((epoch, epoch_loss, val_rmse))
        print(f"epoch {epoch+1}/{epochs}  train_mse={epoch_loss:.4f}  val_rmse(mph)={val_rmse:.4f}")

        # Only the best epoch by validation is used for the final test evaluation — so that
        # if the last epoch happened to drift toward overfitting, that isn't what gets reported as test performance
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_delta = model(X_test.to(device), adj_t).cpu().numpy()
        val_delta = model(X_val.to(device), adj_t).cpu().numpy()
    test_pred = last_test.numpy() + test_delta * speed_std
    val_pred = last_val.numpy() + val_delta * speed_std

    return {
        "model": model,
        "history": history,
        "mean": mean,
        "std": std,
        "metrics_val": _metrics(y_val_raw, val_pred),
        "metrics_test": _metrics(y_test_raw, test_pred),
    }
