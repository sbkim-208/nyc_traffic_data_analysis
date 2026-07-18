"""Pure time-series Transformer based speed prediction — treats each segment (id) as an
independent sequence, with no graph/adjacency matrix. Since spatial information (the
graph) didn't help much on this data in the STGNN experiment ("sparse graph ->
over-smoothing"), this comparison group checks whether boosting temporal expressiveness
via attention instead of spatial structure can beat XGBoost/STGNN.

Reuses build_node_feature_tensor/make_windows from soobin.analysis.stgnn as-is, using
the same calendar (hour_sin/cos, dow, is_weekend) + lag_1d/roll_1d_mean/std + speed
features, but unrolls the [T, N, F] tensor into (sample, segment) units so each segment
gets an independent length-12 (=60 min) sequence. Segment identity (id) is added as a
learnable embedding, playing the same role as the "id" feature in the tree models.
"""

import numpy as np
import torch
import torch.nn as nn


class TransformerForecaster(nn.Module):
    """Transformer encoder (temporal) + segment identity embedding

    Encodes each segment's past seq_len-step feature sequence via self-attention, and
    predicts the normalized speed change (delta) from the last timestep's representation.
    Uses the same "last raw speed + delta" residual structure as STGNNForecaster (delta
    is restored to an absolute value in train_transformer).
    """

    def __init__(self, in_channels: int, num_nodes: int, seq_len: int, d_model: int = 32,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.id_embed = nn.Embedding(num_nodes, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, node_idx: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F], node_idx: [B]
        h = self.input_proj(x) + self.pos_embed[:, : x.shape[1]]
        h = h + self.id_embed(node_idx).unsqueeze(1)
        h = self.encoder(h)
        delta = self.out(h[:, -1]).squeeze(-1)  # [B] — normalized speed change
        return delta


class LSTMForecaster(nn.Module):
    """LSTM (temporal) + segment identity embedding — same interface as TransformerForecaster

    A control group for directly testing the conventional wisdom that "LSTM beats
    Transformer on small data," under identical conditions (features, segment identity
    embedding, residual structure).
    """

    def __init__(self, in_channels: int, num_nodes: int, seq_len: int, d_model: int = 32,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.id_embed = nn.Embedding(num_nodes, d_model)
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, node_idx: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.id_embed(node_idx).unsqueeze(1)
        out, _ = self.lstm(h)
        delta = self.out(out[:, -1]).squeeze(-1)
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


def flatten_by_node(X: np.ndarray, y: np.ndarray, speed_channel: int = 0):
    """Unroll a [S, T, N, F]/[S, N] tensor into (sample x segment) units, converting to independent per-segment sequences

    Returns: X_flat [S*N, T, F], y_flat [S*N], node_idx [S*N] (0~N-1), last_raw [S*N]
    """
    S, T, N, F = X.shape
    X_flat = X.transpose(0, 2, 1, 3).reshape(S * N, T, F)
    y_flat = y.transpose(0, 1).reshape(S * N)
    node_idx = np.tile(np.arange(N), S)
    last_raw = X_flat[:, -1, speed_channel]
    return X_flat, y_flat, node_idx, last_raw


def train_transformer(
    train: tuple, val: tuple, test: tuple, num_nodes: int,
    speed_channel: int = 0, d_model: int = 32, nhead: int = 4, num_layers: int = 2,
    epochs: int = 15, batch_size: int = 512, lr: float = 1e-3, device: str | None = None,
    model_type: str = "transformer",
) -> dict:
    """Train TransformerForecaster/LSTMForecaster and return val/test metrics (using the
    same per-channel standardization + delta-residual + best-val checkpoint approach as STGNNForecaster)

    model_type: "transformer" or "lstm"
    """
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

    X_train_flat, y_train_flat, node_train, last_train = flatten_by_node(train[0], train[1], speed_channel)
    X_val_flat, y_val_flat, node_val, last_val = flatten_by_node(val[0], val[1], speed_channel)
    X_test_flat, y_test_flat, node_test, last_test = flatten_by_node(test[0], test[1], speed_channel)

    mean = X_train_flat.mean(axis=(0, 1))
    std = X_train_flat.std(axis=(0, 1))
    std[std == 0] = 1.0
    speed_std = std[speed_channel]

    def _norm(a):
        return torch.tensor((a - mean) / std, dtype=torch.float32)

    X_train_t, X_val_t, X_test_t = _norm(X_train_flat), _norm(X_val_flat), _norm(X_test_flat)
    node_train_t = torch.tensor(node_train, dtype=torch.long)
    node_val_t = torch.tensor(node_val, dtype=torch.long)
    node_test_t = torch.tensor(node_test, dtype=torch.long)
    y_train_delta = torch.tensor((y_train_flat - last_train) / speed_std, dtype=torch.float32)

    seq_len = X_train_flat.shape[1]
    in_channels = X_train_flat.shape[2]
    if model_type == "lstm":
        model = LSTMForecaster(in_channels, num_nodes, seq_len, d_model, num_layers).to(device)
    else:
        model = TransformerForecaster(in_channels, num_nodes, seq_len, d_model, nhead, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_train = len(X_train_t)
    best_val_rmse = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train_t[idx].to(device)
            nb = node_train_t[idx].to(device)
            yb = y_train_delta[idx].to(device)
            optimizer.zero_grad()
            pred = model(xb, nb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train

        model.eval()
        with torch.no_grad():
            val_delta = model(X_val_t.to(device), node_val_t.to(device)).cpu().numpy()
        val_pred = last_val + val_delta * speed_std
        val_rmse = _metrics(y_val_flat, val_pred)["rmse"]
        print(f"epoch {epoch+1}/{epochs}  train_mse={epoch_loss:.4f}  val_rmse(mph)={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_delta = model(X_test_t.to(device), node_test_t.to(device)).cpu().numpy()
        val_delta = model(X_val_t.to(device), node_val_t.to(device)).cpu().numpy()
    test_pred = last_test + test_delta * speed_std
    val_pred = last_val + val_delta * speed_std

    return {
        "model": model,
        "metrics_val": _metrics(y_val_flat, val_pred),
        "metrics_test": _metrics(y_test_flat, test_pred),
    }
