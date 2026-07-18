import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_feature_importance(feature_importance: pd.DataFrame):
    """Horizontal bar chart of permutation importance

    feature_importance: train_si_forecast_model()["feature_importance"]
    """
    df = feature_importance.sort_values("importance")
    colors = [
        "#377eb8" if f.startswith("lead__") else
        "#4daf4a" if f.startswith("ar_lag") else
        "#999999"
        for f in df["feature"]
    ]
    fig = go.Figure(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=(
            "SI Forecast Model — Permutation Feature Importance<br>"
            "<sup>blue = Granger-significant leader-road lag features | green = follower-road autoregressive features</sup>"
        ),
        xaxis_title="importance (MAE increase)",
        height=max(400, len(df) * 32 + 160),
        width=900,
    )
    fig.show()


def plot_predicted_vs_actual(test_df: pd.DataFrame):
    """Predicted vs actual SI time series per follower road + model vs persistence baseline comparison

    test_df: train_si_forecast_model()["test_df"]
    columns: timestamp, follower_id, target, pred, pred_baseline
    """
    followers = sorted(test_df["follower_id"].unique())
    fig = make_subplots(
        rows=len(followers), cols=1,
        subplot_titles=followers,
        shared_xaxes=False,
        vertical_spacing=0.06,
    )
    for i, follower in enumerate(followers, start=1):
        sub = test_df[test_df["follower_id"] == follower].sort_values("timestamp")
        fig.add_trace(
            go.Scatter(x=sub["timestamp"], y=sub["target"], mode="lines",
                       name="Actual", line=dict(color="#333333", width=1.5),
                       showlegend=(i == 1)),
            row=i, col=1,
        )
        fig.add_trace(
            go.Scatter(x=sub["timestamp"], y=sub["pred"], mode="lines",
                       name="Model prediction", line=dict(color="#e41a1c", width=1.2),
                       showlegend=(i == 1)),
            row=i, col=1,
        )
        fig.add_trace(
            go.Scatter(x=sub["timestamp"], y=sub["pred_baseline"], mode="lines",
                       name="Persistence baseline", line=dict(color="#377eb8", width=1, dash="dot"),
                       showlegend=(i == 1)),
            row=i, col=1,
        )
        fig.update_yaxes(title_text="SI", row=i, col=1)

    fig.update_layout(
        title="SI Forecast by Follower Road — Model vs Persistence Baseline (test period)",
        height=280 * len(followers),
        width=1200,
        legend=dict(orientation="h", y=-0.03),
    )
    fig.show()


def plot_metric_comparison(metrics: dict):
    """Model vs baseline MAE/RMSE/R2 bar chart

    metrics: train_si_forecast_model()["metrics"]
    """
    rows = []
    for name, m in metrics.items():
        rows.append({"model": name, **m})
    df = pd.DataFrame(rows)

    fig = make_subplots(rows=1, cols=3, subplot_titles=["MAE", "RMSE", "R2"])
    colors = {"model": "#e41a1c", "baseline_persistence": "#377eb8"}
    for col_i, metric in enumerate(["mae", "rmse", "r2"], start=1):
        fig.add_trace(
            go.Bar(
                x=df["model"], y=df[metric],
                marker_color=[colors.get(m, "#999999") for m in df["model"]],
                text=df[metric].round(4),
                textposition="outside",
                showlegend=False,
            ),
            row=1, col=col_i,
        )
    fig.update_layout(
        title="SI Forecast Model Performance — Model vs Persistence Baseline",
        height=420, width=1000,
    )
    fig.show()
