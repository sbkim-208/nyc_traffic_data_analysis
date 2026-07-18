import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf as _pacf

_FOLLOWER_COLORS = [
    "#e41a1c", "#ff7f00", "#377eb8", "#984ea3", "#4daf4a",
]


def plot_lag_correlation_curves(lag_df: pd.DataFrame):
    """Lag x Pearson r curve per leader road (colored by follower road, Plotly)

    lag_df: return value of get_lag_correlation()
    """
    leaders   = sorted(lag_df["leader"].unique())
    followers = sorted(lag_df["follower"].unique())
    ncols = 3
    nrows = (len(leaders) + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=leaders,
        shared_yaxes=True,
        vertical_spacing=0.10,
        horizontal_spacing=0.06,
    )

    for i, leader in enumerate(leaders):
        row = i // ncols + 1
        col = i % ncols + 1
        sub = lag_df[lag_df["leader"] == leader]

        for j, follower in enumerate(followers):
            pair = sub[sub["follower"] == follower].sort_values("lag_min")
            if pair.empty:
                continue
            color = _FOLLOWER_COLORS[j % len(_FOLLOWER_COLORS)]
            best_idx = pair["pearson_r"].idxmax()
            best_lag = pair.loc[best_idx, "lag_min"]
            best_r   = pair.loc[best_idx, "pearson_r"]

            fig.add_trace(
                go.Scatter(
                    x=pair["lag_min"],
                    y=pair["pearson_r"],
                    mode="lines+markers",
                    name=follower,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    legendgroup=follower,
                    showlegend=(i == 0),
                    hovertemplate=(
                        f"<b>{follower}</b><br>"
                        "lag: %{x}min<br>r: %{y:.3f}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )
            # Best-lag star marker
            fig.add_trace(
                go.Scatter(
                    x=[best_lag],
                    y=[best_r],
                    mode="markers+text",
                    marker=dict(color=color, size=11, symbol="star"),
                    text=[f"{int(best_lag)}min"],
                    textposition="top center",
                    textfont=dict(size=8, color=color),
                    showlegend=False,
                    hovertemplate=f"best lag: {int(best_lag)}min  r={best_r:.3f}<extra></extra>",
                ),
                row=row, col=col,
            )

        fig.update_xaxes(
            title_text="lag (min)",
            dtick=10, tickmode="linear", tick0=0,
            row=row, col=col,
        )
        fig.update_yaxes(
            title_text="Pearson r" if col == 1 else "",
            range=[-0.05, 1.02],
            row=row, col=col,
        )

    # Hide empty subplots
    total_cells = nrows * ncols
    for k in range(len(leaders), total_cells):
        r = k // ncols + 1
        c = k % ncols + 1
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, row=r, col=c)

    fig.update_layout(
        title=(
            "Tunnel/Bridge -> Manhattan Interior Road Lag Correlation<br>"
            "<sup>&#9733; = best lag | leader road speed drop -> follower road speed drop after lag minutes</sup>"
        ),
        height=360 * nrows,
        width=1200,
        legend=dict(
            title="Follower road",
            orientation="h",
            x=0, y=-0.05,
        ),
    )
    fig.show()


def plot_lag_heatmap(summary: pd.DataFrame):
    """Leader x Follower best-lag heatmap (Plotly)

    summary: return value of get_best_lag_summary()
    Color: shorter lag = faster propagation (darker color)
    Cell text: lag=Xmin / r=0.xx
    """
    leaders   = sorted(summary["leader"].unique())
    followers = sorted(summary["follower"].unique())

    lag_mat  = pd.DataFrame(index=leaders, columns=followers, dtype=float)
    r_mat    = pd.DataFrame(index=leaders, columns=followers, dtype=float)
    text_mat = pd.DataFrame(index=leaders, columns=followers, dtype=object)

    for _, row in summary.iterrows():
        l, f = row["leader"], row["follower"]
        lag_mat.loc[l, f]  = row["best_lag_min"]
        r_mat.loc[l, f]    = row["best_r"]
        text_mat.loc[l, f] = f"lag={int(row['best_lag_min'])}min<br>r={row['best_r']:.2f}"

    fig = go.Figure(
        go.Heatmap(
            z=lag_mat.values.astype(float),
            x=list(followers),
            y=list(leaders),
            colorscale="Blues",
            reversescale=True,          # shorter lag -> darker blue
            zmin=0,
            zmax=60,
            text=text_mat.values,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorbar=dict(
                title="Best lag (min)",
                tickvals=[0, 15, 30, 45, 60],
            ),
            hovertemplate="leader: %{y}<br>follower: %{x}<br>best lag: %{z}min<extra></extra>",
        )
    )

    fig.update_layout(
        title=(
            "Tunnel/Bridge -> Manhattan Interior Road Best Delay Time<br>"
            "<sup>shorter (darker blue) = faster congestion propagation | r = Pearson correlation</sup>"
        ),
        xaxis_title="Follower road (Manhattan interior)",
        yaxis_title="Leader road (tunnel/bridge)",
        yaxis=dict(autorange="reversed"),
        height=max(420, len(leaders) * 50 + 180),
        width=max(750, len(followers) * 130 + 280),
    )
    fig.show()


def plot_leader_best_impact(impact_df: pd.DataFrame):
    """Summary of max r + best lag per leader road (horizontal bar chart)

    impact_df: return value of get_leader_best_impact()
    Color: best_lag_min (shorter=blue, longer=red)
    Text: r=0.xx / lag=Xmin / follower road name
    """
    df = impact_df.sort_values("best_r")

    lag_norm = (df["best_lag_min"] - df["best_lag_min"].min()) / max(
        df["best_lag_min"].max() - df["best_lag_min"].min(), 1
    )
    colors = [
        f"rgb({int(55 + 200 * v)},{int(100 - 80 * v)},{int(200 - 170 * v)})"
        for v in lag_norm
    ]

    labels = [
        f"r={r:.2f}  |  lag={int(lag)}min  |  -> {fol}"
        for r, lag, fol in zip(df["best_r"], df["best_lag_min"], df["best_follower"])
    ]

    fig = go.Figure(
        go.Bar(
            x=df["best_r"],
            y=df["leader"],
            orientation="h",
            marker_color=colors,
            text=labels,
            textposition="outside",
            textfont=dict(size=11),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "max r: %{x:.3f}<br>"
                "best lag: %{customdata[0]}min<br>"
                "follower road: %{customdata[1]}<extra></extra>"
            ),
            customdata=list(zip(df["best_lag_min"].astype(int), df["best_follower"])),
        )
    )
    fig.update_layout(
        title=(
            "Max Impact on Manhattan Interior Roads by Leader Road<br>"
            "<sup>bar length = max Pearson r | color = best lag (blue=fast, red=slow)</sup>"
        ),
        xaxis=dict(title="Max Pearson r", range=[0, min(1.0, df["best_r"].max() + 0.15)]),
        yaxis_title="Leader road",
        height=max(400, len(df) * 52 + 160),
        width=1050,
        margin=dict(r=280),
    )
    fig.show()


def plot_r_heatmap(summary: pd.DataFrame):
    """Leader x Follower r-value heatmap (color = r, text = r / lag)

    summary: return value of get_best_lag_summary()
    Whereas plot_lag_heatmap colors by lag, this function colors by r so
    impact strength can be compared directly.
    """
    leaders   = sorted(summary["leader"].unique())
    followers = sorted(summary["follower"].unique())

    r_mat    = pd.DataFrame(index=leaders, columns=followers, dtype=float)
    text_mat = pd.DataFrame(index=leaders, columns=followers, dtype=object)

    for _, row in summary.iterrows():
        l, f = row["leader"], row["follower"]
        r_mat.loc[l, f]    = row["best_r"]
        text_mat.loc[l, f] = f"r={row['best_r']:.2f}<br>lag={int(row['best_lag_min'])}min"

    fig = go.Figure(
        go.Heatmap(
            z=r_mat.values.astype(float),
            x=list(followers),
            y=list(leaders),
            colorscale="Blues",
            zmin=0,
            zmax=1,
            text=text_mat.values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(
                title="Pearson r",
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            ),
            hovertemplate=(
                "leader: %{y}<br>follower: %{x}<br>"
                "max r: %{z:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=(
            "Tunnel/Bridge -> Manhattan Interior Road Max Correlation (r)<br>"
            "<sup>darker (blue) = stronger impact | lag in cell = delay time where r is maximal</sup>"
        ),
        xaxis_title="Follower road (Manhattan interior)",
        yaxis_title="Leader road (tunnel/bridge)",
        yaxis=dict(autorange="reversed"),
        height=max(420, len(leaders) * 50 + 180),
        width=max(750, len(followers) * 150 + 280),
    )
    fig.show()


_RAW_COLORS = ["#e41a1c", "#ff7f00", "#377eb8", "#984ea3", "#4daf4a", "#a65628", "#f781bf", "#999999"]


def plot_manhattan_raw_series(leader_ts: pd.DataFrame):
    """Visualize raw leader-road speed time series

    leader_ts: result of _build_road_series(resampled, _LEADER_PREFIXES)
    """
    fig = go.Figure()
    for i, col in enumerate(leader_ts.columns):
        fig.add_trace(go.Scatter(
            x=leader_ts.index,
            y=leader_ts[col],
            mode="lines",
            name=col,
            line=dict(width=1, color=_RAW_COLORS[i % len(_RAW_COLORS)]),
            opacity=0.75,
        ))
    fig.update_layout(
        title="Manhattan Leader-Road Speed Time Series (raw)<br><sup>for checking mixed trend/seasonality/noise segments</sup>",
        xaxis_title="Time",
        yaxis_title="Speed",
        height=400, width=1200,
        legend=dict(orientation="h", y=-0.18),
    )
    fig.show()


def plot_stl_decomposition(components: dict, leader: str):
    """STL decomposition 4-panel: original / trend / seasonal / residual

    components: return value of get_stl_components()
    Residual is the criterion for checking stationarity
    """
    keys   = ["original", "trend", "seasonal", "residual"]
    titles = ["Original", "Trend", "Seasonal", "Residual"]
    colors = ["#377eb8", "#e41a1c", "#ff7f00", "#4daf4a"]

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=titles,
        shared_xaxes=True,
        vertical_spacing=0.06,
    )
    for row, (key, color) in enumerate(zip(keys, colors), 1):
        fig.add_trace(
            go.Scatter(
                x=components[key].index,
                y=components[key].values,
                mode="lines",
                line=dict(width=1, color=color),
                showlegend=False,
            ),
            row=row, col=1,
        )
    fig.update_layout(
        title=f"STL Decomposition — {leader}",
        height=900, width=1200,
    )
    fig.show()


def plot_acf_pacf(series: pd.Series, title: str = "", lags: int = 60):
    """ACF/PACF visualization (with 95% confidence interval)

    series: stationary time series (use after removing trend/seasonal)
    Red bars: exceed the confidence interval -> significant lag
    """
    s  = series.dropna()
    n  = len(s)
    ci = 1.96 / np.sqrt(n)

    acf_arr  = acf(s,  nlags=lags, fft=True)[1:]
    pacf_arr = _pacf(s, nlags=lags, method="ywm")[1:]
    lag_x    = np.arange(1, lags + 1)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])
    for col, vals, name in [(1, acf_arr, "ACF"), (2, pacf_arr, "PACF")]:
        bar_colors = ["#e41a1c" if abs(v) > ci else "#377eb8" for v in vals]
        fig.add_trace(
            go.Bar(x=lag_x, y=vals, marker_color=bar_colors, name=name, showlegend=False),
            row=1, col=col,
        )
        for sign in (1, -1):
            fig.add_hline(y=sign * ci, line_dash="dash", line_color="gray", line_width=1, row=1, col=col)
        fig.add_hline(y=0, line_color="black", line_width=0.5, row=1, col=col)
        fig.update_xaxes(title_text="lag (5-min units)", row=1, col=col)
        fig.update_yaxes(title_text="Correlation", row=1, col=col)

    fig.update_layout(
        title=f"ACF / PACF — {title}" if title else "ACF / PACF",
        height=420, width=1000,
    )
    fig.show()
