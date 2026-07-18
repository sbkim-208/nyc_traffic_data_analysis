import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_korean_font = next(
    (f.name for f in fm.fontManager.ttflist if "AppleGothic" in f.name or "NanumGothic" in f.name),
    None,
)
if _korean_font:
    plt.rcParams["font.family"] = _korean_font
plt.rcParams["axes.unicode_minus"] = False


def plot_hour_dow_heatmap(pivot: pd.DataFrame, value_col: str = "speed", aggfunc: str = "mean"):
    """hour x day-of-week heatmap (seaborn)"""
    cmap = "YlOrRd" if aggfunc == "count" else "RdYlGn"
    title_val = "Measurement count" if aggfunc == "count" else f"Average {value_col}"

    plt.figure(figsize=(max(6, len(pivot.columns) * 1.4), 8))
    sns.heatmap(
        pivot,
        cmap=cmap,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": title_val},
    )
    plt.title(f"NYC Traffic Data: {title_val} (hour x day-of-week)", fontsize=14)
    plt.xlabel("Day of week")
    plt.ylabel("Hour (0-23)")
    plt.tight_layout()
    plt.show()


def plot_hour_dow_heatmap_by_borough(pivots: dict, value_col: str = "speed", aggfunc: str = "mean"):
    """hour x day-of-week heatmap by borough (one subplot per borough on the same figure)"""
    cmap = "YlOrRd" if aggfunc == "count" else "RdYlGn"
    title_val = "Measurement count" if aggfunc == "count" else f"Average {value_col}"
    unit = "" if aggfunc == "count" else ("km/h" if value_col == "speed" else "sec" if value_col == "travel_time" else "")
    boroughs = sorted(pivots.keys())
    n = len(boroughs)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 10), sharey=True)
    if n == 1:
        axes = [axes]

    vmin = min(p.min().min() for p in pivots.values())
    vmax = max(p.max().max() for p in pivots.values())

    for ax, borough in zip(axes, boroughs):
        sns.heatmap(
            pivots[borough],
            ax=ax,
            cmap=cmap,
            annot=True,
            fmt=".1f",
            linewidths=0.4,
            vmin=vmin,
            vmax=vmax,
            cbar=borough == boroughs[-1],
            cbar_kws={"label": title_val},
        )
        ax.set_title(f"{borough}\n{title_val}" + (f" ({unit})" if unit else ""),
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Day of week")
        ax.set_ylabel("Hour (0-23)" if borough == boroughs[0] else "")

    fig.suptitle(f"{title_val} by Borough" + (f" ({unit})" if unit else "") + " — hour x day-of-week",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.show()



def plot_weekday_weekend_comparison(comparison_df: pd.DataFrame):
    """Weekday/weekend speed and travel_time comparison by hour, per borough (2 columns: speed | travel_time)"""
    boroughs = sorted(comparison_df["borough"].unique())
    n = len(boroughs)
    colors = {"Weekday": "steelblue", "Weekend": "tomato"}

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), sharex=True)

    for row, borough in enumerate(boroughs):
        g = comparison_df[comparison_df["borough"] == borough]
        for day_type, grp in g.groupby("day_type"):
            grp = grp.sort_values("hour")
            axes[row, 0].plot(grp["hour"], grp["speed"],
                              label=day_type, color=colors[day_type], linewidth=2, marker="o", markersize=3)
            axes[row, 1].plot(grp["hour"], grp["travel_time"],
                              label=day_type, color=colors[day_type], linewidth=2, marker="o", markersize=3,
                              linestyle="--")

        axes[row, 0].set_ylabel(borough, fontsize=11, fontweight="bold")
        axes[row, 0].set_title("Speed (km/h)" if row == 0 else "", fontsize=11)
        axes[row, 1].set_title("Travel Time (sec)" if row == 0 else "", fontsize=11)
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].legend(fontsize=8)
        axes[row, 0].set_xticks(range(24))
        axes[row, 1].set_xticks(range(24))

    axes[-1, 0].set_xlabel("Hour (0-23)")
    axes[-1, 1].set_xlabel("Hour (0-23)")
    fig.suptitle("Weekday vs Weekend: Speed and Travel Time by Hour (per Borough)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/weekday_weekend_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_normalized_comparison(df: pd.DataFrame):
    """Normalized speed vs travel_time comparison by borough (same 0-1 scale, interactive Plotly)

    speed_norm: higher = faster / tt_norm: higher = longer travel time
    Crossing point: traffic-state transition moment
    """
    base = (
        df.groupby(["id", "hour", "borough"])[["speed", "travel_time"]]
        .mean()
        .reset_index()
        .groupby(["borough", "hour"])[["speed", "travel_time"]]
        .mean()
        .reset_index()
    )
    boroughs = sorted(base["borough"].unique())
    n = len(boroughs)

    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=boroughs,
        shared_xaxes=False,
        vertical_spacing=0.06,
    )

    for row_i, borough in enumerate(boroughs, start=1):
        g = base[base["borough"] == borough].sort_values("hour")
        hours = g["hour"].values

        sp = g["speed"]
        tt = g["travel_time"]
        sp_norm = ((sp - sp.min()) / (sp.max() - sp.min())).values
        tt_norm = ((tt - tt.min()) / (tt.max() - tt.min())).values

        show_legend = row_i == 1

        # Normalized speed line
        fig.add_trace(go.Scatter(
            x=hours, y=sp_norm,
            mode="lines+markers",
            name="speed (0-1)",
            line=dict(color="steelblue", width=2),
            marker=dict(size=5),
            legendgroup="speed",
            showlegend=show_legend,
            hovertemplate="Hour: %{x}:00<br>speed normalized: %{y:.3f}<extra></extra>",
        ), row=row_i, col=1)

        # Normalized travel_time line
        fig.add_trace(go.Scatter(
            x=hours, y=tt_norm,
            mode="lines+markers",
            name="travel_time (0-1)",
            line=dict(color="tomato", width=2, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
            legendgroup="travel_time",
            showlegend=show_legend,
            hovertemplate="Hour: %{x}:00<br>travel_time normalized: %{y:.3f}<extra></extra>",
        ), row=row_i, col=1)

        # Detect crossing point -> vertical line + label
        diff = sp_norm - tt_norm
        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] < 0:
                cross = (hours[i] + hours[i + 1]) / 2
                fig.add_vline(
                    x=cross, line=dict(color="gray", dash="dot", width=1.5),
                    row=row_i, col=1,
                )
                fig.add_annotation(
                    x=cross, y=1.05,
                    text=f"crossing ~{cross:.0f}:00",
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    xref=f"x{row_i if row_i > 1 else ''}",
                    yref=f"y{row_i if row_i > 1 else ''}",
                )

        fig.update_yaxes(title_text="Normalized (0-1)", range=[-0.05, 1.15],
                         row=row_i, col=1)
        fig.update_xaxes(dtick=1, tickmode="linear", tick0=0,
                         title_text="Hour (0-23)", row=row_i, col=1)

    fig.update_layout(
        title="Normalized Speed vs Travel Time Comparison (by borough) — same 0-1 scale",
        height=300 * n,
        hovermode="x unified",
    )
    fig.show()


def plot_speed_trend(df: pd.DataFrame):
    base = (
        df.groupby(["id", "hour", "borough"])["speed"]
        .mean()
        .reset_index()
        .groupby(["borough", "hour"])["speed"]
        .agg(mean="mean", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reset_index()
    )
    boroughs = sorted(base["borough"].unique())
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()

    for i, borough in enumerate(boroughs):
        g = base[base["borough"] == borough].sort_values("hour")
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=pd.concat([g["hour"], g["hour"].iloc[::-1]]),
            y=pd.concat([g["q75"], g["q25"].iloc[::-1]]),
            fill="toself", fillcolor=color,
            opacity=0.15, line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=g["hour"], y=g["mean"],
            mode="lines+markers",
            name=borough,
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ))

    fig.update_layout(
        title="Average Speed Trend by Hour (by borough, shading=Q1-Q3)",
        xaxis=dict(title="Hour (0-23)", dtick=1),
        yaxis=dict(title="Average speed (km/h)"),
        height=500,
    )
    fig.show()


def plot_speed_traveltime_trend(df: pd.DataFrame, sharp_changes=None):
    """Speed vs travel time by hour (by borough, dual y-axis)

    sharp_changes: passing the return value of get_sharp_changes() marks sharp-change points
    """
    base = (
        df.groupby(["id", "hour", "borough"])[["speed", "travel_time"]]
        .mean()
        .reset_index()
        .groupby(["borough", "hour"])[["speed", "travel_time"]]
        .mean()
        .reset_index()
    )
    boroughs = sorted(base["borough"].unique())
    colors = px.colors.qualitative.Plotly

    fig = make_subplots(
        rows=len(boroughs), cols=1,
        subplot_titles=boroughs,
        specs=[[{"secondary_y": True}]] * len(boroughs),
        shared_xaxes=False,
        vertical_spacing=0.07,
    )

    for row_i, borough in enumerate(boroughs, start=1):
        g = base[base["borough"] == borough].sort_values("hour")
        show_legend = row_i == 1

        fig.add_trace(go.Scatter(
            x=g["hour"], y=g["speed"],
            mode="lines+markers",
            name="speed (km/h)",
            line=dict(color="steelblue", width=2),
            marker=dict(size=4),
            legendgroup="speed",
            showlegend=show_legend,
        ), row=row_i, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=g["hour"], y=g["travel_time"],
            mode="lines+markers",
            name="travel_time (sec)",
            line=dict(color="tomato", width=2, dash="dash"),
            marker=dict(size=4, symbol="diamond"),
            legendgroup="travel_time",
            showlegend=show_legend,
        ), row=row_i, col=1, secondary_y=True)

        # Mark sharp-change points
        if sharp_changes is not None:
            bc = sharp_changes[sharp_changes["borough"] == borough]

            drops  = bc[bc["type"] == "speed_drop"]
            spikes = bc[bc["type"] == "travel_time_spike"]

            if not drops.empty:
                fig.add_trace(go.Scatter(
                    x=drops["hour"], y=drops["speed"],
                    mode="markers+text",
                    name="Sharp drop",
                    marker=dict(color="navy", size=12, symbol="triangle-down"),
                    text=[f"▼{v:.1f}" for v in drops["change"]],
                    textposition="top center",
                    textfont=dict(size=9, color="navy"),
                    legendgroup="drop",
                    showlegend=show_legend,
                ), row=row_i, col=1, secondary_y=False)

            if not spikes.empty:
                fig.add_trace(go.Scatter(
                    x=spikes["hour"], y=spikes["travel_time"],
                    mode="markers+text",
                    name="Sharp spike",
                    marker=dict(color="darkred", size=12, symbol="triangle-up"),
                    text=[f"▲{v:.0f}s" for v in spikes["change"]],
                    textposition="bottom center",
                    textfont=dict(size=9, color="darkred"),
                    legendgroup="spike",
                    showlegend=show_legend,
                ), row=row_i, col=1, secondary_y=True)

        fig.update_yaxes(title_text="Speed (km/h)", secondary_y=False, row=row_i, col=1)
        fig.update_yaxes(title_text="Travel time (sec)", secondary_y=True, row=row_i, col=1)

    fig.update_xaxes(dtick=1, tickmode="linear", tick0=0, title_text="Hour (0-23)")
    fig.update_layout(
        title="Speed vs Travel Time by Hour (by borough) — ▼speed drop / ▲travel-time spike",
        height=260 * len(boroughs),
    )
    fig.show()


def plot_travel_time_trend(df: pd.DataFrame):
    base = (
        df.groupby(["id", "hour", "borough"])["travel_time"]
        .mean()
        .reset_index()
        .groupby(["borough", "hour"])["travel_time"]
        .agg(mean="mean", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reset_index()
    )
    boroughs = sorted(base["borough"].unique())
    colors = px.colors.qualitative.Plotly

    fig = make_subplots(
        rows=len(boroughs), cols=1,
        subplot_titles=boroughs,
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    for i, borough in enumerate(boroughs):
        g = base[base["borough"] == borough].sort_values("hour")
        color = colors[i % len(colors)]
        row = i + 1

        fig.add_trace(go.Scatter(
            x=pd.concat([g["hour"], g["hour"].iloc[::-1]]),
            y=pd.concat([g["q75"], g["q25"].iloc[::-1]]),
            fill="toself", fillcolor=color,
            opacity=0.15, line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=g["hour"], y=g["mean"],
            mode="lines+markers",
            name=borough,
            line=dict(color=color, width=2),
            marker=dict(size=5),
        ), row=row, col=1)

        fig.update_yaxes(title_text="Travel time (sec)", row=row, col=1)

    fig.update_xaxes(title_text="Hour (0-23)", dtick=1, row=len(boroughs), col=1)
    fig.update_layout(
        title="Average Travel Time Trend by Hour (by borough, shading=Q1-Q3)",
        height=220 * len(boroughs),
        showlegend=False,
    )
    fig.show()


def plot_speed_by_hour(df: pd.DataFrame):
    speed_by_id_hour = (
        df.groupby(["id", "hour", "borough", "link_name"])["speed"]
        .mean()
        .reset_index()
    )
    fig = px.line(
        speed_by_id_hour,
        x="hour", y="speed",
        color="link_name",
        facet_col="borough",
        facet_col_wrap=3,
        title="Segment Speed Trend by Hour (id+hour average)",
        labels={"hour": "Hour (0-23)", "speed": "Average speed (km/h)", "link_name": "Segment"},
        markers=True,
    )
    fig.update_layout(height=700, showlegend=False)
    fig.update_xaxes(dtick=2)
    fig.show()


def plot_borough_road_congestion(weekday_pivot: pd.DataFrame, weekend_pivot: pd.DataFrame, borough: str):
    """SI heatmap by road x hour within a borough (weekday vs weekend, Plotly)"""
    hours   = list(range(24))
    n_roads = len(weekday_pivot)
    si_min  = min(weekday_pivot.min().min(), weekend_pivot.min().min())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Weekday", "Weekend"],
        horizontal_spacing=0.08,
    )
    for col_i, (pivot, label) in enumerate([(weekday_pivot, "Weekday"), (weekend_pivot, "Weekend")], start=1):
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=hours,
                y=list(pivot.index),
                colorscale="RdYlGn",
                reversescale=False,
                zmin=si_min,
                zmax=1.0,
                text=[[f"{v:.2f}" if not pd.isna(v) else "" for v in row] for row in pivot.values],
                texttemplate="%{text}",
                textfont={"size": 8},
                showscale=(col_i == 2),
                colorbar=dict(title="SI", len=0.8, x=1.02),
                hovertemplate=f"{label}<br>Road: %{{y}}<br>Hour: %{{x}}:00<br>SI: %{{z:.3f}}<extra></extra>",
                name=label,
            ),
            row=1, col=col_i,
        )
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=9),
                         showticklabels=(col_i == 1), row=1, col=col_i)
        fig.update_xaxes(dtick=1, tickmode="linear", tick0=0,
                         title_text="Hour (0-23)", tickfont=dict(size=8), row=1, col=col_i)

    fig.update_layout(
        title=f"{borough} — Congestion SI by Road and Hour (Weekday vs Weekend)<br>"
              f"<sup>SI = actual speed / free-flow speed · lower (red) = more congested</sup>",
        height=max(400, n_roads * 28 + 160),
        width=1200,
    )
    fig.show()


def plot_all_boroughs_road_congestion(pivots: dict):
    """SI heatmap by road x hour for all 5 boroughs — stacked vertically by borough in one figure (Plotly)

    pivots: {borough: (weekday_pivot, weekend_pivot)}
            dict collecting get_road_hour_si_by_borough() results per borough
    """
    borough_order = ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]
    hours = list(range(24))

    n_roads = {b: len(pivots[b][0]) for b in borough_order if b in pivots}
    total_roads = sum(n_roads.values())
    row_heights  = [n_roads[b] / total_roads for b in borough_order if b in pivots]
    n_boroughs   = len(row_heights)

    subplot_titles = []
    for b in borough_order:
        if b not in pivots:
            continue
        subplot_titles += [f"<b>{b}</b> — Weekday", f"<b>{b}</b> — Weekend"]

    fig = make_subplots(
        rows=n_boroughs, cols=2,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        horizontal_spacing=0.06,
        vertical_spacing=0.03,
    )

    si_min = min(wd.min().min() for wd, _ in pivots.values())
    si_max = 1.0

    for row_i, borough in enumerate([b for b in borough_order if b in pivots], start=1):
        wd_pivot, we_pivot = pivots[borough]
        is_last = row_i == n_boroughs

        for col_i, (pivot, label) in enumerate([(wd_pivot, "Weekday"), (we_pivot, "Weekend")], start=1):
            show_scale = (col_i == 2 and row_i == 1)
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=hours,
                    y=list(pivot.index),
                    colorscale="RdYlGn",
                    reversescale=False,
                    zmin=si_min,
                    zmax=si_max,
                    text=[[f"{v:.2f}" if not pd.isna(v) else "" for v in row] for row in pivot.values],
                    texttemplate="%{text}",
                    textfont={"size": 7},
                    showscale=show_scale,
                    colorbar=dict(title="SI", thickness=15, len=0.3, y=0.85, x=1.02),
                    hovertemplate=f"{borough} {label}<br>Road: %{{y}}<br>Hour: %{{x}}:00<br>SI: %{{z:.3f}}<extra></extra>",
                    name=f"{borough} {label}",
                ),
                row=row_i, col=col_i,
            )

            fig.update_yaxes(
                autorange="reversed",
                tickfont=dict(size=8),
                showticklabels=(col_i == 1),
                row=row_i, col=col_i,
            )
            fig.update_xaxes(
                dtick=1, tickmode="linear", tick0=0,
                title_text="Hour (0-23)" if is_last else "",
                tickfont=dict(size=8),
                row=row_i, col=col_i,
            )

    fig.update_layout(
        title="NYC Congestion SI by Road and Hour — 5 Boroughs (Weekday vs Weekend)<br>"
              "<sup>SI = actual speed / free-flow speed · lower (red) = more congested</sup>",
        height=max(800, total_roads * 26 + n_boroughs * 60),
        width=1300,
        showlegend=False,
    )
    fig.show()


def plot_tti_si_heatmap(tti_pivot: pd.DataFrame, si_pivot: pd.DataFrame):
    """Interactive borough x hour TTI/SI heatmap (Plotly)

    TTI: higher = more congested (RdYlGn_r)
    SI : lower = more congested (RdYlGn)
    """
    boroughs = list(tti_pivot.columns)
    hours    = list(tti_pivot.index)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "TTI (Travel Time Index) — travel time ratio vs. free-flow<br><sup>higher = more congested</sup>",
            "SI (Speed Index) — speed ratio vs. free-flow<br><sup>lower = more congested</sup>",
        ],
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Heatmap(
            z=tti_pivot.values,
            x=boroughs,
            y=hours,
            colorscale="RdYlGn",
            reversescale=True,
            zmin=1.0,
            zmax=tti_pivot.max().max(),
            text=[[f"{v:.2f}" for v in row] for row in tti_pivot.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="TTI", x=0.44, len=0.9),
            hovertemplate="Hour: %{y}:00<br>Borough: %{x}<br>TTI: %{z:.2f}<extra></extra>",
            name="TTI",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=si_pivot.values,
            x=boroughs,
            y=hours,
            colorscale="RdYlGn",
            reversescale=False,
            zmin=si_pivot.min().min(),
            zmax=1.0,
            text=[[f"{v:.3f}" for v in row] for row in si_pivot.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="SI", x=1.01, len=0.9),
            hovertemplate="Hour: %{y}:00<br>Borough: %{x}<br>SI: %{z:.3f}<extra></extra>",
            name="SI",
        ),
        row=1, col=2,
    )

    fig.update_yaxes(autorange="reversed", dtick=1, title_text="Hour (0-23)", col=1)
    fig.update_yaxes(autorange="reversed", dtick=1, col=2)
    fig.update_xaxes(title_text="Borough", row=1)

    fig.update_layout(
        title="NYC Traffic Congestion — TTI / SI (Borough x Hour)",
        height=700,
        width=1100,
    )
    fig.show()


def plot_connection_road_stats(road_stats: pd.DataFrame, borough_avg: pd.DataFrame):
    """Horizontal bars of avg_speed/avg_tt per connection road + borough average reference lines

    road_stats : return value of get_connection_road_stats()
    borough_avg: columns borough, avg_speed, avg_tt
    """
    TYPE_COLOR = {
        "bridge":     "#2166ac",
        "tunnel":     "#d6604d",
        "expressway": "#4dac26",
        "parkway":    "#9970ab",
    }
    TYPE_LABEL = {
        "bridge": "Bridge", "tunnel": "Tunnel",
        "expressway": "Expressway", "parkway": "Parkway",
    }
    TYPE_ORDER = ["bridge", "tunnel", "expressway", "parkway"]

    rows = []
    for rtype in TYPE_ORDER:
        sub = road_stats[road_stats["road_type"] == rtype].sort_values("avg_speed")
        rows.append(sub)
    df = pd.concat(rows, ignore_index=True)

    labels = df["road_name"].tolist()
    y = range(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, len(labels) * 0.42)))

    # -- Left: average speed --------------------------------
    bar_colors = [TYPE_COLOR[t] for t in df["road_type"]]
    bars = ax1.barh(list(y), df["avg_speed"], color=bar_colors, alpha=0.85, height=0.65)
    ax1.errorbar(
        df["avg_speed"], list(y),
        xerr=[(df["avg_speed"] - df["p25_speed"]).clip(0),
              (df["p75_speed"] - df["avg_speed"]).clip(0)],
        fmt="none", color="black", capsize=3, linewidth=0.9, alpha=0.6,
    )
    boro_colors = ["#e41a1c", "#ff7f00", "#984ea3", "#377eb8", "#4daf4a"]
    for i, (_, r) in enumerate(borough_avg.sort_values("avg_speed").iterrows()):
        ax1.axvline(r["avg_speed"], color=boro_colors[i], linewidth=1.2,
                    linestyle="--", alpha=0.7, label=f"{r['borough']} {r['avg_speed']:.1f}")
    for bar, val in zip(bars, df["avg_speed"]):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=7.5)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Average speed (km/h)", fontsize=11)
    ax1.set_title("Average Speed by Connection Road\n(error: Q25-Q75 / dashed line: borough average)", fontsize=12, fontweight="bold")
    ax1.legend(title="Borough average", fontsize=8, title_fontsize=8, loc="lower right")
    ax1.spines[["top", "right"]].set_visible(False)

    # -- Right: average travel time --------------------------
    bars2 = ax2.barh(list(y), df["avg_tt"], color=bar_colors, alpha=0.85, height=0.65)
    for i, (_, r) in enumerate(borough_avg.sort_values("avg_speed").iterrows()):
        ax2.axvline(r["avg_tt"], color=boro_colors[i], linewidth=1.2,
                    linestyle="--", alpha=0.7, label=f"{r['borough']} {int(r['avg_tt'])}s")
    for bar, val in zip(bars2, df["avg_tt"]):
        ax2.text(val + 5, bar.get_y() + bar.get_height() / 2,
                 f"{int(val)}s", va="center", fontsize=7.5)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("Average travel time (sec)", fontsize=11)
    ax2.set_title("Average Travel Time by Connection Road\n(dashed line: borough average)", fontsize=12, fontweight="bold")
    ax2.legend(title="Borough average", fontsize=8, title_fontsize=8, loc="lower right")
    ax2.spines[["top", "right"]].set_visible(False)

    # -- Legend (road type) -----------------------------------
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=TYPE_COLOR[t], alpha=0.85, label=TYPE_LABEL[t])
                  for t in TYPE_ORDER]
    fig.legend(handles=legend_els, title="Road type", loc="upper center",
               ncol=4, fontsize=9, title_fontsize=9, bbox_to_anchor=(0.5, 1.01))

    # -- road_type separator lines -----------------------------
    for ax in (ax1, ax2):
        cumsum = 0
        for rtype in TYPE_ORDER:
            cnt = (df["road_type"] == rtype).sum()
            if cnt and cumsum > 0:
                ax.axhline(cumsum - 0.5, color="gray", linewidth=0.7, linestyle=":")
            cumsum += cnt

    fig.suptitle("NYC Connection Road Traffic Overview — Bridges, Tunnels, Expressways, Parkways",
                 fontsize=14, fontweight="bold", y=1.04)
    plt.tight_layout()
    plt.savefig("outputs/connection_road_stats.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_within_vs_cross_correlation(corr_df: pd.DataFrame):
    """Within-segment vs cross-segment Speed-TravelTime correlation comparison (Spearman)

    corr_df: return value of get_within_segment_correlation()
    columns: borough, mean, median, min, max, cross_spearman
    """
    df = corr_df.sort_values("mean")
    boroughs = df["borough"].tolist()
    x = range(len(boroughs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars_within = ax.bar(
        [i - width / 2 for i in x],
        df["mean"],
        width,
        label="Within-segment\nvariation by hour",
        color="steelblue",
        alpha=0.85,
    )
    ax.errorbar(
        [i - width / 2 for i in x],
        df["mean"],
        yerr=[df["mean"] - df["min"], df["max"] - df["mean"]],
        fmt="none",
        color="navy",
        capsize=5,
        linewidth=1.2,
    )

    bars_cross = ax.bar(
        [i + width / 2 for i in x],
        df["cross_spearman"],
        width,
        label="Cross-segment\nsegment-average comparison",
        color="tomato",
        alpha=0.85,
    )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(boroughs, fontsize=11)
    ax.set_ylabel("Spearman correlation", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Within-segment vs Cross-segment Speed-TravelTime Correlation (Spearman)\nerror bars: within min/max",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars_within, df["mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, color="navy")
    for bar, val in zip(bars_cross, df["cross_spearman"]):
        offset = 0.03 if val >= 0 else -0.07
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, color="darkred")

    plt.tight_layout()
    plt.savefig("outputs/within_vs_cross_correlation.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_manhattan_connection_si(si_df: pd.DataFrame):
    """SI heatmap by Manhattan connection road x hour (weekday vs weekend, Plotly)

    si_df: return value of get_manhattan_connection_si()
    columns: route_name, group_label, hour, day_type, si
    """
    GROUP_ORDER = [
        "Queens→Manhattan",
        "Brooklyn→Manhattan",
        "Queens/Bronx→Manhattan",
        "Bronx→Manhattan",
        "Bronx/NJ→Manhattan",
        "NJ→Manhattan",
        "Manhattan",
    ]

    # Sort routes in group order + insert NaN separator rows
    route_order = []
    prev_group = None
    for group in GROUP_ORDER:
        routes = si_df[si_df["group_label"] == group]["route_name"].unique()
        if len(routes) == 0:
            continue
        if prev_group is not None:
            route_order.append(f"__sep__{group}")     # dummy row for separator
        for r in sorted(routes):
            route_order.append(r)
        prev_group = group
    # Unclassified routes
    for r in si_df["route_name"].unique():
        if r not in route_order:
            route_order.append(r)

    hours = list(range(24))
    si_min = round(si_df["si"].min() - 0.05, 2)
    n_rows = len(route_order)

    def build_pivot(day_type: str) -> pd.DataFrame:
        sub = si_df[si_df["day_type"] == day_type]
        base = sub.pivot_table(index="route_name", columns="hour", values="si")
        rows = []
        for r in route_order:
            if r.startswith("__sep__"):
                group_label = r[len("__sep__"):]
                sep = pd.Series([float("nan")] * 24, index=range(24),
                                 name=f"--- {group_label} ---")
                rows.append(sep)
            else:
                row = base.reindex([r]).iloc[0].rename(r)
                rows.append(row)
        return pd.DataFrame(rows)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Weekday", "Weekend"],
        horizontal_spacing=0.08,
    )

    for col_i, day_type in enumerate(["Weekday", "Weekend"], start=1):
        pivot = build_pivot(day_type)
        y_labels = list(pivot.index)

        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=hours,
                y=y_labels,
                colorscale="RdYlGn",
                reversescale=False,
                zmin=si_min,
                zmax=1.0,
                text=[[f"{v:.2f}" if not pd.isna(v) else "" for v in row]
                      for row in pivot.values],
                texttemplate="%{text}",
                textfont={"size": 9},
                showscale=(col_i == 2),
                colorbar=dict(title="SI", len=0.85, x=1.02),
                hovertemplate="Road: %{y}<br>Hour: %{x}:00<br>SI: %{z:.3f}<extra></extra>",
                name=day_type,
            ),
            row=1, col=col_i,
        )
        fig.update_yaxes(
            autorange="reversed",
            tickfont=dict(size=9),
            showticklabels=(col_i == 1),
            row=1, col=col_i,
        )
        fig.update_xaxes(
            dtick=1, tickmode="linear", tick0=0,
            title_text="Hour (0-23)",
            tickfont=dict(size=8),
            row=1, col=col_i,
        )

    fig.update_layout(
        title="Congestion SI by Manhattan Connection Road and Hour (Weekday vs Weekend)<br>"
              "<sup>SI = actual speed / free-flow speed · lower (red) = more congested | "
              "--- separator ---: connection-direction group</sup>",
        height=max(500, n_rows * 30 + 160),
        width=1250,
    )
    fig.show()


def plot_hour_correlations(df: pd.DataFrame):
    base = (
        df.groupby(["id", "hour", "borough"])[["speed", "travel_time"]]
        .mean()
        .reset_index()
    )
    boroughs = sorted(base["borough"].unique())
    colors = px.colors.qualitative.Plotly

    fig = make_subplots(
        rows=3, cols=len(boroughs),
        subplot_titles=[f"{b}" for _ in range(3) for b in boroughs],
        row_titles=["hour → speed", "hour → travel_time", "speed → travel_time"],
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    for col_i, borough in enumerate(boroughs, start=1):
        g = base[base["borough"] == borough]
        hour_mean = g.groupby("hour")[["speed", "travel_time"]].mean().reset_index()
        color = colors[(col_i - 1) % len(colors)]

        fig.add_trace(go.Scatter(
            x=hour_mean["hour"], y=hour_mean["speed"],
            mode="lines+markers", line=dict(color=color),
            name=borough, showlegend=(col_i == 1), legendgroup=borough,
        ), row=1, col=col_i)

        fig.add_trace(go.Scatter(
            x=hour_mean["hour"], y=hour_mean["travel_time"],
            mode="lines+markers", line=dict(color=color),
            name=borough, showlegend=False, legendgroup=borough,
        ), row=2, col=col_i)

        fig.add_trace(go.Histogram2dContour(
            x=g["speed"], y=g["travel_time"],
            colorscale="Blues",
            showscale=(col_i == len(boroughs)),
            colorbar=dict(title="Density", x=1.02),
            name=borough, showlegend=False,
        ), row=3, col=col_i)

    fig.update_layout(height=900, title_text="hour / speed / travel_time correlation (id+hour average)")
    fig.show()
