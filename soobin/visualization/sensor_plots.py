import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_korean_font = next(
    (f.name for f in fm.fontManager.ttflist if "AppleGothic" in f.name or "NanumGothic" in f.name),
    None,
)
if _korean_font:
    plt.rcParams["font.family"] = _korean_font
plt.rcParams["axes.unicode_minus"] = False


def plot_sensor_reliability(
    by_borough: pd.DataFrame,
    by_hour: pd.DataFrame,
    pivot: pd.DataFrame,
):
    """Sensor reliability 3-panel visualization
    - Top left: horizontal bar of reliability by borough
    - Top right: line of reliability by hour
    - Bottom: hour (rows) x borough (columns) reliability heatmap
    """
    by_borough = by_borough.copy()
    by_hour = by_hour.copy()

    by_borough["reliability"] = (1 - by_borough["error_rate"]).round(4)
    by_hour["reliability"] = (1 - by_hour["error_rate"]).round(4)
    # pivot: index=hour, columns=borough -> convert to reliability
    reliability_pivot = (1 - pivot).round(4)

    fig = plt.figure(figsize=(20, 26))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[1, 3.5],
                           hspace=0.4, wspace=0.35)

    ax_bar  = fig.add_subplot(gs[0, 0])
    ax_line = fig.add_subplot(gs[0, 1])
    ax_heat = fig.add_subplot(gs[1, :])

    # -- Reliability bar by borough -------------------------
    sorted_df = by_borough.sort_values("reliability")
    bar_colors = [
        "#d73027" if v < 0.6 else "#fc8d59" if v < 0.7 else "#fee090" if v < 0.8 else "#4dac26"
        for v in sorted_df["reliability"]
    ]
    bars = ax_bar.barh(sorted_df["borough"], sorted_df["reliability"],
                       color=bar_colors, edgecolor="white", height=0.55)
    ax_bar.set_xlim(0, 1.05)
    ax_bar.axvline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_bar.set_xlabel("Reliability score", fontsize=11)
    ax_bar.set_title("Sensor Reliability by Borough", fontsize=13, fontweight="bold", pad=12)
    ax_bar.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, sorted_df["reliability"]):
        ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", ha="left", fontsize=11, fontweight="bold")

    # -- Reliability line by hour ----------------------------
    avg = by_hour["reliability"].mean()
    ax_line.plot(by_hour["hour"], by_hour["reliability"],
                 color="steelblue", linewidth=2.5, marker="o", markersize=6, zorder=3)
    ax_line.fill_between(by_hour["hour"], avg, by_hour["reliability"],
                         where=by_hour["reliability"] >= avg,
                         alpha=0.2, color="steelblue", label="Above average")
    ax_line.fill_between(by_hour["hour"], avg, by_hour["reliability"],
                         where=by_hour["reliability"] < avg,
                         alpha=0.25, color="tomato", label="Below average")
    ax_line.axhline(avg, color="tomato", linewidth=1.5, linestyle="--",
                    label=f"Overall average {avg:.1%}")
    ax_line.set_xlim(0, 23)
    ax_line.set_xticks(range(24))
    ax_line.set_ylim(
        max(0, by_hour["reliability"].min() - 0.05),
        min(1.02, by_hour["reliability"].max() + 0.05),
    )
    ax_line.set_xlabel("Hour (0-23)", fontsize=11)
    ax_line.set_ylabel("Reliability score", fontsize=11)
    ax_line.set_title("Sensor Reliability by Hour", fontsize=13, fontweight="bold", pad=12)
    ax_line.spines[["top", "right"]].set_visible(False)
    ax_line.legend(fontsize=9, loc="lower right")

    # -- hour x borough heatmap (24 rows x 5 columns) --------
    # reliability_pivot: index=hour(24), columns=borough(5)
    sns.heatmap(
        reliability_pivot,
        ax=ax_heat,
        cmap="RdYlGn",
        vmin=0.4,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Reliability score", "shrink": 0.5, "pad": 0.02},
    )
    ax_heat.set_title("Sensor Reliability Heatmap — Hour x Borough",
                      fontsize=13, fontweight="bold", pad=12)
    ax_heat.set_xlabel("Borough", fontsize=11)
    ax_heat.set_ylabel("Hour (0-23)", fontsize=11)
    ax_heat.tick_params(axis="x", rotation=0, labelsize=10)
    ax_heat.tick_params(axis="y", rotation=0, labelsize=9)

    fig.suptitle("NYC Traffic Sensor Reliability Analysis", fontsize=16, fontweight="bold", y=1.01)
    plt.savefig("outputs/sensor_reliability.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_borough_reliability(by_borough: pd.DataFrame, overall_reliability: float):
    """Horizontal bar of reliability by borough against overall reliability (bars compare size more accurately than a pie chart)"""
    df = by_borough.copy()
    df["reliability"] = (1 - df["error_rate"]).round(4)
    df = df.sort_values("reliability")

    bar_colors = [
        "#d73027" if v < 0.6 else "#fc8d59" if v < 0.7 else "#fee090" if v < 0.8 else "#4dac26"
        for v in df["reliability"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["borough"], df["reliability"], color=bar_colors, edgecolor="white", height=0.6)
    ax.axvline(overall_reliability, color="steelblue", linewidth=1.5, linestyle="--",
               label=f"Overall reliability {overall_reliability:.1%}")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Reliability score", fontsize=11)
    ax.set_title("Sensor Reliability by Borough vs. Overall", fontsize=13, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="lower right")
    for bar, val in zip(bars, df["reliability"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1%}", va="center", ha="left", fontsize=10, fontweight="bold")

    plt.savefig("outputs/borough_reliability.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_borough_problem_roads(by_borough_segments: pd.DataFrame):
    """Top N error-rate segments by borough — small multiples with one subplot per borough"""
    boroughs = sorted(by_borough_segments["borough"].unique())
    n = len(boroughs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n))
    if n == 1:
        axes = [axes]

    for ax, borough in zip(axes, boroughs):
        sub = by_borough_segments[by_borough_segments["borough"] == borough].sort_values("error_rate")
        labels = [name if len(name) <= 40 else name[:37] + "..." for name in sub["link_name"]]
        bars = ax.barh(labels, sub["error_rate"], color="#d73027", edgecolor="white", height=0.6)
        ax.set_xlim(0, 1.05)
        ax.set_title(borough, fontsize=11, fontweight="bold", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=9)
        for bar, val in zip(bars, sub["error_rate"]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1%}", va="center", ha="left", fontsize=9)

    fig.suptitle("Top Error-Rate Segments by Borough", fontsize=14, fontweight="bold", y=1.005)
    fig.tight_layout()
    plt.savefig("outputs/borough_problem_roads.png", dpi=150, bbox_inches="tight")
    plt.show()


def _severity_ramp(value: float, low_hex: str = "#F3D9B6", high_hex: str = "#8A3B12") -> str:
    """Linear RGB interpolation between a pale and a deep tone of the same warm hue family."""
    lo = tuple(int(low_hex[i : i + 2], 16) for i in (1, 3, 5))
    hi = tuple(int(high_hex[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(lo[i] + (hi[i] - lo[i]) * value) for i in range(3))
    return f"rgb({r},{g},{b})"


def plot_borough_problem_roads_interactive(
    by_borough_segments: pd.DataFrame,
    output_path: str = "outputs/borough_problem_roads.html",
) -> go.Figure:
    """Top N error-rate segments by borough — interactive Plotly small multiples.

    Same data as plot_borough_problem_roads, but as hoverable bars (full segment
    name, exact error rate, sample count) instead of a static PNG.
    """
    boroughs = sorted(by_borough_segments["borough"].unique())
    n = len(boroughs)

    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=boroughs,
        shared_xaxes=True,
        vertical_spacing=0.6 / n,
    )

    for row, borough in enumerate(boroughs, start=1):
        sub = by_borough_segments[by_borough_segments["borough"] == borough].sort_values(
            "error_rate", ascending=True
        )
        colors = [_severity_ramp(v) for v in sub["error_rate"]]
        fig.add_trace(
            go.Bar(
                x=sub["error_rate"],
                y=sub["link_name"],
                orientation="h",
                marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.5)", width=1)),
                text=[f"{v:.1%}" for v in sub["error_rate"]],
                textposition="outside",
                cliponaxis=False,
                customdata=sub[["total", "error_count"]].to_numpy(),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Error rate: %{x:.1%}<br>"
                    "Samples: %{customdata[0]:,}<br>"
                    "Errors: %{customdata[1]:,}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.update_xaxes(range=[0, 1.08], tickformat=".0%", row=row, col=1)
        fig.update_yaxes(automargin=True, row=row, col=1)

    fig.update_layout(
        title=dict(text="Top Error-Rate Segments by Borough", x=0, font=dict(size=18)),
        height=190 * n,
        margin=dict(l=10, r=30, t=70, b=10),
        font=dict(family="ui-sans-serif, Helvetica, Arial, sans-serif", size=12, color="#4B534E"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#131917", font_color="#F1F2ED", font_size=12),
    )
    for ann in fig.layout.annotations:
        ann.font.size = 13
        ann.font.color = "#131917"
        ann.x = 0
        ann.xanchor = "left"

    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    return fig
