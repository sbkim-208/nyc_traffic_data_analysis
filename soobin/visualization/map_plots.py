import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_korean_font = next(
    (f.name for f in fm.fontManager.ttflist if "AppleGothic" in f.name or "NanumGothic" in f.name),
    None,
)
if _korean_font:
    plt.rcParams["font.family"] = _korean_font
plt.rcParams["axes.unicode_minus"] = False

_NYC_LAT = 40.7  # for the lon/lat aspect-ratio correction below


def _severity_hex(value: float, low_hex: str = "#cde2fb", high_hex: str = "#0d366b") -> str:
    """Pale->deep single-hue (blue) severity ramp, as hex (matplotlib wants hex or 0-1 RGB
    tuples, not the CSS rgb(...) string Plotly consumes). Endpoints are steps 100/700 of the
    dataviz skill's documented sequential blue ramp, not eyeballed."""
    lo = tuple(int(low_hex[i : i + 2], 16) for i in (1, 3, 5))
    hi = tuple(int(high_hex[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(lo[i] + (hi[i] - lo[i]) * value) for i in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_congestion_error_map(
    gdf: gpd.GeoDataFrame,
    error_threshold: float = 0.5,
    congestion_quantile: float = 0.25,
    output_path: str = "outputs/congestion_sensor_map.png",
):
    """Static hero map: segment color = PM-peak congestion severity, markers = the two hotspot checklists

    gdf must already carry si/error_rate columns (see geo.add_congestion_and_error). Congestion
    uses a single-hue pale->deep blue ramp (sequential, by magnitude) as background context, plus
    an explicit marker on the most-congested quartile of segments (si <= congestion_quantile) —
    a color gradient alone asks the reader to compare shades to find "the bad ones"; a marker just
    tells them. Sensor error hotspots (error_rate >= error_threshold) get their own marker in ink
    black, kept visually distinct (shape, not just hue) from the congestion marker so the two
    "check this area" layers never get mistaken for each other.

    Segments with no PM-peak readings at all (si is NaN — typically a fully dead sensor, see
    remove_dead_segments) are drawn in flat gray rather than dropped: those are exactly the most
    severe error hotspots (VNB, Lincoln Tunnel, etc. all fall in this group), so silently omitting
    them would erase the map's own headline finding.

    A soft borough-shaped backdrop sits behind the lines: the sensor network only covers
    highways/bridges/tunnels (not the street grid), so on a flat background the segments read as
    disconnected scribbles floating in empty space. The backdrop gives the eye actual ground to
    anchor to instead of bare page color.
    """
    gdf = gdf.copy()
    has_si = gdf["si"].notna()

    fig, ax = plt.subplots(figsize=(13, 9.5))
    ax.set_facecolor("#F7F4EF")

    # Soft per-borough backdrop (buffered convex hull) so gaps between clusters read as
    # "unmonitored ground" rather than blank void.
    proj = gdf.to_crs(epsg=32618)
    for borough, group in proj.groupby("borough"):
        hull = group.geometry.union_all().convex_hull.buffer(700)
        gpd.GeoSeries([hull], crs=proj.crs).to_crs(epsg=4326).plot(
            ax=ax, color="#EDE8DC", edgecolor="#E1DACB", linewidth=1, zorder=0,
        )

    colors = [_severity_hex(v) for v in (1 - gdf.loc[has_si, "si"]).clip(0, 1)]
    gdf[has_si].plot(ax=ax, color=colors, linewidth=2.4, zorder=1)
    gdf[~has_si].plot(ax=ax, color="#B9B6AC", linewidth=2.4, linestyle=(0, (1, 1.5)), zorder=1)

    def _midpoints(rows):
        return rows.to_crs(epsg=32618).geometry.interpolate(0.5, normalized=True).to_crs(epsg=4326)

    congestion_cutoff = gdf["si"].quantile(congestion_quantile)
    congested = gdf[has_si & (gdf["si"] <= congestion_cutoff)]
    if not congested.empty:
        marks = _midpoints(congested)
        ax.scatter(
            marks.x, marks.y, marker="o", s=70,
            facecolor="#E4572E", edgecolor="white", linewidths=1.2, zorder=3,
        )

    hotspots = gdf[gdf["error_rate"] >= error_threshold]
    if not hotspots.empty:
        marks = _midpoints(hotspots)
        ax.scatter(
            marks.x, marks.y, marker="x", s=90, linewidths=2.4,
            color="#131917", zorder=3,
        )

    # The sensor network only covers highways/bridges/tunnels, not the street grid, so the
    # segments alone don't read as "NYC" — a borough label per cluster gives the eye an anchor.
    for borough, group in gdf.groupby("borough"):
        pt = group.geometry.union_all().centroid
        ax.annotate(
            borough, (pt.x, pt.y), fontsize=11.5, color="#898781",
            ha="center", style="italic", zorder=2,
        )

    ax.set_aspect(1 / np.cos(np.radians(_NYC_LAT)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(0.03)

    legend_elems = [
        Line2D([0], [0], color=_severity_hex(0.1), lw=5, label="Free-flowing (PM peak)"),
        Line2D([0], [0], color=_severity_hex(0.9), lw=5, label="Congested (PM peak)"),
        Line2D([0], [0], color="#B9B6AC", lw=5, linestyle=(0, (1, 1.5)), label="No PM-peak data (dead sensor)"),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="#E4572E", markeredgecolor="white",
            markersize=10,
            label=f"Congestion hotspot (worst {congestion_quantile:.0%} of segments)",
        ),
        Line2D(
            [0], [0], marker="x", color="#131917", linestyle="None",
            markersize=10, markeredgewidth=2.4,
            label=f"Sensor error hotspot (≥{error_threshold:.0%} error rate)",
        ),
    ]
    ax.legend(
        handles=legend_elems, loc="lower left", fontsize=10.5,
        frameon=True, facecolor="white", edgecolor="#D8D8D0",
    )

    ax.set_title(
        "NYC Traffic: PM-Peak Congestion & Sensor Error Hotspots",
        fontsize=16, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
