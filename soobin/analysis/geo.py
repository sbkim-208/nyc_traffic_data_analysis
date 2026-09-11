import folium
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

from soobin.analysis.data_analysis import add_tti
from soobin.analysis.sensor_error import get_segment_error_rates


_NYC_BOUNDS = (40.45, 40.95, -74.30, -73.65)  # lat_min, lat_max, lon_min, lon_max


def parse_link_points(s: str) -> list[tuple[float, float]] | None:
    """NYC link_points format: 'lat,lon lat,lon ...' (space-separated pairs).

    Some raw rows have a link_points string truncated at the source, in Socrata itself
    (e.g. "...40.78058,-7"). We discard everything from the truncation point onward and
    keep only the coordinates parsed successfully before it — if fewer than 2 points
    remain (i.e. it was truncated from the start), the whole segment is dropped.
    """
    if not isinstance(s, str) or not s.strip():
        return None
    lat_min, lat_max, lon_min, lon_max = _NYC_BOUNDS
    out = []
    for token in s.split():
        try:
            lat_str, lon_str = token.split(",")
            lat, lon = float(lat_str), float(lon_str)
        except (ValueError, IndexError):
            break
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            break
        out.append((lat, lon))
    return out if len(out) >= 2 else None


def build_segment_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame with a representative geometry (LineString) per segment (id)

    link_points doesn't change over time for a given segment, so only the first row per
    id is used. Coordinates are stored as (lat, lon) in the source, so they're flipped
    into the GIS-standard (lon, lat) order for the geometry.
    """
    segments = df.drop_duplicates(subset="id")[["id", "borough", "link_name", "link_points"]].copy()
    segments["coords"] = segments["link_points"].apply(parse_link_points)
    dropped = segments["coords"].isna().sum()
    if dropped:
        print(f"Segments excluded due to link_points parse failure: {dropped}")
    segments = segments[segments["coords"].notna()]
    segments["geometry"] = segments["coords"].apply(
        lambda pts: LineString([(lon, lat) for lat, lon in pts])
    )
    return gpd.GeoDataFrame(
        segments.drop(columns=["coords", "link_points"]),
        geometry="geometry",
        crs="EPSG:4326",
    )


_BOROUGH_COLOR = {
    "Manhattan": "red",
    "Brooklyn": "blue",
    "Queens": "green",
    "Bronx": "orange",
    "Staten Island": "purple",
}


def build_segment_map(gdf: gpd.GeoDataFrame) -> folium.Map:
    """Render the segment GeoDataFrame via GeoDataFrame.explore() (colored by borough)

    explore() is geopandas' folium integration method — it builds the GeoJson layer,
    categorical coloring, legend, and tooltip appropriate to the geometry type directly.
    This makes actual use of geopandas rather than manually extracting coordinates and
    drawing folium.PolyLine one at a time.
    """
    boroughs = sorted(gdf["borough"].unique())
    colors = [_BOROUGH_COLOR.get(b, "gray") for b in boroughs]
    return gdf.explore(
        column="borough",
        categories=boroughs,
        cmap=colors,
        tooltip=["borough", "link_name"],
        tiles="cartodbpositron",
        style_kwds={"weight": 4, "opacity": 0.8},
    )


def build_congestion_error_map(
    gdf: gpd.GeoDataFrame,
    error_threshold: float = 0.5,
    congestion_quantile: float = 0.25,
) -> folium.Map:
    """Interactive counterpart to soobin.visualization.map_plots.plot_congestion_error_map

    Three things share one map without competing for the same channel:
      - borough (identity): line color, the same fixed 5-color mapping as build_segment_map, so
        "which borough" reads at a glance instead of needing 30+ per-road hues (one per unique
        link_name) that would fail every CVD-safety check a categorical palette has to pass.
      - congestion (magnitude): line weight — thicker means lower PM-peak si (more congested) —
        plus the exact si value in the hover tooltip and a marker on the worst quartile of segments.
      - sensor error (a second, independent hotspot layer): a black marker, kept a different shape
        from the congestion marker so the two "check this spot" layers never get mistaken for
        each other. Segments with no PM-peak reading at all (si is NaN — a dead sensor) are drawn
        thin and dashed rather than dropped, since those are exactly the network's most severe
        error hotspots.

    gdf must already carry si/error_rate columns (see add_congestion_and_error).
    """
    gdf = gdf.copy()
    has_si = gdf["si"].notna()
    congestion_cutoff = gdf["si"].quantile(congestion_quantile)
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)

    m = folium.Map(tiles="cartodbpositron")
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    for _, row in gdf.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        color = _BOROUGH_COLOR.get(row["borough"], "gray")
        if pd.notna(row["si"]):
            severity = min(max(1 - row["si"], 0.0), 1.0)
            weight = 2 + severity * 6  # thicker line = more congested
            dash_array, opacity = None, 0.85
            si_label = f"{row['si']:.2f}"
        else:
            weight, dash_array, opacity = 2.5, "4,4", 0.6
            si_label = "no PM-peak data"
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=opacity,
            dash_array=dash_array,
            tooltip=(
                f"<b>{row['link_name']}</b> ({row['borough']})<br>"
                f"PM-peak SI: {si_label}<br>"
                f"Error rate: {row['error_rate']:.0%}"
            ),
        ).add_to(m)

    def _midpoints(rows: gpd.GeoDataFrame) -> gpd.GeoSeries:
        return rows.to_crs(epsg=32618).geometry.interpolate(0.5, normalized=True).to_crs(epsg=4326)

    congested = gdf[has_si & (gdf["si"] <= congestion_cutoff)]
    for (_, row), point in zip(congested.iterrows(), _midpoints(congested)):
        folium.CircleMarker(
            location=(point.y, point.x), radius=6,
            color="white", weight=1.5, fill=True, fill_color="#E4572E", fill_opacity=0.95,
            tooltip=f"Congestion hotspot — {row['link_name']} (SI {row['si']:.2f})",
        ).add_to(m)

    hotspots = gdf[gdf["error_rate"] >= error_threshold]
    for (_, row), point in zip(hotspots.iterrows(), _midpoints(hotspots)):
        folium.CircleMarker(
            location=(point.y, point.x), radius=6,
            color="white", weight=1.5, fill=True, fill_color="#131917", fill_opacity=0.95,
            tooltip=f"Sensor error hotspot — {row['link_name']} ({row['error_rate']:.0%} error rate)",
        ).add_to(m)

    boroughs_in_view = [b for b in _BOROUGH_COLOR if b in gdf["borough"].unique()]
    legend_rows = "".join(
        f'<div><span style="background:{_BOROUGH_COLOR[b]}"></span>{b}</div>' for b in boroughs_in_view
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: white; padding: 10px 14px; border: 1px solid #D8D8D0;
                border-radius: 6px; font-family: sans-serif; font-size: 12.5px; color: #131917;
                line-height: 1.5;">
      <div style="font-weight: 600; margin-bottom: 4px;">Borough (line color)</div>
      {legend_rows}
      <div style="font-weight: 600; margin: 8px 0 4px;">Line weight = PM-peak congestion</div>
      <div>thin → thick = free-flowing → congested</div>
      <div style="margin-top: 8px;">
        <span style="background:#E4572E; border-radius:50%;"></span>
        Congestion hotspot (worst {congestion_quantile:.0%})
      </div>
      <div>
        <span style="background:#131917; border-radius:50%;"></span>
        Sensor error hotspot (&ge;{error_threshold:.0%} error rate)
      </div>
      <style>
        div span {{ display:inline-block; width:10px; height:10px; margin-right:6px; vertical-align:middle; }}
      </style>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def add_congestion_and_error(
    gdf: gpd.GeoDataFrame, df: pd.DataFrame, pm_peak_hours: range = range(15, 20)
) -> gpd.GeoDataFrame:
    """Attach per-segment PM-peak congestion (si) and overall sensor error rate (error_rate) to gdf

    si: mean Speed Index (free-flow = each segment's own 2-4am average speed) restricted to
    pm_peak_hours — lower means more congested during the evening peak. error_rate: share of all
    rows (any hour) with status == -101, i.e. how often the sensor itself fails, independent of
    traffic conditions. Segments with no PM-peak readings get si = NaN.
    """
    df = df.assign(hour=pd.to_datetime(df["data_as_of"]).dt.hour)
    si_by_id = (
        add_tti(df, method="night")
        .pipe(lambda d: d[d["hour"].isin(pm_peak_hours)])
        .groupby("id")["si"]
        .mean()
        .rename("si")
    )
    error_rate_by_id = get_segment_error_rates(df).set_index("id")["error_rate"]

    out = gdf.set_index("id")
    out["si"] = si_by_id
    out["error_rate"] = error_rate_by_id
    return out.reset_index()


def build_adjacency(gdf: gpd.GeoDataFrame, threshold_m: float = 60.0) -> tuple[np.ndarray, list]:
    """Build an adjacency matrix, treating two segments (id) as physically connected if their LineString endpoints are within threshold_m

    For use as STGNN's graph input. EPSG:4326 (lat/lon, in degrees) is inaccurate for
    distance calculations, so we reproject to UTM 18N (EPSG:32618, in meters) before
    judging by the minimum distance between endpoints.
    Returns: (adj, ids) — adj[i,j]=1 means segments ids[i]-ids[j] are adjacent (diagonal is 0, no self-loops)
    """
    proj = gdf.to_crs(epsg=32618)
    ids = proj["id"].tolist()
    endpoints = [
        (Point(geom.coords[0]), Point(geom.coords[-1])) for geom in proj.geometry
    ]
    n = len(ids)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            min_dist = min(a.distance(b) for a in endpoints[i] for b in endpoints[j])
            if min_dist <= threshold_m:
                adj[i, j] = adj[j, i] = 1.0
    return adj, ids
