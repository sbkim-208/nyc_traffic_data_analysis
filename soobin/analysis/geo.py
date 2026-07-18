import folium
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point


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
