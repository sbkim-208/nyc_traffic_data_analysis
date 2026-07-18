import pandas as pd
from soobin.cleaning.preprocess import drop_duplicate_ids
from soobin.analysis.geo import build_segment_geodataframe, build_segment_map

DATA_PATH = "backend/data/raw/speeds_2024-04-01_2024-08-01.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    df = drop_duplicate_ids(df)
    # The map only shows road locations, so we skip the speed sensor reliability
    # filter (remove_dead_segments) — a dead sensor doesn't make the road coordinates invalid

    gdf = build_segment_geodataframe(df)
    print(f"\nSegment count: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    print(f"Total bounds (lon_min, lat_min, lon_max, lat_max): {gdf.total_bounds}")

    print("\n=== Segment count by borough ===")
    print(gdf["borough"].value_counts().to_string())

    print("\n=== head ===")
    print(gdf.head().to_string())

    m = build_segment_map(gdf)
    m.save("outputs/segment_map.html")
    print("\nMap saved: outputs/segment_map.html")


if __name__ == "__main__":
    main()
