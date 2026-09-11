import pandas as pd
from soobin.cleaning.preprocess import drop_duplicate_ids
from soobin.analysis.geo import build_segment_geodataframe, add_congestion_and_error, build_congestion_error_map
from soobin.visualization.map_plots import plot_congestion_error_map

DATA_PATH = "backend/data/raw/speeds_2024-04-01_2024-08-01.csv"
ERROR_THRESHOLD = 0.5


def main():
    df = pd.read_csv(DATA_PATH)
    df = drop_duplicate_ids(df)

    gdf = build_segment_geodataframe(df)
    gdf = add_congestion_and_error(gdf, df)

    n_no_pm_data = gdf["si"].isna().sum()
    n_hotspots = (gdf["error_rate"] >= ERROR_THRESHOLD).sum()
    print(f"Segments: {len(gdf)}")
    print(f"Segments with no PM-peak readings (excluded from map): {n_no_pm_data}")
    print(f"Sensor error hotspots (error_rate >= {ERROR_THRESHOLD:.0%}): {n_hotspots}")

    path = plot_congestion_error_map(gdf, error_threshold=ERROR_THRESHOLD)
    print(f"\nStatic map saved: {path}")

    m = build_congestion_error_map(gdf, error_threshold=ERROR_THRESHOLD)
    interactive_path = "outputs/congestion_error_map.html"
    m.save(interactive_path)
    print(f"Interactive map saved: {interactive_path}")


if __name__ == "__main__":
    main()
