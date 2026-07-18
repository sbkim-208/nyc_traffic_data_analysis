import pandas as pd
from soobin.analysis.sensor_error import (
    get_sensor_error_rate_by_borough,
    get_most_error_prone_segments,
    get_error_prone_segments_by_borough,
)
from soobin.visualization.sensor_plots import (
    plot_borough_reliability,
    plot_borough_problem_roads,
    plot_borough_problem_roads_interactive,
)

DATA_PATH = "backend/data/raw/speeds_2024-04-01_2024-08-01.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    total = len(df)
    error_count = (df["status"] == -101).sum()
    reliability = (total - error_count) / total

    print(f"Total rows: {total:,}")
    print(f"status=-101 count: {error_count:,}")
    print(f"Sensor reliability: {reliability:.4f}")

    by_borough = get_sensor_error_rate_by_borough(df)
    print("\n=== Sensor error rate by borough ===")
    print(by_borough.to_string(index=False))

    print("\n=== Top 10 highest error-rate segments (roads) ===")
    print(get_most_error_prone_segments(df, top_n=10).to_string(index=False))

    by_borough_segments = get_error_prone_segments_by_borough(df, top_n=5, min_total=100)
    print("\n=== Top 5 highest error-rate segments by borough ===")
    print(by_borough_segments.to_string(index=False))

    plot_borough_reliability(by_borough, reliability)
    plot_borough_problem_roads(by_borough_segments)
    plot_borough_problem_roads_interactive(by_borough_segments)


if __name__ == "__main__":
    main()
