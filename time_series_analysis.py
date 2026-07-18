import pandas as pd
from soobin.cleaning.preprocess import (
    drop_duplicate_ids,
    remove_dead_segments,
    fill_missing_speed,
    fill_missing_travel_time,
)
from soobin.analysis.data_analysis import (
    add_tti,
    get_weekday_weekend_comparison,
    get_tti_si_by_borough_hour,
)
from soobin.visualization.traffic_plots import plot_tti_si_heatmap

DATA_PATH = "backend/data/raw/speeds_2024-04-01_2024-08-01.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    df = drop_duplicate_ids(df)
    df = remove_dead_segments(df)
    df = fill_missing_speed(df)
    df = fill_missing_travel_time(df)

    # Step 1 — convert to datetime and set as index
    df["data_as_of"] = pd.to_datetime(df["data_as_of"])
    df = df.set_index("data_as_of").sort_index()

    # Step 2 — resample (network-wide average speed)
    hourly = df["speed"].resample("1h").mean()
    daily = df["speed"].resample("D").agg(["mean", "count", "std"])
    print("=== Hourly average speed (head) ===")
    print(hourly.head(10).to_string())
    print("\n=== Daily speed aggregates (head) ===")
    print(daily.head(10).to_string())

    # Step 3 — rolling window (moving avg/std) + shift (week-over-week change)
    daily["mean_7d_avg"] = daily["mean"].rolling(window="7D").mean()
    daily["mean_14d_std"] = daily["mean"].rolling(window="14D").std()
    daily["change_vs_lastweek"] = daily["mean"] - daily["mean"].shift(7)
    print("\n=== After adding rolling/shift features (head 14) ===")
    print(daily.head(14).to_string())

    # Traffic domain application — find peak hour (lowest speed hour = most congested, instead of trip_count)
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    hourly_avg_speed = df.groupby("hour")["speed"].mean()
    am_peak = hourly_avg_speed[6:10].idxmin()
    pm_peak = hourly_avg_speed[15:20].idxmin()
    print(f"\nAM peak (most congested): {am_peak}:00, PM peak (most congested): {pm_peak}:00")
    print(hourly_avg_speed.to_string())

    # Peak hour split by borough x weekday/weekend
    comparison = get_weekday_weekend_comparison(df)
    print("\n=== Peak hour by borough/weekday-weekend (lowest-speed hour) ===")
    for (borough, day_type), g in comparison.groupby(["borough", "day_type"]):
        g = g.sort_values("hour")
        am = g[g["hour"].between(6, 9)].loc[lambda x: x["speed"].idxmin()]
        pm = g[g["hour"].between(15, 19)].loc[lambda x: x["speed"].idxmin()]
        print(f"  {borough:<14} {day_type:<8} AM peak {int(am['hour'])}:00({am['speed']}mph)  "
              f"PM peak {int(pm['hour'])}:00({pm['speed']}mph)")

    # plot_weekday_weekend_comparison(comparison)  # skipped when running the script since
    # plt.show() opens a GUI window and blocks forever if not closed (PNG is already saved)

    # Peak time vs free-flow (2-4am) SI comparison
    # method="night": uses the 2-4am average speed as each segment's free-flow baseline
    # (instead of add_tti's default max_speed quantile, using "no traffic hours" directly as the baseline)
    df = add_tti(df, method="night")
    free_flow_speed = hourly_avg_speed[[2, 3, 4]].mean()
    print(f"\n=== Network-wide: free-flow (2-4am) vs peak comparison ===")
    print(f"Free-flow (2-4am) average speed: {free_flow_speed:.2f}mph")
    for label, h in (("AM peak", am_peak), ("PM peak", pm_peak)):
        peak_speed = hourly_avg_speed[h]
        peak_si = df.loc[df["hour"] == h, "si"].mean()
        print(f"  {label}({h}:00): {peak_speed:.2f}mph  ({peak_speed / free_flow_speed:.1%} of free-flow)  si={peak_si:.3f}")

    tti_pivot, si_pivot = get_tti_si_by_borough_hour(df)
    print("\n=== SI by borough and hour (closer to free-flow=1.0 means less congested) ===")
    print(si_pivot.loc[[2, 3, 4, 6, 7, 8, 9, 15, 16, 17, 18, 19]].to_string())
    plot_tti_si_heatmap(tti_pivot, si_pivot)  # Plotly, so fig.show() opens a browser tab without blocking

    # Check with absolute mph whether Manhattan is "just slow at free-flow to begin with"
    print("\n=== Free-flow (2-4am) vs PM peak (4pm) absolute speed by borough (mph, weekday) ===")
    wd = comparison[comparison["day_type"] == "Weekday"]
    night_speed = wd[wd["hour"].isin([2, 3, 4])].groupby("borough")["speed"].mean().round(2)
    pm16_speed = wd[wd["hour"] == 16].groupby("borough")["speed"].mean().round(2)
    abs_compare = pd.DataFrame({"free_flow_mph": night_speed, "pm16_mph": pm16_speed})
    abs_compare["drop_mph"] = (abs_compare["free_flow_mph"] - abs_compare["pm16_mph"]).round(2)
    print(abs_compare.sort_values("free_flow_mph").to_string())


if __name__ == "__main__":
    main()
