import pandas as pd
from scipy import stats


def get_peak_hour_by_borough(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["borough", "hour", "is_weekend", "dow"])[["speed", "travel_time"]].mean().reset_index()


def get_borough_traffic(df: pd.DataFrame, borough: str) -> pd.DataFrame:
    result = get_peak_hour_by_borough(df)
    return result[result["borough"] == borough]


def get_speed_by_owner_segment(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["owner", "link_name", "hour"])["speed"].mean().reset_index()
    result = {}
    for owner, owner_group in grouped.groupby("owner"):
        result[owner] = {}
        for link_name, link_group in owner_group.groupby("link_name"):
            result[owner][link_name] = link_group
    return result


def get_segment(df: pd.DataFrame, owner: str, link_name: str) -> pd.DataFrame:
    return df[(df["owner"] == owner) & (df["link_name"] == link_name)]


def get_correlation_by_borough(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for borough in df["borough"].unique():
        borough_df = get_borough_traffic(df, borough)
        corr = borough_df[["hour", "speed", "travel_time"]].corr()
        result.append({
            "borough": borough,
            "speed_travel_time": corr.loc["speed", "travel_time"],
            "hour_speed": corr.loc["hour", "speed"],
            "hour_travel_time": corr.loc["hour", "travel_time"],
        })
    return pd.DataFrame(result)


def get_correlation_by_borough_id_hour(df: pd.DataFrame) -> pd.DataFrame:
    id_hour_mean = (
        df.groupby(["id", "hour", "borough"])[["speed", "travel_time"]]
        .mean()
        .reset_index()
    )
    result = []
    for borough, group in id_hour_mean.groupby("borough"):
        corr = group[["speed", "travel_time"]].corr()
        result.append({
            "borough": borough,
            "n": len(group),
            "speed_travel_time_corr": round(corr.loc["speed", "travel_time"], 4),
        })
    return pd.DataFrame(result)


def get_within_segment_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Speed-travel_time correlation over time within the same segment (id), removing the effect of segment length"""
    within = []
    for (seg_id, borough), g in df.groupby(["id", "borough"]):
        if len(g) < 5:
            continue
        sp = g["speed"].corr(g["travel_time"], method="spearman")
        within.append({
            "id": seg_id,
            "borough": borough,
            "n": len(g),
            "within_spearman": round(sp, 3),
        })

    within_df = pd.DataFrame(within)

    summary = (
        within_df.groupby("borough")["within_spearman"]
        .agg(mean="mean", median="median", min="min", max="max")
        .round(3)
        .reset_index()
    )

    seg = df.groupby(["id", "borough"])[["speed", "travel_time"]].mean().reset_index()
    cross = (
        seg.groupby("borough")
        .apply(lambda g: round(g["speed"].corr(g["travel_time"], method="spearman"), 3), include_groups=False)
        .rename("cross_spearman")
        .reset_index()
    )

    return summary.merge(cross, on="borough").sort_values("mean")


def get_segment_correlation_summary(df: pd.DataFrame) -> pd.DataFrame:
    seg = df.groupby(["id", "borough"])[["speed", "travel_time"]].mean().reset_index()

    result = []
    for borough, g in seg.groupby("borough"):
        spearman = g["speed"].corr(g["travel_time"], method="spearman")
        result.append({
            "borough": borough,
            "segment_count": len(g),
            "travel_time_std": round(g["travel_time"].std(), 1),
            "spearman": round(spearman, 3),
        })

    return pd.DataFrame(result).sort_values("spearman")


def compare_mean_median(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    if cols is None:
        cols = ["speed", "travel_time"]
    result = []
    for (borough, hour), group in df.groupby(["borough", "hour"]):
        for col in cols:
            values = group[col].dropna()
            mean = values.mean()
            median = values.median()
            result.append({
                "borough": borough,
                "hour": hour,
                "column": col,
                "mean": round(mean, 2),
                "median": round(median, 2),
                "mean_minus_median": round(mean - median, 2),
                "skewness": round(stats.skew(values), 3),
                "pct_below_mean": round((values < mean).mean(), 3),
            })
    return pd.DataFrame(result)


def check_normality_by_borough_hour(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    if cols is None:
        cols = ["speed", "travel_time"]
    result = []
    for (borough, hour), group in df.groupby(["borough", "hour"]):
        for col in cols:
            values = group[col].dropna()
            _, p_value = stats.shapiro(values)
            result.append({
                "borough": borough,
                "hour": hour,
                "column": col,
                "n": len(values),
                "mean": round(values.mean(), 2),
                "std": round(values.std(), 2),
                "skewness": round(stats.skew(values), 3),
                "kurtosis": round(stats.kurtosis(values), 3),
                "shapiro_p": round(p_value, 4),
                "is_normal": p_value > 0.05,
            })
    return pd.DataFrame(result)


def check_normality_after_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.groupby(["id", "hour", "borough"])[["speed", "travel_time"]]
        .mean()
        .reset_index()
    )
    result = []
    for borough, group in base.groupby("borough"):
        for col in ["speed", "travel_time"]:
            values = group[col].dropna()
            _, p_value = stats.shapiro(values)
            result.append({
                "borough": borough,
                "column": col,
                "n": len(values),
                "skewness": round(stats.skew(values), 3),
                "kurtosis": round(stats.kurtosis(values), 3),
                "shapiro_p": round(p_value, 4),
                "is_normal": p_value > 0.05,
            })
    return pd.DataFrame(result)
