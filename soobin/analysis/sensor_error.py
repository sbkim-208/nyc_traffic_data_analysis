import pandas as pd


def get_sensor_error_rate_by_borough(df: pd.DataFrame) -> pd.DataFrame:
    """Sensor error (status=-101) rate by borough"""
    total = df.groupby("borough").size().rename("total")
    errors = df[df["status"] == -101].groupby("borough").size().rename("error_count")
    result = pd.concat([total, errors], axis=1).fillna(0).astype({"error_count": int})
    result["error_rate"] = (result["error_count"] / result["total"]).round(4)
    return result.reset_index().sort_values("error_rate", ascending=False)


def get_sensor_error_rate_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Sensor error rate by hour"""
    total = df.groupby("hour").size().rename("total")
    errors = df[df["status"] == -101].groupby("hour").size().rename("error_count")
    result = pd.concat([total, errors], axis=1).fillna(0).astype({"error_count": int})
    result["error_rate"] = (result["error_count"] / result["total"]).round(4)
    return result.reset_index().sort_values("hour")


def get_sensor_error_by_borough_hour(df: pd.DataFrame) -> pd.DataFrame:
    """borough x hour error count pivot table"""
    errors = df[df["status"] == -101].groupby(["borough", "hour"]).size().rename("error_count")
    total = df.groupby(["borough", "hour"]).size().rename("total")
    combined = pd.concat([total, errors], axis=1).fillna(0).astype({"error_count": int})
    combined["error_rate"] = (combined["error_count"] / combined["total"]).round(4)
    return combined.reset_index()


def get_sensor_error_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """borough x hour error rate pivot (for heatmap)"""
    detail = get_sensor_error_by_borough_hour(df)
    return detail.pivot(index="hour", columns="borough", values="error_rate")


def get_most_error_prone_segments(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top N segments (id) by error rate"""
    total = df.groupby(["id", "borough", "link_name"]).size().rename("total")
    errors = df[df["status"] == -101].groupby(["id", "borough", "link_name"]).size().rename("error_count")
    result = pd.concat([total, errors], axis=1).fillna(0).astype({"error_count": int})
    result["error_rate"] = (result["error_count"] / result["total"]).round(4)
    return result.reset_index().sort_values("error_rate", ascending=False).head(top_n)


def get_error_prone_segments_by_borough(
    df: pd.DataFrame, top_n: int = 5, min_total: int = 100
) -> pd.DataFrame:
    """Top N error-rate segments by borough (excludes noisy segments with fewer than min_total samples)"""
    total = df.groupby(["borough", "id", "link_name"]).size().rename("total")
    errors = df[df["status"] == -101].groupby(["borough", "id", "link_name"]).size().rename("error_count")
    result = pd.concat([total, errors], axis=1).fillna(0).astype({"error_count": int})
    result["error_rate"] = (result["error_count"] / result["total"]).round(4)
    result = result[result["total"] >= min_total].reset_index()
    return (
        result.sort_values(["borough", "error_rate"], ascending=[True, False])
        .groupby("borough")
        .head(top_n)
        .reset_index(drop=True)
    )


def get_sensor_error_summary(df: pd.DataFrame) -> dict:
    """Overall sensor error summary statistics"""
    total = len(df)
    error_rows = (df["status"] == -101).sum()
    error_segments = df[df["status"] == -101]["id"].nunique()
    clean_segments = df[df["status"] != -101]["id"].nunique()

    return {
        "total_rows": total,
        "error_rows": int(error_rows),
        "overall_error_rate": round(error_rows / total, 4),
        "segments_with_errors": int(error_segments),
        "clean_segments": int(clean_segments),
        "most_error_hour": int(
            df[df["status"] == -101].groupby("hour").size().idxmax()
        ),
        "most_error_borough": (
            df[df["status"] == -101].groupby("borough").size().idxmax()
        ),
    }
