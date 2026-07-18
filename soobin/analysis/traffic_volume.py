import pandas as pd


def get_hourly_volume_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze row count, unique timestamps, and active segment count by hour"""
    total = df.groupby("hour").size().rename("data_count")
    timestamps = df.groupby("hour")["data_as_of"].nunique().rename("unique_timestamps")
    segments = df.groupby("hour")["id"].nunique().rename("active_segments")
    interval = (60 / timestamps).round(1).rename("avg_interval_min")

    return (
        pd.concat([total, timestamps, segments, interval], axis=1)
        .reset_index()
        .sort_values("hour")
    )


def get_hourly_volume_by_borough(df: pd.DataFrame) -> pd.DataFrame:
    """Unique timestamp count by borough and hour (measurement frequency as a traffic volume proxy)"""
    return (
        df.groupby(["borough", "hour"])["data_as_of"]
        .nunique()
        .rename("unique_timestamps")
        .reset_index()
        .sort_values(["borough", "hour"])
    )
