from pathlib import Path

import numpy as np
import pandas as pd


def drop_duplicate_ids(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["link_id", "transcom_id", "encoded_poly_line", "encoded_poly_line_lvls"])


def remove_dead_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully-dead segments (by id) where every row has status=-101 — never once a valid reading"""
    dead_mask = (df["status"] == -101).groupby(df["id"]).transform("all")
    print(f"Removing fully-dead segments: {df.loc[dead_mask, 'id'].nunique()} segments, {dead_mask.sum():,} rows")
    return df[~dead_mask]


def _fill_with_source(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Fill col's status=-101 gaps per segment (id) via time interpolation -> ffill/bfill, in that order,
    and record whether each row is a real observation or a substituted value in `{col}_source`.

    Why distinguish them: an interpolated value is estimated between real measurements on
    either side, so it has some grounding, whereas an edge_filled value sits at the start/end
    of a segment with no anchor on one side at all, so ffill/bfill just pushes the nearest
    value straight through. Both count as "missing-value correction" but differ in
    reliability, so this lets later lag/rolling features or correlation calculations trace
    back how much of the data is actually measured.
    """
    missing = (df["status"] == -101) & (df[col] == 0)
    df.loc[missing, col] = None
    pre_na = df[col].isna().to_numpy()

    dt_index = pd.to_datetime(df["data_as_of"])
    df = df.set_index(dt_index)
    df[col] = df.groupby("id")[col].transform(lambda x: x.interpolate(method="time"))
    still_na = df[col].isna().to_numpy()
    interpolated = pre_na & ~still_na

    if still_na.any():
        df[col] = df.groupby("id")[col].transform(lambda x: x.ffill().bfill())
    edge_filled = still_na

    source = np.full(len(df), "observed", dtype=object)
    source[interpolated] = "interpolated"
    source[edge_filled] = "edge_filled"
    df[f"{col}_source"] = pd.Categorical(
        source, categories=["observed", "interpolated", "edge_filled"]
    )

    return df.reset_index(drop=True)


def fill_missing_speed(df: pd.DataFrame) -> pd.DataFrame:
    return _fill_with_source(df, "speed")


def get_imputation_summary(df: pd.DataFrame, source_col: str = "speed_source") -> dict:
    """Summarize the share of values that are not real observations (interpolated/edge-filled) after preprocessing

    Unlike the stuck-run diagnostic (ml_feature_analysis.py), which only catches segments
    "frozen after interpolation," this captures the full substituted-value share including
    interpolated values that vary normally without freezing — because not being frozen
    doesn't mean it's a real measurement.
    """
    counts = df[source_col].value_counts()
    total = len(df)
    observed = int(counts.get("observed", 0))
    interpolated = int(counts.get("interpolated", 0))
    edge_filled = int(counts.get("edge_filled", 0))

    by_segment = (
        df.assign(_imputed=df[source_col] != "observed")
        .groupby("id")["_imputed"].mean()
        .sort_values(ascending=False)
    )

    return {
        "total_rows": total,
        "observed": observed,
        "interpolated": interpolated,
        "edge_filled": edge_filled,
        "interpolated_rate": round(interpolated / total, 4),
        "edge_filled_rate": round(edge_filled / total, 4),
        "imputed_rate": round((interpolated + edge_filled) / total, 4),
        "top_imputed_segments": by_segment.head(10),
    }


def load_and_clean(csv_paths: list[str], cache_path: str, force: bool = False) -> pd.DataFrame:
    """Combine csv_paths, clean through drop_duplicate_ids -> remove_dead_segments ->
    fill_missing_speed, save the result to cache_path (parquet), and read only the cache on subsequent calls.

    A helper to avoid the waste of every analysis script re-reading multi-GB raw CSVs and
    repeating the same cleaning. force=True ignores the cache and rebuilds from the raw source.
    """
    cache = Path(cache_path)
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        df["data_as_of"] = pd.to_datetime(df["data_as_of"])
        return df

    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    df = drop_duplicate_ids(df)
    df = remove_dead_segments(df)
    df = fill_missing_speed(df)
    df["data_as_of"] = pd.to_datetime(df["data_as_of"])

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def fill_missing_travel_time(df: pd.DataFrame) -> pd.DataFrame:
    return _fill_with_source(df, "travel_time")
