import warnings

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# Full list of Manhattan connection roads (route_name, group_label, link_name_prefix)
_MANHATTAN_BOUND_ROUTES = [
    ("Queens-Midtown Tunnel",   "Queens->Manhattan",       "QMT"),
    ("Brooklyn Bridge",          "Brooklyn->Manhattan",     "BKN Bridge"),
    ("Hugh L. Carey Tunnel",     "Brooklyn->Manhattan",     "BBT"),
    ("RFK Bridge (Triborough)",  "Queens/Bronx->Manhattan", "TBB"),
    ("Major Deegan Expwy",       "Bronx->Manhattan",        "MDE"),
    ("Cross Bronx Expwy",        "Bronx->Manhattan",        "CBE"),
    ("Geo. Washington Bridge",   "Bronx/NJ->Manhattan",     "GWB"),
    ("Lincoln Tunnel",           "NJ->Manhattan",           "LINCOLN TUNNEL"),
    ("FDR Drive",                "Manhattan",              "FDR"),
]

# Roads not present in the data
_MANHATTAN_MISSING = [
    ("Queensboro Bridge (59th St)", "Queens->Manhattan"),
    ("Williamsburg Bridge",         "Queens/Brooklyn->Manhattan"),
    ("Manhattan Bridge",            "Brooklyn->Manhattan"),
    ("Madison Ave Bridge",          "Bronx->Manhattan"),
    ("3rd Avenue Bridge",           "Bronx->Manhattan"),
    ("Willis Avenue Bridge",        "Bronx->Manhattan"),
    ("Harlem River Drive",          "Bronx->Manhattan"),
]

# Leader roads: tunnels/bridges (link_name prefixes)
_LEADER_PREFIXES = {
    "Lincoln Tunnel":          "LINCOLN TUNNEL",
    "Queens-Midtown Tunnel":   "QMT",
    "Hugh L. Carey Tunnel":    "BBT",
    "Brooklyn Bridge":         "BKN Bridge",
    "RFK Bridge (Triborough)": "TBB",
    "Major Deegan Expwy":      "MDE",
    "Cross Bronx Expwy":       "CBE",
    "Geo. Washington Bridge":  "GWB",
}

# Follower roads: Manhattan interior (11th/12th Ave, FDR, West St)
_FOLLOWER_PREFIXES = {
    "11th/12th Ave (N)": ["11th ave n", "12th Ave N"],
    "11th/12th Ave (S)": ["12th Ave S", "12th ave @"],
    "FDR (N)":           ["FDR N"],
    "FDR (S)":           ["FDR S"],
    "West St":           ["West St"],
}


def resample_at(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Resample a preprocessed df to the given time resolution (freq)

    Floors the data_as_of timestamp to freq, then returns the mean speed/travel_time per
    (timestamp, id, link_name, borough) group. Includes tti/si columns if present.
    freq: a pandas offset string ("5min", "10min", "30min", "1h", etc.)

    If the `{speed,travel_time}_source` columns left by preprocess.py's
    fill_missing_speed/fill_missing_travel_time are present, also aggregates the "not
    observed" fraction per bucket as `{col}_imputed_frac` — so downstream lag correlation
    calculations can carry forward how much a bucket's average relied on
    interpolation/edge-fill versus real measurements.
    """
    tmp = df.copy()
    tmp["timestamp"] = tmp["data_as_of"].dt.floor(freq)
    agg_cols = ["speed", "travel_time"]
    for col in ("tti", "si"):
        if col in tmp.columns:
            agg_cols.append(col)

    group_cols = ["timestamp", "id", "link_name", "borough"]
    out = tmp.groupby(group_cols)[agg_cols].mean().reset_index()

    for value_col, source_col in (("speed", "speed_source"), ("travel_time", "travel_time_source")):
        if source_col in tmp.columns:
            frac_col = f"{value_col}_imputed_frac"
            frac = (
                tmp.assign(_imputed=tmp[source_col] != "observed")
                .groupby(group_cols)["_imputed"]
                .mean()
                .rename(frac_col)
                .reset_index()
            )
            out = out.merge(frac, on=group_cols, how="left")

    return out


def resample_5min(df: pd.DataFrame) -> pd.DataFrame:
    """5-minute resampling — backward-compatible alias for resample_at(df, "5min")"""
    return resample_at(df, "5min")


def _build_road_series(
    resampled: pd.DataFrame,
    prefix_map: dict,
    value_col: str = "speed",
) -> pd.DataFrame:
    """Return the average value_col time series per prefix group

    prefix_map: {label: prefix_str or [prefix_str, ...]}
    value_col: column to aggregate ("speed", "si", etc. — must be present in resample_5min's output)
    Returns: index=timestamp, columns=label
    """
    series = {}
    for label, pref in prefix_map.items():
        if isinstance(pref, str): # check whether this is a string type
            pref = [pref]
        mask = resampled["link_name"].apply(
            lambda x: any(x.startswith(p) for p in pref)
        )
        sub = resampled[mask]
        if sub.empty:
            continue
        series[label] = sub.groupby("timestamp")[value_col].mean()
    return pd.DataFrame(series).sort_index()


def get_lag_correlation(
    resampled: pd.DataFrame,
    value_col: str = "speed",
    max_lag_min: int = 60,
) -> pd.DataFrame:
    """Analyze delayed correlation from leader (tunnel/bridge) to follower (Manhattan interior)

    Computes corr(leader[t], follower[t + lag]) for lag = 0, 5, ..., max_lag_min.
    Higher means: when leader-road speed is high, follower-road speed is also high lag
    minutes later (conversely: leader congestion -> follower congestion lag minutes later).

    value_col: "speed" or "si" — si is a per-segment ratio against free-flow, so use it
    when you want to compare with segment-length and free-flow-speed differences removed.

    If resampled has the `{value_col}_imputed_frac` produced by resample_at(), this also
    returns leader_imputed_frac/follower_imputed_frac — how much, on average, the
    leader/follower data relied on interpolation/edge-fill during the common window
    actually used for comparison — so you can tell whether a correlation is driven mostly
    by real measurements or largely by correlated imputed values.

    Return columns: leader, follower, lag_min, pearson_r
    (adds leader_imputed_frac, follower_imputed_frac if the input has imputed_frac columns)
    """
    leader_ts   = _build_road_series(resampled, _LEADER_PREFIXES, value_col)
    follower_ts = _build_road_series(resampled, _FOLLOWER_PREFIXES, value_col)

    frac_col = f"{value_col}_imputed_frac"
    has_frac = frac_col in resampled.columns
    if has_frac:
        leader_frac   = _build_road_series(resampled, _LEADER_PREFIXES, frac_col)
        follower_frac = _build_road_series(resampled, _FOLLOWER_PREFIXES, frac_col)

    common_idx  = leader_ts.index.intersection(follower_ts.index)
    leader_ts   = leader_ts.loc[common_idx]
    follower_ts = follower_ts.loc[common_idx]

    steps = max_lag_min // 5
    results = []

    for leader in leader_ts.columns:
        for follower in follower_ts.columns:
            l_s = leader_ts[leader].dropna()
            f_s = follower_ts[follower].dropna()
            common = l_s.index.intersection(f_s.index)
            if len(common) < 50:
                continue
            l = l_s.loc[common]
            f = f_s.loc[common]

            leader_frac_avg = follower_frac_avg = None
            if has_frac:
                leader_frac_avg = round(float(leader_frac[leader].reindex(common).mean()), 4)
                follower_frac_avg = round(float(follower_frac[follower].reindex(common).mean()), 4)

            for step in range(steps + 1):
                lag_min = step * 5
                # f.shift(-step): pull the time axis forward -> compare f[t+lag] against l[t]
                f_shifted = f.shift(-step)
                valid = l.index[f_shifted.notna()]
                if len(valid) < 50:
                    continue
                r = l.loc[valid].corr(f_shifted.loc[valid])
                row = {
                    "leader":    leader,
                    "follower":  follower,
                    "lag_min":   lag_min,
                    "pearson_r": round(r, 4),
                }
                if has_frac:
                    row["leader_imputed_frac"] = leader_frac_avg
                    row["follower_imputed_frac"] = follower_frac_avg
                results.append(row)

    return pd.DataFrame(results)


def get_best_lag_summary(lag_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the best lag for each (leader, follower) pair

    Return columns:
      leader, follower, best_lag_min, best_r, lag_0_r, r_gain
      (passes through leader_imputed_frac/follower_imputed_frac if get_lag_correlation
      returned them — the value is fixed per pair, so just taking the first row is enough)
    """
    has_frac = "leader_imputed_frac" in lag_df.columns
    rows = []
    for (leader, follower), g in lag_df.groupby(["leader", "follower"]):
        best_idx  = g["pearson_r"].idxmax()
        lag0_vals = g[g["lag_min"] == 0]["pearson_r"].values
        lag0_r    = round(float(lag0_vals[0]), 4) if len(lag0_vals) else None
        best_lag  = int(g.loc[best_idx, "lag_min"])
        best_r    = g.loc[best_idx, "pearson_r"]
        r_gain    = round(best_r - lag0_r, 4) if lag0_r is not None else None
        row = {
            "leader":       leader,
            "follower":     follower,
            "best_lag_min": best_lag,
            "best_r":       best_r,
            "lag_0_r":      lag0_r,
            "r_gain":       r_gain,   # correlation improvement from introducing the lag
        }
        if has_frac:
            row["leader_imputed_frac"] = g["leader_imputed_frac"].iloc[0]
            row["follower_imputed_frac"] = g["follower_imputed_frac"].iloc[0]
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["leader", "best_lag_min"])
        .reset_index(drop=True)
    )


def get_stationarity_report(
    resampled: pd.DataFrame,
    value_col: str = "speed",
) -> pd.DataFrame:
    """ADF stationarity test results for each leader road

    p_value < 0.05 means no unit root -> stationarity confirmed.
    Return columns: road, adf_stat, p_value, is_stationary
    """
    leader_ts = _build_road_series(resampled, _LEADER_PREFIXES, value_col)
    rows = []
    for road in leader_ts.columns:
        s = leader_ts[road].interpolate("time").dropna()
        if len(s) < 50:
            continue
        adf_stat, p_val, *_ = adfuller(s, autolag="AIC")
        rows.append({
            "road":          road,
            "adf_stat":      round(adf_stat, 4),
            "p_value":       round(p_val, 4),
            "is_stationary": bool(p_val < 0.05),
        })
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def get_granger_causality(
    resampled: pd.DataFrame,
    value_col: str = "speed",
    max_lag_steps: int = 6,
) -> pd.DataFrame:
    """Test whether a leader road Granger-causally predicts follower-road speed

    The Pearson correlation from get_lag_correlation() can come out high just because
    two series share the same daily pattern (spurious correlation). The Granger causality
    test compares "a model predicting from the follower road's own past values" against
    "a model predicting from the follower's plus the leader road's past values," using an
    F-test to check whether the leader road's past values actually add predictive power.

    If either series fails the ADF stationarity test (p >= 0.05), both are first-order
    differenced before testing — to avoid the spurious-regression problem where
    non-stationary series can show significant results even without a real causal
    relationship.

    Tests lag=1..max_lag_steps (in 5-minute units) and adopts the lag with the smallest
    p-value as the representative result for that (leader, follower) pair.

    Return columns: leader, follower, best_lag_min, p_value, significant, differenced
    """
    leader_ts   = _build_road_series(resampled, _LEADER_PREFIXES, value_col)
    follower_ts = _build_road_series(resampled, _FOLLOWER_PREFIXES, value_col)

    common_idx  = leader_ts.index.intersection(follower_ts.index)
    leader_ts   = leader_ts.loc[common_idx]
    follower_ts = follower_ts.loc[common_idx]

    min_len = 50 + max_lag_steps
    rows = []

    for leader in leader_ts.columns:
        for follower in follower_ts.columns:
            l = leader_ts[leader].interpolate("time").dropna()
            f = follower_ts[follower].interpolate("time").dropna()
            common = l.index.intersection(f.index)
            if len(common) < min_len:
                continue
            l, f = l.loc[common], f.loc[common]

            _, l_p, *_ = adfuller(l, autolag="AIC")
            _, f_p, *_ = adfuller(f, autolag="AIC")
            differenced = (l_p >= 0.05) or (f_p >= 0.05)
            if differenced:
                l = l.diff().dropna()
                f = f.diff().dropna()
                common = l.index.intersection(f.index)
                l, f = l.loc[common], f.loc[common]
                if len(common) < min_len:
                    continue

            # grangercausalitytests(data, maxlag): tests whether data's 2nd column
            # Granger-causes the 1st, so we pass in [follower, leader] order.
            data = pd.concat(
                [f.rename("follower"), l.rename("leader")], axis=1
            ).dropna()

            try:
                with warnings.catch_warnings():
                    # verbose=False silences per-lag console output; statsmodels
                    # emits a FutureWarning about the (unrelated) print behavior
                    # being deprecated, which we don't need to surface here.
                    warnings.simplefilter("ignore", FutureWarning)
                    result = grangercausalitytests(
                        data.to_numpy(), maxlag=max_lag_steps, verbose=False
                    )
            except (ValueError, np.linalg.LinAlgError):
                continue

            best_lag, best_p = min(
                ((lag, res[0]["ssr_ftest"][1]) for lag, res in result.items()),
                key=lambda item: item[1],
            )
            rows.append({
                "leader":       leader,
                "follower":     follower,
                "best_lag_min": best_lag * 5,
                "p_value":      round(best_p, 4),
                "significant":  bool(best_p < 0.05),
                "differenced":  differenced,
            })

    return (
        pd.DataFrame(rows)
        .sort_values(["leader", "p_value"])
        .reset_index(drop=True)
    )


def get_stl_components(
    resampled: pd.DataFrame,
    leader: str,
    value_col: str = "speed",
    period: int = 288,
) -> dict:
    """STL decomposition for a specific leader road

    period: one day at 5-minute intervals = 288 (default)
    robust=True performs a decomposition robust to outliers

    Returns: {"original", "trend", "seasonal", "residual"}
    detrended    = original - trend    (seasonal + residual)
    deseasoned   = original - seasonal (trend + residual)
    """
    leader_ts = _build_road_series(resampled, _LEADER_PREFIXES, value_col)
    if leader not in leader_ts.columns:
        raise ValueError(f"'{leader}' not found. Available: {list(leader_ts.columns)}")
    s = leader_ts[leader].interpolate("time").dropna()
    res = STL(s, period=period, robust=True).fit()
    return {
        "original": s,
        "trend":    res.trend,
        "seasonal": res.seasonal,
        "residual": res.resid,
    }


def get_manhattan_connection_si(df: pd.DataFrame) -> pd.DataFrame:
    """Return average SI by hour for each Manhattan connection road

    Return columns: route_name, group_label, hour, day_type, si
    """
    label_map = {0: "Weekday", 1: "Weekend"}
    tmp = df.copy()
    tmp["day_type"] = tmp["is_weekend"].map(label_map)

    rows = []
    for route_name, group_label, prefix in _MANHATTAN_BOUND_ROUTES:
        subset = tmp[tmp["link_name"].str.startswith(prefix)]
        if subset.empty:
            continue
        agg = (
            subset.groupby(["day_type", "hour"])["si"]
            .mean()
            .reset_index()
        )
        agg["route_name"] = route_name
        agg["group_label"] = group_label
        rows.append(agg)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
