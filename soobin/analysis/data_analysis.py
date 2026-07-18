import pandas as pd


def add_tti(df: pd.DataFrame, method: str = "max_speed") -> pd.DataFrame:
    """Add a TTI (Travel Time Index) column to df

    TTI = actual travel time / free-flow travel time
    The length variable cancels out in numerator and denominator, leaving pure congestion.

    method:
      'max_speed' : per-segment shortest travel time (= peak-speed moment) as the free-flow baseline (default)
      'night'     : per-segment average travel time during 2-4am as the free-flow baseline
    """
    if method == "max_speed":
        free_flow_tt = (
            df[df["travel_time"] > 0]
            .groupby("id")["travel_time"]
            .quantile(0.05) # bottom 5% travel time
            .rename("free_flow_tt")
        )
        free_flow_sp = (
            df[df["speed"] > 0]
            .groupby("id")["speed"]
            .quantile(0.95) # top 95th percentile speed
            .rename("free_flow_sp")
        )
    else:
        free_flow_tt = (
            df[df["hour"].between(2, 4)]
            .groupby("id")["travel_time"]
            .mean()
            .rename("free_flow_tt")
        )
        free_flow_sp = (
            df[df["hour"].between(2, 4)] # filter to 2-4am
            .groupby("id")["speed"]
            .mean()
            .rename("free_flow_sp")
        )
    df = df.join(free_flow_tt, on="id").join(free_flow_sp, on="id")
    valid_tt = df["travel_time"] > 0
    valid_sp = df["speed"] > 0
    df["tti"] = df["travel_time"].where(valid_tt) / df["free_flow_tt"] # exclude non-positive travel time values
    df["si"]  = df["speed"].where(valid_sp)       / df["free_flow_sp"]
    df = df.drop(columns=["free_flow_tt", "free_flow_sp"])
    return df


def get_count_by_borough(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("borough")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )


def get_count_by_borough_id(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["borough", "id"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["borough", "id"])
    )


def get_count_by_borough_id_hour(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["borough", "id", "hour"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["borough", "id", "hour"])
    )


def get_count_pivot_by_id_hour(df: pd.DataFrame, borough: str) -> pd.DataFrame:
    """id x hour row-count pivot table for a specific borough"""
    subset = df[df["borough"] == borough]
    return (
        subset.groupby(["id", "hour"])
        .size()
        .unstack(level="hour") # unstack: turns rows into columns
        .fillna(0)
        .astype(int)
    )


DOW_LABELS = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
              4: "Friday", 5: "Saturday", 6: "Sunday"}
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def get_hour_dow_heatmap(df: pd.DataFrame, value_col: str = "speed", aggfunc: str = "mean") -> pd.DataFrame:
    """hour x day-of-week pivot table (for heatmap data; missing days are filled with NaN)"""
    tmp = df.copy()
    tmp["dow_name"] = tmp["dow"].map(DOW_LABELS)
    pivot = tmp.pivot_table(index="hour", columns="dow_name", values=value_col, aggfunc=aggfunc)
    for day in DOW_ORDER:
        if day not in pivot.columns:
            pivot[day] = float("nan")
    return pivot[DOW_ORDER]


def get_congestion_start(df: pd.DataFrame, threshold_ratio: float = 0.85) -> pd.DataFrame:
    """Detect congestion-start hour by borough x day_type

    Defines congestion start as the first afternoon (12:00+) hour whose speed drops to or below threshold_ratio of the morning (0-11:00) peak speed
    """
    label_map = {0: "Weekday", 1: "Weekend"}
    base = (
        df.groupby(["borough", "hour", "is_weekend"])["speed"]
        .mean()
        .reset_index()
        .assign(day_type=lambda x: x["is_weekend"].map(label_map))
    )

    result = []
    for (borough, day_type), g in base.groupby(["borough", "day_type"]):
        g = g.sort_values("hour")
        morning_max = g[g["hour"] <= 11]["speed"].max()
        threshold = morning_max * threshold_ratio
        afternoon = g[g["hour"] >= 12]
        below = afternoon[afternoon["speed"] < threshold]
        congestion_hour = int(below["hour"].iloc[0]) if not below.empty else None
        peak_hour = int(g.loc[g["speed"].idxmin(), "hour"])
        min_speed = round(g["speed"].min(), 2)
        result.append({
            "borough": borough,
            "day_type": day_type,
            "morning_max_speed": round(morning_max, 2),
            "threshold": round(threshold, 2),
            "congestion_start": congestion_hour,
            "peak_congestion_hour": peak_hour,
            "min_speed": min_speed,
        })
    return pd.DataFrame(result).sort_values(["day_type", "borough"])


def get_sharp_changes(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Detect hour-over-hour speed-drop and travel-time-spike windows by borough

    Computes the hour-over-hour change (diff) and returns the hours with the sharpest changes
    diff_ratio = (diff / base) * 100
    """
    base = (
        df.groupby(["borough", "hour"])[["speed", "travel_time"]]
        .mean()
        .reset_index()
        .sort_values(["borough", "hour"])
    )

    result = []
    for borough, g in base.groupby("borough"):
        g = g.set_index("hour").reindex(range(24))
        g["speed_diff"]       = g["speed"].diff()        # negative = speed drop
        g["travel_time_diff"] = g["travel_time"].diff()  # positive = travel time increase

        # Top_n sharpest speed drops
        """
        Drops missing values via dropna(), then uses nsmallest(top_n) to pick the sharpest speed drops:
        the top_n hours where speed_diff is most negative (speed drop),

        and the top_n hours where travel_time_diff is most positive (travel-time spike),
        collecting the hour, change amount, and speed/travel-time at that hour into one DataFrame.
        """
        drops = g["speed_diff"].dropna().nsmallest(top_n) # pick the sharpest speed drops
        for hour, val in drops.items():
            result.append({
                "borough":   borough,
                "hour":      int(hour),
                "type":      "speed_drop",
                "change":    round(val, 2),
                "speed":     round(g.loc[hour, "speed"], 2),
                "travel_time": round(g.loc[hour, "travel_time"], 2),
            })

        # Top_n sharpest travel-time spikes
        spikes = g["travel_time_diff"].dropna().nlargest(top_n)
        for hour, val in spikes.items():
            result.append({
                "borough":   borough,
                "hour":      int(hour),
                "type":      "travel_time_spike",
                "change":    round(val, 2),
                "speed":     round(g.loc[hour, "speed"], 2),
                "travel_time": round(g.loc[hour, "travel_time"], 2),
            })

    return pd.DataFrame(result).sort_values(["borough", "type", "change"])


_CONNECTION_MAP = {
    "VNB":            ("Verrazzano-Narrows Bridge",  "Staten Island <-> Brooklyn",     "bridge"),
    "TBB":            ("RFK Bridge (Triborough)",     "Manhattan <-> Queens <-> Bronx", "bridge"),
    "BKN Bridge":     ("Brooklyn Bridge",             "Manhattan <-> Brooklyn",         "bridge"),
    "MAN Bridge":     ("Manhattan Bridge",            "Manhattan <-> Brooklyn",         "bridge"),
    "QMT":            ("Queens-Midtown Tunnel",       "Manhattan <-> Queens",           "tunnel"),
    "BBT":            ("Hugh L. Carey Tunnel",        "Manhattan <-> Brooklyn",         "tunnel"),
    "BWB":            ("Bronx-Whitestone Bridge",     "Queens <-> Bronx",               "bridge"),
    "TNB":            ("Throgs Neck Bridge",          "Queens <-> Bronx",               "bridge"),
    "LINCOLN TUNNEL": ("Lincoln Tunnel",              "Manhattan <-> NJ",               "tunnel"),
    "GWB":            ("George Washington Bridge",   "Manhattan/Bronx <-> NJ",         "bridge"),
    "BQE":            ("Brooklyn-Queens Expwy",       "Brooklyn <-> Queens",            "expressway"),
    "LIE":            ("Long Island Expressway",      "Manhattan <-> Queens",           "expressway"),
    "CBE":            ("Cross Bronx Expressway",      "Manhattan <-> Bronx",            "expressway"),
    "MDE":            ("Major Deegan Expressway",     "Manhattan <-> Bronx",            "expressway"),
    "SIE":            ("Staten Island Expressway",    "Staten Island <-> Brooklyn",     "expressway"),
    "WSE":            ("West Shore Expressway",       "Staten Island internal",         "expressway"),
    "MLK":            ("MLK Expressway",              "Staten Island <-> NJ",           "expressway"),
    "BE ":            ("Bruckner Expressway",         "Bronx <-> Queens",               "expressway"),
    "GOW":            ("Gowanus Expressway",          "Brooklyn internal",              "expressway"),
    "FDR":            ("FDR Drive",                   "Manhattan (Bridge access)",      "expressway"),
    "Westside Hwy":   ("Westside Highway",            "Manhattan (GWB access)",         "expressway"),
    "West St":        ("West Street",                 "Manhattan (Tunnel access)",      "expressway"),
    "I-87":           ("I-87 NY State Thruway",      "Bronx <-> Upstate",              "expressway"),
    "HRP":            ("Harlem River Park",           "Bronx internal",                 "expressway"),
    "Belt Pkwy":      ("Belt Parkway",                "Brooklyn <-> Queens",            "parkway"),
    "BRP":            ("Bronx River Parkway",         "Bronx internal",                 "parkway"),
    "CIP":            ("Cross Island Parkway",        "Queens internal",                "parkway"),
    "CVE":            ("College Point Blvd Expwy",    "Queens internal",                "expressway"),
    "VWE":            ("Van Wyck Expressway",         "Queens internal",                "expressway"),
    "Laurelton Pkwy": ("Laurelton Parkway",           "Queens internal",                "parkway"),
}

_TYPE_ORDER = ["bridge", "tunnel", "expressway", "parkway"]


def _classify_connection(link_name: str):
    for prefix, info in _CONNECTION_MAP.items():
        if link_name.startswith(prefix):
            return info
    return None


def get_congested_segments(df: pd.DataFrame, si_threshold: float = 0.7) -> pd.DataFrame:
    """Return df filtered to only congested segments whose average SI is below si_threshold

    Requires an si column in df (must run add_tti first)
    """
    avg_si = df.groupby("id")["si"].mean()
    congested_ids = avg_si[avg_si < si_threshold].index
    return df[df["id"].isin(congested_ids)].copy()

# Used in the weekday/weekend si heatmap

def get_road_hour_si_by_borough(df: pd.DataFrame, borough: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return average SI pivot by road (link_name) x hour for a specific borough

    Returns: (weekday_pivot, weekend_pivot)
    index=link_name (ascending avg_si — most congested road at the top)
    columns=hour(0-23)
    """
    sub = df[df["borough"] == borough].copy()
    label_map = {0: "Weekday", 1: "Weekend"}
    sub["day_type"] = sub["is_weekend"].map(label_map)

    pivots = {}
    for day_type, group in sub.groupby("day_type"):
        pivot = (
            group.groupby(["link_name", "hour"])["si"]
            .mean()
            .reset_index()
            .pivot(index="link_name", columns="hour", values="si")
        )
        order = pivot.mean(axis=1).sort_values().index
        pivots[day_type] = pivot.loc[order]

    return pivots.get("Weekday", pd.DataFrame()), pivots.get("Weekend", pd.DataFrame())


def get_tti_si_by_borough_hour(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return average TTI/SI pivot by borough x hour (requires tti, si columns)

    Returns: (tti_pivot, si_pivot) — index=hour(0-23), columns=borough
    """
    base = (
        df.groupby(["borough", "hour"])[["tti", "si"]]
        .mean()
        .round(3)
        .reset_index()
    )
    tti_pivot = base.pivot(index="hour", columns="borough", values="tti")
    si_pivot  = base.pivot(index="hour", columns="borough", values="si")
    return tti_pivot, si_pivot


def get_connection_road_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Speed/travel_time statistics per connection road

    Classifies bridges/tunnels/expressways/parkways by link_name prefix and returns
    n_records, avg_speed, avg_tt, p25_speed, p75_speed.
    road_type order: bridge -> tunnel -> expressway -> parkway
    """
    tmp = df.copy()
    cls = tmp["link_name"].apply(_classify_connection)
    tmp["road_name"] = cls.apply(lambda x: x[0] if x else None)
    tmp["connects"]  = cls.apply(lambda x: x[1] if x else None)
    tmp["road_type"] = cls.apply(lambda x: x[2] if x else None)

    tagged = tmp[tmp["road_name"].notna()]

    agg_dict = {
        "n_records": ("speed", "count"),
        "avg_speed":  ("speed", "mean"),
        "avg_tt":     ("travel_time", "mean"),
        "p25_speed":  ("speed", lambda x: x.quantile(0.25)),
        "p75_speed":  ("speed", lambda x: x.quantile(0.75)),
    }
    if "tti" in tagged.columns:
        agg_dict["avg_tti"] = ("tti", "mean")
    if "si" in tagged.columns:
        agg_dict["avg_si"] = ("si", "mean")

    result = tagged.groupby(["road_type", "road_name", "connects"]).agg(**agg_dict).reset_index()

    result["avg_speed"] = result["avg_speed"].round(1)
    result["avg_tt"]    = result["avg_tt"].round(0).astype(int)
    result["p25_speed"] = result["p25_speed"].round(1)
    result["p75_speed"] = result["p75_speed"].round(1)
    if "avg_tti" in result.columns:
        result["avg_tti"] = result["avg_tti"].round(2)
    if "avg_si" in result.columns:
        result["avg_si"] = result["avg_si"].round(3)

    result["_order"] = result["road_type"].map(
        {t: i for i, t in enumerate(_TYPE_ORDER)}
    ).fillna(99)
    return (
        result.sort_values(["_order", "avg_speed"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def get_weekday_weekend_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Weekday/weekend average speed and travel_time comparison by borough and hour"""
    label_map = {0: "Weekday", 1: "Weekend"}
    return (
        df.groupby(["borough", "hour", "is_weekend"])[["speed", "travel_time"]]
        .mean()
        .round(2)
        .reset_index()
        .assign(day_type=lambda x: x["is_weekend"].map(label_map))
    )


def get_hour_dow_heatmap_by_borough(df: pd.DataFrame, value_col: str = "speed", aggfunc: str = "mean") -> dict:
    """Return a dict of hour x day-of-week pivot tables by borough {borough: DataFrame}"""
    tmp = df.copy()
    tmp["dow_name"] = tmp["dow"].map(DOW_LABELS)
    result = {}
    for borough, group in tmp.groupby("borough"):
        pivot = group.pivot_table(index="hour", columns="dow_name", values=value_col, aggfunc=aggfunc)
        for day in DOW_ORDER:
            if day not in pivot.columns:
                pivot[day] = float("nan")
        result[borough] = pivot[DOW_ORDER]
    return result
