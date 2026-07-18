import pandas as pd


def get_outliers_by_id_hour(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    if cols is None:
        cols = ["speed", "travel_time"]
    result = []
    for (seg_id, hour), group in df.groupby(["id", "hour"]):
        borough = group["borough"].iloc[0]
        for col in cols:
            Q1 = group[col].quantile(0.25)
            Q3 = group[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_mask = (group[col] < lower) | (group[col] > upper)
            outlier_count = outlier_mask.sum()
            result.append({
                "id": seg_id,
                "borough": borough,
                "hour": hour,
                "column": col,
                "total_count": len(group),
                "outlier_count": int(outlier_count),
                "outlier_ratio": round(outlier_count / len(group), 4),
                "Q1": round(Q1, 2),
                "Q3": round(Q3, 2),
                "IQR": round(IQR, 2),
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
            })
    return pd.DataFrame(result)
