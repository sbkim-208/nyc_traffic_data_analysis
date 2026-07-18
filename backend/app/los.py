"""Level of Service classification.

Speed-based simplification adapted from HCM 6th edition urban-street thresholds.
For a school project we use easy-to-explain bands; the README documents the
methodology and its limitations.
"""
from __future__ import annotations

import pandas as pd

# Bands ordered by lower bound, descending. (lower_mph, label, hex_color, range_label)
LOS_BANDS = [
    (30.0, "A", "#1a9850", "≥ 30 mph"),
    (25.0, "B", "#66bd63", "25–30 mph"),
    (18.0, "C", "#fee08b", "18–25 mph"),
    (12.0, "D", "#fdae61", "12–18 mph"),
    (7.0,  "E", "#f46d43", "7–12 mph"),
    (0.0,  "F", "#a50026", "< 7 mph"),
]

_BIN_EDGES = [0.0, 7.0, 12.0, 18.0, 25.0, 30.0, 1000.0]
_BIN_LABELS = ["F", "E", "D", "C", "B", "A"]
_COLOR_BY_LABEL = {b[1]: b[2] for b in LOS_BANDS}


def classify_series(speeds: pd.Series) -> pd.Series:
    """Vectorized LOS classification. NaNs become 'F'."""
    cats = pd.cut(speeds, bins=_BIN_EDGES, labels=_BIN_LABELS, right=False, include_lowest=True)
    return cats.astype(object).fillna("F")


def color_for(label: str) -> str:
    return _COLOR_BY_LABEL.get(label, "#888888")


def color_series(labels: pd.Series) -> pd.Series:
    return labels.map(_COLOR_BY_LABEL).fillna("#888888")
