"""Rotation risk features — minutes variance and starts consistency.

These features are already computed inside engineer.py (`minutes_std_5`,
`starts_rolling_5`).  This module exposes them as standalone functions for
use by MinutesModel feature pipelines that operate independently of the full
engineer.py pipeline.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import DataFrame

from fplt.schemas import GameweekSchema


@pa.check_types
def add_rotation_risk(
    df: DataFrame[GameweekSchema], window: int = 5
) -> DataFrame[GameweekSchema]:
    """Add minutes standard deviation and starts consistency features.

    Both features use shift(1) before rolling to prevent data leakage.

    Args:
        df: Player-level gameweek DataFrame sorted by season, element, GW.
            Must contain `minutes` and optionally `starts`.
        window: Rolling window size (default 5).

    Returns:
        Copy of df with added columns:
            minutes_std_{window}     — rolling std dev of minutes played
            starts_pct_{window}      — fraction of recent appearances as starter
    """
    df = df.copy()
    df = df.sort_values(["season", "element", "GW"])

    grouped = df.groupby(["season", "element"], sort=False)

    shifted_mins = grouped["minutes"].transform(lambda s: s.shift(1))
    df[f"minutes_std_{window}"] = shifted_mins.groupby(df["element"]).transform(
        lambda s: s.rolling(window, min_periods=2).std()
    )

    if "starts" in df.columns:
        shifted_starts = grouped["starts"].transform(lambda s: s.shift(1))
        df[f"starts_pct_{window}"] = shifted_starts.groupby(df["element"]).transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

    return df  # type: ignore[return-value]
