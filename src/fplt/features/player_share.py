"""Per-player xG/shot share features within each fixture.

Computes each player's rolling expected-goal share of their team's total
expected goals.  This is the primary input for GoalAllocationModel: a player
who accounts for 40% of their team's xG should be allocated ~40% of the
team's predicted goals from the Dixon-Coles distribution.

Falls back to zero share for seasons without xG data (pre-2019-20).
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from fplt.schemas import FeatureSchema


@pa.check_types
def add_player_xg_share(gw_df: DataFrame[FeatureSchema]) -> DataFrame[FeatureSchema]:
    """Add `xg_share` column: player rolling xG / team rolling xG per fixture.

    Uses `expected_goals_rolling_5` if available; otherwise sets xg_share to
    NaN so GoalAllocationModel can handle the missing-data case.

    Args:
        gw_df: Player-level gameweek DataFrame.  Must contain `fixture` and
               `team` (or `opponent_team` + `was_home` as a fallback).
               `expected_goals_rolling_5` is optional.

    Returns:
        Copy of gw_df with an additional `xg_share` column in [0, 1].
    """
    df = gw_df.copy()

    xg_col = "expected_goals_rolling_5"
    if xg_col not in df.columns:
        df["xg_share"] = float("nan")
        return df  # type: ignore[return-value]

    # Team identifier: prefer `team` column, fall back to derivation
    if "team" not in df.columns:
        df["xg_share"] = float("nan")
        return df  # type: ignore[return-value]

    # Per fixture, sum each team's total rolling xG across players
    team_xg = (
        df.groupby(["fixture", "team"])[xg_col]
        .sum()
        .rename("team_xg_total")
        .reset_index()
    )
    df = df.merge(team_xg, on=["fixture", "team"], how="left")

    team_total = df["team_xg_total"].fillna(0.0)
    player_xg = df[xg_col].fillna(0.0)

    # Avoid division by zero when no player has any xG
    df["xg_share"] = (
        player_xg.where(team_total > 0)
        .div(team_total.where(team_total > 0))
        .fillna(0.0)
    )

    df = df.drop(columns=["team_xg_total"])
    return df  # type: ignore[return-value]
