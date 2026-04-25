"""Feature engineering for FPL prediction models."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import DataFrame

from fplt.schemas import FeatureSchema, GameweekSchema, TeamSchema
from fplt.types import Position
from fplt.utils import timed

# Maps raw position strings → Position enum (AM treated as MID)
POSITION_ENC = {
    "GK": Position.GK,
    "DEF": Position.DEF,
    "MID": Position.MID,
    "AM": Position.MID,
    "FWD": Position.FWD,
}

# Rolling windows to compute
WINDOWS = [3, 5, 10]

# Stats to roll — (column, min_periods_override or None)
_ROLL_STATS = [
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "bps",
    "ict_index",
    "threat",
    "creativity",
    "influence",
]

_ROLL_STATS_OPTIONAL = [
    "xP",
    "starts",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]


@timed
def _add_rolling_stats(df: DataFrame[GameweekSchema]) -> DataFrame[FeatureSchema]:
    """Add rolling mean features per player per season.

    Uses shift(1) before rolling so GW-N features are based on GWs 1..N-1,
    preventing data leakage into the target.
    """
    df = df.sort_values(["season", "element", "GW"]).copy()

    stats = [c for c in _ROLL_STATS if c in df.columns]
    optional = [c for c in _ROLL_STATS_OPTIONAL if c in df.columns]

    grouped = df.groupby(["season", "element"], sort=False)

    for col in stats + optional:
        shifted = grouped[col].transform(lambda s: s.shift(1))
        for w in WINDOWS:
            feat = f"{col}_rolling_{w}"
            df[feat] = shifted.groupby(df["element"]).transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean()
            )

    # Minutes std as rotation-risk proxy
    shifted_mins = grouped["minutes"].transform(lambda s: s.shift(1))
    df["minutes_std_5"] = shifted_mins.groupby(df["element"]).transform(
        lambda s: s.rolling(5, min_periods=2).std()
    )

    return df  # type: ignore[return-value]


@timed
def _add_opponent_strength(
    df: DataFrame[FeatureSchema], teams: DataFrame[TeamSchema]
) -> DataFrame[FeatureSchema]:
    """Join opponent team strength ratings onto the GW dataframe.

    Picks home/away strength based on the player's perspective:
    - player at home → opponent is the away side → use opponent's away strength
    - player away    → opponent is the home side → use opponent's home strength
    """
    strength_cols = [
        "id",
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]
    teams_slim = teams[[c for c in strength_cols if c in teams.columns]].copy()

    df = df.merge(
        teams_slim,
        left_on="opponent_team",
        right_on="id",
        how="left",
    ).drop(columns=["id"])

    # From the player's perspective, the opponent's relevant strength flips
    df["opp_strength_overall"] = df.apply(
        lambda r: (
            r["strength_overall_away"] if r["was_home"] else r["strength_overall_home"]
        ),
        axis=1,
    )
    df["opp_strength_attack"] = df.apply(
        lambda r: (
            r["strength_attack_away"] if r["was_home"] else r["strength_attack_home"]
        ),
        axis=1,
    )
    df["opp_strength_defence"] = df.apply(
        lambda r: (
            r["strength_defence_away"] if r["was_home"] else r["strength_defence_home"]
        ),
        axis=1,
    )

    drop = [
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
    ]
    return df.drop(columns=[c for c in drop if c in df.columns])  # type: ignore[return-value]


@timed
def _encode_position(df: DataFrame[FeatureSchema]) -> DataFrame[FeatureSchema]:
    """Encode position as an integer. Rows without a known position get -1."""
    if "position" not in df.columns:
        df["position_enc"] = -1
    else:
        df["position_enc"] = df["position"].map(POSITION_ENC).fillna(-1).astype(int)
    return df  # type: ignore[return-value]


@pa.check_types
@timed
def build_features(
    gw_df: DataFrame[GameweekSchema],
    teams_df: DataFrame[TeamSchema],
) -> DataFrame[FeatureSchema]:
    """Build the full feature set for model training or inference.

    Args:
        gw_df: Raw gameweek DataFrame from load_gameweeks / load_all_seasons.
        teams_df: Teams DataFrame from load_teams for the relevant season(s).

    Returns:
        DataFrame with all engineered features appended. The target column
        ``total_points`` is retained so callers can split X/y as needed.
    """
    df = _add_rolling_stats(gw_df)
    df = _add_opponent_strength(df, teams_df)
    df = _encode_position(df)
    return df  # type: ignore[return-value]
