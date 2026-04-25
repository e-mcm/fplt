"""FPL scoring rules and component → points conversion.

Official rules: https://fantasy.premierleague.com/help/rules
"""

import numpy as np
import pandas as pd

from fplt.types import Position

# Points per goal scored
GOAL_POINTS = {
    Position.GK: 6,
    Position.DEF: 6,
    Position.MID: 5,
    Position.FWD: 4,
}

# Points for a clean sheet
CLEAN_SHEET_POINTS = {
    Position.GK: 4,
    Position.DEF: 4,
    Position.MID: 1,
    Position.FWD: 0,
}

# Points for playing at all vs playing >= 60 minutes
APPEARANCE_POINTS = {
    "sub": 1,  # played but < 60 min
    "full": 2,  # played >= 60 min
}

# Fixed point values
ASSIST_POINTS = 3
SAVE_BONUS_EVERY = 3  # 1pt per 3 saves (GK only)
PENALTY_SAVE_POINTS = 5
PENALTY_MISS_POINTS = -2
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
OWN_GOAL_POINTS = -2


def score_components(
    position_enc: Position | int | np.integer,
    minutes: float,
    goals_scored: float = 0.0,
    assists: float = 0.0,
    clean_sheets: float = 0.0,
    saves: float = 0.0,
    yellow_cards: float = 0.0,
    red_cards: float = 0.0,
    own_goals: float = 0.0,
    penalties_missed: float = 0.0,
    penalties_saved: float = 0.0,
    bonus: float = 0.0,
) -> float:
    """Convert raw component values into predicted FPL points.

    All arguments may be fractional (expected values from regression models).

    Args:
        position_enc: Position enum or equivalent integer.
        minutes: Expected minutes played.
        goals_scored: Expected goals.
        assists: Expected assists.
        clean_sheets: Expected clean sheet probability (0–1).
        saves: Expected saves (GK only).
        yellow_cards: Expected yellow cards.
        red_cards: Expected red cards.
        own_goals: Expected own goals.
        penalties_missed: Expected penalties missed.
        penalties_saved: Expected penalties saved (GK only).
        bonus: Expected bonus points.

    Returns:
        Expected total FPL points.
    """
    pts = 0.0

    # Appearance points
    if minutes >= 60:
        pts += APPEARANCE_POINTS["full"]
    elif minutes > 0:
        pts += APPEARANCE_POINTS["sub"]

    pos = Position(int(position_enc))

    # Goals
    pts += goals_scored * GOAL_POINTS.get(pos, 4)

    # Assists
    pts += assists * ASSIST_POINTS

    # Clean sheet
    pts += clean_sheets * CLEAN_SHEET_POINTS.get(pos, 0)

    # GK save bonus
    pts += saves // SAVE_BONUS_EVERY

    # GK penalty save
    pts += penalties_saved * PENALTY_SAVE_POINTS

    # Penalty miss
    pts += penalties_missed * PENALTY_MISS_POINTS

    # Cards
    pts += yellow_cards * YELLOW_CARD_POINTS
    pts += red_cards * RED_CARD_POINTS

    # Own goals
    pts += own_goals * OWN_GOAL_POINTS

    # Bonus
    pts += bonus

    return pts


def score_dataframe(df: pd.DataFrame) -> pd.Series:
    """Apply score_components row-wise to a DataFrame.

    Expects columns: position_enc, minutes, and any of the optional component
    columns (missing ones default to 0).

    Returns:
        Series of expected points, same index as df.
    """
    optional = [
        "goals_scored",
        "assists",
        "clean_sheets",
        "saves",
        "yellow_cards",
        "red_cards",
        "own_goals",
        "penalties_missed",
        "penalties_saved",
        "bonus",
    ]

    def _row(r: pd.Series) -> float:
        kwargs = {c: r[c] for c in optional if c in r.index and pd.notna(r[c])}
        return score_components(r["position_enc"], r["minutes"], **kwargs)

    return df.apply(_row, axis=1)
