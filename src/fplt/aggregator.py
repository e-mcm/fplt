"""Aggregator — combines all model outputs into Xpts per player per GW.

Integrates over the full ScoreDistribution rather than using point estimates
for score-dependent components (clean sheets, goals).

Key principle from design.md:
    "Do not compute P(clean sheet) from E[goals].
     Integrate over the full score distribution."

The aggregator is intentionally stateless: it takes fitted model objects and
data as arguments, so it can be used both in the training pipeline (for
held-out validation) and in the prediction pipeline (for upcoming GWs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from fplt.models.base import PointModel
from fplt.models.goal_allocation.model import GoalAllocationModel
from fplt.models.score_distribution import ScoreDistribution
from fplt.rules import (
    APPEARANCE_POINTS,
    CLEAN_SHEET_POINTS,
    GOAL_POINTS,
    RED_CARD_POINTS,
    SAVE_BONUS_EVERY,
    YELLOW_CARD_POINTS,
)
from fplt.schemas import FeatureSchema
from fplt.types import Position

_MAX_GOALS_ITER = 10  # cap the k summation matching GoalAllocationModel


def compute_xpts(
    players: DataFrame[FeatureSchema],
    score_dist: ScoreDistribution,
    models: dict[str, PointModel],
    fixture_col: str = "fixture",
) -> pd.Series:
    """Compute expected points per player by integrating over the score distribution.

    Args:
        players: Player feature DataFrame.  Must contain:
                   - position_enc (int Position enum value)
                   - was_home (bool)
                   - fixture (or whatever fixture_col names) → links to score_dist
                   - xg_share (float, may be NaN for pre-xG seasons)
        score_dist: ScoreDistribution from DixonColesModel for these fixtures.
        models: Dict with keys:
                   "minutes"        → MinutesModel
                   "goal_allocation"→ GoalAllocationModel
                   "bonus"          → BonusModel
                   "cards"          → CardModel
                   "saves"          → SavesModel
        fixture_col: Column name in players that maps to score_dist.fixture_ids.

    Returns:
        Series of expected FPL points, same index as players.
    """
    minutes_model = models["minutes"]
    goal_model = models["goal_allocation"]
    bonus_model = models["bonus"]
    card_model = models["cards"]
    saves_model = models["saves"]

    assert isinstance(goal_model, GoalAllocationModel)

    # --- Run each model ---
    min_preds = minutes_model.predict(players)  # p_starts, e_minutes
    goal_preds = goal_model.predict(
        players, score_dist=score_dist, fixture_col=fixture_col
    )
    bonus_preds = bonus_model.predict(players)  # e_bonus
    card_preds = card_model.predict(players)  # p_yellow, p_red
    saves_preds = saves_model.predict(players)  # e_saves

    p_starts = min_preds["p_starts"].to_numpy()
    e_minutes = min_preds["e_minutes"].to_numpy()
    e_goals = goal_preds["expected_goals_scored"].to_numpy()
    e_bonus = bonus_preds["e_bonus"].to_numpy()
    p_yellow = card_preds["p_yellow"].to_numpy()
    p_red = card_preds["p_red"].to_numpy()
    e_saves = saves_preds["e_saves"].to_numpy()

    pos_enc = players["position_enc"].to_numpy()
    was_home = (
        players["was_home"].to_numpy()
        if "was_home" in players.columns
        else np.ones(len(players), dtype=bool)
    )
    fixture_ids = (
        players[fixture_col].to_numpy()
        if fixture_col in players.columns
        else np.arange(len(players))
    )

    xpts = np.zeros(len(players), dtype=np.float64)

    for i in range(len(players)):
        pos = (
            Position(int(pos_enc[i]))
            if pos_enc[i] in [p.value for p in Position]
            else Position.MID
        )

        # --- Appearance points (unconditional on score) ---
        if e_minutes[i] >= 60:
            app_pts = APPEARANCE_POINTS["full"]
        elif e_minutes[i] > 0:
            app_pts = APPEARANCE_POINTS["sub"]
        else:
            app_pts = 0.0

        # --- Clean sheet probability (integrated over full distribution) ---
        fid = fixture_ids[i]
        try:
            fi = score_dist.fixture_index(fid)
            side = "home" if was_home[i] else "away"
            p_cs = float(score_dist.p_clean_sheet(side)[fi])
        except (ValueError, IndexError):
            p_cs = 0.0

        cs_pts = CLEAN_SHEET_POINTS.get(pos, 0) * p_cs

        # --- Goal points ---
        goal_pts = e_goals[i] * GOAL_POINTS.get(pos, 4)

        # --- Save bonus (GK only) ---
        save_pts = float(e_saves[i] // SAVE_BONUS_EVERY)

        # --- Card penalties ---
        card_pts = p_yellow[i] * YELLOW_CARD_POINTS + p_red[i] * RED_CARD_POINTS

        # --- Combine ---
        xpts[i] = (
            app_pts
            + p_starts[i] * (cs_pts + goal_pts + e_bonus[i])
            + save_pts
            + card_pts
        )

    return pd.Series(xpts, index=players.index, name="xpts")
