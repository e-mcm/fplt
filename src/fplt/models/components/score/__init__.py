"""Scoreline components. Importing this module registers them."""

from fplt.models.components.score.dixon_coles import MatchScoreModel
from fplt.models.components.score.odds_1x2 import Odds1X2Score
from fplt.models.components.score.odds_lambda import (
    blend_lambda,
    de_vig,
    odds_to_lambda,
    outcome_probs_from_joint,
    rescale_joint_to_1x2,
)
from fplt.models.components.score.odds_poisson import OddsPoissonScore

__all__ = [
    "MatchScoreModel",
    "Odds1X2Score",
    "OddsPoissonScore",
    "blend_lambda",
    "de_vig",
    "odds_to_lambda",
    "outcome_probs_from_joint",
    "rescale_joint_to_1x2",
]
