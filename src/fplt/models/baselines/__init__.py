"""Point baselines that implement ``PointsPredictor``."""

from fplt.models.baselines.ep_next import EpNextPredictor
from fplt.models.baselines.rolling_mean import RollingMeanPredictor

__all__ = ["EpNextPredictor", "RollingMeanPredictor"]
