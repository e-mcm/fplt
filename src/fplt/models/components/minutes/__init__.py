"""Minutes components. Importing this module registers them."""

from fplt.models.components.minutes.empirical import EmpiricalMinutes
from fplt.models.components.minutes.empirical_v2 import EmpiricalMinutesV2

__all__ = ["EmpiricalMinutes", "EmpiricalMinutesV2"]
