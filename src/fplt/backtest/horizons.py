"""Shared horizon length defaults for projections and skill curves.

``DEFAULT_PLAN_HORIZON`` lives in ``fplt.optimise.plan`` and stays at 3 until
a chips-on season-points retune (FDB-120). These constants only cover
prediction / eval lead times.
"""

from __future__ import annotations

# Live ``fplt predict`` default: enough steps for H=8–10 planning without
# forcing the MILP default.
DEFAULT_PROJECTION_HORIZON = 10

# Skill-by-horizon eval length (``eval-components --points-models``,
# ``fplt horizon-skill``).
DEFAULT_SKILL_STEPS = 10
