"""Squad and transfer optimisation."""

from fplt.optimise.attainable import (
    ORACLE_MODEL_ID,
    ChipTimingRegret,
    attainable_share,
    chip_timing_regret,
    component_regret,
    enrich_decision_summary,
    foresight_predictions,
    format_attainable_scoreboard,
    run_hindsight_oracle,
    simulate_hindsight_oracle,
    summarise_attainable,
)
from fplt.optimise.engine import TransferPlan, horizon_points, optimize_transfers
from fplt.optimise.plan import GameweekPlan, plan_squad
from fplt.optimise.season_optimiser import (
    FplOptimizationToolsAdapter,
    IdentityReferenceOptimiser,
    OursSeasonOptimiser,
    freeze_reference_state,
)
from fplt.optimise.squad import SquadSelection, aggregate_gameweek, select_squad
from fplt.optimise.state import (
    ChipInventory,
    ChipPlay,
    SquadState,
    load_squad_state,
    rebuild_squad_state,
)

__all__ = [
    "ORACLE_MODEL_ID",
    "ChipInventory",
    "ChipPlay",
    "ChipTimingRegret",
    "FplOptimizationToolsAdapter",
    "GameweekPlan",
    "IdentityReferenceOptimiser",
    "OursSeasonOptimiser",
    "SquadSelection",
    "SquadState",
    "TransferPlan",
    "aggregate_gameweek",
    "attainable_share",
    "chip_timing_regret",
    "component_regret",
    "enrich_decision_summary",
    "foresight_predictions",
    "format_attainable_scoreboard",
    "freeze_reference_state",
    "horizon_points",
    "load_squad_state",
    "optimize_transfers",
    "plan_squad",
    "rebuild_squad_state",
    "run_hindsight_oracle",
    "select_squad",
    "simulate_hindsight_oracle",
    "summarise_attainable",
]
