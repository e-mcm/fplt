"""Typed pydantic params for registered component kinds.

Each component kind has a ``BaseModel`` with ``extra="forbid"`` so unknown
keys fail loudly and defaults are explicit for future ``spec_hash`` work
(data-model step 2b / rollout B3).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Pin of today's ``spec.params.get("…", default)`` values before typing.
# Changing a field default here (or on the model) must update this map in the
# same PR so the default-equality tests stay intentional.
# Dixon–Coles ``use_closing_proxy`` is the *resolved* empty-params value
# (``market_1x2_strength`` default 0 → proxy False).
LEGACY_GET_DEFAULTS: dict[str, dict[str, Any]] = {
    "involvement_shares": {
        "window": 10,
        "shrinkage": 5.0,
        "penalty_bonus": 0.3,
        "use_penalties_order": True,
        "use_penalty_event": False,
        "p_penalty": 0.12,
        "use_set_piece_orders": False,
        "set_piece_bonus": 0.1,
    },
    "involvement_xg": {
        "window": 10,
        "shrinkage": 5.0,
        "penalty_bonus": 0.3,
        "xg_blend": 0.7,
        "use_penalties_order": True,
        "use_penalty_event": False,
        "p_penalty": 0.12,
        "use_set_piece_orders": False,
        "set_piece_bonus": 0.1,
    },
    "empirical_minutes": {
        "window": 5,
        "shrinkage": 5.0,
        "doubtful_chance": 0.5,
        "use_injury_overlay": False,
        "use_availability_gate": True,
        "availability_recovery_games": 3.0,
    },
    "empirical_minutes_v2": {
        "windows": [3, 5, 10],
        "shrinkage": 5.0,
        "doubtful_chance": 0.5,
        "soft_squad": True,
        "target_starters": 11.0,
        "soft_squad_strength": 0.5,
        "logistic_blend": 0.5,
        "l2": 1.0,
        "return_ramp_games": 2.0,
        "availability_recovery_games": 3.0,
        "use_injury_overlay": False,
        "use_availability_gate": True,
        "calibrate_chance_flags": True,
        "chance_cal_min_count": 20,
        "predicted_xi": False,
        "predicted_xi_window_hours": 72.0,
        "predicted_xi_strength": 0.75,
    },
    "other_player": {
        "window": 10,
        "shrinkage": 5.0,
        "bps_bonus_weight": 0.5,
        "use_defensive_contribution": True,
        "defcon_mode": "mean",
        "defcon_bucket_minutes": None,
    },
    "other_match_conditioned": {
        "window": 10,
        "shrinkage": 5.0,
        "bps_bonus_weight": 0.5,
        "use_defensive_contribution": True,
        "defcon_mode": "mean",
        "defcon_bucket_minutes": None,
        "match_shrinkage": 20.0,
    },
    "other_priors": {},
    "dixon_coles": {
        "xi": 0.0,
        "home_advantage": 1.0,
        "rho": 0.0,
        "fit_rho": False,
        "fit_home_advantage": False,
        "new_team_prior_strength": 0.0,
        "promoted_prior_strength": 0.0,
        "market_prior_strength": 0.0,
        "market_1x2_strength": 0.0,
        "market_cs_strength": 0.0,
        "rest_coef": 0.0,
        "congestion_coef": 0.0,
        "form_window": 0,
        "form_strength": 0.0,
        "market_prior_source": None,
        "use_closing_proxy": False,
    },
    "poisson_score": {
        "xi": 0.0,
        "home_advantage": 1.0,
        "rho": 0.0,
        "fit_rho": False,
        "fit_home_advantage": False,
        "new_team_prior_strength": 0.0,
        "promoted_prior_strength": 0.0,
        "market_prior_strength": 0.0,
        "market_1x2_strength": 0.0,
        "market_cs_strength": 0.0,
        "rest_coef": 0.0,
        "congestion_coef": 0.0,
        "form_window": 0,
        "form_strength": 0.0,
        "market_prior_source": None,
        "use_closing_proxy": False,
    },
    "odds_1x2": {
        "use_closing_proxy": True,
        "fallback_p_home": 1.0 / 3.0,
        "fallback_p_draw": 1.0 / 3.0,
        "fallback_p_away": 1.0 / 3.0,
    },
    "odds_poisson": {
        "use_closing_proxy": True,
        "use_over_under": True,
        "fallback_lambda_home": 1.35,
        "fallback_lambda_away": 1.1,
    },
}


class InvolvementSharesParams(BaseModel):
    """Params for ``involvement_shares``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    window: int = Field(default=10, ge=1)
    shrinkage: float = Field(default=5.0, ge=0.0)
    penalty_bonus: float = Field(default=0.3, ge=0.0)
    use_penalties_order: bool = True
    use_penalty_event: bool = False
    p_penalty: float = Field(default=0.12, ge=0.0, le=1.0)
    use_set_piece_orders: bool = False
    set_piece_bonus: float = Field(default=0.1, ge=0.0)


class InvolvementXgParams(BaseModel):
    """Params for ``involvement_xg``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    window: int = Field(default=10, ge=1)
    shrinkage: float = Field(default=5.0, ge=0.0)
    penalty_bonus: float = Field(default=0.3, ge=0.0)
    xg_blend: float = Field(default=0.7, ge=0.0, le=1.0)
    use_penalties_order: bool = True
    use_penalty_event: bool = False
    p_penalty: float = Field(default=0.12, ge=0.0, le=1.0)
    use_set_piece_orders: bool = False
    set_piece_bonus: float = Field(default=0.1, ge=0.0)


class EmpiricalMinutesParams(BaseModel):
    """Params for ``empirical_minutes``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    window: int = Field(default=5, ge=1)
    shrinkage: float = Field(default=5.0, ge=0.0)
    doubtful_chance: float = Field(default=0.5, ge=0.0, le=1.0)
    use_injury_overlay: bool = False
    use_availability_gate: bool = True
    availability_recovery_games: float = Field(default=3.0, ge=0.0)


class EmpiricalMinutesV2Params(BaseModel):
    """Params for ``empirical_minutes_v2``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    windows: list[int] = Field(default_factory=lambda: [3, 5, 10])
    shrinkage: float = Field(default=5.0, ge=0.0)
    doubtful_chance: float = Field(default=0.5, ge=0.0, le=1.0)
    soft_squad: bool = True
    target_starters: float = Field(default=11.0, gt=0.0)
    soft_squad_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    logistic_blend: float = Field(default=0.5, ge=0.0, le=1.0)
    l2: float = Field(default=1.0, ge=0.0)
    return_ramp_games: float = Field(default=2.0, ge=0.0)
    availability_recovery_games: float = Field(default=3.0, ge=0.0)
    use_injury_overlay: bool = False
    use_availability_gate: bool = True
    calibrate_chance_flags: bool = True
    chance_cal_min_count: int = Field(default=20, ge=1)
    predicted_xi: bool = False
    predicted_xi_window_hours: float = Field(default=72.0, ge=0.0)
    predicted_xi_strength: float = Field(default=0.75, ge=0.0, le=1.0)


class OtherPlayerParams(BaseModel):
    """Params for ``other_player``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    window: int = Field(default=10, ge=1)
    shrinkage: float = Field(default=5.0, ge=0.0)
    bps_bonus_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    use_defensive_contribution: bool = True
    defcon_mode: Literal["mean", "rate"] = "mean"
    defcon_bucket_minutes: dict[str, float] | None = None


class OtherMatchConditionedParams(OtherPlayerParams):
    """Params for ``other_match_conditioned`` (includes other_player fields)."""

    match_shrinkage: float = Field(default=20.0, ge=0.0)


class OtherPriorsParams(BaseModel):
    """Params for ``other_priors`` (no hyperparameters today)."""

    model_config = ConfigDict(extra="forbid", strict=True)


class DixonColesParams(BaseModel):
    """Params for ``dixon_coles`` / ``poisson_score``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    xi: float = 0.0
    home_advantage: float = 1.0
    rho: float = 0.0
    fit_rho: bool = False
    fit_home_advantage: bool = False
    new_team_prior_strength: float = 0.0
    promoted_prior_strength: float = 0.0
    market_prior_strength: float = Field(default=0.0, ge=0.0)
    market_1x2_strength: float = Field(default=0.0, ge=0.0)
    market_cs_strength: float = Field(default=0.0, ge=0.0)
    rest_coef: float = Field(default=0.0, ge=0.0)
    congestion_coef: float = Field(default=0.0, ge=0.0)
    form_window: int = Field(default=0, ge=0)
    form_strength: float = Field(default=0.0, ge=0.0)
    market_prior_source: str | None = None
    # None means "derive from market_1x2_strength > 0" (legacy ``.get`` default).
    use_closing_proxy: bool | None = None

    @field_validator("market_prior_source", mode="before")
    @classmethod
    def _normalize_market_source(cls, value: object) -> object:
        """Treat empty string as absent; only ``history`` is allowed."""
        if value is None or value == "":
            return None
        if value != "history":
            raise ValueError(
                f"market_prior_source must be absent or 'history', got {value!r}"
            )
        return value


class Odds1X2Params(BaseModel):
    """Params for ``odds_1x2``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    use_closing_proxy: bool = True
    fallback_p_home: float = 1.0 / 3.0
    fallback_p_draw: float = 1.0 / 3.0
    fallback_p_away: float = 1.0 / 3.0


class OddsPoissonParams(BaseModel):
    """Params for ``odds_poisson``."""

    model_config = ConfigDict(extra="forbid", strict=True)

    use_closing_proxy: bool = True
    use_over_under: bool = True
    fallback_lambda_home: float = 1.35
    fallback_lambda_away: float = 1.1


PARAMS_BY_KIND: dict[str, type[BaseModel]] = {
    "involvement_shares": InvolvementSharesParams,
    "involvement_xg": InvolvementXgParams,
    "empirical_minutes": EmpiricalMinutesParams,
    "empirical_minutes_v2": EmpiricalMinutesV2Params,
    "other_player": OtherPlayerParams,
    "other_match_conditioned": OtherMatchConditionedParams,
    "other_priors": OtherPriorsParams,
    "dixon_coles": DixonColesParams,
    "poisson_score": DixonColesParams,
    "odds_1x2": Odds1X2Params,
    "odds_poisson": OddsPoissonParams,
}


def resolve_component_params(kind: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Validate ``raw`` and return a fully resolved params dict.

    Applies Dixon–Coles' dynamic ``use_closing_proxy`` default so the dump
    matches today's ``.get(..., default_proxy)`` behaviour.

    Args:
        kind: Registered component kind.
        raw: ``ModelSpec.params`` mapping.

    Returns:
        Resolved params including defaults.

    Raises:
        KeyError: Unknown component kind.
        ValueError: Validation failed.
    """
    model = PARAMS_BY_KIND[kind]
    params = parse_component_params(model, raw)
    dump = params.model_dump()
    if kind in {"dixon_coles", "poisson_score"} and dump["use_closing_proxy"] is None:
        dump["use_closing_proxy"] = float(dump["market_1x2_strength"]) > 0
    return dump


def parse_component_params[T: BaseModel](model: type[T], raw: dict[str, Any]) -> T:
    """Validate ``raw`` into ``model``, re-raising as ``ValueError``.

    Component ``__init__`` methods historically raise ``ValueError`` for bad
    params; wrapping keeps that contract while pydantic supplies defaults and
    ``extra="forbid"``.

    Args:
        model: Params class for one component kind.
        raw: ``ModelSpec.params`` mapping.

    Returns:
        Validated params instance.

    Raises:
        ValueError: Validation failed (unknown key, wrong type, or constraint).
    """
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
