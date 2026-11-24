"""Per-step objective weights for the transfer planner.

Live control uses geometric ``SquadState.decay ** step`` (YAML ``decay: 0.9``).
FDB-242 treatment replaces that schedule with weights from measured
skill-by-horizon RMSE ratios (``1 / rmse_vs_h1``), which decay far more
slowly than ``0.9^t`` on current ledgers. Does **not** flip live defaults
until a season-points keep.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

HorizonDecayMethod = Literal["geometric", "skill"]

_DEFAULT_SKILL_WEIGHTS = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "optimise"
    / "horizon_skill_weights.yaml"
)


class SkillWeightConfig(BaseModel):
    """Measured RMSE ratios used to build skill step weights."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: Literal["rmse"] = "rmse"
    power: float = Field(default=1.0, gt=0.0)
    rmse_vs_h1: tuple[float, ...] = Field(min_length=1)
    model: str | None = None
    model_id: str | None = None
    season: str | None = None
    notes: str | None = None

    @field_validator("rmse_vs_h1")
    @classmethod
    def _ratios_positive(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(ratio <= 0.0 for ratio in value):
            raise ValueError("rmse_vs_h1 entries must be positive")
        if abs(value[0] - 1.0) > 1e-6:
            raise ValueError("rmse_vs_h1[0] (H=1) must be 1.0")
        return value


def geometric_step_weights(decay: float, horizon: int) -> tuple[float, ...]:
    """Return ``decay ** step`` for ``step`` in ``0..horizon-1``.

    Args:
        decay: Planner scalar in ``(0, 1]`` (``SquadState.decay``).
        horizon: Number of steps.

    Returns:
        Length-``horizon`` weights with ``weights[0] == 1.0`` when
        ``decay`` is finite.

    Raises:
        ValueError: ``decay`` or ``horizon`` is invalid.
    """
    if not 0.0 < decay <= 1.0:
        raise ValueError(f"decay must be in (0, 1], got {decay}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return tuple(float(decay**step) for step in range(horizon))


def skill_step_weights(
    rmse_vs_h1: Sequence[float],
    *,
    horizon: int,
    power: float = 1.0,
) -> tuple[float, ...]:
    """Map RMSE-vs-H1 ratios to objective step weights.

    ``weight[step] = (1 / rmse_vs_h1[step]) ** power``. When ``horizon``
    exceeds the measured curve, the last ratio is reused (flat
    extrapolation). H=1 must be ratio ``1.0`` so ``weight[0] == 1.0``.

    Args:
        rmse_vs_h1: Ratios for lead times H=1, H=2, … (index 0 = H=1).
        horizon: Number of planner steps to emit.
        power: Exponent on the inverse ratio (default 1).

    Returns:
        Length-``horizon`` positive weights.

    Raises:
        ValueError: Empty ratios, non-positive entries, bad power/horizon.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if power <= 0.0:
        raise ValueError(f"power must be > 0, got {power}")
    if not rmse_vs_h1:
        raise ValueError("rmse_vs_h1 must be non-empty")
    ratios = [float(r) for r in rmse_vs_h1]
    if any(r <= 0.0 for r in ratios):
        raise ValueError("rmse_vs_h1 entries must be positive")
    weights: list[float] = []
    for step in range(horizon):
        ratio = ratios[step] if step < len(ratios) else ratios[-1]
        weights.append((1.0 / ratio) ** power)
    return tuple(weights)


def weights_from_skill_curve(
    curve: pl.DataFrame,
    *,
    horizon: int,
    model_id: str | None = None,
    power: float = 1.0,
) -> tuple[float, ...]:
    """Build step weights from a ``HorizonSkillSchema`` curve.

    Prefers ``rmse_vs_h1``. When ``model_id`` is set, filters to that arm.

    Args:
        curve: Skill-by-horizon frame with ``horizon`` and ``rmse_vs_h1``.
        horizon: Planner steps to emit.
        model_id: Optional model filter when several arms are present.
        power: Exponent forwarded to :func:`skill_step_weights`.

    Returns:
        Length-``horizon`` weights.

    Raises:
        ValueError: Curve empty / missing columns, or no rows after filter.
    """
    if curve.is_empty():
        raise ValueError("skill curve is empty")
    needed = {"horizon", "rmse_vs_h1"}
    missing = needed - set(curve.columns)
    if missing:
        raise ValueError(f"skill curve missing {sorted(missing)}")
    frame = curve
    if model_id is not None:
        if "model_id" not in frame.columns:
            raise ValueError("skill curve has no model_id column to filter")
        frame = frame.filter(pl.col("model_id") == model_id)
        if frame.is_empty():
            raise ValueError(f"no skill rows for model_id={model_id!r}")
    ordered = (
        frame.select(
            pl.col("horizon").cast(pl.Int64),
            pl.col("rmse_vs_h1").cast(pl.Float64),
        )
        .unique(subset=["horizon"], keep="first")
        .sort("horizon")
    )
    horizons = ordered["horizon"].to_list()
    if horizons != list(range(1, len(horizons) + 1)):
        raise ValueError(
            f"skill curve horizons must be 1..N contiguous, got {horizons}"
        )
    return skill_step_weights(
        ordered["rmse_vs_h1"].to_list(),
        horizon=horizon,
        power=power,
    )


def load_skill_weight_config(path: Path | None = None) -> SkillWeightConfig:
    """Load measured RMSE ratios from YAML.

    Args:
        path: Config file. Defaults to
            ``configs/optimise/horizon_skill_weights.yaml``.

    Returns:
        Validated :class:`SkillWeightConfig`.

    Raises:
        FileNotFoundError: The file is missing.
        ValueError: The document is not a mapping / fails validation.
    """
    source = _DEFAULT_SKILL_WEIGHTS if path is None else path
    loaded_raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded_raw, dict):
        raise ValueError(f"{source.name} must contain a mapping")
    payload = cast(dict[str, Any], loaded_raw)
    # Flatten optional provenance block into top-level optional fields.
    source_meta = payload.pop("source", None)
    if isinstance(source_meta, dict):
        meta = cast(dict[str, Any], source_meta)
        for key in ("model", "model_id", "season", "notes"):
            if key in meta and key not in payload:
                payload[key] = meta[key]
    return SkillWeightConfig.model_validate(payload)


def resolve_step_weights(
    horizon: int,
    *,
    decay: float,
    method: HorizonDecayMethod = "geometric",
    skill_config: SkillWeightConfig | None = None,
    skill_rmse_vs_h1: Sequence[float] | None = None,
    power: float | None = None,
) -> tuple[float, ...]:
    """Resolve planner step weights for ``method``.

    Args:
        horizon: Plan length.
        decay: Geometric control scalar (``SquadState.decay``).
        method: ``geometric`` (live control) or ``skill`` (measured).
        skill_config: Optional loaded YAML config for the skill path.
        skill_rmse_vs_h1: Explicit ratios; overrides ``skill_config`` ratios.
        power: Explicit power; overrides ``skill_config.power`` when set.

    Returns:
        Length-``horizon`` weights.

    Raises:
        ValueError: Unknown method, or skill path missing ratios.
    """
    if method == "geometric":
        return geometric_step_weights(decay, horizon)
    if method != "skill":
        raise ValueError(f"unknown horizon decay method {method!r}")
    ratios: Sequence[float]
    resolved_power: float
    if skill_rmse_vs_h1 is not None:
        ratios = skill_rmse_vs_h1
        resolved_power = 1.0 if power is None else float(power)
    elif skill_config is not None:
        ratios = skill_config.rmse_vs_h1
        resolved_power = float(skill_config.power if power is None else power)
    else:
        cfg = load_skill_weight_config()
        ratios = cfg.rmse_vs_h1
        resolved_power = float(cfg.power if power is None else power)
    return skill_step_weights(ratios, horizon=horizon, power=resolved_power)


def validate_step_weights(
    weights: Sequence[float],
    *,
    horizon: int,
) -> tuple[float, ...]:
    """Validate caller-supplied step weights for ``plan_squad``.

    Args:
        weights: Per-step objective weights (0-indexed).
        horizon: Required length.

    Returns:
        The weights as a tuple.

    Raises:
        ValueError: Length mismatch or non-positive entries.
    """
    if len(weights) < horizon:
        raise ValueError(f"step_weights length {len(weights)} < horizon {horizon}")
    out = tuple(float(w) for w in weights[:horizon])
    if any(w <= 0.0 for w in out):
        raise ValueError("step_weights must be positive")
    return out


__all__ = [
    "HorizonDecayMethod",
    "SkillWeightConfig",
    "geometric_step_weights",
    "load_skill_weight_config",
    "resolve_step_weights",
    "skill_step_weights",
    "validate_step_weights",
    "weights_from_skill_curve",
]
