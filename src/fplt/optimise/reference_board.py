"""Load the versioned reference board flags for season-points reports.

Committed defaults live in ``configs/boards/reference.yaml``. Live planner
constants must match these flags (see ``docs/eval-protocol.md`` §4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fplt.optimise.highs_options import HighsSolveOptions

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "boards" / "reference.yaml"
)

ChipArmName = Literal["off", "rule_based", "optimised"]
PriceForecastName = Literal["flat", "mean_gw_delta"]


class ReferenceHighs(BaseModel):
    """Deterministic HiGHS knobs for published boards (FDB-228)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    time_limit: float | None = None
    threads: int = Field(default=1, ge=1)
    parallel: Literal["on", "off"] = "off"
    random_seed: int = Field(default=0, ge=0)
    mip_rel_gap: float = Field(default=0.0, ge=0.0)
    accept_time_limit: bool = False
    presolve: Literal["choose", "on", "off"] = "on"
    output_flag: bool = False

    def to_highs_options(self) -> HighsSolveOptions:
        """Build ``HighsSolveOptions`` for ``plan_squad``."""
        return HighsSolveOptions(
            time_limit=self.time_limit,
            threads=self.threads,
            parallel=self.parallel,
            random_seed=self.random_seed,
            mip_rel_gap=self.mip_rel_gap,
            accept_time_limit=self.accept_time_limit,
            presolve=self.presolve,
            output_flag=self.output_flag,
        )


class ReferenceBoard(BaseModel):
    """Frozen flags boards report against; must match live defaults."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1)
    horizon: int = Field(ge=1)
    chip_arm: ChipArmName
    force_play: bool
    lineup_reopt: bool
    price_forecast: PriceForecastName
    bench_weight: float = Field(ge=0.0)
    model: str = Field(min_length=1)
    highs: ReferenceHighs = Field(default_factory=ReferenceHighs)

    @field_validator("model")
    @classmethod
    def _non_blank_model(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model must be a non-empty spec name")
        return cleaned


def default_reference_board_path() -> Path:
    """Return the repo-root path to ``configs/boards/reference.yaml``."""
    return _DEFAULT_PATH


def load_reference_board(path: Path | None = None) -> ReferenceBoard:
    """Load and validate the reference board YAML.

    Args:
        path: Config file. Defaults to ``configs/boards/reference.yaml``.

    Returns:
        Validated ``ReferenceBoard``.

    Raises:
        FileNotFoundError: The YAML path does not exist.
        ValueError: The file is empty, not a mapping, or fails validation.
    """
    source = default_reference_board_path() if path is None else path
    if not source.is_file():
        raise FileNotFoundError(f"reference board config not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{source.name} must contain a mapping")
    return ReferenceBoard.model_validate(loaded)
