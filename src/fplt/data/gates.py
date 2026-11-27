"""Pure capture sanity gates for ops-06 / A10.

Compares each new bootstrap+fixtures capture with the previous good one.
Failures become ``QuarantineRecordSchema`` rows; raw stays write-once.
Thresholds: ``configs/ops/data_gates.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.quarantine import QuarantineRecordSchema
from fplt.utils.frames import require_dataframe

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "ops" / "data_gates.yaml"
)
GATE_PLAYER_COUNT = "player_count"
GATE_NULL_SHARE = "null_share"
GATE_NEXT_GW_FIXTURES = "next_gw_fixtures"
GATE_DEADLINES_MONOTONE = "deadlines_monotone"
GATE_DATA_CHECKED = "data_checked_monotone"
_NULL_FIELDS = ("now_cost", "status", "team")
_EMPTY_QUARANTINE: dict[str, pl.DataType] = {
    "season": pl.Utf8(),
    "source": pl.Utf8(),
    "capture_path": pl.Utf8(),
    "captured_at": pl.Datetime(time_zone="UTC"),
    "gate": pl.Utf8(),
    "detail": pl.Utf8(),
    "previous_capture_path": pl.Utf8(),
    "available_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Int64(),
}


class DataGatesConfig(BaseModel):
    """Thresholds for bootstrap capture sanity gates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_count_max_delta: int = Field(ge=0)
    max_null_share: dict[str, float]
    expected_team_count: int = Field(ge=1)

    @field_validator("max_null_share")
    @classmethod
    def _null_shares(cls, value: dict[str, float]) -> dict[str, float]:
        missing = [name for name in _NULL_FIELDS if name not in value]
        if missing:
            raise ValueError(f"max_null_share missing keys: {missing}")
        for name, share in value.items():
            if not 0.0 <= float(share) <= 1.0:
                raise ValueError(f"max_null_share.{name} must be in [0, 1]")
        return {key: float(share) for key, share in value.items()}


@dataclass(frozen=True)
class GateResult:
    """Outcome of one named sanity gate."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class CaptureGateReport:
    """All gate results for one capture."""

    results: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        """True when every gate passed."""
        return all(result.passed for result in self.results)

    @property
    def failures(self) -> tuple[GateResult, ...]:
        """Gates that failed."""
        return tuple(result for result in self.results if not result.passed)


def default_data_gates_path() -> Path:
    """Return the repo-root path to ``configs/ops/data_gates.yaml``."""
    return _DEFAULT_PATH


def load_data_gates_config(path: Path | None = None) -> DataGatesConfig:
    """Load gate thresholds from YAML (default ``configs/ops/data_gates.yaml``)."""
    source = default_data_gates_path() if path is None else path
    if not source.is_file():
        raise FileNotFoundError(f"data gates config not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{source.name} must contain a mapping")
    return DataGatesConfig.model_validate(cast(dict[str, Any], loaded))


def run_bootstrap_gates(
    bootstrap: Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
    *,
    previous: Mapping[str, Any] | None = None,
    config: DataGatesConfig | None = None,
) -> CaptureGateReport:
    """Run all bootstrap capture sanity gates vs an optional previous good capture."""
    cfg = load_data_gates_config() if config is None else config
    return CaptureGateReport(
        results=(
            _gate_player_count(bootstrap, previous, cfg),
            _gate_null_share(bootstrap, cfg),
            _gate_next_gw_fixtures(bootstrap, fixtures, cfg),
            _gate_deadlines_monotone(bootstrap),
            _gate_data_checked(bootstrap, previous),
        )
    )


def quarantine_records(
    report: CaptureGateReport,
    *,
    season: str,
    capture_path: str,
    captured_at: datetime,
    available_at: datetime,
    source: str = "fpl_api",
    previous_capture_path: str | None = None,
) -> pl.DataFrame:
    """Build a validated ``QuarantineRecordSchema`` frame from gate failures."""
    if report.passed:
        return pl.DataFrame(schema=_EMPTY_QUARANTINE)
    stamp = captured_at.astimezone(UTC)
    written = available_at.astimezone(UTC)
    frame = pl.DataFrame(
        [
            {
                "season": season,
                "source": source,
                "capture_path": capture_path,
                "captured_at": stamp,
                "gate": failure.name,
                "detail": failure.detail,
                "previous_capture_path": previous_capture_path,
                "available_at": written,
                "schema_version": SCHEMA_VERSION,
            }
            for failure in report.failures
        ]
    )
    checked = QuarantineRecordSchema.validate(frame)
    checked = require_dataframe(checked)
    return checked


def _rows(bootstrap: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    raw = bootstrap.get(key)
    if not isinstance(raw, list):
        return []
    return [cast(dict[str, Any], row) for row in cast(list[Any], raw)]


def _gate_player_count(
    bootstrap: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    config: DataGatesConfig,
) -> GateResult:
    current_n = len(_rows(bootstrap, "elements"))
    if previous is None:
        return GateResult(
            GATE_PLAYER_COUNT,
            True,
            f"no previous capture; current player count={current_n}",
        )
    previous_n = len(_rows(previous, "elements"))
    delta = abs(current_n - previous_n)
    return GateResult(
        GATE_PLAYER_COUNT,
        delta <= config.player_count_max_delta,
        (
            f"player count {previous_n} → {current_n} "
            f"(|Δ|={delta}, max={config.player_count_max_delta})"
        ),
    )


def _gate_null_share(
    bootstrap: Mapping[str, Any], config: DataGatesConfig
) -> GateResult:
    elements = _rows(bootstrap, "elements")
    if not elements:
        return GateResult(GATE_NULL_SHARE, False, "bootstrap has zero elements")
    parts: list[str] = []
    failed = False
    for field in _NULL_FIELDS:
        share = sum(1 for row in elements if row.get(field) is None) / len(elements)
        limit = config.max_null_share[field]
        parts.append(f"{field}={share:.4f} (max={limit})")
        failed = failed or share > limit
    return GateResult(GATE_NULL_SHARE, not failed, "; ".join(parts))


def _next_event_id(events: Sequence[Mapping[str, Any]]) -> int | None:
    for event in events:
        if bool(event.get("is_next")):
            return int(event["id"])
    for event in events:
        if not bool(event.get("finished")):
            return int(event["id"])
    return max((int(event["id"]) for event in events), default=None)


def _gate_next_gw_fixtures(
    bootstrap: Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
    config: DataGatesConfig,
) -> GateResult:
    team_n = len(_rows(bootstrap, "teams"))
    next_id = _next_event_id(_rows(bootstrap, "events"))
    if next_id is None:
        return GateResult(GATE_NEXT_GW_FIXTURES, False, "bootstrap has no events")
    gw_fixtures = [row for row in fixtures if row.get("event") == next_id]
    if not gw_fixtures:
        return GateResult(
            GATE_NEXT_GW_FIXTURES,
            False,
            f"no fixtures for next gw={next_id}; teams={team_n}",
        )
    unique: set[int] = set()
    for row in gw_fixtures:
        for key in ("team_h", "team_a"):
            value = row.get(key)
            if value is not None:
                unique.add(int(value))
    expected = config.expected_team_count
    return GateResult(
        GATE_NEXT_GW_FIXTURES,
        team_n == expected and len(unique) == expected,
        (
            f"next gw={next_id}; bootstrap_teams={team_n}; "
            f"fixture_teams={len(unique)}; expected={expected}; "
            f"fixtures={len(gw_fixtures)}"
        ),
    )


def _gate_deadlines_monotone(bootstrap: Mapping[str, Any]) -> GateResult:
    events = sorted(_rows(bootstrap, "events"), key=lambda row: int(row["id"]))
    if len(events) < 2:
        return GateResult(
            GATE_DEADLINES_MONOTONE,
            True,
            f"fewer than 2 events ({len(events)}); monotone check skipped",
        )
    deadlines: list[datetime] = []
    for event in events:
        raw = event.get("deadline_time")
        if not isinstance(raw, str):
            return GateResult(
                GATE_DEADLINES_MONOTONE,
                False,
                f"event id={event.get('id')} missing deadline_time",
            )
        deadlines.append(
            datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
        )
    for left, right in zip(deadlines, deadlines[1:], strict=False):
        if right < left:
            return GateResult(
                GATE_DEADLINES_MONOTONE,
                False,
                f"deadlines not monotone: {left.isoformat()} → {right.isoformat()}",
            )
    return GateResult(
        GATE_DEADLINES_MONOTONE,
        True,
        f"{len(deadlines)} event deadlines are non-decreasing",
    )


def _gate_data_checked(
    bootstrap: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
) -> GateResult:
    if previous is None:
        return GateResult(
            GATE_DATA_CHECKED,
            True,
            "no previous capture; data_checked regression check skipped",
        )
    previous_by_id = {
        int(event["id"]): bool(event.get("data_checked"))
        for event in _rows(previous, "events")
    }
    regressions = [
        str(int(event["id"]))
        for event in _rows(bootstrap, "events")
        if previous_by_id.get(int(event["id"])) and not bool(event.get("data_checked"))
    ]
    if regressions:
        return GateResult(
            GATE_DATA_CHECKED,
            False,
            f"data_checked flipped true→false for gw={','.join(regressions)}",
        )
    return GateResult(
        GATE_DATA_CHECKED,
        True,
        "no data_checked true→false regressions vs previous capture",
    )
