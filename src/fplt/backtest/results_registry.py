"""Committed promote-results registry (verdict ↔ EvalSpec ↔ audit ↔ hashes).

Written keep/drop/tune calls used to live only in Project-store markdown and
git commit messages. This module loads ``configs/evals/results/registry.yaml``
so each call also links the EvalSpec (when present), window, committed
``audit.json`` path, and staged-data SHA-256 digests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WrittenVerdict = Literal[
    "keep",
    "tune",
    "drop",
    "inconclusive",
    "blocked",
    "aborted",
]
PromotePath = Literal["modelspec", "decision_policy"]


class DataDigest(BaseModel):
    """SHA-256 digest for one staged parquet named in an audit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    table: str
    sha256: str = Field(min_length=64, max_length=64)


class PromoteWindow(BaseModel):
    """One primary / confirm / combined window for a promote call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = Field(min_length=1)
    seasons: list[str] = Field(default_factory=list)
    from_gw: int | None = Field(default=None, ge=1)
    to_gw: int | None = Field(default=None, ge=1)
    audit: str | None = None
    git_sha: str | None = None
    git_dirty: bool | None = None
    uv_lock_sha256: str | None = None
    runner_verdict: str | None = None
    control_model_id: str | None = None
    treatment_model_id: str | None = None
    control_spec_sha256: str | None = None
    treatment_spec_sha256: str | None = None
    model_id: str | None = None
    data_sha256: dict[str, str] = Field(default_factory=dict)
    reconstructed_from: str | None = None
    notes: str | None = None
    note: str | None = None

    @field_validator("seasons")
    @classmethod
    def _season_names(cls, value: list[str]) -> list[str]:
        cleaned = [s.strip() for s in value]
        if any(not s for s in cleaned):
            raise ValueError("seasons entries must be non-empty")
        return cleaned

    @field_validator("data_sha256")
    @classmethod
    def _digest_hex(cls, value: dict[str, str]) -> dict[str, str]:
        for table, digest in value.items():
            if not table.strip():
                raise ValueError("data_sha256 keys must be non-empty table names")
            if len(digest) != 64:
                raise ValueError(
                    f"data_sha256[{table!r}] must be a 64-char sha256 hex digest"
                )
        return value

    @model_validator(mode="after")
    def _gw_order(self) -> PromoteWindow:
        if (
            self.from_gw is not None
            and self.to_gw is not None
            and self.from_gw > self.to_gw
        ):
            raise ValueError("from_gw must be <= to_gw")
        return self


class PromoteResult(BaseModel):
    """One written promote call recorded in the committed registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    date: str = Field(min_length=1)
    task: str | None = None
    path: PromotePath
    verdict: WrittenVerdict
    provisional: bool = False
    provisional_reason: str | None = None
    eval_spec: str | None = None
    experiment_id: str | None = None
    control: str | None = None
    treatment: str | None = None
    store_doc: str | None = None
    windows: list[PromoteWindow] = Field(default_factory=list[PromoteWindow])
    notes: str | None = None

    @field_validator("id")
    @classmethod
    def _id_slug(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(ch.isspace() for ch in cleaned):
            raise ValueError("id must be a non-empty slug without whitespace")
        return cleaned

    @model_validator(mode="after")
    def _provisional_reason(self) -> PromoteResult:
        if self.provisional and not (self.provisional_reason or "").strip():
            raise ValueError("provisional entries require provisional_reason")
        if not self.provisional and self.provisional_reason:
            raise ValueError("provisional_reason requires provisional: true")
        return self


class ResultsRegistry(BaseModel):
    """Top-level committed promote-results registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1)
    description: str = ""
    entries: list[PromoteResult] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> ResultsRegistry:
        seen: set[str] = set()
        for entry in self.entries:
            if entry.id in seen:
                raise ValueError(f"duplicate promote result id: {entry.id!r}")
            seen.add(entry.id)
        return self


def default_results_registry_path() -> Path:
    """Return the repo-root path to ``configs/evals/results/registry.yaml``."""
    return (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "evals"
        / "results"
        / "registry.yaml"
    )


def default_evals_root() -> Path:
    """Return ``configs/evals`` at the repo root (parent of ``results/``)."""
    return Path(__file__).resolve().parents[3] / "configs" / "evals"


def load_results_registry(
    path: Path | None = None,
    *,
    evals_root: Path | None = None,
    require_audits: bool = True,
) -> ResultsRegistry:
    """Load and validate the committed promote-results registry.

    Args:
        path: Registry YAML. Defaults to ``configs/evals/results/registry.yaml``.
        evals_root: Root for resolving ``window.audit`` paths (defaults to
            ``configs/evals``).
        require_audits: When True, every ``window.audit`` path must exist and
            its ``data`` digests (when present) must match ``data_sha256``.

    Returns:
        Validated ``ResultsRegistry``.

    Raises:
        FileNotFoundError: Registry YAML or a referenced audit is missing.
        ValueError: YAML shape / pydantic / audit digest mismatch.
    """
    source = default_results_registry_path() if path is None else path
    if not source.is_file():
        raise FileNotFoundError(f"promote results registry not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{source.name} must contain a mapping")
    registry = ResultsRegistry.model_validate(loaded)
    if require_audits:
        root = default_evals_root() if evals_root is None else evals_root
        _validate_audits(registry, evals_root=root)
    return registry


def entry_by_id(registry: ResultsRegistry, entry_id: str) -> PromoteResult:
    """Return one registry entry by ``id``.

    Args:
        registry: Loaded registry.
        entry_id: Stable slug (e.g. ``defcon_rate``).

    Returns:
        Matching ``PromoteResult``.

    Raises:
        KeyError: No entry with that id.
    """
    for entry in registry.entries:
        if entry.id == entry_id:
            return entry
    raise KeyError(entry_id)


def provisional_entries(registry: ResultsRegistry) -> list[PromoteResult]:
    """Return entries whose keep (or other call) is still provisional."""
    return [entry for entry in registry.entries if entry.provisional]


def _validate_audits(registry: ResultsRegistry, *, evals_root: Path) -> None:
    for entry in registry.entries:
        for window in entry.windows:
            if window.audit is None:
                continue
            audit_path = evals_root / window.audit
            if not audit_path.is_file():
                raise FileNotFoundError(
                    f"{entry.id}/{window.role}: audit not found: {audit_path}"
                )
            payload = _read_audit(audit_path)
            data = _mapping(payload.get("data"))
            if data is None or not window.data_sha256:
                continue
            for table, expected in window.data_sha256.items():
                actual = _mapping(data.get(table))
                if actual is None:
                    raise ValueError(
                        f"{entry.id}/{window.role}: audit missing data[{table!r}]"
                    )
                digest = actual.get("sha256")
                if digest != expected:
                    raise ValueError(
                        f"{entry.id}/{window.role}: data_sha256[{table!r}] "
                        f"mismatch vs {audit_path.name}"
                    )


def _mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    items = cast(dict[object, object], value)
    return {str(key): item for key, item in items.items()}


def _read_audit(path: Path) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    mapped = _mapping(raw)
    if mapped is None:
        raise ValueError(f"{path.name} must contain a JSON object")
    return mapped


__all__ = [
    "DataDigest",
    "PromotePath",
    "PromoteResult",
    "PromoteWindow",
    "ResultsRegistry",
    "WrittenVerdict",
    "default_evals_root",
    "default_results_registry_path",
    "entry_by_id",
    "load_results_registry",
    "provisional_entries",
]
