"""Run identity fingerprints: required tables, fp, fit_fp, and ``by_fp`` index.

A run's identity is what it ran on (#346 / #374). ``fp`` hashes the resolved
spec, prod tag, target GW, run type, run kind, and content shas of
``required_tables(spec)``. Kind is part of the hash so mid-week
``kind=adhoc`` fingerprint runs do not collide with official T−90m
``by_fp`` entries (AGENTS.md §7). When ``runs/by_fp/fp=<fp>.json`` already
exists, ops skips the work.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fplt.models.spec import ModelSpec
from fplt.models.spec_identity import spec_hash
from fplt.schemas.manifests import BuildManifest, RunKind, RunType
from fplt.schemas.staged import STAGED_SCHEMAS
from fplt.types import GWKey

logger = logging.getLogger(__name__)

# Fixed tables every ops predict/plan path needs (planner + outcomes + PIT).
PLANNER_AND_OUTCOME_TABLES: frozenset[str] = frozenset(
    {
        "players",
        "teams",
        "gameweeks",
        "fixtures",
        "player_status",
        "player_fixture_stats",
        "player_fixture_outcomes",
        "fixture_outcomes",
    }
)

# Live-ish tables excluded from the fit fingerprint so mid-week price/news
# churn does not force a weekly refit (#346: fit_fp changes ~weekly).
_FIT_EXCLUDE_TABLES: frozenset[str] = frozenset(
    {
        "player_status",
        "predicted_lineups",
    }
)

# Declared staged inputs per model/component kind (kinds that read PIT tables
# without going through ``@feature(inputs=…)``). Unioned into required_tables.
_KIND_INPUT_TABLES: dict[str, frozenset[str]] = {
    "rolling_mean": frozenset({"player_fixture_stats", "players", "fixtures"}),
    "ep_next": frozenset({"player_status", "players", "fixtures"}),
    "openfpl": frozenset(
        {
            "players",
            "fixtures",
            "player_fixture_stats",
            "player_status",
            "understat_player_matches",
            "understat_team_matches",
        }
    ),
    "dc_layered": frozenset(
        {
            "players",
            "teams",
            "gameweeks",
            "fixtures",
            "player_status",
            "player_fixture_stats",
            "understat_player_matches",
            "understat_team_matches",
            "match_odds",
            "player_injuries",
            "competition_fixtures",
            "predicted_lineups",
        }
    ),
    "dixon_coles": frozenset(
        {"fixtures", "players", "match_odds", "player_fixture_stats", "teams"}
    ),
    "empirical_minutes": frozenset(
        {"players", "fixtures", "player_status", "player_fixture_stats"}
    ),
    "empirical_minutes_v2": frozenset(
        {
            "players",
            "fixtures",
            "player_status",
            "player_fixture_stats",
            "gameweeks",
            "player_injuries",
            "competition_fixtures",
            "predicted_lineups",
        }
    ),
    "involvement_xg": frozenset(
        {
            "players",
            "fixtures",
            "player_status",
            "player_fixture_stats",
            "understat_player_matches",
        }
    ),
    "other_player": frozenset(
        {"players", "fixtures", "player_fixture_stats", "player_status"}
    ),
    "forecast_combine": frozenset({"players", "fixtures"}),
}


class FpIndexEntry(BaseModel):
    """Write-once index row at ``runs/by_fp/fp=<fp>.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)


def ensure_features_registered() -> None:
    """Import feature modules so ``@feature`` registrations are visible."""
    import fplt.features.availability as availability_rows
    import fplt.features.form as form_features
    import fplt.features.understat as understat_rows

    if form_features.context.__name__ != "context":
        raise RuntimeError("context did not register")
    if understat_rows.understat_share.__name__ != "understat_share":
        raise RuntimeError("understat_share did not register")
    if availability_rows.availability_v1.__name__ != "availability_v1":
        raise RuntimeError("availability_v1 did not register")


def required_tables(spec: ModelSpec) -> set[str]:
    """Return staged table names a predict/plan for ``spec`` must read.

    Union of ``@feature(inputs=…)`` for every feature the spec (and nested
    components) declares, kind-declared PIT inputs, and the fixed planner /
    outcomes set. Every feature input must be a key of ``STAGED_SCHEMAS``.

    Args:
        spec: Model spec (features + nested components walked recursively).

    Returns:
        Set of staged table names.

    Raises:
        KeyError: A declared feature is not registered.
        ValueError: A feature lists an input missing from ``STAGED_SCHEMAS``.
    """
    ensure_features_registered()
    from fplt.features.registry import get_feature

    tables = set(PLANNER_AND_OUTCOME_TABLES)
    for item in _iter_specs(spec):
        tables |= set(_KIND_INPUT_TABLES.get(item.kind, frozenset()))
        for feature_id in item.features:
            name, version = _parse_feature(feature_id)
            feature = get_feature(name, version)
            for table in feature.inputs:
                if table not in STAGED_SCHEMAS:
                    raise ValueError(
                        f"feature {name}@{version} input {table!r} is not in "
                        "STAGED_SCHEMAS"
                    )
                tables.add(table)
    return tables


def run_fingerprint(
    *,
    spec: ModelSpec,
    build: BuildManifest,
    target_gw: GWKey,
    run_type: RunType,
    prod_tag: str | None = None,
    kind: RunKind = "official",
    tables: set[str] | None = None,
    extra: Mapping[str, object] | None = None,
) -> str:
    """Compute the run identity fingerprint for one spec / build / GW.

    ``fp = sha256(spec_hash, prod_tag, target_gw, run_type, kind, table shas)``.
    Kind is part of the hash (#374) so mid-week ``adhoc`` fingerprint runs
    do not collide with official T−90m. Plan runs also pass ``extra`` (#380):
    official line, horizon, squad mode and a squad-state digest, so lines
    sharing a spec never collide and a changed squad re-plans. ``extra`` is
    hashed under ``"extra"`` with ``"v": 2``.

    Args:
        spec: Model that will run.
        build: Curated build supplying table content shas.
        target_gw: Deadline gameweek the run covers.
        run_type: ``predict`` / ``plan`` / ``eval``.
        prod_tag: Moving ops tag label, or ``None``.
        kind: Run kind (``official`` vs ``adhoc`` fingerprint, …). Part of
            the hash so light mid-week runs do not block official PMF runs
            (#374 / AGENTS.md §7).
        tables: Override for ``required_tables(spec)`` (tests).
        extra: Additional identity fields (JSON-serialisable), or ``None``.

    Returns:
        64-character lowercase hex digest. Required tables absent from the
        build (empty at curate time) contribute ``sha=\"\"``.
    """
    needed = required_tables(spec) if tables is None else tables
    digest = spec_hash(spec)
    pairs = _table_sha_pairs(build, needed)
    payload: dict[str, object] = {
        "spec_hash": digest,
        "prod_tag": prod_tag,
        "target_gw": {"season": target_gw.season, "gw": target_gw.gw},
        "run_type": run_type,
        "kind": kind,
        "tables": pairs,
    }
    if extra is not None:
        payload["v"] = 2
        payload["extra"] = dict(extra)
    return _sha256_canonical(payload)


def fit_fingerprint(
    *,
    spec: ModelSpec,
    build: BuildManifest,
    last_completed: GWKey | None,
    tables: set[str] | None = None,
) -> str:
    """Fingerprint for reusing a fitted artifact across mid-week ticks.

    Same shape as ``run_fingerprint`` over the spec and training inputs, but
    restricted to completed-GW identity: live tables (``player_status``,
    ``predicted_lineups``) are excluded so price/news churn does not refit.
    Advances when ``last_completed`` moves (about weekly).

    Args:
        spec: Model being fit.
        build: Curated build supplying table content shas.
        last_completed: Latest finished / data-checked GW, or ``None``.
        tables: Override for ``required_tables(spec)`` (tests).

    Returns:
        64-character lowercase hex digest.
    """
    needed = required_tables(spec) if tables is None else tables
    fit_tables = needed - _FIT_EXCLUDE_TABLES
    digest = spec_hash(spec)
    pairs = _table_sha_pairs(build, fit_tables)
    completed = (
        None
        if last_completed is None
        else {"season": last_completed.season, "gw": last_completed.gw}
    )
    payload = {
        "spec_hash": digest,
        "last_completed": completed,
        "tables": pairs,
    }
    return _sha256_canonical(payload)


def by_fp_path(data_dir: Path, fp: str) -> Path:
    """Local path ``data_dir/runs/by_fp/fp=<fp>.json``."""
    return data_dir / "runs" / "by_fp" / f"fp={fp}.json"


def by_fp_remote_key(fp: str) -> str:
    """Remote object key for the write-once fp index entry."""
    return f"runs/by_fp/fp={fp}.json"


def load_fp_index(data_dir: Path, fp: str) -> FpIndexEntry | None:
    """Load a local ``by_fp`` entry, or ``None`` when absent."""
    path = by_fp_path(data_dir, fp)
    if not path.is_file():
        return None
    return FpIndexEntry.model_validate_json(path.read_text(encoding="utf-8"))


def write_fp_index(data_dir: Path, fp: str, run_id: str) -> Path:
    """Write ``runs/by_fp/fp=<fp>.json`` once after the run's ``_SUCCESS``.

    Args:
        data_dir: Local data root.
        fp: Run fingerprint.
        run_id: Succeeded run id to record.

    Returns:
        Path written.

    Raises:
        FileExistsError: An index entry for ``fp`` already exists.
    """
    path = by_fp_path(data_dir, fp)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    entry = FpIndexEntry(run_id=run_id)
    path.write_text(entry.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("by_fp: wrote fp=%s → run_id=%s", fp[:12], run_id)
    return path


def fp_index_exists(
    data_dir: Path,
    fp: str,
    *,
    remote_listing: set[str] | None = None,
) -> bool:
    """Return whether ``by_fp`` already records ``fp``.

    Prefers an optional remote ``lsf`` listing of ``runs/by_fp/``; falls back
    to the local index file.
    """
    if remote_listing is not None:
        name = f"fp={fp}.json"
        if name in remote_listing or by_fp_remote_key(fp) in remote_listing:
            return True
    return by_fp_path(data_dir, fp).is_file()


def last_completed_gw(gameweeks: object) -> GWKey | None:
    """Latest season/gw with ``finished`` or ``data_checked`` true.

    Args:
        gameweeks: Polars DataFrame of staged ``gameweeks`` rows.

    Returns:
        The max ``(season, gw)`` among completed rows, or ``None``.
    """
    import polars as pl

    if not isinstance(gameweeks, pl.DataFrame) or gameweeks.is_empty():
        return None
    frame: pl.DataFrame = gameweeks
    if "finished" in frame.columns and "data_checked" in frame.columns:
        done = frame.filter(pl.col("finished") | pl.col("data_checked"))
    elif "data_checked" in frame.columns:
        done = frame.filter(pl.col("data_checked"))
    elif "finished" in frame.columns:
        done = frame.filter(pl.col("finished"))
    else:
        return None
    if done.is_empty():
        return None
    latest = done.sort(["season", "gw"]).tail(1)
    return GWKey(str(latest["season"][0]), int(latest["gw"][0]))


def _iter_specs(spec: ModelSpec) -> list[ModelSpec]:
    out = [spec]
    for child in spec.components.values():
        out.extend(_iter_specs(child))
    return out


def _parse_feature(item: str) -> tuple[str, int]:
    name, version = item.split("@", maxsplit=1)
    return name, int(version)


def _table_sha_pairs(build: BuildManifest, tables: set[str]) -> list[tuple[str, str]]:
    # Empty staged tables are omitted from builds (#345). Treat absence as
    # sha="" so the fingerprint still covers every required name.
    pairs: list[tuple[str, str]] = []
    for name in sorted(tables):
        entry = build.tables.get(name)
        pairs.append((name, entry.sha if entry is not None else ""))
    return pairs


def _sha256_canonical(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


RunFpKind = Literal["predict", "plan"]
