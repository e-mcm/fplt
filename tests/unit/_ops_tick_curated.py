"""Shared curated-context stubs for ops-tick unit tests (#346)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from fplt.schemas.manifests import (
    BuildManifest,
    BuildTableEntry,
    CuratedLatestPointer,
)


def stub_build_context(*, build_id: str = "c" * 64) -> Any:
    """Minimal ``OpsBuildContext``-shaped object for monkeypatching."""
    # Fresh pointer so T−90m / T−30m freshness guard does not skip.
    now = datetime.now(UTC)
    pointer = CuratedLatestPointer(
        build_id=build_id,
        created_at=now,
        raw_cutoff=now,
        code_sha="abc1234",
    )
    build = BuildManifest(
        build_id=build_id,
        created_at=now,
        staging_version=1,
        code_sha="abc1234",
        raw_manifest_hash="r" * 64,
        raw_cutoff=now,
        tables={
            "gameweeks": BuildTableEntry(sha="g" * 64, rows=1, schema_hash="s" * 64)
        },
    )
    return type(
        "Ctx",
        (),
        {
            "pointer": pointer,
            "build": build,
            "remote_by_fp": frozenset(),
        },
    )()


def patch_curated_ops_path(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Replace curated load / table pulls; return a list of pull call records."""
    pulls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_ops_build_context",
        lambda *a, **k: stub_build_context(),
    )

    def fake_pull(*args: Any, **kwargs: Any) -> list[str]:
        pulls.append(args)
        return []

    monkeypatch.setattr("fplt.automation.ops_tick.pull_required_tables", fake_pull)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.decisions_for_slate",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.tables_for_slate",
        lambda *a, **k: {"gameweeks"},
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.ensure_slate_runs",
        lambda *a, **k: ("c" * 64, ()),
    )
    return pulls
