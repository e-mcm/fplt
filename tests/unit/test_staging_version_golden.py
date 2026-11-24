"""Golden hash: staging output must not drift without bumping STAGING_VERSION.

Records the SHA-256 of ``stage_bootstrap`` + ``stage_event_live`` over the
committed snapshot fixtures. If a staging transform changes output, bump
``STAGING_VERSION`` in ``fplt.store.builds`` and update ``_GOLDEN`` here.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fplt.data.staging import stage_bootstrap, stage_event_live
from fplt.store.builds import STAGING_VERSION

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "staging_golden"
_CAPTURED = datetime(2024, 8, 10, 12, 0, 0, tzinfo=UTC)
_SEASON = "2024-25"

# Map STAGING_VERSION → content digest of the recorded fixture staging output.
# When bumping STAGING_VERSION, add a new entry with the new digest; keep old
# entries so history of intentional bumps stays auditable.
_GOLDEN: dict[int, str] = {
    1: "98486842a59fb2163a17a79270f7c2070984c5c3969f509f7c01477aeeba7c6d",
}


def _canonical_frame_digest(frame: pl.DataFrame) -> str:
    """Stable hex digest of one frame (column-sorted CSV bytes)."""
    if frame.is_empty():
        return hashlib.sha256(b"").hexdigest()
    ordered = frame.select(sorted(frame.columns)).sort(sorted(frame.columns))
    return hashlib.sha256(ordered.write_csv().encode()).hexdigest()


def _stage_fixture_digest() -> str:
    bootstrap = json.loads((_FIXTURES / "bootstrap_static.json").read_text())
    fixtures = json.loads((_FIXTURES / "fixtures.json").read_text())
    live = json.loads((_FIXTURES / "event_live.json").read_text())
    frames = stage_bootstrap(bootstrap, fixtures, _CAPTURED, _SEASON)
    players = frames["players"]
    element_to_player = {
        int(row["element_id"]): int(row["player_key"])
        for row in players.iter_rows(named=True)
    }
    frames["player_fixture_stats"] = stage_event_live(
        live,
        _SEASON,
        gw=1,
        available_at=_CAPTURED,
        element_to_player=element_to_player,
    )
    hasher = hashlib.sha256()
    hasher.update(f"staging_version={STAGING_VERSION}\n".encode())
    for name in sorted(frames):
        hasher.update(f"{name}:{_canonical_frame_digest(frames[name])}\n".encode())
    return hasher.hexdigest()


def test_staging_version_golden_hash() -> None:
    digest = _stage_fixture_digest()
    assert STAGING_VERSION in _GOLDEN, (
        f"STAGING_VERSION={STAGING_VERSION} missing from _GOLDEN; add digest {digest!r}"
    )
    expected = _GOLDEN[STAGING_VERSION]
    if expected == "REPLACE_ME":
        # First landing: print the digest so the PR can pin it.
        raise AssertionError(f"pin _GOLDEN[{STAGING_VERSION}] = {digest!r} then re-run")
    assert digest == expected, (
        "staging output changed without a STAGING_VERSION bump. "
        f"got {digest!r}, expected {expected!r}. "
        "If the change is intentional, bump STAGING_VERSION in "
        "fplt.store.builds and add the new digest to _GOLDEN."
    )
