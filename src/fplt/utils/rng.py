"""Seeded generators keyed by model, deadline, and fixture."""

from __future__ import annotations

import hashlib

import numpy as np

from fplt.types import GWKey


def seed_for(model_id: str, as_of: GWKey, fixture_id: int) -> np.random.Generator:
    """Return a generator determined by model, as-of gameweek, and fixture.

    The same triple always yields the same stream. Nothing else draws from it.
    """
    payload = f"{model_id}|{as_of.season}|{as_of.gw}|{fixture_id}".encode()
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)
