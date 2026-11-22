"""Seeded generators keyed by model, deadline, and fixture.

``seed_for`` is the single-model production stream. Promote-eval pairing uses
``eval_seed_for`` (no ``model_id``) plus per-player ``player_seed_for`` streams
so two arms share Monte Carlo noise and an extra player does not shift others.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

from fplt.types import GWKey

_paired_eval_seed: ContextVar[int | None] = ContextVar(
    "fplt_paired_eval_seed", default=None
)


def seed_for(model_id: str, as_of: GWKey, fixture_id: int) -> np.random.Generator:
    """Return a generator determined by model, as-of gameweek, and fixture.

    The same triple always yields the same stream. Nothing else draws from it.
    """
    payload = f"{model_id}|{as_of.season}|{as_of.gw}|{fixture_id}".encode()
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def eval_seed_for(seed: int, as_of: GWKey, fixture_id: int) -> np.random.Generator:
    """Return a fixture-level generator for promote-eval pairing.

    Unlike ``seed_for``, the key omits ``model_id`` so control and treatment
    arms share the same scoreline and match-sim draws when they use the same
    eval ``seed``.
    """
    payload = f"{int(seed)}|{as_of.season}|{as_of.gw}|{fixture_id}".encode()
    digest = hashlib.sha256(payload).digest()
    derived = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(derived)


def player_seed_for(
    seed: int,
    as_of: GWKey,
    fixture_id: int,
    player_key: int,
) -> np.random.Generator:
    """Return a per-player generator spawned from the eval seed.

    Built with ``numpy.random.SeedSequence`` on
    ``(seed, season, gw, fixture_id, player_key)`` so adding another player
    does not consume entropy from this stream.
    """
    season_key = int.from_bytes(
        hashlib.sha256(as_of.season.encode()).digest()[:8], "little"
    )
    sequence = np.random.SeedSequence(
        [
            int(seed),
            season_key,
            int(as_of.gw),
            int(fixture_id),
            int(player_key),
        ]
    )
    return np.random.default_rng(sequence)


def active_eval_seed() -> int | None:
    """Return the paired eval seed when promote-eval CRN is active."""
    return _paired_eval_seed.get()


@contextmanager
def paired_eval_seed(seed: int) -> Iterator[None]:
    """Activate CRN streams for layered simulation inside the block.

    While active, ``CompositePredictor`` draws fixture noise from
    ``eval_seed_for`` and per-player minutes from ``player_seed_for``.
    Production callers that never enter this context keep using ``seed_for``.
    """
    token = _paired_eval_seed.set(int(seed))
    try:
        yield
    finally:
        _paired_eval_seed.reset(token)
