#!/usr/bin/env python3
"""Per-step Monte Carlo micro-benchmark (native-mc-build-spec §1.1 / §5.5).

Synthetic fixture: two squads of 20, ``n_sims=10_000``, 33×33 scoreline grid.
Reports best-of-``repeats`` wall time per step so PR0 (and later ``mc/`` ports)
can compare before/after on the same host.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

import fplt.models.composite as composite_mod
from fplt.models.distributions import (
    MAX_GOALS,
    MINUTE_BINS,
    MinutesDist,
    ScorelineDist,
    enforce_hard_starters,
)
from fplt.models.simulation import simulate_match
from fplt.scoring.rules import Rules
from fplt.types import GWKey

# Bench the CRN helper used by CompositePredictor (private module attribute).
_sample_minutes_crn: Callable[..., np.ndarray] = getattr(
    composite_mod, "_sample_minutes_crn"
)


@dataclass(frozen=True)
class _Fixture:
    """Synthetic inputs matching the §1.1 profile."""

    scorelines: ScorelineDist
    minutes: MinutesDist
    team_keys: np.ndarray
    positions: np.ndarray
    start_weight: np.ndarray
    players: pl.DataFrame
    other_points: np.ndarray
    rules: Rules


def _build_fixture(players_per_side: int = 20) -> _Fixture:
    """Build a 40-player, 33×33-grid fixture for the micro-bench."""
    goals = MAX_GOALS + 1
    rng = np.random.default_rng(0)
    raw = rng.random((1, goals, goals))
    raw /= raw.sum()
    scorelines = ScorelineDist(np.array([1], dtype=np.int64), raw)

    n_players = 2 * players_per_side
    pmf = rng.random((n_players, len(MINUTE_BINS)))
    pmf /= pmf.sum(axis=1, keepdims=True)
    minutes = MinutesDist(
        pl.DataFrame(
            {
                "player_key": list(range(n_players)),
                "fixture_id": [1] * n_players,
            }
        ),
        MINUTE_BINS,
        pmf,
    )
    team_keys = np.array(
        [0] * players_per_side + [1] * players_per_side, dtype=np.int64
    )
    positions = np.array(
        ["GK"]
        + ["DEF"] * 4
        + ["MID"] * 8
        + ["FWD"] * 7
        + ["GK"]
        + ["DEF"] * 4
        + ["MID"] * 8
        + ["FWD"] * 7,
        dtype=object,
    )
    start_weight = minutes.start_weights()
    players = pl.DataFrame(
        {
            "player_key": list(range(n_players)),
            "side": ["home"] * players_per_side + ["away"] * players_per_side,
            "position": positions.tolist(),
            "goal_weight": rng.random(n_players).tolist(),
            "assist_weight": rng.random(n_players).tolist(),
        }
    )
    other_points = rng.random((n_players, 3))
    rules = Rules.for_season("2025-26")
    return _Fixture(
        scorelines=scorelines,
        minutes=minutes,
        team_keys=team_keys,
        positions=positions,
        start_weight=start_weight,
        players=players,
        other_points=other_points,
        rules=rules,
    )


def _best_ms(fn: Callable[[], Any], *, repeats: int) -> float:
    """Return the best-of-``repeats`` runtime of ``fn`` in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best


def run_bench(*, n_sims: int, repeats: int) -> list[tuple[str, float]]:
    """Time each §1.1 step; return ``(name, best_ms)`` rows."""
    fixture = _build_fixture()
    as_of = GWKey(season="2025-26", gw=1)
    results: list[tuple[str, float]] = []

    def scoreline() -> None:
        fixture.scorelines.sample(np.random.default_rng(1), n_sims)

    results.append(("ScorelineDist.sample", _best_ms(scoreline, repeats=repeats)))

    def minutes_indep() -> None:
        fixture.minutes.sample(np.random.default_rng(2), n_sims)

    results.append(("MinutesDist.sample", _best_ms(minutes_indep, repeats=repeats)))

    def minutes_crn() -> None:
        _sample_minutes_crn(
            fixture.minutes, seed=0, as_of=as_of, fixture_id=1, n_sims=n_sims
        )

    results.append(("_sample_minutes_crn", _best_ms(minutes_crn, repeats=repeats)))

    drawn = fixture.minutes.sample(np.random.default_rng(3), n_sims)

    def hard_xi() -> None:
        enforce_hard_starters(
            drawn,
            fixture.team_keys,
            fixture.start_weight,
            np.random.default_rng(4),
            positions=fixture.positions,
        )

    results.append(("enforce_hard_starters", _best_ms(hard_xi, repeats=repeats)))

    score_draws = fixture.scorelines.sample(np.random.default_rng(5), n_sims)[0]
    minutes_adj = enforce_hard_starters(
        drawn,
        fixture.team_keys,
        fixture.start_weight,
        np.random.default_rng(6),
        positions=fixture.positions,
    )

    def match_sim() -> None:
        simulate_match(
            score_draws,
            fixture.players,
            minutes_adj,
            fixture.other_points,
            fixture.rules,
            np.random.default_rng(7),
            sample_other=True,
            cs_red_void_prob=0.05,
        )

    results.append(
        ("simulate_match (alloc+points)", _best_ms(match_sim, repeats=repeats))
    )
    return results


def main(argv: list[str] | None = None) -> int:
    """Run the micro-bench and print a table to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sims", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    rows = run_bench(n_sims=args.n_sims, repeats=args.repeats)
    total = sum(ms for _, ms in rows)
    print(f"mc-bench  n_sims={args.n_sims}  repeats={args.repeats}  (best of)")
    print(f"{'step':<36} {'ms':>10} {'share':>8}")
    for name, ms in rows:
        share = 100.0 * ms / total if total > 0 else 0.0
        print(f"{name:<36} {ms:10.1f} {share:7.1f}%")
    print(f"{'total (sum of steps)':<36} {total:10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
