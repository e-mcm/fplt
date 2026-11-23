"""L1/L2/L3 tests for native K2 allocate_side vs the Python inject surface.

Requires an installed ``fplt-mc`` wheel (CI reinstalls the artifact on ``mc/``
PRs) and the repo's Python reference in ``fplt.models.simulation``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("fplt_mc")

from fplt.models.simulation import SideAllocateUniforms, _allocate_side, _pick
from fplt_mc import _testing as mc_testing

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _side_uniforms(max_goals: int, n_sims: int, seed: int) -> SideAllocateUniforms:
    """Build ``(G, n)`` inject grids in the same order as the RNG path."""
    rng = np.random.default_rng(seed)
    return SideAllocateUniforms(
        scorer=rng.random((max_goals, n_sims)),
        assist_flag=rng.random((max_goals, n_sims)),
        assister=rng.random((max_goals, n_sims)),
    )


def _philox_side_uniforms(
    max_goals: int, n_sims: int, seed: int
) -> SideAllocateUniforms:
    """Fill inject grids from Philox (native RNG) for L3 property tests."""
    scorer = np.empty((max_goals, n_sims), dtype=np.float64)
    assist_flag = np.empty((max_goals, n_sims), dtype=np.float64)
    assister = np.empty((max_goals, n_sims), dtype=np.float64)
    for g in range(max_goals):
        for j in range(n_sims):
            # Distinct streams so goal-k draws stay independent.
            base = g * n_sims + j
            scorer[g, j] = mc_testing.philox_u01_batch(seed, 1, base, 1)[0]
            assist_flag[g, j] = mc_testing.philox_u01_batch(seed, 2, base, 1)[0]
            assister[g, j] = mc_testing.philox_u01_batch(seed, 3, base, 1)[0]
    return SideAllocateUniforms(
        scorer=scorer, assist_flag=assist_flag, assister=assister
    )


def _native_allocate(
    team_goals: np.ndarray,
    minutes: np.ndarray,
    goal_weight: np.ndarray,
    assist_weight: np.ndarray,
    own_goal_weight: float,
    p_assist: float,
    uniforms: SideAllocateUniforms,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call ``allocate_side_with_uniforms`` with contiguous float64/int64."""
    return mc_testing.allocate_side_with_uniforms(
        np.ascontiguousarray(team_goals, dtype=np.int64),
        np.ascontiguousarray(minutes, dtype=np.float64),
        np.ascontiguousarray(goal_weight, dtype=np.float64),
        np.ascontiguousarray(assist_weight, dtype=np.float64),
        float(own_goal_weight),
        float(p_assist),
        np.ascontiguousarray(uniforms.scorer, dtype=np.float64),
        np.ascontiguousarray(uniforms.assist_flag, dtype=np.float64),
        np.ascontiguousarray(uniforms.assister, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# L2 — exact equivalence with injected uniforms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99, 123, 999])
def test_l2_allocate_side_matches_python_inject(seed: int) -> None:
    """Native and Python produce identical integer outputs on the same u's."""
    rng = np.random.default_rng(seed)
    players, n_sims = 11, 64
    team_goals = rng.integers(0, 5, size=n_sims, dtype=np.int64)
    max_goals = int(team_goals.max(initial=0))
    minutes = rng.choice([0.0, 45.0, 60.0, 90.0], size=(players, n_sims))
    goal_weight = rng.random(players) + 0.01
    assist_weight = rng.random(players) + 0.01
    own_goal_weight = 0.05
    p_assist = 0.75
    uniforms = _side_uniforms(max(max_goals, 1), n_sims, seed + 21)

    py = _allocate_side(
        team_goals,
        minutes,
        goal_weight,
        assist_weight,
        own_goal_weight,
        p_assist,
        np.random.default_rng(0),
        uniforms=uniforms,
    )
    native = _native_allocate(
        team_goals,
        minutes,
        goal_weight,
        assist_weight,
        own_goal_weight,
        p_assist,
        uniforms,
    )
    for left, right in zip(py, native, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("seed", [2, 8, 17])
def test_l2_pick_matches_python(seed: int) -> None:
    """``pick_with_uniforms`` matches ``_pick`` bit-for-bit."""
    rng = np.random.default_rng(seed)
    players, n = 9, 128
    weights = rng.random((players, n))
    # Zero some columns / rows to hit edge mass.
    weights[rng.random(players) < 0.2, :] = 0.0
    weights[:, rng.random(n) < 0.1] = 0.0
    draw = rng.random(n)
    for extra in (0.0, 0.05, 1.0):
        py = _pick(weights, draw, extra)
        native = mc_testing.pick_with_uniforms(
            np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(draw, dtype=np.float64),
            float(extra),
        )
        np.testing.assert_array_equal(py, native)


def test_l2_zero_players_copies_team_goals_to_missed() -> None:
    """Empty side: all goals are unattributed."""
    team_goals = np.array([2, 0, 1], dtype=np.int64)
    minutes = np.zeros((0, 3), dtype=np.float64)
    gw = np.zeros(0, dtype=np.float64)
    aw = np.zeros(0, dtype=np.float64)
    # max_goals=2; grids must be wide enough even though players=0.
    uniforms = SideAllocateUniforms(
        scorer=np.zeros((2, 3)),
        assist_flag=np.zeros((2, 3)),
        assister=np.zeros((2, 3)),
    )
    py = _allocate_side(
        team_goals, minutes, gw, aw, 0.05, 0.5, np.random.default_rng(0),
        uniforms=uniforms,
    )
    native = _native_allocate(team_goals, minutes, gw, aw, 0.05, 0.5, uniforms)
    for left, right in zip(py, native, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(native[2], team_goals)


# ---------------------------------------------------------------------------
# L1 — behavioural properties (parametrised over python / native)
# ---------------------------------------------------------------------------


def _run_allocate(backend: str, **kwargs):  # type: ignore[no-untyped-def]
    uniforms = kwargs.pop("uniforms")
    if backend == "python":
        return _allocate_side(
            **kwargs, rng=np.random.default_rng(0), uniforms=uniforms
        )
    return _native_allocate(**kwargs, uniforms=uniforms)


@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("seed", [1, 5, 13])
def test_l1_goals_plus_missed_equal_scoreline(backend: str, seed: int) -> None:
    """Player goals + unattributed == team_goals for every sim."""
    rng = np.random.default_rng(seed)
    players, n = 10, 48
    team_goals = rng.integers(0, 6, size=n, dtype=np.int64)
    max_goals = int(team_goals.max(initial=0))
    minutes = rng.choice([0.0, 30.0, 90.0], size=(players, n))
    uniforms = _side_uniforms(max(max_goals, 1), n, seed + 3)
    scored, _assisted, missed = _run_allocate(
        backend,
        team_goals=team_goals,
        minutes=minutes,
        goal_weight=rng.random(players) + 0.05,
        assist_weight=rng.random(players) + 0.05,
        own_goal_weight=0.1,
        p_assist=0.8,
        uniforms=uniforms,
    )
    np.testing.assert_array_equal(scored.sum(axis=0) + missed, team_goals)


@pytest.mark.parametrize("backend", ["python", "native"])
def test_l1_zero_minute_players_never_score_or_assist(backend: str) -> None:
    """Minutes == 0 ⇒ rate 0 ⇒ never scorer or assister."""
    team_goals = np.array([4], dtype=np.int64)
    minutes = np.array([[0.0], [90.0], [0.0]], dtype=np.float64)
    uniforms = _side_uniforms(4, 1, seed=42)
    scored, assisted, missed = _run_allocate(
        backend,
        team_goals=team_goals,
        minutes=minutes,
        goal_weight=np.array([1.0, 1.0, 1.0]),
        assist_weight=np.array([1.0, 1.0, 1.0]),
        own_goal_weight=0.0,
        p_assist=1.0,
        uniforms=uniforms,
    )
    assert scored[0, 0] == 0 and scored[2, 0] == 0
    assert assisted[0, 0] == 0 and assisted[2, 0] == 0
    assert scored[1, 0] + missed[0] == 4


@pytest.mark.parametrize("backend", ["python", "native"])
def test_l1_assister_is_never_the_scorer(backend: str) -> None:
    """When an assist is awarded, assister ≠ scorer (rate zeroed)."""
    rng = np.random.default_rng(7)
    players, n = 8, 32
    team_goals = np.ones(n, dtype=np.int64)  # exactly one goal each
    minutes = np.full((players, n), 90.0)
    uniforms = _side_uniforms(1, n, seed=11)
    scored, assisted, _missed = _run_allocate(
        backend,
        team_goals=team_goals,
        minutes=minutes,
        goal_weight=np.ones(players),
        assist_weight=np.ones(players),
        own_goal_weight=0.0,
        p_assist=1.0,
        uniforms=uniforms,
    )
    for j in range(n):
        scorer = int(scored[:, j].argmax()) if scored[:, j].sum() else -1
        if scorer < 0:
            continue
        # Assister cannot be the scorer.
        assert assisted[scorer, j] == 0


# ---------------------------------------------------------------------------
# L3 — invariants under Philox-generated uniforms
# ---------------------------------------------------------------------------


@given(
    seed=st.integers(0, 10_000),
    players=st.integers(1, 16),
    n_sims=st.integers(1, 24),
    own_goal_weight=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    p_assist=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
def test_l3_native_invariants_under_philox(
    seed: int,
    players: int,
    n_sims: int,
    own_goal_weight: float,
    p_assist: float,
) -> None:
    """Goals conserve; zero-minute silent; assister ≠ scorer; finite ints."""
    rng = np.random.default_rng(seed)
    team_goals = rng.integers(0, 5, size=n_sims, dtype=np.int64)
    max_goals = int(team_goals.max(initial=0))
    minutes = rng.choice([0.0, 45.0, 90.0], size=(players, n_sims))
    goal_weight = rng.random(players)
    assist_weight = rng.random(players)
    uniforms = _philox_side_uniforms(max(max_goals, 1), n_sims, seed)

    scored, assisted, missed = _native_allocate(
        team_goals,
        minutes,
        goal_weight,
        assist_weight,
        own_goal_weight,
        p_assist,
        uniforms,
    )

    np.testing.assert_array_equal(scored.sum(axis=0) + missed, team_goals)
    assert np.isfinite(scored).all() and np.isfinite(assisted).all()
    assert (scored >= 0).all() and (assisted >= 0).all() and (missed >= 0).all()

    zero_min = minutes <= 0
    assert (scored[zero_min] == 0).all()
    assert (assisted[zero_min] == 0).all()

    # Per goal-event: a player who scored on a one-goal sim cannot also assist.
    for j in range(n_sims):
        if team_goals[j] != 1:
            continue
        if scored[:, j].sum() != 1:
            continue
        scorer = int(np.flatnonzero(scored[:, j] == 1)[0])
        assert assisted[scorer, j] == 0


@given(seed=st.integers(0, 5000))
@settings(max_examples=20, deadline=None)
def test_l3_same_philox_seed_is_deterministic(seed: int) -> None:
    """Identical Philox uniforms ⇒ identical native allocate output."""
    rng = np.random.default_rng(seed)
    players, n = 6, 16
    team_goals = rng.integers(0, 4, size=n, dtype=np.int64)
    max_goals = int(team_goals.max(initial=0))
    minutes = rng.choice([0.0, 90.0], size=(players, n))
    gw = rng.random(players) + 0.1
    aw = rng.random(players) + 0.1
    uniforms = _philox_side_uniforms(max(max_goals, 1), n, seed + 99)
    a = _native_allocate(team_goals, minutes, gw, aw, 0.05, 0.7, uniforms)
    b = _native_allocate(team_goals, minutes, gw, aw, 0.05, 0.7, uniforms)
    for left, right in zip(a, b, strict=True):
        np.testing.assert_array_equal(left, right)


def test_l3_different_philox_seed_can_differ() -> None:
    """Different seeds should (almost always) change at least one cell."""
    players, n = 8, 32
    team_goals = np.full(n, 2, dtype=np.int64)
    minutes = np.full((players, n), 90.0)
    gw = np.ones(players)
    aw = np.ones(players)
    u0 = _philox_side_uniforms(2, n, seed=1)
    u1 = _philox_side_uniforms(2, n, seed=2)
    a = _native_allocate(team_goals, minutes, gw, aw, 0.0, 1.0, u0)
    b = _native_allocate(team_goals, minutes, gw, aw, 0.0, 1.0, u1)
    assert not (
        np.array_equal(a[0], b[0])
        and np.array_equal(a[1], b[1])
        and np.array_equal(a[2], b[2])
    )
