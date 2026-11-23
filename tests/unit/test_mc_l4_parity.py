"""L4 statistical parity: Python RNG vs native Philox on shared fixtures.

CI uses moderate ``n_sims`` (still under the 4·SE / chi-square gates). Full
ledger-scale sizes (50k Python / 200k native) are for the FDB-278 VM run —
override with ``FPLT_MC_L4_N_PYTHON`` / ``FPLT_MC_L4_N_NATIVE``.
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest
from scipy import stats

pytest.importorskip("fplt_mc")

from fplt.models.simulation import simulate_match
from fplt.models.simulation_native import simulate_match_native
from fplt.scoring.rules import Rules

_RULES = Rules.for_season("2024-25")
_N_PYTHON = int(os.environ.get("FPLT_MC_L4_N_PYTHON", "8000"))
_N_NATIVE = int(os.environ.get("FPLT_MC_L4_N_NATIVE", "20000"))


def _fixture_bank() -> list[dict]:
    """Synthetic stand-ins for ledger exports (DGW-ish, thin, high-scoring)."""
    rng = np.random.default_rng(20261123)
    bank: list[dict] = []
    templates = [
        {"players": 8, "home": 4, "max_g": 2, "thin": False},
        {"players": 6, "home": 3, "max_g": 4, "thin": False},
        {"players": 10, "home": 5, "max_g": 1, "thin": True},
        {"players": 12, "home": 6, "max_g": 3, "thin": False},
        {"players": 7, "home": 3, "max_g": 5, "thin": True},
    ]
    positions = [
        "GK",
        "DEF",
        "DEF",
        "MID",
        "MID",
        "MID",
        "FWD",
        "FWD",
        "DEF",
        "MID",
        "FWD",
        "GK",
    ]
    for index, tmpl in enumerate(templates * 4):  # 20 fixtures
        p = tmpl["players"]
        home_n = tmpl["home"]
        sides = ["home"] * home_n + ["away"] * (p - home_n)
        pos = positions[:p]
        minutes = rng.choice(
            [0.0, 45.0, 60.0, 90.0] if not tmpl["thin"] else [0.0, 0.0, 60.0, 90.0],
            size=(p, 1),
        )
        bank.append(
            {
                "seed": 1000 + index,
                "sides": sides,
                "positions": pos,
                "goal_weight": rng.random(p) + 0.05,
                "assist_weight": rng.random(p) + 0.05,
                "minutes_row": minutes[:, 0],
                "other": rng.random((p, 3)) * 0.4,
                "home_goals": int(rng.integers(0, tmpl["max_g"] + 1)),
                "away_goals": int(rng.integers(0, tmpl["max_g"] + 1)),
                "own_goal_weight": 0.05,
                "p_assist": 0.75,
                "cs_red_void_prob": 0.02,
                "sample_other": True,
            }
        )
    return bank


def _players_df(fx: dict) -> pl.DataFrame:
    p = len(fx["sides"])
    return pl.DataFrame(
        {
            "player_key": list(range(p)),
            "side": fx["sides"],
            "position": fx["positions"],
            "goal_weight": fx["goal_weight"].tolist(),
            "assist_weight": fx["assist_weight"].tolist(),
        }
    )


def _run_python(fx: dict, n: int) -> np.ndarray:
    rng = np.random.default_rng(fx["seed"])
    scorelines = np.column_stack(
        [
            np.full(n, fx["home_goals"], dtype=np.int64),
            np.full(n, fx["away_goals"], dtype=np.int64),
        ]
    )
    minutes = np.broadcast_to(fx["minutes_row"][:, None], (len(fx["sides"]), n)).copy()
    result = simulate_match(
        scorelines,
        _players_df(fx),
        minutes,
        fx["other"],
        _RULES,
        rng,
        own_goal_weight=fx["own_goal_weight"],
        p_assist=fx["p_assist"],
        cs_red_void_prob=fx["cs_red_void_prob"],
        conceded_mode="played",
        sample_other=fx["sample_other"],
    )
    return result.points


def _run_native(fx: dict, n: int) -> np.ndarray:
    scorelines = np.column_stack(
        [
            np.full(n, fx["home_goals"], dtype=np.int64),
            np.full(n, fx["away_goals"], dtype=np.int64),
        ]
    )
    minutes = np.broadcast_to(fx["minutes_row"][:, None], (len(fx["sides"]), n)).copy()
    result = simulate_match_native(
        scorelines,
        _players_df(fx),
        minutes,
        fx["other"],
        _RULES,
        kernel_seed=int(fx["seed"]),
        own_goal_weight=fx["own_goal_weight"],
        p_assist=fx["p_assist"],
        cs_red_void_prob=fx["cs_red_void_prob"],
        conceded_mode="played",
        sample_other=fx["sample_other"],
    )
    return result.points


def _se_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean difference and SE of the difference of independent means."""
    mean_a = float(a.mean())
    mean_b = float(b.mean())
    se = float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))
    return mean_a - mean_b, se


@pytest.mark.slow
def test_l4_statistical_parity_points_means() -> None:
    """Per-player |Δ mean points| < 4·SE across the fixture bank."""
    failures: list[str] = []
    for fx in _fixture_bank():
        py = _run_python(fx, _N_PYTHON)
        nat = _run_native(fx, _N_NATIVE)
        for player in range(py.shape[0]):
            delta, se = _se_diff(py[player], nat[player])
            if se == 0.0:
                if abs(delta) > 1e-9:
                    failures.append(f"seed={fx['seed']} p={player} se=0 delta={delta}")
                continue
            if abs(delta) >= 4.0 * se:
                failures.append(
                    f"seed={fx['seed']} p={player} |delta|={abs(delta):.4f} "
                    f"4se={4 * se:.4f}"
                )
    assert not failures, "L4 mean gates failed:\n" + "\n".join(failures[:20])


@pytest.mark.slow
def test_l4_points_histogram_chi_square() -> None:
    """Pooled points histogram chi-square is not rejected at α=0.001."""
    # Pool a few fixtures to keep bins populated.
    py_chunks: list[np.ndarray] = []
    nat_chunks: list[np.ndarray] = []
    for fx in _fixture_bank()[:8]:
        py_chunks.append(_run_python(fx, _N_PYTHON).ravel())
        nat_chunks.append(_run_native(fx, _N_NATIVE).ravel())
    py = np.concatenate(py_chunks)
    nat = np.concatenate(nat_chunks)
    # Integer-ish bins via floor of points (appearance/involvement scale).
    py_bins = np.clip(np.floor(py).astype(np.int64), -5, 25)
    nat_bins = np.clip(np.floor(nat).astype(np.int64), -5, 25)
    values = np.arange(-5, 26)
    py_counts = np.array([(py_bins == v).sum() for v in values], dtype=np.float64)
    nat_counts = np.array([(nat_bins == v).sum() for v in values], dtype=np.float64)
    # 2×K contingency: backend × bin (avoids float rescale tolerance issues).
    table = np.vstack([py_counts, nat_counts])
    keep = table.sum(axis=0) >= 10
    if keep.sum() < 3:
        pytest.skip("too few histogram bins for chi-square")
    statistic, p_value, _, _ = stats.chi2_contingency(table[:, keep])
    assert p_value >= 0.001, f"chi2={statistic:.3f} p={p_value:.4g}"
