"""Native Monte Carlo backend: validate, convert enums, call ``fplt_mc``.

Python remains the seed authority. After minutes (and optional hard starters)
are drawn, the caller samples ``kernel_seed`` from the fixture RNG and passes
it here. The C++ kernel draws Philox uniforms from that seed — no silent
fallback if ``fplt_mc`` is missing.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import polars as pl

from fplt.models.simulation import MatchOtherOffsets, MatchSimulation
from fplt.scoring.rules import Rules

McBackend = Literal["python", "native"]

_POS = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
_EXPECTED_KERNEL_API = 2


def resolve_mc_backend(
    spec_value: str | None,
    *,
    env: Mapping[str, str] | None = None,
) -> McBackend:
    """Resolve ``mc_backend`` from ModelSpec with optional env override.

    ``FPLT_MC_BACKEND`` is a debugging override only (python|native). Spec
    default is ``python`` when the param is absent.
    """
    environ: Mapping[str, str] = env if env is not None else os.environ
    override = str(environ.get("FPLT_MC_BACKEND", "")).strip().lower()
    if override:
        if override not in ("python", "native"):
            raise ValueError(
                f"FPLT_MC_BACKEND must be 'python' or 'native', got {override!r}"
            )
        return override  # type: ignore[return-value]
    value = "python" if spec_value is None else str(spec_value).strip().lower()
    if value not in ("python", "native"):
        raise ValueError(f"mc_backend must be 'python' or 'native', got {value!r}")
    return value  # type: ignore[return-value]


def require_native_kernel() -> Any:
    """Import ``fplt_mc`` or raise (no silent fallback).

    Raises:
        ImportError: The wheel is not installed.
        RuntimeError: ``KERNEL_API`` is older than this fplt expects.
    """
    try:
        import fplt_mc
    except ImportError as exc:
        raise ImportError(
            "mc_backend=native requires the fplt-mc wheel "
            "(https://e-mcm.github.io/fplt/simple/); install via uv sync / CI artifact"
        ) from exc
    if int(fplt_mc.KERNEL_API) < _EXPECTED_KERNEL_API:
        raise RuntimeError(
            f"fplt-mc KERNEL_API={fplt_mc.KERNEL_API} < {_EXPECTED_KERNEL_API}; "
            "upgrade the pinned wheel"
        )
    return fplt_mc


def native_provenance() -> dict[str, Any]:
    """Fields for ``audit.json`` / artifact provenance when native is in play."""
    override = os.environ.get("FPLT_MC_BACKEND")
    payload: dict[str, Any] = {
        "fplt_mc_backend_env": override,
    }
    try:
        fplt_mc = require_native_kernel()
    except (ImportError, RuntimeError) as exc:
        payload["fplt_mc_error"] = str(exc)
        return payload
    payload["fplt_mc"] = dict(fplt_mc.build_info())
    return payload


def simulate_match_native(
    scorelines: np.ndarray,
    players: pl.DataFrame,
    minutes: np.ndarray,
    other_points: np.ndarray,
    rules: Rules,
    *,
    kernel_seed: int,
    own_goal_weight: float = 0.05,
    p_assist: float = 0.75,
    cs_red_void_prob: float = 0.0,
    conceded_mode: Literal["scaled", "played"] = "scaled",
    sample_other: bool = False,
    match_other: MatchOtherOffsets | None = None,
    return_samples: bool = True,
) -> MatchSimulation:
    """Run allocate + points + aggregate in ``fplt_mc.simulate_fixture``.

    Args:
        scorelines: ``(n, 2)`` home/away goals.
        players: Rows with ``player_key``, ``side``, ``position``, weights.
        minutes: Final minutes ``(P, n)`` (hard starters already applied).
        other_points: Minutes-bucket means ``(P, 3)``.
        rules: Season scoring table.
        kernel_seed: Philox key drawn from the fixture RNG in Python.
        own_goal_weight / p_assist / cs_red_void_prob / conceded_mode /
            sample_other: Same knobs as ``simulate_match``.
        match_other: Optional match-conditioned tables (duck-typed deltas).
        return_samples: When true, materialise ``(P, n)`` sample planes.

    Returns:
        ``MatchSimulation`` with the same fields as the Python path.

    Raises:
        ImportError / RuntimeError: Native kernel unavailable or too old.
        ValueError: Shapes or params are invalid.
    """
    fplt_mc = require_native_kernel()
    # Nanobind rejects non-writable numpy views (e.g. polars .to_numpy()).
    home = np.array(scorelines[:, 0], dtype=np.int64, copy=True, order="C")
    away = np.array(scorelines[:, 1], dtype=np.int64, copy=True, order="C")
    n = int(home.shape[0])
    keys = np.asarray(players["player_key"].to_list(), dtype=np.int64)
    sides = np.asarray(players["side"].to_list(), dtype=object)
    positions = np.asarray(players["position"].to_list(), dtype=object)
    goal_weight = np.array(
        players["goal_weight"].to_list(), dtype=np.float64, copy=True, order="C"
    )
    assist_weight = np.array(
        players["assist_weight"].to_list(), dtype=np.float64, copy=True, order="C"
    )
    players_n = int(keys.shape[0])
    played = np.array(minutes, dtype=np.float64, copy=True, order="C")
    if played.shape != (players_n, n):
        raise ValueError(
            f"minutes must have shape {(players_n, n)}, got {played.shape}"
        )
    other = np.array(other_points, dtype=np.float64, copy=True, order="C")
    if other.shape != (players_n, 3):
        raise ValueError(
            f"other_points must have shape {(players_n, 3)}, got {other.shape}"
        )

    side_u8 = np.array(
        [0 if s == "home" else 1 for s in sides], dtype=np.uint8, copy=True
    )
    pos_u8 = np.array([_POS[str(p)] for p in positions], dtype=np.uint8, copy=True)
    rules_kw = _rules_kwargs(rules)
    call_kw: dict[str, Any] = {
        **rules_kw,
        "own_goal_weight": float(own_goal_weight),
        "p_assist": float(p_assist),
        "cs_red_void_prob": float(cs_red_void_prob),
        "conceded_scaled": (conceded_mode == "scaled"),
        "sample_other": bool(sample_other),
        "kernel_seed": int(kernel_seed) & ((1 << 63) - 1),
        "return_samples": return_samples,
    }
    call_kw.update(_match_other_tables(match_other))

    out = fplt_mc.simulate_fixture(
        home,
        away,
        side_u8,
        pos_u8,
        played,
        goal_weight,
        assist_weight,
        other,
        **call_kw,
    )
    if not return_samples:
        # Summary-only path still needs sample-shaped arrays for MatchSimulation;
        # re-run is avoided by always requesting samples from the composite.
        raise ValueError("simulate_match_native requires return_samples=True")
    return MatchSimulation(
        player_key=keys,
        points=np.asarray(out["points"], dtype=np.float64),
        goals=np.asarray(out["goals"], dtype=np.int64),
        assists=np.asarray(out["assists"], dtype=np.int64),
        clean_sheets=np.asarray(out["clean_sheets"], dtype=np.bool_),
        unattributed_home=np.asarray(out["unattributed_home"], dtype=np.int64),
        unattributed_away=np.asarray(out["unattributed_away"], dtype=np.int64),
    )


def _rules_kwargs(rules: Rules) -> dict[str, Any]:
    order = ("GK", "DEF", "MID", "FWD")
    conceded = np.array(
        [
            float(rules.goals_conceded.points)
            if pos in rules.goals_conceded.positions
            else 0.0
            for pos in order
        ],
        dtype=np.float64,
    )
    return {
        "appearance_long_minutes": int(rules.appearance.long_minutes),
        "appearance_short": float(rules.appearance.short),
        "appearance_long": float(rules.appearance.long),
        "goals_by_pos": np.array(
            [float(rules.goals[p]) for p in order], dtype=np.float64
        ),
        "assists_pts": float(rules.assists),
        "cs_by_pos": np.array(
            [float(rules.clean_sheets[p]) for p in order], dtype=np.float64
        ),
        "conceded_every": int(rules.goals_conceded.every),
        "conceded_by_pos": conceded,
    }


def _match_other_tables(match_other: MatchOtherOffsets | None) -> dict[str, Any]:
    """Return kwargs for optional K4 tables (omit keys when unused)."""
    if match_other is None:
        return {}
    # OtherMatchConditioned exposes fitted tables as private attrs.
    delta_goals = getattr(match_other, "_delta_goals", None)
    delta_cs = getattr(match_other, "_delta_cs", None)
    delta_scored = getattr(match_other, "_delta_scored", None)
    if delta_goals is None or delta_cs is None or delta_scored is None:
        raise TypeError(
            "native match_other requires _delta_goals/_delta_cs/_delta_scored tables"
        )
    return {
        "delta_goals": np.ascontiguousarray(delta_goals, dtype=np.float64),
        "delta_cs": np.ascontiguousarray(delta_cs, dtype=np.float64),
        "delta_scored": np.ascontiguousarray(delta_scored, dtype=np.float64),
    }
