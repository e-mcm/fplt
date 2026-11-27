"""Tests for measured skill-by-horizon step weights (FDB-242)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from fplt.backtest.horizon_skill import skill_curve_from_ledger
from fplt.optimise.horizon_decay import (
    geometric_step_weights,
    load_skill_weight_config,
    resolve_step_weights,
    skill_step_weights,
    validate_step_weights,
    weights_from_skill_curve,
)
from fplt.optimise.plan import plan_squad
from fplt.optimise.state import SquadState


def test_geometric_matches_decay_power() -> None:
    """Geometric weights follow decay**step."""
    assert geometric_step_weights(0.9, 4) == pytest.approx((1.0, 0.9, 0.81, 0.729))


def test_skill_step_weights_inverse_rmse() -> None:
    """Skill weights are inverse RMSE ratios (optional power)."""
    weights = skill_step_weights([1.0, 1.25, 2.0], horizon=3, power=1.0)
    assert weights == pytest.approx((1.0, 0.8, 0.5))
    squared = skill_step_weights([1.0, 2.0], horizon=2, power=2.0)
    assert squared == pytest.approx((1.0, 0.25))


def test_skill_step_weights_extrapolate_last_ratio() -> None:
    """Horizons past the curve reuse the last measured ratio."""
    weights = skill_step_weights([1.0, 1.25], horizon=4)
    assert weights == pytest.approx((1.0, 0.8, 0.8, 0.8))


def test_weights_from_skill_curve() -> None:
    """Curve helper maps rmse_vs_h1 into step weights."""
    curve = pl.DataFrame(
        {
            "model_name": ["m"] * 3,
            "model_id": ["abc"] * 3,
            "horizon": [1, 2, 3],
            "mae": [1.0, 1.1, 1.2],
            "rmse": [2.0, 2.2, 2.4],
            "n": [10, 10, 10],
            "mae_vs_h1": [1.0, 1.1, 1.2],
            "rmse_vs_h1": [1.0, 1.1, 1.2],
        }
    )
    assert weights_from_skill_curve(curve, horizon=3) == pytest.approx(
        (1.0, 1.0 / 1.1, 1.0 / 1.2)
    )


def test_skill_curve_from_ledger() -> None:
    """Ledger helper scores RMSE by horizon without re-fitting models."""
    ledger = pl.DataFrame(
        {
            "model_id": ["abc"] * 4,
            "horizon": [1, 1, 2, 2],
            "xpts": [2.0, 4.0, 2.0, 4.0],
            "actual_pts": [3.0, 5.0, 4.0, 6.0],
            "target_season": ["2025-26"] * 4,
            "target_gw": [1, 1, 2, 2],
            "player_key": [1, 2, 1, 2],
            "position": ["MID"] * 4,
        }
    )
    curve = skill_curve_from_ledger(ledger, model_name="m")
    assert curve["horizon"].to_list() == [1, 2]
    assert curve["rmse_vs_h1"][0] == pytest.approx(1.0)
    assert curve["rmse_vs_h1"][1] > 1.0


def test_load_default_skill_config_and_resolve() -> None:
    """Committed YAML loads and skill weights beat geometric at far steps."""
    cfg = load_skill_weight_config()
    assert cfg.rmse_vs_h1[0] == pytest.approx(1.0)
    assert len(cfg.rmse_vs_h1) >= 8
    geometric = resolve_step_weights(5, decay=0.9, method="geometric")
    assert geometric == geometric_step_weights(0.9, 5)
    skill = resolve_step_weights(5, decay=0.9, method="skill", skill_config=cfg)
    assert skill[0] == pytest.approx(1.0)
    # Measured skill is flatter than 0.9^t at far steps.
    assert skill[4] > geometric[4]


def test_validate_step_weights_rejects_bad() -> None:
    """Short or non-positive weight vectors raise."""
    with pytest.raises(ValueError, match="length"):
        validate_step_weights([1.0], horizon=2)
    with pytest.raises(ValueError, match="positive"):
        validate_step_weights([1.0, 0.0], horizon=2)


def test_plan_squad_accepts_step_weights() -> None:
    """Explicit step_weights replace geometric decay in the objective."""
    rows: list[dict[str, object]] = []
    player = 1
    for club in range(1, 7):
        for position in ("GK", "DEF", "DEF", "MID", "FWD"):
            rows.append(
                {
                    "player_key": player,
                    "position": position,
                    "team_key": club,
                    "price": 40 + club,
                }
            )
            player += 1
    pool = pl.DataFrame(rows)
    keys = pool["player_key"].to_list()
    # Front-load value on step 0; back-load a different player on step 3.
    point_rows: list[dict[str, object]] = []
    for key in keys:
        for step in range(4):
            xpts = 5.0 if step == 0 else 0.1
            if key == keys[0] and step == 3:
                xpts = 50.0
            point_rows.append({"player_key": key, "step": step, "xpts": xpts})
    points = pl.DataFrame(point_rows)
    state = SquadState.model_validate(
        {
            "bank": 1000,
            "free_transfers": 1,
            "ft_bank_cap": 5,
            "hit_cost": 4,
            "decay": 0.9,
            "bench_weight": 0.0,
            "owned": [],
        }
    )
    control = plan_squad(pool, points, state, initial=True, horizon=4, prune=False)
    flat = plan_squad(
        pool,
        points,
        state,
        initial=True,
        horizon=4,
        prune=False,
        step_weights=(1.0, 1.0, 1.0, 1.0),
    )
    # Flat weights keep far-step mass; geometric 0.9^3 shrinks it → different
    # decayed objective (and often a different look-ahead path).
    assert flat.points != pytest.approx(control.points)


def test_skill_config_yaml_roundtrip(tmp_path: Path) -> None:
    """YAML loader accepts a provenance ``source`` block."""
    path = tmp_path / "weights.yaml"
    path.write_text(
        "metric: rmse\npower: 1.0\nrmse_vs_h1: [1.0, 1.1, 1.2]\n"
        "source:\n  season: 2024-25\n  model: dc_layered_v4\n",
        encoding="utf-8",
    )
    cfg = load_skill_weight_config(path)
    assert cfg.season == "2024-25"
    assert cfg.model == "dc_layered_v4"
    assert cfg.rmse_vs_h1 == pytest.approx((1.0, 1.1, 1.2))


def test_skill_weight_config_rejects_bad_ratios() -> None:
    from fplt.optimise.horizon_decay import SkillWeightConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="positive"):
        SkillWeightConfig.model_validate({"rmse_vs_h1": [1.0, 0.0]})
    with pytest.raises(ValidationError, match="1.0"):
        SkillWeightConfig.model_validate({"rmse_vs_h1": [1.1, 1.2]})


def test_geometric_and_skill_validation_branches() -> None:
    with pytest.raises(ValueError, match="decay must be"):
        geometric_step_weights(0.0, 2)
    with pytest.raises(ValueError, match="decay must be"):
        geometric_step_weights(1.5, 2)
    with pytest.raises(ValueError, match="horizon must be"):
        geometric_step_weights(0.9, 0)

    with pytest.raises(ValueError, match="horizon must be"):
        skill_step_weights([1.0], horizon=0)
    with pytest.raises(ValueError, match="power must be"):
        skill_step_weights([1.0], horizon=1, power=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        skill_step_weights([], horizon=1)
    with pytest.raises(ValueError, match="positive"):
        skill_step_weights([1.0, -0.5], horizon=2)


def test_weights_from_skill_curve_errors() -> None:
    with pytest.raises(ValueError, match="empty"):
        weights_from_skill_curve(pl.DataFrame(), horizon=1)
    with pytest.raises(ValueError, match="missing"):
        weights_from_skill_curve(pl.DataFrame({"horizon": [1]}), horizon=1)

    curve = pl.DataFrame(
        {
            "model_id": ["a", "a", "b", "b"],
            "horizon": [1, 2, 1, 2],
            "rmse_vs_h1": [1.0, 1.2, 1.0, 1.5],
        }
    )
    with pytest.raises(ValueError, match="no model_id"):
        weights_from_skill_curve(
            curve.select("horizon", "rmse_vs_h1"),
            horizon=2,
            model_id="a",
        )
    with pytest.raises(ValueError, match="no skill rows"):
        weights_from_skill_curve(curve, horizon=2, model_id="missing")
    assert weights_from_skill_curve(curve, horizon=2, model_id="b") == pytest.approx(
        (1.0, 1.0 / 1.5)
    )

    gap = pl.DataFrame({"horizon": [1, 3], "rmse_vs_h1": [1.0, 1.2]})
    with pytest.raises(ValueError, match="contiguous"):
        weights_from_skill_curve(gap, horizon=2)


def test_load_skill_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_skill_weight_config(path)


def test_resolve_step_weights_skill_paths() -> None:
    with pytest.raises(ValueError, match="unknown horizon decay"):
        resolve_step_weights(2, decay=0.9, method="nope")  # type: ignore[arg-type]

    explicit = resolve_step_weights(
        3,
        decay=0.9,
        method="skill",
        skill_rmse_vs_h1=[1.0, 2.0, 4.0],
        power=2.0,
    )
    assert explicit == pytest.approx((1.0, 0.25, 0.0625))

    from fplt.optimise.horizon_decay import SkillWeightConfig

    cfg = SkillWeightConfig(rmse_vs_h1=(1.0, 1.25), power=1.0)
    from_cfg = resolve_step_weights(2, decay=0.9, method="skill", skill_config=cfg)
    assert from_cfg == pytest.approx((1.0, 0.8))
    overridden = resolve_step_weights(
        2, decay=0.9, method="skill", skill_config=cfg, power=2.0
    )
    assert overridden == pytest.approx((1.0, 0.64))

    # Falls back to committed YAML when neither ratios nor config given.
    defaulted = resolve_step_weights(2, decay=0.9, method="skill")
    assert defaulted[0] == pytest.approx(1.0)
    assert len(defaulted) == 2
