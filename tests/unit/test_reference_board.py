"""Reference board YAML must match live planner defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fplt.automation.production import load_production_ops
from fplt.optimise.decisions import (
    DEFAULT_LIVE_CHIP_ARM,
    DEFAULT_LIVE_PRICE_FORECAST_METHOD,
)
from fplt.optimise.highs_options import HighsSolveOptions
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON
from fplt.optimise.reference_board import (
    ReferenceBoard,
    default_reference_board_path,
    load_reference_board,
)
from fplt.optimise.state import load_squad_state


def test_committed_reference_board_matches_live_defaults() -> None:
    """Reference flags stay aligned with live plan / automation constants."""
    board = load_reference_board()
    assert default_reference_board_path().is_file()
    assert board.version == 1
    assert board.horizon == DEFAULT_PLAN_HORIZON
    assert board.chip_arm == DEFAULT_LIVE_CHIP_ARM
    assert board.force_play is True
    assert board.lineup_reopt is True
    assert board.price_forecast == DEFAULT_LIVE_PRICE_FORECAST_METHOD
    assert board.bench_weight == pytest.approx(load_squad_state().bench_weight)
    assert board.model == load_production_ops().models[0]
    assert load_production_ops().plan_horizon == board.horizon


def test_reference_highs_match_board_deterministic() -> None:
    """Board YAML HiGHS knobs equal ``HighsSolveOptions.board_deterministic``."""
    board = load_reference_board()
    expected = HighsSolveOptions.board_deterministic()
    got = board.highs.to_highs_options()
    assert got == expected
    assert got.time_limit is None
    assert got.threads is None
    assert got.parallel == "off"
    assert got.random_seed == 0
    assert got.mip_rel_gap == pytest.approx(0.0)
    assert got.accept_time_limit is False


def test_load_reference_board_rejects_bad_yaml(tmp_path: Path) -> None:
    """Missing / non-mapping / invalid files raise clearly."""
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="reference board"):
        load_reference_board(missing)
    bad = tmp_path / "reference.yaml"
    bad.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_reference_board(bad)
    empty_model = tmp_path / "empty_model.yaml"
    empty_model.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "horizon": 3,
                "chip_arm": "optimised",
                "force_play": True,
                "lineup_reopt": True,
                "price_forecast": "mean_gw_delta",
                "bench_weight": 0.0,
                "model": "  ",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model"):
        load_reference_board(empty_model)


def test_reference_board_rejects_unknown_chip_arm() -> None:
    """Extra chip arms are refused at validation time."""
    with pytest.raises(ValueError):
        ReferenceBoard(
            version=1,
            horizon=3,
            chip_arm="mystery",  # type: ignore[arg-type]
            force_play=True,
            lineup_reopt=True,
            price_forecast="mean_gw_delta",
            bench_weight=0.0,
            model="dc_layered_v2",
        )
