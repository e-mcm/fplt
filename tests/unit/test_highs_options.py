"""HiGHS solve-option policy for the transfer MILP."""

from __future__ import annotations

import highspy
import pytest

from fplt.optimise.highs_options import HighsSolveOptions


def test_open_fpl_defaults_accept_time_limit() -> None:
    """Default options use 600s, a small MIP gap, and accept incumbents."""
    options = HighsSolveOptions.open_fpl_defaults()
    assert options.time_limit == 600.0
    assert options.mip_rel_gap == pytest.approx(1e-3)
    assert options.accepts_status(highspy.HighsModelStatus.kOptimal)
    assert options.accepts_status(highspy.HighsModelStatus.kTimeLimit)


def test_unlimited_optimal_rejects_time_limit() -> None:
    """Pre-policy behaviour requires proven optimality."""
    options = HighsSolveOptions.unlimited_optimal()
    assert options.time_limit is None
    assert options.accepts_status(highspy.HighsModelStatus.kOptimal)
    assert not options.accepts_status(highspy.HighsModelStatus.kTimeLimit)


def test_board_deterministic_is_single_threaded_seeded() -> None:
    """Chip boards use no time limit, one thread, and a fixed random seed."""
    options = HighsSolveOptions.board_deterministic()
    assert options.time_limit is None
    assert options.threads == 1
    assert options.parallel == "off"
    assert options.random_seed == 0
    assert options.mip_rel_gap == pytest.approx(0.0)
    assert options.accept_time_limit is False
    assert not options.accepts_status(highspy.HighsModelStatus.kTimeLimit)
    assert options.accepts_status(highspy.HighsModelStatus.kOptimal)

    solver = highspy.Highs()
    options.apply(solver)

    def _opt(name: str) -> object:
        status, value = solver.getOptionValue(name)
        assert status == highspy.HighsStatus.kOk
        return value

    assert _opt("threads") == 1
    assert _opt("random_seed") == 0
    assert _opt("parallel") == "off"


def test_time_limit_must_be_positive() -> None:
    """Zero / negative time limits are rejected."""
    with pytest.raises(ValueError, match="time_limit"):
        HighsSolveOptions(time_limit=0.0)


def test_apply_writes_options_onto_solver() -> None:
    """apply() sets the HiGHS option bag the MILP will use."""
    solver = highspy.Highs()
    options = HighsSolveOptions(
        time_limit=12.5,
        mip_rel_gap=1e-3,
        output_flag=False,
        presolve="on",
        parallel="off",
        threads=1,
        random_seed=7,
    )
    options.apply(solver)

    def _opt(name: str) -> object:
        status, value = solver.getOptionValue(name)
        assert status == highspy.HighsStatus.kOk
        return value

    assert _opt("time_limit") == 12.5
    assert _opt("mip_rel_gap") == pytest.approx(1e-3)
    assert _opt("presolve") == "on"
    assert _opt("parallel") == "off"
    assert _opt("threads") == 1
    assert _opt("random_seed") == 7
