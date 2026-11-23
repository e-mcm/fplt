"""Phase timing aggregation for walk-forward wall-clock logs."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from fplt.backtest.walk_forward import walk_forward
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.baselines import RollingMeanPredictor
from fplt.models.spec import ModelSpec
from fplt.types import GWKey
from fplt.utils.timing import (
    TimingSession,
    format_gw_breakdown,
    format_run_summary,
    timed_phase,
    timing_session,
)
from tests.builders import build_league


def test_timing_session_merges_and_shares() -> None:
    first = TimingSession()
    first.add("fit", 2.0)
    first.add("predict", 8.0)
    first.add("fit/score", 1.5)
    second = TimingSession()
    second.add("fit", 1.0)
    second.add("predict", 4.0)
    second.add("write", 0.5)
    second.add("fit/score", 0.5)
    first.merge(second)
    assert first.get("fit") == 3.0
    assert first.get("predict") == 12.0
    assert first.get("write") == 0.5
    assert first.total() == 15.5
    assert first.nested() == {"fit/score": 2.0}
    summary = format_run_summary("demo", first, n_gw=2)
    assert "walk_forward timing model=demo gws=2 total=15.5s" in summary
    assert "fit=3.0s (19%)" in summary
    assert "predict=12.0s (77%)" in summary
    assert "write=0.5s (3%)" in summary
    assert "| fit/score=2.0s" in summary
    assert format_gw_breakdown(first) == "fit=3.0s predict=12.0s write=0.5s"


def test_timed_phase_records_only_with_active_session() -> None:
    orphan = TimingSession()
    with timed_phase("fit"):
        time.sleep(0.01)
    assert orphan.seconds == {}
    with timing_session(orphan):
        with timed_phase("fit"):
            time.sleep(0.01)
        with timed_phase("predict"):
            time.sleep(0.01)
        with timed_phase("fit/score"):
            time.sleep(0.01)
    assert orphan.get("fit") > 0
    assert orphan.get("predict") > 0
    assert orphan.get("fit/score") > 0


class _SlowMean(RollingMeanPredictor):
    """Baseline that sleeps in fit/predict so wall times are measurable."""

    def fit(self, view: PITView) -> None:
        time.sleep(0.02)
        super().fit(view)

    def predict(self, view: PITView, horizon: list[GWKey]):
        time.sleep(0.03)
        return super().predict(view, horizon)


def test_walk_forward_logs_fit_predict_split_and_summary(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = build_league(DataStore(tmp_path))
    spec = ModelSpec(
        name="rolling_mean",
        kind="rolling_mean",
        params={"window": 5},
        refit="never",
    )
    as_ofs = [GWKey("2024-25", 1), GWKey("2024-25", 2)]
    with caplog.at_level(logging.INFO, logger="fplt.backtest.walk_forward"):
        walk_forward(_SlowMean(spec), PITData(store), as_ofs, steps=1)
    assert "gw 1 done in" in caplog.text
    assert "fit=" in caplog.text
    assert "predict=" in caplog.text
    assert "write=" in caplog.text
    assert "rolling_mean 2024-25 gw 2 (2/2) predicting" in caplog.text
    summary_lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("walk_forward timing model=rolling_mean")
    ]
    assert len(summary_lines) == 1
    summary = summary_lines[0]
    assert "gws=2" in summary
    assert "fit=" in summary and "predict=" in summary
    assert "%" in summary
