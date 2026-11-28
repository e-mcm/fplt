"""Renderer registry registration and lookup."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.ui import renderers


def test_table_renderer_registered() -> None:
    assert "table" in renderers.registered_views()
    frame = pl.DataFrame({"a": [1]})
    assert renderers.get_renderer("table")(frame).equals(frame)


def test_timing_renderers() -> None:
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    phases = pl.DataFrame(
        {
            "phase": ["ingest", "setup"],
            "seconds": [10.0, 2.0],
            "share_pct": [80.0, 16.0],
        }
    )
    stacked = renderers.get_renderer("timing_stacked_bar")(phases)
    assert isinstance(stacked, go.Figure)
    assert len(stacked.data) == 2
    trend = pl.DataFrame(
        {
            "season": ["2026-27", "2026-27"],
            "gw": [12, 13],
            "checkpoint": ["T-90m", "T-90m"],
            "total_s": [5.0, 8.0],
            "ingest": [4.0, 6.0],
        }
    )
    trend_fig = renderers.get_renderer("timing_trend")(trend)
    assert isinstance(trend_fig, go.Figure)
    assert "timing_stacked_bar" in renderers.registered_views()
    assert "timing_trend" in renderers.registered_views()


def test_pit_hist_and_reliability_renderers() -> None:
    pit = pl.DataFrame(
        {
            "bin_lo": [0.0, 0.1],
            "bin_hi": [0.1, 0.2],
            "density": [0.1, 0.2],
            "label": ["0.0-0.1", "0.1-0.2"],
        }
    )
    pit_fig = renderers.get_renderer("pit_hist")(pit)
    assert pit_fig is not None
    assert "pit_hist" in renderers.registered_views()

    rel = pl.DataFrame(
        {
            "event": ["played", "played"],
            "pred_bin": ["0.0-0.1", "0.9-1.0"],
            "mean_pred": [0.05, 0.95],
            "observed": [0.1, 0.9],
            "n": [10, 12],
        }
    )
    rel_fig = renderers.get_renderer("reliability")(rel)
    assert rel_fig is not None
    assert "reliability" in renderers.registered_views()
    # No ``event`` column → single ``all`` trace.
    plain = rel.drop("event")
    plain_fig = renderers.get_renderer("reliability")(plain)
    assert plain_fig is not None


def test_points_pmf_bar_renderer() -> None:
    pytest.importorskip("plotly")
    frame = pl.DataFrame({"points_bin": [0, 5], "probability": [0.4, 0.6]})
    fig = renderers.get_renderer("points_pmf_bar")(frame)
    assert fig is not None
    assert "points_pmf_bar" in renderers.registered_views()


def test_unknown_renderer_raises() -> None:
    with pytest.raises(KeyError, match="unknown renderer"):
        renderers.get_renderer("not_a_view")


def test_duplicate_registration_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @renderers.renderer("table")
        def _dup(frame: pl.DataFrame) -> pl.DataFrame:
            return frame
