"""Renderer registry registration and lookup."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.ui import renderers


def test_table_renderer_registered() -> None:
    assert "table" in renderers.registered_views()
    frame = pl.DataFrame({"a": [1]})
    assert renderers.get_renderer("table")(frame).equals(frame)


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
