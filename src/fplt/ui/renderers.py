"""View-name → renderer registry (``fplt-ui-design.md`` §3.2).

Renderers are thin Streamlit/Plotly wrappers. Data shaping stays in
``fplt.analysis``. This first slice only registers a table renderer so the
registry exists for later stage panels.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

# plotly is imported inside the points_pmf renderer so non-UI imports stay light.

Renderer = Callable[[pl.DataFrame], Any]

_REGISTRY: dict[str, Renderer] = {}


def renderer(name: str) -> Callable[[Renderer], Renderer]:
    """Register a UI renderer under ``name``.

    Args:
        name: View name declared by a ``Measure`` (e.g. ``\"table\"``).

    Returns:
        Decorator that stores the function in the registry.

    Raises:
        ValueError: ``name`` is already registered.
    """

    def decorator(fn: Renderer) -> Renderer:
        if name in _REGISTRY:
            raise ValueError(f"renderer already registered: {name!r}")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_renderer(name: str) -> Renderer:
    """Return the renderer for ``name``.

    Args:
        name: Registered view name.

    Returns:
        The renderer callable.

    Raises:
        KeyError: Unknown view name.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown renderer: {name!r}") from exc


def registered_views() -> frozenset[str]:
    """Return the set of registered view names."""
    return frozenset(_REGISTRY)


@renderer("table")
def render_table(frame: pl.DataFrame) -> pl.DataFrame:
    """Pass-through table view (Streamlit shows the frame directly).

    Args:
        frame: Any overview / stage table.

    Returns:
        The same frame (identity renderer for AppTest / non-Plotly views).
    """
    return frame


@renderer("points_pmf_bar")
def render_points_pmf_bar(frame: pl.DataFrame) -> Any:
    """Plotly bar chart of a player–gameweek points PMF.

    Args:
        frame: Columns ``points_bin``, ``probability``.

    Returns:
        A Plotly ``Figure``.
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    return go.Figure(
        data=[
            go.Bar(
                x=frame["points_bin"].to_list(),
                y=frame["probability"].to_list(),
                name="P(points)",
            )
        ],
        layout=go.Layout(xaxis_title="Points", yaxis_title="P"),
    )


@renderer("timing_stacked_bar")
def render_timing_stacked_bar(frame: pl.DataFrame) -> Any:
    """Horizontal stacked bar of top-level ops timing phases.

    Args:
        frame: Columns ``phase``, ``seconds`` (``RunTimingPhaseSchema``).

    Returns:
        A Plotly ``Figure``.
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    bars = [
        go.Bar(
            x=[float(row["seconds"])],
            y=["ops"],
            name=str(row["phase"]),
            orientation="h",
            hovertemplate=f"{row['phase']}: %{{x:.1f}}s<extra></extra>",
        )
        for row in frame.iter_rows(named=True)
    ]
    return go.Figure(
        data=bars,
        layout=go.Layout(
            barmode="stack",
            xaxis_title="Seconds",
            yaxis_title="",
            showlegend=True,
            height=160,
            margin={"l": 40, "r": 20, "t": 20, "b": 40},
        ),
    )


@renderer("pit_hist")
def render_pit_hist(frame: pl.DataFrame) -> Any:
    """Plotly bar chart of a randomised PIT histogram.

    Args:
        frame: Columns ``label``, ``density`` (``LastWeekPitHistSchema``).

    Returns:
        A Plotly ``Figure``.
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    return go.Figure(
        data=[
            go.Bar(
                x=frame["label"].to_list(),
                y=frame["density"].to_list(),
                name="PIT density",
            )
        ],
        layout=go.Layout(
            xaxis_title="PIT bin",
            yaxis_title="Density",
            height=280,
            margin={"l": 40, "r": 20, "t": 20, "b": 40},
        ),
    )


@renderer("reliability")
def render_reliability(frame: pl.DataFrame) -> Any:
    """Plotly reliability diagram (mean predicted vs observed rate).

    Args:
        frame: Columns ``mean_pred``, ``observed``, ``event``, ``n``
            (``LastWeekReliabilitySchema``). One event at a time, or multiple
            events as separate traces.

    Returns:
        A Plotly ``Figure``.
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    traces: list[Any] = []
    if "event" in frame.columns:
        groups = [
            (str(event), frame.filter(pl.col("event") == event))
            for event in frame["event"].unique().to_list()
        ]
    else:
        groups = [("all", frame)]
    for name, part in groups:
        traces.append(
            go.Scatter(
                x=part["mean_pred"].to_list(),
                y=part["observed"].to_list(),
                mode="markers+lines",
                name=name,
                text=[f"n={int(n)}" for n in part["n"].to_list()],
                hovertemplate="%{text}<br>pred=%{x:.2f}<br>obs=%{y:.2f}<extra></extra>",
            )
        )
    traces.append(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="perfect",
            line={"dash": "dash", "color": "gray"},
            hoverinfo="skip",
        )
    )
    return go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis_title="Mean predicted",
            yaxis_title="Observed rate",
            xaxis={"range": [0, 1]},
            yaxis={"range": [0, 1]},
            height=320,
            margin={"l": 40, "r": 20, "t": 20, "b": 40},
        ),
    )


@renderer("timing_trend")
def render_timing_trend(frame: pl.DataFrame) -> Any:
    """Line chart of ``total_s`` (and optional phase columns) across gameweeks.

    Args:
        frame: ``RunTimingTrendSchema`` plus optional phase second columns.

    Returns:
        A Plotly ``Figure``.
    """
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    x_labels = [f"GW{int(gw)}" for gw in frame["gw"].to_list()]
    traces: list[Any] = [
        go.Scatter(
            x=x_labels,
            y=frame["total_s"].to_list(),
            mode="lines+markers",
            name="total_s",
        )
    ]
    skip = {"season", "gw", "checkpoint", "total_s"}
    for column in frame.columns:
        if column in skip:
            continue
        traces.append(
            go.Scatter(
                x=x_labels,
                y=frame[column].to_list(),
                mode="lines+markers",
                name=str(column),
            )
        )
    return go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis_title="Gameweek",
            yaxis_title="Seconds",
            height=280,
            margin={"l": 40, "r": 20, "t": 20, "b": 40},
        ),
    )
