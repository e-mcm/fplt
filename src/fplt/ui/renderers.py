"""View-name → renderer registry (``fplt-ui-design.md`` §3.2).

Renderers are thin Streamlit/Plotly wrappers. Data shaping stays in
``fplt.analysis``. This first slice only registers a table renderer so the
registry exists for later stage panels.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl

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
