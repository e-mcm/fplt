"""Versioned feature registry.

A released ``(name, version)`` is not replaced. Change the calculation by
registering the next version.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandera.polars as pa
import polars as pl

from fplt.data.pit import PITView

FeatureFn = Callable[[PITView], pl.DataFrame]


@dataclass(frozen=True)
class Feature:
    """One registered feature version."""

    name: str
    version: int
    grain: str
    inputs: tuple[str, ...]
    schema: type[pa.DataFrameModel]
    fn: FeatureFn


_FEATURES: dict[tuple[str, int], Feature] = {}


def feature(
    name: str,
    version: int,
    grain: str,
    inputs: list[str],
    schema: type[pa.DataFrameModel],
) -> Callable[[FeatureFn], FeatureFn]:
    """Register ``fn`` as ``name`` at ``version``.

    Args:
        name: Feature name, stable across versions.
        version: Positive integer. A repeat of ``(name, version)`` is refused.
        grain: ``player_fixture``, ``team_fixture``, or ``player_gw``.
        inputs: Staged tables the function reads.
        schema: Pandera model the output must match.

    Returns:
        A decorator that leaves the function callable.
    """

    def decorate(fn: FeatureFn) -> FeatureFn:
        key = (name, version)
        if key in _FEATURES:
            raise ValueError(f"feature {name} v{version} is already registered")
        _FEATURES[key] = Feature(name, version, grain, tuple(inputs), schema, fn)
        return fn

    return decorate


def get_feature(name: str, version: int) -> Feature:
    """Return one registered feature version.

    Args:
        name: Feature name.
        version: Feature version.

    Returns:
        The registration.

    Raises:
        KeyError: That version was never registered.
    """
    try:
        return _FEATURES[(name, version)]
    except KeyError:
        raise KeyError(f"feature {name} v{version} is not registered") from None


def materialise_feature(view: PITView, name: str, version: int, root: Path) -> Path:
    """Validate a feature and write ``part.parquet``. The same deadline is kept.

    Args:
        view: Point-in-time view passed to the feature.
        name: Feature name.
        version: Feature version.
        root: Data directory that contains ``features/``.

    Returns:
        The parquet path.

    Raises:
        KeyError: The feature version is not registered.
        FileExistsError: This deadline was already materialised.
    """
    import fplt.features.understat as understat_rows

    if understat_rows.understat_share.__name__ != "understat_share":
        raise RuntimeError("understat_share did not register")
    spec = get_feature(name, version)
    checked = spec.schema.validate(spec.fn(view))
    assert isinstance(checked, pl.DataFrame)
    stamp = f"as_of={view.as_of.season}_gw{view.as_of.gw:02d}"
    path = root / "features" / name / f"v={version}" / stamp / "part.parquet"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checked.write_parquet(path)
    return path
