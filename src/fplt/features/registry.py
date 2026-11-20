"""Versioned feature registry.

A released ``(name, version)`` is not replaced. Change the calculation by
registering the next version. Models ask for ``name@version`` strings and join
them through ``build_feature_matrix``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandera.polars as pa
import polars as pl

from fplt.data.pit import PITView
from fplt.models.universe import player_fixtures
from fplt.types import GWKey

FeatureFn = Callable[..., pl.DataFrame]


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


def registered_features() -> list[Feature]:
    """Features in name and version order."""
    return [spec for _, spec in sorted(_FEATURES.items())]


def call_feature(
    spec: Feature, view: PITView, horizon: Sequence[GWKey] | None = None
) -> pl.DataFrame:
    """Invoke a feature, passing a horizon when the function declares one."""
    parameters = list(inspect.signature(spec.fn).parameters)
    if len(parameters) >= 2:
        keys = list(horizon) if horizon is not None else [view.as_of]
        return spec.fn(view, keys)
    return spec.fn(view)


def build_feature_matrix(
    view: PITView,
    specs: Sequence[str],
    horizon: Sequence[GWKey] | None = None,
) -> pl.DataFrame:
    """Join ``name@version`` features onto the player-fixture horizon.

    Args:
        view: Point-in-time tables visible at one deadline.
        specs: Feature ids such as ``rolling_form@1``.
        horizon: Gameweeks to score. Defaults to ``view.as_of`` alone.

    Returns:
        ``player_fixtures`` rows with each feature's value columns attached.
    """
    import fplt.features.form as form_features
    import fplt.features.understat as understat_rows

    if form_features.context.__name__ != "context":
        raise RuntimeError("context did not register")
    if understat_rows.understat_share.__name__ != "understat_share":
        raise RuntimeError("understat_share did not register")
    keys = list(horizon) if horizon is not None else [view.as_of]
    frame = player_fixtures(view, keys)
    for item in specs:
        name, version = _parse_spec(item)
        extra = call_feature(get_feature(name, version), view, keys)
        value_columns = [
            column
            for column in extra.columns
            if column not in {"player_key", "fixture_id", "season", "gw"}
        ]
        if not value_columns:
            continue
        frame = frame.join(
            extra.select(["player_key", "fixture_id", *value_columns]),
            on=["player_key", "fixture_id"],
            how="left",
        )
    return frame


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
    import fplt.features.form as form_features
    import fplt.features.understat as understat_rows

    if form_features.context.__name__ != "context":
        raise RuntimeError("context did not register")
    if understat_rows.understat_share.__name__ != "understat_share":
        raise RuntimeError("understat_share did not register")
    spec = get_feature(name, version)
    checked = spec.schema.validate(call_feature(spec, view))
    assert isinstance(checked, pl.DataFrame)
    stamp = f"as_of={view.as_of.season}_gw{view.as_of.gw:02d}"
    path = root / "features" / name / f"v={version}" / stamp / "part.parquet"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checked.write_parquet(path)
    return path


def _parse_spec(item: str) -> tuple[str, int]:
    name, version = item.split("@", maxsplit=1)
    return name, int(version)
