"""Walk-forward blend of registered PointsPredictors.

Members are ordinary ``ModelSpec``s (typically ``dc_layered`` + ``lgbm_direct``).
The loader expands ``params.members`` name lists into ``components`` so the
ensemble ``model_id`` tracks member identity. Combination is a convex blend of
``xpts`` (and optional fields) — equal weights, or fixed per-position weights.
No OpenFPL arm until Deep/PPDA stages cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import polars as pl

from fplt.data.pit import PITView
from fplt.models.predictions import prediction_frame
from fplt.models.spec import ModelSpec, get_model, register_model, write_identity
from fplt.models.universe import player_fixtures
from fplt.types import GWKey

_BLEND_COLS = ("xpts", "xpts_sd", "p_haul", "p_60")
_POSITIONS = ("GKP", "DEF", "MID", "FWD")
_MODES = frozenset({"equal", "position"})


@register_model("forecast_combine")
class ForecastCombinePredictor:
    """Convex blend of member ``PointsPredictor`` forecasts.

    ``params.mode`` is ``equal`` (uniform weights) or ``position`` (per-position
    weights from ``params.position_weights``). Missing position rows fall back
    to equal weights. Members live under ``spec.components`` after the YAML
    loader expands ``params.members``.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        mode = spec.params.get("mode", "equal")
        if mode not in _MODES:
            raise ValueError(
                f"forecast_combine mode must be one of {_MODES}, got {mode!r}"
            )
        if not spec.components:
            raise ValueError(
                "forecast_combine needs components; set params.members to "
                "ModelSpec names so the loader can expand them"
            )
        for name, member in spec.components.items():
            if member.kind == "forecast_combine":
                raise ValueError(
                    f"forecast_combine member {name!r} cannot itself be "
                    "forecast_combine"
                )
        self._mode = str(mode)
        self._member_names = list(spec.components)
        self._arms: list[tuple[str, Any]] = [
            (name, get_model(member.kind)(member))
            for name, member in spec.components.items()
        ]
        self._position_weights = _parse_position_weights(
            spec.params.get("position_weights"),
            self._member_names,
        )
        n_arms = len(self._arms)
        self._equal_weights = {name: 1.0 / n_arms for name in self._member_names}

    def fit(self, view: PITView) -> None:
        """Fit every member on the same PIT view."""
        for _, arm in self._arms:
            arm.fit(view)

    def predict(self, view: PITView, horizon: list[GWKey]) -> pl.DataFrame:
        """Blend member predictions. Returns ``PredictionSchema``."""
        rows = player_fixtures(view, horizon)
        if rows.is_empty():
            zero = pl.lit(0.0)
            return prediction_frame(self.spec, view.as_of, rows, zero, zero, zero, zero)

        frames: list[pl.DataFrame] = []
        for name, arm in self._arms:
            predicted = arm.predict(view, horizon).select(
                "target_season",
                "target_gw",
                "player_key",
                "fixture_id",
                "xpts",
                "xpts_sd",
                "p_haul",
                "p_60",
            )
            renamed = predicted.rename(
                {
                    "xpts": f"xpts__{name}",
                    "xpts_sd": f"xpts_sd__{name}",
                    "p_haul": f"p_haul__{name}",
                    "p_60": f"p_60__{name}",
                }
            )
            frames.append(renamed)

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.join(
                frame,
                on=["target_season", "target_gw", "player_key", "fixture_id"],
                how="inner",
            )

        players = view.latest("players", by=["season", "player_key"]).collect()
        positions = players.select(
            pl.col("season").alias("target_season"),
            "player_key",
            "position",
        )
        merged = merged.join(positions, on=["target_season", "player_key"], how="left")

        weight_frame = _weight_frame(
            merged,
            self._member_names,
            self._mode,
            self._equal_weights,
            self._position_weights,
        )
        blended = merged.join(
            weight_frame,
            on=["target_season", "target_gw", "player_key", "fixture_id"],
            how="left",
        )
        for column in _BLEND_COLS:
            blended = blended.with_columns(
                _weighted_mean_expr(column, self._member_names).alias(column)
            )

        joined = rows.join(
            blended.select(
                pl.col("target_season").alias("season"),
                pl.col("target_gw").alias("gw"),
                "player_key",
                "fixture_id",
                "xpts",
                "xpts_sd",
                "p_haul",
                "p_60",
            ),
            on=["season", "gw", "player_key", "fixture_id"],
            how="left",
        )
        return prediction_frame(
            self.spec,
            view.as_of,
            joined,
            pl.col("xpts").fill_null(0.0),
            pl.col("xpts_sd"),
            pl.col("p_haul"),
            pl.col("p_60"),
        )

    def save(self, path: Path) -> None:
        """Write member artifacts and blend metadata. Refuses overwrite."""
        write_identity(path, self.spec)
        meta = {
            "mode": self._mode,
            "members": self._member_names,
            "position_weights": self._position_weights,
        }
        (path / "combine.json").write_text(
            json.dumps(meta, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        for name, arm in self._arms:
            dest = path / "members" / name
            dest.mkdir(parents=True, exist_ok=True)
            arm.save(dest)

    @classmethod
    def load(cls, path: Path, spec: ModelSpec) -> ForecastCombinePredictor:
        """Load members written by ``save``."""
        model = cls(spec)
        loaded: list[tuple[str, Any]] = []
        for name, arm in model._arms:
            member_spec = spec.components[name]
            restored = arm.load(path / "members" / name, member_spec)
            loaded.append((name, restored))
        model._arms = loaded
        return model


def _parse_position_weights(
    raw: object,
    members: list[str],
) -> dict[str, dict[str, float]]:
    """Validate optional per-position weight maps.

    Args:
        raw: YAML ``position_weights`` mapping or ``None``.
        members: Member component names in blend order.

    Returns:
        Position → member → non-negative weight (not necessarily normalised).

    Raises:
        ValueError: Bad shape, unknown member, or empty weights.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("position_weights must be a mapping of position → weights")
    parsed: dict[str, dict[str, float]] = {}
    for position_obj, weights_obj in cast(dict[object, object], raw).items():
        position = str(position_obj)
        if position not in _POSITIONS:
            raise ValueError(f"unknown position {position!r} in position_weights")
        if not isinstance(weights_obj, dict) or not weights_obj:
            raise ValueError(
                f"position_weights[{position!r}] must be a non-empty mapping"
            )
        row: dict[str, float] = {}
        for member_obj, value_obj in cast(dict[object, object], weights_obj).items():
            member = str(member_obj)
            if member not in members:
                raise ValueError(
                    f"position_weights[{position!r}] names unknown member {member!r}"
                )
            weight = float(cast(Any, value_obj))
            if weight < 0:
                raise ValueError(
                    f"position_weights[{position!r}][{member!r}] must be >= 0"
                )
            row[member] = weight
        if sum(row.values()) <= 0:
            raise ValueError(
                f"position_weights[{position!r}] must sum to a positive value"
            )
        parsed[position] = row
    return parsed


def _weight_frame(
    merged: pl.DataFrame,
    members: list[str],
    mode: str,
    equal: dict[str, float],
    position_weights: dict[str, dict[str, float]],
) -> pl.DataFrame:
    """Build one weight column per member for each prediction row."""
    keys = ["target_season", "target_gw", "player_key", "fixture_id", "position"]
    base = merged.select(keys)
    if mode == "equal" or not position_weights:
        return base.select(
            "target_season",
            "target_gw",
            "player_key",
            "fixture_id",
            *[pl.lit(equal[name]).alias(f"w__{name}") for name in members],
        )

    # Start from equal, overwrite known positions.
    weight_rows: list[dict[str, object]] = []
    for position in _POSITIONS:
        row_weights = position_weights.get(position, equal)
        total = sum(row_weights.get(name, 0.0) for name in members)
        if total <= 0:
            normalised = equal
        else:
            normalised = {name: row_weights.get(name, 0.0) / total for name in members}
        entry: dict[str, object] = {"position": position}
        for name in members:
            entry[f"w__{name}"] = normalised[name]
        weight_rows.append(entry)
    by_pos = pl.DataFrame(weight_rows)
    joined = base.join(by_pos, on="position", how="left")
    for name in members:
        joined = joined.with_columns(
            pl.col(f"w__{name}").fill_null(equal[name]).alias(f"w__{name}")
        )
    return joined.select(
        "target_season",
        "target_gw",
        "player_key",
        "fixture_id",
        *[f"w__{name}" for name in members],
    )


def _weighted_mean_expr(column: str, members: list[str]) -> pl.Expr:
    """Weighted mean of ``column__member``, ignoring null member values.

    When every member is null for a row, the result is null. Otherwise weights
    of non-null members are renormalised to sum to one.
    """
    parts: list[pl.Expr] = []
    weight_parts: list[pl.Expr] = []
    for name in members:
        value = pl.col(f"{column}__{name}")
        weight = pl.col(f"w__{name}")
        present = value.is_not_null()
        parts.append(pl.when(present).then(value * weight).otherwise(0.0))
        weight_parts.append(pl.when(present).then(weight).otherwise(0.0))
    numer = parts[0]
    for part in parts[1:]:
        numer = numer + part
    denom = weight_parts[0]
    for part in weight_parts[1:]:
        denom = denom + part
    return pl.when(denom > 0.0).then(numer / denom).otherwise(None)
