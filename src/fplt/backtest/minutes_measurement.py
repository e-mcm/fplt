"""Snapshot-backed minutes measurement: error split, weighted Brier, misses.

Diagnostic board for the minutes / availability lever (FDB-246). Restricts to
``source=fpl_api`` rows, splits Brier / log score by availability / rotation /
early-sub regimes (plus FPL flag, position, returning-from-absence), reports
xPts-weighted Brier on the decision-relevant pool, and writes a weekly misses
table ranked by ``|ΔP(start)| × xPts``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

import fplt.models.components.minutes  # noqa: F401  # pyright: ignore[reportUnusedImport]
from fplt.backtest.decision_pool import (
    POOL_FILTER_DESCRIPTION,
    decision_pool_keys,
)
from fplt.backtest.walk_forward import allowed_seasons, should_refit
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.components.minutes.common import UNAVAILABLE, availability, gate_for
from fplt.models.distributions import MinutesDist
from fplt.models.spec import ModelSpec, get_component, load_model_specs
from fplt.models.universe import player_fixtures
from fplt.schemas.minutes_measurement import (
    MINUTES_MEASUREMENT_SUMMARY_COLUMNS,
    MINUTES_MISS_COLUMNS,
    MinutesMeasurementSummarySchema,
    MinutesMissSchema,
)
from fplt.types import GWKey

logger = logging.getLogger(__name__)

_EPS = 1e-15
_DEFAULT_MINUTES = "empirical_minutes_v2"
_SNAPSHOT_SOURCE = "fpl_api"
_REGIMES = ("availability", "rotation", "early_sub")


@dataclass(frozen=True)
class MinutesMeasurementResult:
    """Summary metrics, misses table, and the directory they were written to."""

    summary: pl.DataFrame
    misses: pl.DataFrame
    detail: pl.DataFrame
    output_dir: Path


def run_minutes_measurement(
    data_dir: Path,
    season: str,
    *,
    from_gw: int = 1,
    to_gw: int = 10,
    minutes_model: str = _DEFAULT_MINUTES,
    output_dir: Path | None = None,
    top_misses_per_gw: int = 25,
    doubtful_chance: float = 0.25,
) -> MinutesMeasurementResult:
    """Walk-forward minutes diagnostics on snapshot-backed gameweeks.

    Args:
        data_dir: Root with ``staged/``.
        season: Evaluation season such as ``"2025-26"``.
        from_gw: First as-of gameweek scored.
        to_gw: Last as-of gameweek, inclusive.
        minutes_model: Minutes component YAML name.
        output_dir: Where to write parquet / JSON / markdown.
        top_misses_per_gw: Cap for the weekly misses report.
        doubtful_chance: Gate used when status is ``d`` and chance is null
            (must match the minutes model default for fair slicing).

    Returns:
        Validated summary / misses frames and the output path.

    Raises:
        ValueError: Empty window or unknown YAML name.
        KeyError: Minutes component kind is not registered.
    """
    if from_gw > to_gw:
        raise ValueError(f"from_gw ({from_gw}) must be <= to_gw ({to_gw})")
    if top_misses_per_gw < 1:
        raise ValueError(f"top_misses_per_gw must be positive, got {top_misses_per_gw}")
    specs = load_model_specs()
    if minutes_model not in specs:
        raise ValueError(f"unknown model spec {minutes_model!r}")
    spec = specs[minutes_model]
    store = DataStore(data_dir)
    pit = PITData(store)
    as_ofs = [GWKey(season, gw) for gw in range(from_gw, to_gw + 1)]
    detail = _collect_detail(
        pit,
        store,
        as_ofs,
        spec,
        doubtful_chance=doubtful_chance,
    )
    summary = _summarise(detail, store, season=season, model_name=spec.name)
    misses = weekly_minutes_misses(detail, top_per_gw=top_misses_per_gw)
    out = output_dir or (
        data_dir / "minutes_measurement" / f"{season}_gw{from_gw}-{to_gw}"
    )
    out.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(out / "summary.parquet")
    misses.write_parquet(out / "misses.parquet")
    detail.write_parquet(out / "detail.parquet")
    (out / "misses.md").write_text(format_misses_markdown(misses) + "\n")
    payload = {
        "season": season,
        "from_gw": from_gw,
        "to_gw": to_gw,
        "minutes_model": spec.name,
        "snapshot_source": _SNAPSHOT_SOURCE,
        "decision_pool": POOL_FILTER_DESCRIPTION,
        "summary": summary.to_dicts(),
        "misses": misses.to_dicts(),
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote minutes measurement under %s", out)
    return MinutesMeasurementResult(
        summary=summary,
        misses=misses,
        detail=detail,
        output_dir=out,
    )


def classify_error_regime(
    *,
    fully_available: bool,
    started: bool,
) -> str:
    """Assign availability / rotation / early-sub for one player–fixture.

    Priority partition:

    1. ``availability`` — FPL gate < 1 at the deadline (flagged / reduced).
    2. ``early_sub`` — fully available and started (score ≥60 given a start).
    3. ``rotation`` — fully available and did not start.

    Args:
        fully_available: True when the FPL availability gate is 1.0.
        started: True when official ``starts >= 1``.

    Returns:
        One of ``availability``, ``rotation``, ``early_sub``.
    """
    if not fully_available:
        return "availability"
    if started:
        return "early_sub"
    return "rotation"


def fpl_flag_label(status: str | None, chance: float | None) -> str:
    """Compact FPL flag slice label for reports.

    Args:
        status: Bootstrap ``status`` code (``a`` / ``d`` / ``i`` / …).
        chance: Effective chance-of-playing percent, or null.

    Returns:
        Slice token such as ``a``, ``d``, ``unavailable``, or ``chance_lt_100``.
    """
    if status in UNAVAILABLE:
        return "unavailable"
    if status == "d":
        return "d"
    if status is not None and status != "a":
        return str(status)
    if chance is not None and chance < 100.0:
        return "chance_lt_100"
    return "a"


def weighted_brier(
    probs: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
) -> float | None:
    """xPts-weighted mean Bernoulli Brier score.

    Args:
        probs: Predicted probabilities in ``[0, 1]``.
        outcomes: Realised 0/1 outcomes.
        weights: Non-negative importance weights (typically xPts / ep_next).

    Returns:
        Weighted mean of squared errors, or ``None`` when no positive weight.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if p.shape != y.shape or p.shape != w.shape:
        raise ValueError("probs, outcomes, and weights must share a shape")
    mask = np.isfinite(p) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return None
    ww = w[mask]
    return float(np.sum(ww * (p[mask] - y[mask]) ** 2) / np.sum(ww))


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float | None:
    """Mean Bernoulli Brier score, or ``None`` when empty."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if p.size == 0:
        return None
    return float(np.mean((p - y) ** 2))


def log_score(probs: np.ndarray, outcomes: np.ndarray) -> float | None:
    """Mean Bernoulli log loss (lower is better), or ``None`` when empty."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if p.size == 0:
        return None
    clipped = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def weekly_minutes_misses(
    detail: pl.DataFrame,
    *,
    top_per_gw: int = 25,
) -> pl.DataFrame:
    """Rank player–fixtures by ``|ΔP(start)| × xPts`` within each gameweek.

    Args:
        detail: Per player–fixture measurement rows.
        top_per_gw: Maximum rows retained per GW.

    Returns:
        Validated ``MinutesMissSchema`` rows, highest impact first within GW.
    """
    if detail.is_empty():
        return _validated_misses(pl.DataFrame(schema=MINUTES_MISS_COLUMNS))
    ranked = (
        detail.with_columns(
            (pl.col("p_start") - pl.col("started")).abs().alias("abs_delta_p_start"),
            (
                (pl.col("p_start") - pl.col("started")).abs() * pl.col("xpts_weight")
            ).alias("impact"),
        )
        .sort(["gw", "impact"], descending=[False, True])
        .with_columns(
            pl.col("impact").rank("ordinal", descending=True).over("gw").alias("_rank")
        )
        .filter(pl.col("_rank") <= top_per_gw)
        .drop("_rank")
        .select(list(MINUTES_MISS_COLUMNS))
    )
    return _validated_misses(ranked)


def format_misses_markdown(misses: pl.DataFrame) -> str:
    """Render the weekly misses table as Markdown.

    Args:
        misses: ``MinutesMissSchema`` rows.

    Returns:
        Markdown string (empty table note when there are no rows).
    """
    if misses.is_empty():
        return "# Minutes misses\n\n_No snapshot-backed rows._"
    lines = [
        "# Minutes misses",
        "",
        "Ranked by `|ΔP(start)| × xPts` within each gameweek "
        "(snapshot `source=fpl_api` only).",
        "",
    ]
    for gw in sorted(misses["gw"].unique().to_list()):
        part = misses.filter(pl.col("gw") == gw).sort("impact", descending=True)
        lines.append(f"## GW{gw}")
        lines.append("")
        lines.append(
            "| player_key | pos | regime | flag | "
            "P(start) | started | |ΔP| | xPts | impact | pool |"
        )
        lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|:---:|")
        for row in part.iter_rows(named=True):
            lines.append(
                "| {player_key} | {position} | {regime} | {fpl_flag} | "
                "{p_start:.3f} | {started:.0f} | {abs_delta_p_start:.3f} | "
                "{xpts_weight:.2f} | {impact:.3f} | {pool} |".format(
                    player_key=row["player_key"],
                    position=row["position"],
                    regime=row["regime"],
                    fpl_flag=row["fpl_flag"],
                    p_start=row["p_start"],
                    started=row["started"],
                    abs_delta_p_start=row["abs_delta_p_start"],
                    xpts_weight=row["xpts_weight"],
                    impact=row["impact"],
                    pool="Y" if row["in_decision_pool"] else "",
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_summary_table(summary: pl.DataFrame) -> str:
    """Pretty-print the measurement summary for the CLI."""
    if summary.is_empty():
        return "(no minutes measurement rows)"
    return str(
        summary.sort(["metric", "slice"]).select(
            "model_name", "metric", "slice", "value", "n"
        )
    )


def _collect_detail(
    pit: PITData,
    store: DataStore,
    as_ofs: list[GWKey],
    spec: ModelSpec,
    *,
    doubtful_chance: float,
) -> pl.DataFrame:
    model = get_component(spec.kind)(spec)
    rows: list[dict[str, object]] = []
    prior_blank = _prior_blank_map(store, as_ofs[0].season)
    pool_keys = decision_pool_keys(
        store,
        seasons=sorted({as_of.season for as_of in as_ofs}),
    )
    pool_set: set[tuple[str, int, int]] = set()
    if not pool_keys.is_empty():
        pool_set = {
            (str(season), int(gw), int(player))
            for season, gw, player in pool_keys.select(
                "as_of_season", "as_of_gw", "player_key"
            ).iter_rows()
        }
    for index, as_of in enumerate(as_ofs):
        view = _view(pit, as_of, spec)
        if should_refit(spec, index):
            model.fit(view)
        asked = player_fixtures(view, [as_of])
        if asked.is_empty():
            continue
        dist = model.predict(view, asked.select("player_key", "fixture_id"))
        probs = _minutes_probs(dist)
        actuals = _minutes_actuals(store, as_of)
        joined = probs.join(actuals, on=["player_key", "fixture_id"], how="inner")
        joined = joined.filter(pl.col("source") == _SNAPSHOT_SOURCE)
        if joined.is_empty():
            continue
        status = _deadline_status(view)
        positions = _positions(view)
        ep_next = _ep_next(view)
        chance_map = availability(view)
        blanks = prior_blank.get(as_of.gw, set())
        for row in joined.iter_rows(named=True):
            player = int(row["player_key"])
            tags = chance_map.get(player)
            gate = gate_for(tags, doubtful_chance) if tags is not None else 1.0
            fully = gate >= 1.0 - 1e-12
            started = float(row["started"]) >= 1.0
            status_code = None if tags is None else tags.status
            chance_pct = _effective_chance_pct(tags)
            flag = fpl_flag_label(status_code, chance_pct)
            position = positions.get(player, "UNK")
            weight = ep_next.get(player)
            if weight is None or not np.isfinite(weight) or weight < 0:
                weight = float(row["p_60"])
            rows.append(
                {
                    "season": as_of.season,
                    "gw": as_of.gw,
                    "player_key": player,
                    "fixture_id": int(row["fixture_id"]),
                    "position": position,
                    "source": str(row["source"]),
                    "regime": classify_error_regime(
                        fully_available=fully, started=started
                    ),
                    "fpl_flag": flag,
                    "returning_from_absence": player in blanks,
                    "p_played": float(row["p_played"]),
                    "p_60": float(row["p_60"]),
                    "p_start": float(row["p_60"]),
                    "played": float(row["played"]),
                    "is_60": float(row["is_60"]),
                    "started": float(started),
                    "xpts_weight": float(weight),
                    "status": status.get(player, "missing"),
                    "in_decision_pool": (
                        as_of.season,
                        as_of.gw,
                        player,
                    )
                    in pool_set,
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "gw": pl.Int64(),
                "player_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "position": pl.Utf8(),
                "source": pl.Utf8(),
                "regime": pl.Utf8(),
                "fpl_flag": pl.Utf8(),
                "returning_from_absence": pl.Boolean(),
                "p_played": pl.Float64(),
                "p_60": pl.Float64(),
                "p_start": pl.Float64(),
                "played": pl.Float64(),
                "is_60": pl.Float64(),
                "started": pl.Float64(),
                "xpts_weight": pl.Float64(),
                "status": pl.Utf8(),
                "in_decision_pool": pl.Boolean(),
            }
        )
    return pl.DataFrame(rows)


def _summarise(
    detail: pl.DataFrame,
    store: DataStore,
    *,
    season: str,
    model_name: str,
) -> pl.DataFrame:
    del store, season  # reserved for future pool rebuilds; pool flag is on detail
    rows: list[dict[str, object]] = []
    if detail.is_empty():
        empty = pl.DataFrame(schema=MINUTES_MEASUREMENT_SUMMARY_COLUMNS)
        return _validated_summary(empty)

    def add_slice(label: str, part: pl.DataFrame) -> None:
        rows.extend(_event_metrics(model_name, label, part))

    add_slice(f"source={_SNAPSHOT_SOURCE}", detail)
    for regime in _REGIMES:
        add_slice(
            f"source={_SNAPSHOT_SOURCE}|regime={regime}",
            detail.filter(pl.col("regime") == regime),
        )
    for flag in sorted(detail["fpl_flag"].unique().to_list()):
        add_slice(
            f"source={_SNAPSHOT_SOURCE}|flag={flag}",
            detail.filter(pl.col("fpl_flag") == flag),
        )
    for position in sorted(detail["position"].unique().to_list()):
        add_slice(
            f"source={_SNAPSHOT_SOURCE}|position={position}",
            detail.filter(pl.col("position") == position),
        )
    for returning in (True, False):
        add_slice(
            f"source={_SNAPSHOT_SOURCE}|returning={str(returning).lower()}",
            detail.filter(pl.col("returning_from_absence") == returning),
        )
    pool = detail.filter(pl.col("in_decision_pool"))
    add_slice("decision_pool", pool)
    rows.extend(_weighted_metrics(model_name, "decision_pool|weighted", pool))
    return _validated_summary(
        pl.DataFrame(rows, schema=MINUTES_MEASUREMENT_SUMMARY_COLUMNS)
        if rows
        else pl.DataFrame(schema=MINUTES_MEASUREMENT_SUMMARY_COLUMNS)
    )


def _event_metrics(
    model_name: str, slice_name: str, part: pl.DataFrame
) -> list[dict[str, object]]:
    if part.is_empty():
        return [
            _metric_row(model_name, name, slice_name, None, 0)
            for name in (
                "brier_played",
                "log_played",
                "brier_60",
                "log_60",
                "brier_start",
                "log_start",
            )
        ]
    events = (
        ("played", "p_played", "played"),
        ("60", "p_60", "is_60"),
        ("start", "p_start", "started"),
    )
    out: list[dict[str, object]] = []
    for label, p_col, y_col in events:
        p = part[p_col].to_numpy()
        y = part[y_col].to_numpy()
        out.append(
            _metric_row(
                model_name, f"brier_{label}", slice_name, brier_score(p, y), part.height
            )
        )
        out.append(
            _metric_row(
                model_name, f"log_{label}", slice_name, log_score(p, y), part.height
            )
        )
    return out


def _weighted_metrics(
    model_name: str, slice_name: str, part: pl.DataFrame
) -> list[dict[str, object]]:
    if part.is_empty():
        return [
            _metric_row(model_name, "weighted_brier_played", slice_name, None, 0),
            _metric_row(model_name, "weighted_brier_60", slice_name, None, 0),
            _metric_row(model_name, "weighted_brier_start", slice_name, None, 0),
        ]
    weights = part["xpts_weight"].to_numpy()
    return [
        _metric_row(
            model_name,
            "weighted_brier_played",
            slice_name,
            weighted_brier(
                part["p_played"].to_numpy(), part["played"].to_numpy(), weights
            ),
            part.height,
        ),
        _metric_row(
            model_name,
            "weighted_brier_60",
            slice_name,
            weighted_brier(part["p_60"].to_numpy(), part["is_60"].to_numpy(), weights),
            part.height,
        ),
        _metric_row(
            model_name,
            "weighted_brier_start",
            slice_name,
            weighted_brier(
                part["p_start"].to_numpy(), part["started"].to_numpy(), weights
            ),
            part.height,
        ),
    ]


def _metric_row(
    model_name: str,
    metric: str,
    slice_name: str,
    value: float | None,
    n: int,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "metric": metric,
        "slice": slice_name,
        "value": value,
        "n": n,
    }


def _minutes_probs(dist: MinutesDist) -> pl.DataFrame:
    bins = np.asarray(dist.bins)
    pmf = np.asarray(dist.pmf, dtype=np.float64)
    zero = np.where(bins == 0)[0]
    p_zero = pmf[:, int(zero[0])] if len(zero) else np.zeros(pmf.shape[0])
    p_60 = (
        pmf[:, bins >= 60].sum(axis=1) if np.any(bins >= 60) else np.zeros(pmf.shape[0])
    )
    return dist.keys.with_columns(
        pl.Series("p_played", 1.0 - p_zero),
        pl.Series("p_60", p_60),
    )


def _minutes_actuals(store: DataStore, as_of: GWKey) -> pl.DataFrame:
    return (
        store.read("player_fixture_stats")
        .filter((pl.col("season") == as_of.season) & (pl.col("gw") == as_of.gw))
        .sort("available_at")
        .unique(subset=["player_key", "fixture_id"], keep="last")
        .select(
            "player_key",
            "fixture_id",
            "source",
            (pl.col("minutes") > 0).cast(pl.Float64).alias("played"),
            (pl.col("minutes") >= 60).cast(pl.Float64).alias("is_60"),
            (pl.col("starts") > 0).cast(pl.Float64).alias("started"),
        )
    )


def _prior_blank_map(store: DataStore, season: str) -> dict[int, set[int]]:
    """Map target GW → players whose previous appearance was a blank."""
    stats = (
        store.read("player_fixture_stats")
        .filter(pl.col("season") == season)
        .sort(["player_key", "gw", "available_at"])
        .unique(subset=["player_key", "gw"], keep="last")
        .select("player_key", "gw", "minutes")
    )
    by_gw: dict[int, set[int]] = {}
    if stats.is_empty():
        return by_gw
    last_minutes: dict[int, int] = {}
    for gw in sorted(stats["gw"].unique().to_list()):
        blanks = {player for player, minutes in last_minutes.items() if minutes <= 0}
        by_gw[int(gw)] = blanks
        for row in stats.filter(pl.col("gw") == gw).iter_rows(named=True):
            last_minutes[int(row["player_key"])] = int(row["minutes"])
    return by_gw


def _deadline_status(view: PITView) -> dict[int, str]:
    frame = (
        view.latest("player_status", ["season", "player_key"])
        .collect()
        .sort("available_at")
        .unique(subset=["player_key"], keep="last")
    )
    return {
        int(player): str(status)
        for player, status in zip(
            frame["player_key"].to_list(), frame["status"].to_list(), strict=True
        )
    }


def _positions(view: PITView) -> dict[int, str]:
    frame = (
        view.latest("players", ["season", "player_key"])
        .collect()
        .sort("available_at")
        .unique(subset=["player_key"], keep="last")
    )
    return {
        int(player): str(position)
        for player, position in zip(
            frame["player_key"].to_list(), frame["position"].to_list(), strict=True
        )
    }


def _ep_next(view: PITView) -> dict[int, float]:
    frame = (
        view.latest("player_status", ["season", "player_key"])
        .collect()
        .sort("available_at")
        .unique(subset=["player_key"], keep="last")
    )
    if "ep_next" not in frame.columns:
        return {}
    found: dict[int, float] = {}
    for row in frame.select("player_key", "ep_next").iter_rows(named=True):
        value = row["ep_next"]
        if value is None:
            continue
        found[int(row["player_key"])] = float(value)
    return found


def _effective_chance_pct(tags: object) -> float | None:
    if tags is None:
        return None
    chance_this = getattr(tags, "chance_this", None)
    chance_next = getattr(tags, "chance_next", None)
    if chance_this is not None:
        return float(chance_this)
    if chance_next is not None:
        return float(chance_next)
    return None


def _view(pit: PITData, as_of: GWKey, spec: ModelSpec) -> PITView:
    seasons = allowed_seasons(as_of, spec.train_window.seasons_back)
    return pit.view(as_of).within_seasons(seasons)


def _validated_summary(frame: pl.DataFrame) -> pl.DataFrame:
    checked = MinutesMeasurementSummarySchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def _validated_misses(frame: pl.DataFrame) -> pl.DataFrame:
    checked = MinutesMissSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


__all__ = [
    "MinutesMeasurementResult",
    "brier_score",
    "classify_error_regime",
    "format_misses_markdown",
    "format_summary_table",
    "fpl_flag_label",
    "log_score",
    "run_minutes_measurement",
    "weekly_minutes_misses",
    "weighted_brier",
]
