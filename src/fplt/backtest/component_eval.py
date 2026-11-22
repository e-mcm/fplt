"""Walk-forward eval for layered components and an ``n_sims`` stability sweep.

Scorelines: mean RPS, categorical log-loss (home/draw/away), and clean-sheet
Brier (home & away) for Dixon-Coles, independent Poisson, and the odds
Poisson baseline. Minutes: Brier on P(played) and P(≥60), sliced by stats
``source``. Involvement: MAE of predicted goal share vs realised share among
players with minutes. The sims sweep refits ``dc_layered`` once and
re-predicts one deadline under several ``n_sims`` values.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

import fplt.models.baselines  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.components.involvement  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.components.minutes  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.components.score  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.composite  # noqa: F401  # pyright: ignore[reportUnusedImport]
from fplt.backtest.components import ranked_probability_score, record_scorelines
from fplt.backtest.walk_forward import allowed_seasons, should_refit
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.distributions import MinutesDist
from fplt.models.spec import ModelSpec, get_component, get_model, load_model_specs
from fplt.models.universe import player_fixtures
from fplt.schemas.component_eval import (
    COMPONENT_EVAL_COLUMNS,
    PER_MATCH_COLUMNS,
    SIMS_SWEEP_COLUMNS,
    ComponentEvalPerMatchSchema,
    ComponentEvalSummarySchema,
    SimsSweepSchema,
)
from fplt.types import GWKey

logger = logging.getLogger(__name__)

DEFAULT_SIMS = (1000, 2000, 5000, 10_000, 20_000)
_DEFAULT_SCORE = ("dixon_coles_v1", "poisson_score_v1", "odds_poisson_v1")
_DEFAULT_MINUTES = ("empirical_minutes_v1",)
_DEFAULT_INVOLVEMENT = ("involvement_shares_v1",)
_EPS = 1e-12


@dataclass(frozen=True)
class ComponentEvalResult:
    """Summary tables and the directory they were written to."""

    summary: pl.DataFrame
    sims: pl.DataFrame
    output_dir: Path
    per_match: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame(schema=PER_MATCH_COLUMNS)
    )


def run_component_eval(
    data_dir: Path,
    season: str,
    *,
    from_gw: int = 1,
    to_gw: int = 10,
    fit_gw: int | None = None,
    score_models: list[str] | None = None,
    minutes_model: str | list[str] | None = "empirical_minutes_v1",
    involvement_model: str | list[str] | None = "involvement_shares_v1",
    layered_model: str | None = "dc_layered_v1",
    sims: tuple[int, ...] | list[int] | None = None,
    output_dir: Path | None = None,
    write_per_match: bool = False,
    extra_score_specs: list[ModelSpec] | None = None,
) -> ComponentEvalResult:
    """Evaluate layered components over ``from_gw``..``to_gw`` and sweep sims.

    Args:
        data_dir: Root with ``staged/``.
        season: Evaluation season such as ``"2024-25"``.
        from_gw: First as-of gameweek scored.
        to_gw: Last as-of gameweek, inclusive.
        fit_gw: Deadline used for the sims sweep fit (default: ``from_gw``).
        score_models: YAML score specs; default DC, Poisson, and odds Poisson.
            Pass ``[]`` to skip the named score suite.
        minutes_model: Minutes YAML name(s); default ``empirical_minutes_v1``.
            Pass ``[]`` to skip.
        involvement_model: Involvement YAML name(s); default
            ``involvement_shares_v1``. Pass ``[]`` to skip.
        layered_model: Composite YAML used for the ``n_sims`` sweep. Ignored
            when ``sims`` is empty. Pass ``None`` with a non-empty sweep to
            raise.
        sims: ``n_sims`` values to sweep; ``None`` uses defaults; ``[]`` skips.
        output_dir: Where to write summary parquet/JSON.
        write_per_match: When true, also write fixture × arm × metric rows for
            scoreline models under ``per_match.parquet``.
        extra_score_specs: Inline score ``ModelSpec`` objects scored alongside
            ``score_models`` (e.g. a raw de-vig 1X2 market arm).

    Returns:
        Validated summary frames and the output path.

    Raises:
        ValueError: Empty window or unknown YAML names.
        KeyError: A component kind is not registered.
    """
    if from_gw > to_gw:
        raise ValueError(f"from_gw ({from_gw}) must be <= to_gw ({to_gw})")
    specs = load_model_specs()
    score_names = list(_DEFAULT_SCORE) if score_models is None else list(score_models)
    minutes_names = _as_name_list(minutes_model, default=_DEFAULT_MINUTES)
    involvement_names = _as_name_list(involvement_model, default=_DEFAULT_INVOLVEMENT)
    needed = [*score_names, *minutes_names, *involvement_names]
    sweep_ns = tuple(sims) if sims is not None else DEFAULT_SIMS
    if sweep_ns:
        if layered_model is None:
            raise ValueError("layered_model is required when running the sims sweep")
        needed.append(layered_model)
    for name in needed:
        if name not in specs:
            raise ValueError(f"unknown model spec {name!r}")
    score_spec_list = [specs[name] for name in score_names]
    if extra_score_specs:
        score_spec_list.extend(extra_score_specs)

    store = DataStore(data_dir)
    pit = PITData(store)
    as_ofs = [GWKey(season, gw) for gw in range(from_gw, to_gw + 1)]
    rows: list[dict[str, object]] = []
    per_match_rows: list[dict[str, object]] = []
    if score_spec_list:
        summary_rows, match_rows = _eval_scorelines(
            pit,
            as_ofs,
            score_spec_list,
            store,
            collect_per_match=write_per_match,
        )
        rows.extend(summary_rows)
        per_match_rows.extend(match_rows)
    for name in minutes_names:
        rows.extend(_eval_minutes(pit, as_ofs, specs[name], store))
    for name in involvement_names:
        rows.extend(_eval_involvement(pit, as_ofs, specs[name], store))

    if sweep_ns:
        assert layered_model is not None
        sims_frame = _sweep_n_sims(
            pit,
            specs[layered_model],
            season=season,
            fit_gw=from_gw if fit_gw is None else fit_gw,
            sims=sweep_ns,
        )
    else:
        sims_frame = pl.DataFrame(schema=SIMS_SWEEP_COLUMNS)

    summary = _validated_summary(
        pl.DataFrame(rows, schema=COMPONENT_EVAL_COLUMNS)
        if rows
        else pl.DataFrame(schema=COMPONENT_EVAL_COLUMNS)
    )
    sims_out = _validated_sims(sims_frame)
    per_match = _validated_per_match(
        pl.DataFrame(per_match_rows, schema=PER_MATCH_COLUMNS)
        if per_match_rows
        else pl.DataFrame(schema=PER_MATCH_COLUMNS)
    )
    out = output_dir or (data_dir / "component_eval" / f"{season}_gw{from_gw}-{to_gw}")
    out.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(out / "summary.parquet")
    sims_out.write_parquet(out / "sims_sweep.parquet")
    if write_per_match:
        per_match.write_parquet(out / "per_match.parquet")
    payload = {
        "season": season,
        "from_gw": from_gw,
        "to_gw": to_gw,
        "fit_gw": from_gw if fit_gw is None else fit_gw,
        "score_models": score_names,
        "minutes_models": minutes_names,
        "involvement_models": involvement_names,
        "layered_model": layered_model,
        "summary": summary.to_dicts(),
        "sims_sweep": sims_out.to_dicts(),
        "per_match": per_match.to_dicts() if write_per_match else [],
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote component eval under %s", out)
    return ComponentEvalResult(
        summary=summary, sims=sims_out, output_dir=out, per_match=per_match
    )


def _as_name_list(
    value: str | list[str] | None, *, default: tuple[str, ...]
) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return list(value)


def format_component_table(summary: pl.DataFrame) -> str:
    """Pretty-print the summary for the CLI."""
    if summary.is_empty():
        return "(no component metrics)"
    return str(
        summary.sort(["component", "model_name", "metric", "slice"]).select(
            "component", "model_name", "metric", "slice", "value", "n"
        )
    )


def _eval_scorelines(
    pit: PITData,
    as_ofs: list[GWKey],
    specs: list[ModelSpec],
    store: DataStore,
    *,
    collect_per_match: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    per_match: list[dict[str, object]] = []
    for spec in specs:
        model = get_component(spec.kind)(spec)
        rps_vals: list[float] = []
        logloss_vals: list[float] = []
        cs_home_vals: list[float] = []
        cs_away_vals: list[float] = []
        for index, as_of in enumerate(as_ofs):
            view = _view(pit, as_of, spec)
            if should_refit(spec, index):
                model.fit(view)
            fixtures = _target_fixtures(view, as_of)
            if fixtures.is_empty():
                continue
            frame = record_scorelines(model, view, fixtures)
            dist = model.predict(view, fixtures)
            # Actuals come from the store (post-deadline), not the PIT view.
            actuals = _score_actuals(store, as_of)
            joined = frame.join(actuals, on="fixture_id", how="inner")
            if joined.is_empty():
                continue
            home = joined["actual_home"].to_numpy().astype(np.float64)
            away = joined["actual_away"].to_numpy().astype(np.float64)
            rps = ranked_probability_score(
                joined["p_home"].to_numpy(),
                joined["p_draw"].to_numpy(),
                joined["p_away"].to_numpy(),
                home,
                away,
            )
            logloss = _outcome_logloss(
                joined["p_home"].to_numpy(),
                joined["p_draw"].to_numpy(),
                joined["p_away"].to_numpy(),
                home,
                away,
            )
            rps_vals.extend(float(value) for value in rps)
            logloss_vals.extend(logloss)
            order = {int(fid): i for i, fid in enumerate(dist.fixture_ids.tolist())}
            idx = [order[int(fid)] for fid in joined["fixture_id"].to_list()]
            p_cs_home = dist.p_clean_sheet("home")[idx]
            p_cs_away = dist.p_clean_sheet("away")[idx]
            cs_home = ((p_cs_home - (away == 0.0)) ** 2).tolist()
            cs_away = ((p_cs_away - (home == 0.0)) ** 2).tolist()
            cs_home_vals.extend(cs_home)
            cs_away_vals.extend(cs_away)
            if collect_per_match:
                fixture_ids = joined["fixture_id"].to_list()
                for i, fixture_id in enumerate(fixture_ids):
                    for metric, value in (
                        ("rps", float(rps[i])),
                        ("logloss", float(logloss[i])),
                        ("brier_cs_home", float(cs_home[i])),
                        ("brier_cs_away", float(cs_away[i])),
                    ):
                        per_match.append(
                            {
                                "season": as_of.season,
                                "gw": as_of.gw,
                                "fixture_id": int(fixture_id),
                                "arm": spec.name,
                                "metric": metric,
                                "value": value,
                            }
                        )
        rows.append(_metric("score", "rps", spec.name, "overall", rps_vals))
        rows.append(_metric("score", "logloss", spec.name, "overall", logloss_vals))
        rows.append(
            _metric("score", "brier_cs_home", spec.name, "overall", cs_home_vals)
        )
        rows.append(
            _metric("score", "brier_cs_away", spec.name, "overall", cs_away_vals)
        )
    return rows, per_match


def _eval_minutes(
    pit: PITData, as_ofs: list[GWKey], spec: ModelSpec, store: DataStore
) -> list[dict[str, object]]:
    model = get_component(spec.kind)(spec)
    by_slice: dict[str, dict[str, list[float]]] = {}
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
        if joined.is_empty():
            continue
        for source in sorted(joined["source"].unique().to_list()):
            part = joined.filter(pl.col("source") == source)
            label = f"source={source}"
            bucket = by_slice.setdefault(label, {"brier_played": [], "brier_60": []})
            bucket["brier_played"].extend(
                ((part["p_played"] - part["played"]) ** 2).to_list()
            )
            bucket["brier_60"].extend(((part["p_60"] - part["is_60"]) ** 2).to_list())
        overall = by_slice.setdefault("overall", {"brier_played": [], "brier_60": []})
        overall["brier_played"].extend(
            ((joined["p_played"] - joined["played"]) ** 2).to_list()
        )
        overall["brier_60"].extend(((joined["p_60"] - joined["is_60"]) ** 2).to_list())
    rows: list[dict[str, object]] = []
    for label, metrics in sorted(by_slice.items()):
        rows.append(
            _metric(
                "minutes", "brier_played", spec.name, label, metrics["brier_played"]
            )
        )
        rows.append(
            _metric("minutes", "brier_60", spec.name, label, metrics["brier_60"])
        )
    return rows


def _eval_involvement(
    pit: PITData, as_ofs: list[GWKey], spec: ModelSpec, store: DataStore
) -> list[dict[str, object]]:
    model = get_component(spec.kind)(spec)
    errors: list[float] = []
    for index, as_of in enumerate(as_ofs):
        view = _view(pit, as_of, spec)
        if should_refit(spec, index):
            model.fit(view)
        asked = player_fixtures(view, [as_of])
        if asked.is_empty():
            continue
        weights = model.predict(view, asked.select("player_key", "fixture_id"))
        actuals = _goal_actuals(store, as_of)
        joined = (
            weights.join(actuals, on=["player_key", "fixture_id"], how="inner")
            .filter(pl.col("minutes") > 0)
            .with_columns(
                pl.col("goals_scored")
                .sum()
                .over("fixture_id", "team_key")
                .alias("team_goals")
            )
            .filter(pl.col("team_goals") > 0)
            .with_columns(
                (pl.col("goals_scored") / pl.col("team_goals")).alias("actual_share")
            )
        )
        if joined.is_empty():
            continue
        errors.extend((joined["goal_weight"] - joined["actual_share"]).abs().to_list())
    return [_metric("involvement", "mae_goal_share", spec.name, "overall", errors)]


def _sweep_n_sims(
    pit: PITData,
    spec: ModelSpec,
    *,
    season: str,
    fit_gw: int,
    sims: tuple[int, ...] | list[int],
) -> pl.DataFrame:
    if not sims:
        return pl.DataFrame(schema=SIMS_SWEEP_COLUMNS)
    as_of = GWKey(season, fit_gw)
    view = _view(pit, as_of, spec)
    ordered = sorted(int(value) for value in sims)
    ref_n = max(ordered)
    predictions: dict[int, pl.DataFrame] = {}
    times: dict[int, float] = {}
    for n_sims in ordered:
        variant = spec.model_copy(update={"params": {**spec.params, "n_sims": n_sims}})
        model = get_model(variant.kind)(variant)
        started = time.perf_counter()
        model.fit(view)
        frame = model.predict(view, [as_of])
        times[n_sims] = time.perf_counter() - started
        predictions[n_sims] = frame.select(
            "player_key", "fixture_id", "xpts", "xpts_sd"
        )
        logger.info("n_sims=%s wall=%.2fs rows=%s", n_sims, times[n_sims], frame.height)
    ref = predictions[ref_n].rename({"xpts": "xpts_ref", "xpts_sd": "xpts_sd_ref"})
    rows: list[dict[str, object]] = []
    for n_sims in ordered:
        frame = predictions[n_sims].join(
            ref, on=["player_key", "fixture_id"], how="inner"
        )
        if frame.is_empty():
            rows.append(
                {
                    "n_sims": n_sims,
                    "wall_seconds": times[n_sims],
                    "mean_abs_delta_xpts": None,
                    "max_abs_delta_xpts": None,
                    "mean_xpts_sd": None,
                    "analytic_se": None,
                    "n_players": 0,
                }
            )
            continue
        delta = (frame["xpts"] - frame["xpts_ref"]).abs()
        mean_sd = _as_float(frame["xpts_sd"].drop_nulls().mean())
        rows.append(
            {
                "n_sims": n_sims,
                "wall_seconds": times[n_sims],
                "mean_abs_delta_xpts": _as_float(delta.mean()),
                "max_abs_delta_xpts": _as_float(delta.max()),
                "mean_xpts_sd": mean_sd,
                "analytic_se": mean_sd / math.sqrt(n_sims) if n_sims else None,
                "n_players": frame.height,
            }
        )
    return pl.DataFrame(rows, schema=SIMS_SWEEP_COLUMNS)


def _as_float(value: object) -> float:
    if value is None or isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return 0.0


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


def _score_actuals(store: DataStore, as_of: GWKey) -> pl.DataFrame:
    return (
        store.read("fixtures")
        .filter(
            (pl.col("season") == as_of.season)
            & (pl.col("gw") == as_of.gw)
            & pl.col("finished")
            & pl.col("team_h_score").is_not_null()
        )
        .sort("available_at")
        .unique(subset=["fixture_id"], keep="last")
        .select(
            "fixture_id",
            pl.col("team_h_score").alias("actual_home"),
            pl.col("team_a_score").alias("actual_away"),
        )
    )


def _minutes_actuals(store: DataStore, as_of: GWKey) -> pl.DataFrame:
    stats = (
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
        )
    )
    return stats


def _goal_actuals(store: DataStore, as_of: GWKey) -> pl.DataFrame:
    stats = (
        store.read("player_fixture_stats")
        .filter((pl.col("season") == as_of.season) & (pl.col("gw") == as_of.gw))
        .sort("available_at")
        .unique(subset=["player_key", "fixture_id"], keep="last")
        .select("player_key", "fixture_id", "minutes", "goals_scored")
    )
    players = (
        store.read("players")
        .filter(pl.col("season") == as_of.season)
        .sort("available_at")
        .unique(subset=["player_key"], keep="last")
        .select("player_key", "team_key")
    )
    return stats.join(players, on="player_key", how="inner")


def _target_fixtures(view: PITView, as_of: GWKey) -> pl.DataFrame:
    return (
        view.latest("fixtures", by=["season", "fixture_id"])
        .filter((pl.col("season") == as_of.season) & (pl.col("gw") == as_of.gw))
        .select("season", "gw", "fixture_id", "team_h_key", "team_a_key")
        .collect()
    )


def _view(pit: PITData, as_of: GWKey, spec: ModelSpec) -> PITView:
    seasons = allowed_seasons(as_of, spec.train_window.seasons_back)
    return pit.view(as_of).within_seasons(seasons)


def _outcome_logloss(
    p_home: np.ndarray,
    p_draw: np.ndarray,
    p_away: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> list[float]:
    home_win = home_goals > away_goals
    draw = home_goals == away_goals
    probs = np.where(home_win, p_home, np.where(draw, p_draw, p_away))
    return (-np.log(np.clip(probs, _EPS, 1.0))).tolist()


def _metric(
    component: str,
    metric: str,
    model_name: str,
    slice_name: str,
    values: list[float],
) -> dict[str, object]:
    if not values:
        return {
            "component": component,
            "metric": metric,
            "model_name": model_name,
            "slice": slice_name,
            "value": None,
            "n": 0,
        }
    return {
        "component": component,
        "metric": metric,
        "model_name": model_name,
        "slice": slice_name,
        "value": float(np.mean(np.asarray(values, dtype=np.float64))),
        "n": len(values),
    }


def _validated_summary(frame: pl.DataFrame) -> pl.DataFrame:
    checked = ComponentEvalSummarySchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def _validated_sims(frame: pl.DataFrame) -> pl.DataFrame:
    checked = SimsSweepSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def _validated_per_match(frame: pl.DataFrame) -> pl.DataFrame:
    checked = ComponentEvalPerMatchSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


__all__ = [
    "ComponentEvalResult",
    "DEFAULT_SIMS",
    "format_component_table",
    "run_component_eval",
]
