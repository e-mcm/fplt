"""Post-GW analysis run for ops-tick (#393 / Spec PR5).

``data_checked`` → write-once ``run_type=eval`` / ``purpose=post_gw`` run,
fingerprinted on outcomes shas + official pointers; done = ``by_fp`` exists.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from fplt.automation.cycle import current_season
from fplt.automation.notify import notify_recommendation
from fplt.automation.production import ProductionOps, load_production_ops
from fplt.data.overall_rank import OverallRankSnapshot, load_latest_overall_rank
from fplt.data.store import DataStore
from fplt.models.spec import ModelSpec, load_model_specs
from fplt.models.spec_identity import spec_hash
from fplt.optimise.attainable import component_regret
from fplt.optimise.decision_scoring import (
    score_gameweek_decisions,
    score_gameweeks,
)
from fplt.optimise.decisions import read_decisions
from fplt.optimise.realised import score_gameweek
from fplt.optimise.weekly_report import rollup_season_to_date
from fplt.schemas.manifests import AsOfDeadline, BuildManifest
from fplt.schemas.post_gw import PostGwAttributionSchema, PostGwScorecardSchema
from fplt.settings import Settings, get_settings
from fplt.store.fingerprint import fp_index_exists, run_fingerprint
from fplt.store.official import official_checkpoint_dir, pointer_path
from fplt.store.push import maybe_push_run
from fplt.store.runs import RunWriter, run_dir
from fplt.types import GWKey
from fplt.utils.frames import require_dataframe
from fplt.utils.rng import seed_for

logger = logging.getLogger(__name__)

_PURPOSE = "post_gw"
_OUTCOMES = frozenset({"player_fixture_outcomes", "fixture_outcomes"})
_CP = "T-90m"
_N = 2_000

__all__ = [
    "PostGwResult",
    "format_post_gw_ntfy",
    "format_post_gw_summary",
    "has_pending_post_gw",
    "pending_post_gw",
    "post_gw_fingerprint",
    "run_pending_post_gw",
    "run_post_gw",
]


@dataclass(frozen=True)
class PostGwResult:
    """One post_gw attempt result."""

    key: GWKey
    run_id: str | None
    fp: str
    skipped: bool
    notified: bool


def post_gw_fingerprint(
    *,
    spec: ModelSpec,
    build: BuildManifest,
    target_gw: GWKey,
    data_dir: Path,
    prod_tag: str | None = None,
    checkpoint: str = _CP,
) -> str:
    """Return fp of outcomes shas + official pointer digests for ``target_gw``."""
    base = official_checkpoint_dir(
        data_dir, season=target_gw.season, gw=target_gw.gw, checkpoint=checkpoint
    )
    pointers: dict[str, str] = {}
    if base.is_dir():
        for path in sorted(base.glob("line=*.json")):
            pointers[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        applied = base / "applied.json"
        if applied.is_file():
            pointers["applied.json"] = hashlib.sha256(applied.read_bytes()).hexdigest()
    return run_fingerprint(
        spec=spec,
        build=build,
        target_gw=target_gw,
        run_type="eval",
        prod_tag=prod_tag,
        kind="official",
        tables=set(_OUTCOMES),
        extra={
            "purpose": _PURPOSE,
            "outcomes": {
                n: (build.tables[n].sha if n in build.tables else "")
                for n in sorted(_OUTCOMES)
            },
            "pointers": pointers,
            "checkpoint": checkpoint,
        },
    )


def pending_post_gw(
    data_dir: Path,
    store: DataStore,
    build: BuildManifest,
    *,
    ops: ProductionOps | None = None,
    prod_tag: str | None = None,
    remote_by_fp: set[str] | frozenset[str] | None = None,
    specs: Mapping[str, ModelSpec] | None = None,
) -> list[tuple[GWKey, str]]:
    """Return checked GWs whose post_gw fp is missing from ``by_fp``."""
    cfg = load_production_ops() if ops is None else ops
    season = current_season(store)
    if season is None:
        return []
    spec = (load_model_specs() if specs is None else specs)[cfg.primary_line().spec]
    remote = set(remote_by_fp) if remote_by_fp is not None else None
    out: list[tuple[GWKey, str]] = []
    seen: set[GWKey] = set()
    frame = store.read("gameweeks").filter(
        pl.col("data_checked")
        & (pl.col("season") == season)
        & (pl.col("gw") >= cfg.catch_up_from_gw)
    )
    for s, g in frame.select("season", "gw").sort("gw").iter_rows():
        key = GWKey(str(s), int(g))
        if key in seen:
            continue
        seen.add(key)
        fp = post_gw_fingerprint(
            spec=spec, build=build, target_gw=key, data_dir=data_dir, prod_tag=prod_tag
        )
        if not fp_index_exists(data_dir, fp, remote_listing=remote):
            out.append((key, fp))
    return out


def has_pending_post_gw(
    data_dir: Path,
    store: DataStore,
    build: BuildManifest,
    *,
    ops: ProductionOps | None = None,
    prod_tag: str | None = None,
    remote_by_fp: set[str] | frozenset[str] | None = None,
    specs: Mapping[str, ModelSpec] | None = None,
) -> bool:
    """Return True when any checked GW still needs a post_gw run."""
    return bool(
        pending_post_gw(
            data_dir,
            store,
            build,
            ops=ops,
            prod_tag=prod_tag,
            remote_by_fp=remote_by_fp,
            specs=specs,
        )
    )


def run_post_gw(
    data_dir: Path,
    key: GWKey,
    *,
    build: BuildManifest,
    fp: str | None = None,
    ops: ProductionOps | None = None,
    prod_tag: str | None = None,
    settings: Settings | None = None,
    notify: bool = True,
    decisions: pl.DataFrame | None = None,
    n_fixtures: int | None = None,
    rank_snapshot: OverallRankSnapshot | None = None,
    specs: Mapping[str, ModelSpec] | None = None,
    post: Any | None = None,
    push: bool = True,
) -> PostGwResult:
    """Write one post_gw analysis run (skip when ``by_fp`` already has ``fp``).

    Raises:
        ValueError: Unchecked GW or missing decisions for the primary model.
    """
    cfg = get_settings() if settings is None else settings
    prod = load_production_ops() if ops is None else ops
    primary = prod.primary_line()
    spec = (load_model_specs() if specs is None else specs)[primary.spec]
    digest = fp or post_gw_fingerprint(
        spec=spec, build=build, target_gw=key, data_dir=data_dir, prod_tag=prod_tag
    )
    if fp_index_exists(data_dir, digest):
        return PostGwResult(
            key=key, run_id=None, fp=digest, skipped=True, notified=False
        )
    store = DataStore(data_dir)
    gw_row = store.read("gameweeks").filter(
        (pl.col("season") == key.season)
        & (pl.col("gw") == key.gw)
        & pl.col("data_checked")
    )
    if gw_row.is_empty():
        raise ValueError(f"post_gw refuses unchecked gameweek {key}")
    frame = (decisions if decisions is not None else read_decisions(data_dir)).filter(
        pl.col("season") == key.season
    )
    model_id, model_frame = _model_frame(frame, spec.model_id, key)
    fixtures_n = n_fixtures if n_fixtures is not None else _n_fixtures(store, key)
    snap = rank_snapshot or load_latest_overall_rank(data_dir / "raw", key.season)
    history = score_gameweeks(
        model_frame.filter(pl.col("gw") <= key.gw),
        n_fixtures_by_gw={key.gw: fixtures_n} if fixtures_n is not None else None,
        rank_snapshot=snap,
    )
    if history.filter(pl.col("gw") == key.gw).is_empty():
        raise ValueError(f"no scored week for {key}")
    std = rollup_season_to_date(history, through_gw=key.gw)
    picks = model_frame.filter(pl.col("gw") == key.gw)
    score = score_gameweek_decisions(
        model_frame,
        gw=key.gw,
        model_id=model_id,
        n_fixtures=fixtures_n,
        rank_snapshot=snap,
    )
    scorecard, attribution = _tables(
        data_dir, key, picks, score, primary.line, model_id
    )
    summary = format_post_gw_summary(scorecard, std, attribution)
    deadline = gw_row["deadline_time"][0]
    if not isinstance(deadline, datetime):
        deadline = datetime.now(UTC)
    elif deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    with RunWriter(
        data_dir,
        kind="official",
        mode="live",
        run_type="eval",
        spec_hash=spec_hash(spec),
        spec_label=spec.name,
        model_id=spec.model_id,
        as_of=[AsOfDeadline(season=key.season, gw=key.gw, deadline_utc=deadline)],
        horizon=max(1, prod.plan_horizon),
        purpose=_PURPOSE,
        tags={"gw": f"{key.season}-{key.gw}", "purpose": _PURPOSE},
        build_id=build.build_id,
        fp=digest,
        prod_tag=prod_tag,
        line=primary.line,
        squad_mode=primary.squad_mode,
        auto_build_id=False,
    ) as writer:
        writer.write_parquet("scorecard.parquet", scorecard)
        writer.write_parquet("attribution.parquet", attribution)
        writer.write_parquet("season_to_date.parquet", std)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(summary)
            tmp = Path(handle.name)
        try:
            writer.copy_file("summary.md", tmp)
        finally:
            tmp.unlink(missing_ok=True)
        run_id = writer.run_id
    if push:
        maybe_push_run(data_dir, run_id, settings=cfg)
    notified = False
    if notify:
        notified = bool(
            notify_recommendation(
                cfg,
                f"fplt post_gw {key.season} GW{key.gw}",
                format_post_gw_ntfy(scorecard, std),
                post=post,
            )
        )
    return PostGwResult(
        key=key, run_id=run_id, fp=digest, skipped=False, notified=notified
    )


def run_pending_post_gw(
    data_dir: Path,
    store: DataStore,
    build: BuildManifest,
    *,
    ops: ProductionOps | None = None,
    prod_tag: str | None = None,
    settings: Settings | None = None,
    notify: bool = True,
    remote_by_fp: set[str] | frozenset[str] | None = None,
    decisions: pl.DataFrame | None = None,
    post: Any | None = None,
    push: bool = True,
) -> list[PostGwResult]:
    """Write post_gw runs for every pending checked gameweek."""
    results: list[PostGwResult] = []
    for key, fp in pending_post_gw(
        data_dir, store, build, ops=ops, prod_tag=prod_tag, remote_by_fp=remote_by_fp
    ):
        try:
            results.append(
                run_post_gw(
                    data_dir,
                    key,
                    build=build,
                    fp=fp,
                    ops=ops,
                    prod_tag=prod_tag,
                    settings=settings,
                    notify=notify,
                    decisions=decisions,
                    post=post,
                    push=push,
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.exception("post_gw: failed for %s: %s", key, exc)
    return results


def format_post_gw_summary(
    scorecard: pl.DataFrame,
    season_to_date: pl.DataFrame,
    attribution: pl.DataFrame,
) -> str:
    """Render ``summary.md`` with season-to-date first, then this GW."""
    if scorecard.is_empty():
        return "_No post_gw scorecard._\n"
    gw, season = int(scorecard["gw"][0]), str(scorecard["season"][0])
    lines = [f"# Post-GW analysis — {season} GW{gw}", "", "## Season to date", ""]
    if season_to_date.is_empty():
        lines.append("_No season-to-date rows._")
    else:
        r = season_to_date.row(0, named=True)
        lines.append(
            f"- **{r['model_id']}** through GW{int(r['through_gw'])}: "
            f"{int(r['season_points'])} pts "
            f"(plan EV Σ {float(r['plan_ev_sum']):.1f})"
        )
    lines.extend(["", "## This gameweek", ""])
    for r in scorecard.iter_rows(named=True):
        pmf = r.get("pmf_percentile")
        ex = r.get("realised_ex_chips")
        delta = float(r["realised_minus_plan_ev"])
        lines.append(
            f"- **{r['model_id']}** ({r['gw_kind']}): "
            f"{int(r['realised_points'])} pts "
            f"(exp {float(r['plan_ev']):.1f}, Δ {delta:+.1f}; "
            f"PMF {'-' if pmf is None else f'{float(pmf):.0%}'}; "
            f"chip {r.get('chip') or '-'}; "
            f"ex-chips {'-' if ex is None else ex})"
        )
    if not attribution.is_empty():
        lines.extend(["", "## Decision attribution", ""])
        for r in attribution.iter_rows(named=True):
            cv = r.get("chip_value")
            lines.append(
                f"- luck {float(r['luck_vs_model']):+.1f}; "
                f"cap {int(r['captaincy_regret'])}; "
                f"XI/bench {int(r['xi_bench_regret'])}; "
                f"xfer {int(r['transfer_regret'])}; "
                f"chip {'-' if cv is None else f'{float(cv):+.1f}'}; "
                f"regret {int(r['regret'])}"
            )
    lines.append("")
    return "\n".join(lines)


def format_post_gw_ntfy(scorecard: pl.DataFrame, season_to_date: pl.DataFrame) -> str:
    """Return one ntfy line for a post_gw run."""
    if scorecard.is_empty():
        return "post_gw: no scorecard"
    r = scorecard.row(0, named=True)
    std = (
        int(season_to_date["season_points"][0]) if not season_to_date.is_empty() else 0
    )
    pmf = r.get("pmf_percentile")
    return (
        f"GW{int(r['gw'])}: {int(r['realised_points'])} pts "
        f"(exp {float(r['plan_ev']):.0f}, "
        f"{'-' if pmf is None else f'{float(pmf):.0%}'} PMF); season {std}"
    )


def _model_frame(
    frame: pl.DataFrame, model_id: str, key: GWKey
) -> tuple[str, pl.DataFrame]:
    part = frame.filter(pl.col("model_id") == model_id)
    if not part.filter(pl.col("gw") == key.gw).is_empty():
        return model_id, part
    models = frame["model_id"].unique().to_list() if not frame.is_empty() else []
    if len(models) != 1:
        raise ValueError(
            f"no decisions for model_id={model_id} season={key.season} gw={key.gw}"
        )
    mid = str(models[0])
    return mid, frame.filter(pl.col("model_id") == mid)


def _tables(
    data_dir: Path,
    key: GWKey,
    picks: pl.DataFrame,
    score: Any,
    line: str,
    model_id: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    chip = _chip(picks)
    hit = float(picks["hit"][0])
    ex = None if chip in (None, "freehit") else score_gameweek(picks, hit=hit, chip="")
    regret = (
        None if picks.filter(pl.col("in_xi")).is_empty() else component_regret(picks)
    )
    chip_value = None
    if chip is not None and chip not in {"wildcard", "freehit"}:
        chip_value = float(score.realised_points) - float(
            score_gameweek(picks, hit=hit, chip="")
        )
    scorecard = require_dataframe(
        PostGwScorecardSchema.validate(
            pl.DataFrame(
                [
                    {
                        "model_id": score.model_id,
                        "season": score.season,
                        "gw": score.gw,
                        "gw_kind": score.gw_kind,
                        "plan_ev": score.plan_ev,
                        "realised_points": score.realised_points,
                        "realised_minus_plan_ev": score.realised_minus_plan_ev,
                        "pmf_percentile": _pmf_pct(
                            data_dir,
                            key,
                            picks,
                            realised=score.realised_points,
                            line=line,
                            model_id=model_id,
                        ),
                        "fpl_average": None,
                        "fpl_highest": None,
                        "rank_change": None,
                        "season_points_to_date": score.season_points_to_date,
                        "points_at_rank": score.points_at_rank,
                        "percentile": score.percentile,
                        "realised_ex_chips": ex,
                        "chip": chip,
                    }
                ]
            )
        )
    )
    attribution = require_dataframe(
        PostGwAttributionSchema.validate(
            pl.DataFrame(
                [
                    {
                        "model_id": score.model_id,
                        "season": score.season,
                        "gw": score.gw,
                        "gw_kind": score.gw_kind,
                        "plan_ev": score.plan_ev,
                        "realised_points": score.realised_points,
                        "luck_vs_model": float(score.realised_minus_plan_ev),
                        "captaincy_regret": score.captaincy_regret,
                        "xi_bench_regret": (
                            0 if regret is None else int(regret["xi_bench_regret"][0])
                        ),
                        "transfer_regret": (
                            0 if regret is None else int(regret["transfer_regret"][0])
                        ),
                        "chip_timing_regret": (
                            None
                            if regret is None or regret["chip_timing_regret"][0] is None
                            else int(regret["chip_timing_regret"][0])
                        ),
                        "hindsight_best": score.hindsight_best,
                        "regret": score.regret,
                        "hit": score.hit,
                        "hit_payback_to_date": score.hit_payback_to_date,
                        "chip_value": chip_value,
                        "intervention_delta": None,
                    }
                ]
            )
        )
    )
    return scorecard, attribution


def _n_fixtures(store: DataStore, key: GWKey) -> int | None:
    try:
        fixtures = store.read("fixtures")
    except (OSError, KeyError, ValueError):
        return None
    if fixtures.is_empty() or "gw" not in fixtures.columns:
        return None
    part = fixtures.filter((pl.col("season") == key.season) & (pl.col("gw") == key.gw))
    return int(part.height) if not part.is_empty() else 0


def _chip(picks: pl.DataFrame) -> str | None:
    if "chip" not in picks.columns or picks.is_empty():
        return None
    for value in picks["chip"].to_list():
        if value is not None and str(value).strip():
            return str(value)
    return None


def _pmf_pct(
    data_dir: Path,
    key: GWKey,
    picks: pl.DataFrame,
    *,
    realised: int,
    line: str,
    model_id: str,
) -> float | None:
    path = pointer_path(
        data_dir, season=key.season, gw=key.gw, line=line, checkpoint=_CP
    )
    if not path.is_file():
        return None
    try:
        rid = str(json.loads(path.read_text(encoding="utf-8"))["predict_run_id"])
    except (OSError, KeyError, TypeError, ValueError):
        return None
    pmf_path = run_dir(data_dir, rid) / "predictions" / "points_pmf_gw.parquet"
    if not pmf_path.is_file():
        return None
    pmf = pl.read_parquet(pmf_path).filter(
        (pl.col("target_season") == key.season) & (pl.col("target_gw") == key.gw)
    )
    if pmf.is_empty():
        return None
    kept = picks.filter(~pl.col("sold")) if "sold" in picks.columns else picks
    xi = kept.filter(pl.col("in_xi"))
    if xi.is_empty():
        return 0.0 if realised <= 0 else 1.0
    rng = seed_for(model_id, key, fixture_id=key.gw)
    totals = np.zeros(_N, dtype=np.float64)
    for row in xi.iter_rows(named=True):
        part = pmf.filter(pl.col("player_key") == int(row["player_key"]))
        if part.is_empty():
            continue
        bins = part["points_bin"].to_numpy().astype(np.float64)
        probs = part["probability"].to_numpy().astype(np.float64)
        probs = probs / probs.sum()
        w = 2.0 if bool(row["captain"]) else 1.0
        totals += w * rng.choice(bins, size=_N, p=probs)
    return float(np.mean(totals <= float(realised)))
