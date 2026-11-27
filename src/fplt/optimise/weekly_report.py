"""Post-GW weekly decision report writer (ops-08).

Builds ``reports/gw=<n>/weekly.md`` + parquet from pure decision scores,
maintains a season-to-date table, and notifies once per distinct report
body (ntfy/SMTP via ``notify_recommendation``; missing secrets are a no-op).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request

import polars as pl

from fplt.automation.notify import notify_recommendation
from fplt.data.overall_rank import (
    OverallRankSnapshot,
    load_latest_overall_rank,
)
from fplt.optimise.decision_scoring import score_gameweeks
from fplt.optimise.decisions import read_decisions
from fplt.schemas.weekly_report import (
    SeasonToDateSchema,
    WeeklyDecisionScoreSchema,
)
from fplt.settings import Settings, get_settings
from fplt.utils.frames import require_dataframe

_Post = Callable[[Request], object]
_SmtpFactory = Callable[[str, int], Any]

__all__ = [
    "WeeklyReportResult",
    "format_weekly_markdown",
    "report_dir",
    "rollup_season_to_date",
    "run_weekly_report",
    "write_weekly_report",
]


@dataclass(frozen=True)
class WeeklyReportResult:
    """Scored week, season-to-date rollup, markdown, and notify status."""

    score: pl.DataFrame
    season_to_date: pl.DataFrame
    report_md: str
    report_dir: Path
    content_sha256: str
    notified: bool
    notify_skipped_reason: str | None


def report_dir(data_dir: Path, gw: int) -> Path:
    """Return ``data_dir/reports/gw=<n>``."""
    return data_dir / "reports" / f"gw={int(gw)}"


def rollup_season_to_date(weekly: pl.DataFrame, *, through_gw: int) -> pl.DataFrame:
    """Aggregate weekly scores into a season-to-date row per model.

    Args:
        weekly: Rows matching ``WeeklyDecisionScoreSchema`` for one season
            through ``through_gw``.
        through_gw: Inclusive gameweek cutoff written onto the rollup.

    Returns:
        Validated ``SeasonToDateSchema`` frame.
    """
    if weekly.is_empty():
        return _empty_std()
    window = weekly.filter(pl.col("gw") <= through_gw)
    if window.is_empty():
        return _empty_std()
    # Latest row per model supplies points-at-rank / hit payback.
    latest = window.sort("gw").group_by("model_id", "season").agg(pl.all().last())
    totals = window.group_by("model_id", "season").agg(
        pl.col("gw").n_unique().alias("gameweeks"),
        pl.col("realised_points").sum().alias("season_points"),
        pl.col("plan_ev").sum().alias("plan_ev_sum"),
        pl.col("regret").sum().alias("regret_sum"),
        pl.col("captaincy_regret").sum().alias("captaincy_regret_sum"),
        pl.col("hit").sum().alias("hits_sum"),
    )
    joined = totals.join(
        latest.select(
            "model_id",
            "season",
            "hit_payback_to_date",
            "points_at_rank",
            "percentile",
        ),
        on=["model_id", "season"],
        how="left",
    ).with_columns(pl.lit(int(through_gw)).cast(pl.Int64).alias("through_gw"))
    ordered = joined.select(
        "model_id",
        "season",
        "through_gw",
        "gameweeks",
        "season_points",
        "plan_ev_sum",
        "regret_sum",
        "captaincy_regret_sum",
        "hits_sum",
        "hit_payback_to_date",
        "points_at_rank",
        "percentile",
    ).sort("model_id")
    return SeasonToDateSchema.validate(ordered)  # type: ignore[return-value]


def format_weekly_markdown(
    score: pl.DataFrame,
    season_to_date: pl.DataFrame,
    *,
    links: dict[str, str] | None = None,
) -> str:
    """Render one screen of weekly decision metrics plus season-to-date.

    Args:
        score: ``WeeklyDecisionScoreSchema`` rows for this gameweek.
        season_to_date: ``SeasonToDateSchema`` rollup.
        links: Optional label → URL map appended under Links.

    Returns:
        Markdown body suitable for ``weekly.md`` and ntfy.
    """
    if score.is_empty():
        return "_No decision scores for this gameweek._\n"
    gw = int(score["gw"][0])
    season = str(score["season"][0])
    lines = [
        f"# Weekly decision report — {season} GW{gw}",
        "",
        "## This gameweek",
        "",
        (
            "| Model | Kind | Plan EV | Realised | Δ vs EV | "
            "Hindsight | Regret | Cap regret | Hit | Hit payback |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in score.iter_rows(named=True):
        lines.append(
            "| {model} | {kind} | {ev:.1f} | {real} | {delta:+.1f} | "
            "{best} | {regret} | {cap} | {hit:.0f} | {pay:+.1f} |".format(
                model=row["model_id"],
                kind=row["gw_kind"],
                ev=float(row["plan_ev"]),
                real=int(row["realised_points"]),
                delta=float(row["realised_minus_plan_ev"]),
                best=int(row["hindsight_best"]),
                regret=int(row["regret"]),
                cap=int(row["captaincy_regret"]),
                hit=float(row["hit"]),
                pay=float(row["hit_payback_to_date"]),
            )
        )
    lines.extend(["", "## Season to date", ""])
    if season_to_date.is_empty():
        lines.append("_No season-to-date rows._")
    else:
        lines.extend(
            [
                (
                    "| Model | Through | GWs | Points | Plan EV Σ | "
                    "Regret Σ | Cap Σ | Hits Σ | Hit payback | "
                    "OR equiv | Percentile |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in season_to_date.iter_rows(named=True):
            rank = row.get("points_at_rank")
            pct = row.get("percentile")
            lines.append(
                "| {model} | {through} | {gws} | {pts} | {ev:.1f} | "
                "{reg} | {cap} | {hits:.0f} | {pay:+.1f} | {rank} | "
                "{pct} |".format(
                    model=row["model_id"],
                    through=int(row["through_gw"]),
                    gws=int(row["gameweeks"]),
                    pts=int(row["season_points"]),
                    ev=float(row["plan_ev_sum"]),
                    reg=int(row["regret_sum"]),
                    cap=int(row["captaincy_regret_sum"]),
                    hits=float(row["hits_sum"]),
                    pay=float(row["hit_payback_to_date"]),
                    rank="-" if rank is None else f"{float(rank):,.0f}",
                    pct="-" if pct is None else f"{float(pct):.4f}",
                )
            )
    if links:
        lines.extend(["", "## Links", ""])
        for label, url in links.items():
            lines.append(f"- [{label}]({url})")
    lines.append("")
    return "\n".join(lines)


def write_weekly_report(
    *,
    score: pl.DataFrame,
    season_to_date: pl.DataFrame,
    report_md: str,
    out_dir: Path,
    season_dir: Path,
) -> str:
    """Write weekly.md / parquet and update season-to-date (idempotent).

    Args:
        score: This gameweek's scores.
        season_to_date: Rollup through this gameweek.
        report_md: Rendered markdown body.
        out_dir: ``reports/gw=<n>/``.
        season_dir: ``reports/`` root for the season-to-date table.

    Returns:
        SHA-256 hex digest of ``report_md`` (UTF-8).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    season_dir.mkdir(parents=True, exist_ok=True)
    checked = WeeklyDecisionScoreSchema.validate(score)
    checked = require_dataframe(checked)
    std = SeasonToDateSchema.validate(season_to_date)
    std = require_dataframe(std)
    checked.write_parquet(out_dir / "weekly.parquet")
    (out_dir / "weekly.md").write_text(report_md, encoding="utf-8")
    std.write_parquet(season_dir / "season_to_date.parquet")
    digest = hashlib.sha256(report_md.encode("utf-8")).hexdigest()
    (out_dir / "meta.json").write_text(
        json.dumps({"content_sha256": digest}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return digest


def run_weekly_report(
    data_dir: Path,
    *,
    season: str,
    gw: int,
    model_id: str | None = None,
    decisions: pl.DataFrame | None = None,
    hindsight_best: int | None = None,
    n_fixtures: int | None = None,
    rank_snapshot: OverallRankSnapshot | None = None,
    links: dict[str, str] | None = None,
    settings: Settings | None = None,
    notify: bool = True,
    post: _Post | None = None,
    smtp_factory: _SmtpFactory | None = None,
    load_rank: Callable[[Path, str], OverallRankSnapshot | None] | None = None,
) -> WeeklyReportResult:
    """Score a checked gameweek, write the report, and notify at most once.

    Notification is skipped when the content SHA matches a previous
    ``notified.json``, or when ``notify=False``. Missing ntfy/SMTP config
    is a no-op (still recorded so a later secret does not re-spam an
    unchanged report).

    Args:
        data_dir: ``$FPLT_DATA_DIR`` root.
        season: Season label.
        gw: Checked gameweek.
        model_id: Optional ledger model filter.
        decisions: Optional in-memory decisions (else ``read_decisions``).
        hindsight_best: Optional oracle net points for ``gw``.
        n_fixtures: Optional fixture count for DGW/BGW tagging.
        rank_snapshot: Optional ladder; else latest under ``raw/``.
        links: Optional URLs for the markdown Links section.
        settings: Process settings for notify channels.
        notify: When false, never send (still writes the report).
        post: Optional ntfy fake for tests.
        smtp_factory: Optional SMTP fake for tests.
        load_rank: Optional rank loader (tests inject a stub).

    Returns:
        ``WeeklyReportResult`` with paths and notify status.

    Raises:
        ValueError: No decision rows for the requested gameweek/model.
    """
    frame = decisions if decisions is not None else read_decisions(data_dir)
    frame = frame.filter(pl.col("season") == season)
    if model_id is not None:
        frame = frame.filter(pl.col("model_id") == model_id)
    if frame.filter(pl.col("gw") == gw).is_empty():
        raise ValueError(f"no decisions for season={season} gw={gw}")
    snap = rank_snapshot
    if snap is None:
        loader = load_rank or load_latest_overall_rank
        snap = loader(data_dir / "raw", season)
    # Score every GW through ``gw`` so season-to-date and hit payback are
    # consistent, then keep this week's rows for the weekly parquet.
    hindsight_map = {gw: hindsight_best} if hindsight_best is not None else None
    fixtures_map = {gw: n_fixtures} if n_fixtures is not None else None
    history = score_gameweeks(
        frame.filter(pl.col("gw") <= gw),
        hindsight_by_gw=hindsight_map,
        n_fixtures_by_gw=fixtures_map,
        rank_snapshot=snap,
    )
    week = history.filter(pl.col("gw") == gw)
    std = rollup_season_to_date(history, through_gw=gw)
    body = format_weekly_markdown(week, std, links=links)
    out = report_dir(data_dir, gw)
    digest = write_weekly_report(
        score=week,
        season_to_date=std,
        report_md=body,
        out_dir=out,
        season_dir=data_dir / "reports",
    )
    notified = False
    skip_reason: str | None = None
    if not notify:
        skip_reason = "notify_disabled"
    elif _already_notified(out, digest):
        skip_reason = "already_notified"
    else:
        cfg = settings or get_settings()
        title = f"fplt weekly GW{gw} ({season})"
        sent = notify_recommendation(
            cfg, title, body, post=post, smtp_factory=smtp_factory
        )
        notified = sent
        if not sent:
            skip_reason = "no_notify_channel"
        _write_notified(out, digest, sent=sent)
    return WeeklyReportResult(
        score=week,
        season_to_date=std,
        report_md=body,
        report_dir=out,
        content_sha256=digest,
        notified=notified,
        notify_skipped_reason=skip_reason,
    )


def _already_notified(out_dir: Path, digest: str) -> bool:
    path = out_dir / "notified.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return str(payload.get("content_sha256", "")) == digest


def _write_notified(out_dir: Path, digest: str, *, sent: bool) -> None:
    payload = {"content_sha256": digest, "sent": sent}
    (out_dir / "notified.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _empty_std() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "season": pl.Utf8(),
            "through_gw": pl.Int64(),
            "gameweeks": pl.Int64(),
            "season_points": pl.Int64(),
            "plan_ev_sum": pl.Float64(),
            "regret_sum": pl.Int64(),
            "captaincy_regret_sum": pl.Int64(),
            "hits_sum": pl.Float64(),
            "hit_payback_to_date": pl.Float64(),
            "points_at_rank": pl.Float64(),
            "percentile": pl.Float64(),
        }
    )
