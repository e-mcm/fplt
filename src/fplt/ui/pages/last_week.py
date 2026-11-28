"""Last week page: post_gw season-to-date first, then week detail (#396 / #397)."""

from __future__ import annotations

from typing import Any

import polars as pl
import streamlit as st

from fplt.analysis.last_week import (
    LastWeekDetailPanel,
    LastWeekSeasonPanel,
    header_fields,
    ops_review_table,
    season_panel,
    selection_from_catalog,
    week_detail,
)
from fplt.schemas.manifests import AppliedRecord
from fplt.ui.data import DashboardStore
from fplt.ui.renderers import get_renderer


def render(store: DashboardStore | None = None) -> None:
    """Lay out the Last-week page (read-only; local files only).

    Args:
        store: Optional adapter (tests inject a fixture-backed instance).
    """
    source = store if store is not None else DashboardStore.from_settings()
    st.title("Last week")
    st.caption(f"Read-only · data_dir=`{source.data_dir}` · localhost only")

    catalog = source.post_gw_runs()
    if catalog.is_empty():
        st.info(
            "No post-GW analysis runs under `runs/`. "
            "They appear after ops-tick sees `data_checked`."
        )
        return

    seasons = sorted({str(s) for s in catalog["season"].to_list()}, reverse=True)
    season = st.selectbox("Season", seasons, index=0)
    at_season = catalog.filter(pl.col("season") == season)
    gw_options = sorted({int(g) for g in at_season["gw"].to_list()}, reverse=True)
    gw = st.selectbox("Gameweek", gw_options, index=0)
    selection = selection_from_catalog(catalog, season=season, gw=gw)
    if selection is None:
        st.warning("Could not resolve a post_gw run for that gameweek.")
        return

    finished = None
    match = catalog.filter(pl.col("run_id") == selection.run_id)
    if not match.is_empty():
        finished = match["finished_at"][0]
    header = header_fields(
        selection,
        finished_at=None if finished is None else str(finished),
    )
    st.markdown(
        f"**{header['season']} GW{header['gw']}** · model `{header['model_id']}` · "
        f"line {header['line']}"
    )
    st.caption(f"run `{header['run_id']}` · finished `{header['finished_at']}`")

    scorecards = source.post_gw_scorecards(
        season=selection.season, through_gw=selection.gw
    )
    season_to_date = source.load_table("season_to_date", run_id=selection.run_id)
    forecast_agg = source.load_table("forecast_score_agg", run_id=selection.run_id)
    forecast_scores = _prior_forecast_scores(
        source, season=selection.season, through_gw=selection.gw
    )

    panel = season_panel(
        scorecards=scorecards,
        season_to_date=season_to_date,
        forecast_agg=forecast_agg,
        forecast_scores=forecast_scores,
        season=selection.season,
        through_gw=selection.gw,
    )
    _render_season(panel)

    detail = week_detail(
        scorecard=source.load_table("scorecard", run_id=selection.run_id),
        attribution=source.load_table("attribution", run_id=selection.run_id),
        biggest_misses=source.load_table("biggest_misses", run_id=selection.run_id),
        primary=source.official_predict_table(
            season=selection.season, gw=selection.gw, line=selection.line
        ),
        benchmarks=source.benchmark_projections(
            season=selection.season, gw=selection.gw
        ),
        outcomes=source.player_outcomes(season=selection.season, gw=selection.gw),
        season=selection.season,
        gw=selection.gw,
        ops_rows=_ops_rows(source, season=selection.season, gw=selection.gw),
    )
    _render_week(detail)


def _render_season(panel: LastWeekSeasonPanel) -> None:
    st.header("Season to date")
    st.subheader("Lines (chips / ex-chips)")
    if panel.lines.is_empty():
        st.caption("No scorecard rows to roll up yet.")
    else:
        get_renderer("table")
        st.dataframe(panel.lines, hide_index=True)  # pyright: ignore[reportUnknownMemberType]

    st.subheader("Forecast skill")
    if panel.skill.is_empty():
        st.caption("No forecast_score_agg skill rows.")
    else:
        get_renderer("table")
        st.dataframe(panel.skill, hide_index=True)  # pyright: ignore[reportUnknownMemberType]

    st.subheader("PIT histogram")
    if panel.pit_hist.is_empty():
        st.caption("No PIT histogram in forecast_score_agg.")
    else:
        fig = get_renderer("pit_hist")(panel.pit_hist)
        st.plotly_chart(fig, use_container_width=True)  # pyright: ignore[reportUnknownMemberType]

    st.subheader("Reliability")
    if panel.reliability.is_empty():
        st.caption("No forecast_scores for a reliability diagram.")
    else:
        event_options = sorted({str(e) for e in panel.reliability["event"].to_list()})
        event = st.selectbox("Event", event_options, index=0)
        part = panel.reliability.filter(pl.col("event") == event)
        fig = get_renderer("reliability")(part)
        st.plotly_chart(fig, use_container_width=True)  # pyright: ignore[reportUnknownMemberType]
        st.dataframe(part, hide_index=True)  # pyright: ignore[reportUnknownMemberType]

    st.subheader("Selection bias")
    if panel.selection_bias.is_empty():
        st.caption("No selection-conditional bias rows.")
    else:
        get_renderer("table")
        st.dataframe(  # pyright: ignore[reportUnknownMemberType]
            panel.selection_bias, hide_index=True
        )

    if not panel.season_to_date.is_empty():
        with st.expander("Season-to-date rollup (raw)", expanded=False):
            st.dataframe(  # pyright: ignore[reportUnknownMemberType]
                panel.season_to_date, hide_index=True
            )


def _render_week(detail: LastWeekDetailPanel) -> None:
    st.header("This gameweek")
    st.subheader("Scorecard")
    if detail.scorecard.is_empty():
        st.caption("No scorecard for this run.")
    else:
        get_renderer("table")
        st.dataframe(  # pyright: ignore[reportUnknownMemberType]
            detail.scorecard, hide_index=True
        )

    st.subheader("Rank context")
    if detail.rank_context.is_empty():
        st.caption("No rank context.")
    else:
        get_renderer("table")
        st.dataframe(  # pyright: ignore[reportUnknownMemberType]
            detail.rank_context, hide_index=True
        )

    st.subheader("Attribution")
    if detail.attribution.is_empty():
        st.caption("No attribution for this run.")
    else:
        get_renderer("table")
        st.dataframe(  # pyright: ignore[reportUnknownMemberType]
            detail.attribution, hide_index=True
        )

    st.subheader("Biggest misses")
    if detail.biggest_misses.is_empty():
        st.caption("No biggest-misses table for this run.")
    else:
        get_renderer("table")
        st.dataframe(  # pyright: ignore[reportUnknownMemberType]
            detail.biggest_misses, hide_index=True
        )

    st.subheader("Benchmarks")
    if detail.benchmarks.is_empty():
        st.caption("No benchmark comparison for this gameweek.")
    else:
        get_renderer("table")
        st.dataframe(  # pyright: ignore[reportUnknownMemberType]
            detail.benchmarks, hide_index=True
        )

    st.subheader("Ops review")
    if detail.ops_review.is_empty():
        st.caption("No ops review rows.")
    else:
        get_renderer("table")
        st.dataframe(  # pyright: ignore[reportUnknownMemberType]
            detail.ops_review, hide_index=True
        )


def _ops_rows(source: DashboardStore, *, season: str, gw: int) -> pl.DataFrame:
    """Assemble ops-review rows from applied / status / timing / alarms."""
    applied_payload = _applied_payload(source, season=season, gw=gw)
    timings: list[dict[str, Any]] = []
    checkpoint = None if applied_payload is None else applied_payload.get("checkpoint")
    cps = [str(checkpoint)] if checkpoint else ["T-90m", "T-6h", "T-24h"]
    for cp in cps:
        record = source.ops_timing(season=season, gw=gw, checkpoint=cp)
        if record is None:
            continue
        timings.append(
            {
                "key": cp,
                "value": f"{record.total_s:.1f}s",
                "detail": record.kind,
            }
        )
        break
    return ops_review_table(
        applied=applied_payload,
        freshness=source.status_freshness_rows(),
        timings=timings,
        alarms=source.gw_alarms(season=season, gw=gw),
        last_error=source.status_last_error(),
    )


def _applied_payload(
    source: DashboardStore, *, season: str, gw: int
) -> dict[str, Any] | None:
    pointers = source.official_lines().filter(
        (pl.col("season") == season) & (pl.col("gw") == gw)
    )
    cps = sorted({str(c) for c in pointers["checkpoint"].to_list()}, reverse=True)
    ordered = ["T-90m", *[c for c in cps if c != "T-90m"]] if cps else ["T-90m"]
    for checkpoint in ordered:
        record = source.applied_record(season=season, gw=gw, checkpoint=checkpoint)
        if record is None:
            continue
        return _applied_dict(record, checkpoint=checkpoint)
    # No pointers: still try the live apply target.
    record = source.applied_record(season=season, gw=gw, checkpoint="T-90m")
    if record is None:
        return None
    return _applied_dict(record, checkpoint="T-90m")


def _applied_dict(record: AppliedRecord, *, checkpoint: str) -> dict[str, Any]:
    submitted = record.submitted_at
    stamp = submitted.isoformat() if hasattr(submitted, "isoformat") else str(submitted)
    return {
        "checkpoint": checkpoint,
        "line": record.line,
        "plan_run_id": record.plan_run_id,
        "submitted_at": stamp,
    }


def _prior_forecast_scores(
    source: DashboardStore, *, season: str, through_gw: int
) -> pl.DataFrame:
    """Concat forecast_scores from post_gw runs through ``through_gw``."""
    catalog = source.post_gw_runs().filter(
        (pl.col("season") == season) & (pl.col("gw") <= through_gw)
    )
    frames: list[pl.DataFrame] = []
    for run_id in catalog["run_id"].to_list():
        frame = source.load_table("forecast_scores", run_id=str(run_id))
        if not frame.is_empty():
            frames.append(frame)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")
