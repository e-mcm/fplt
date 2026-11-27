"""This week page: official predictions, plan, and per-player drill-down."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import polars as pl
import streamlit as st

from fplt.analysis.run_timing import RunTimingPanel, run_timing_panel
from fplt.analysis.this_week import (
    OfficialSelection,
    default_official_selection,
    header_fields,
    plan_panel,
    player_drilldown,
    players_table,
    squad_keys_from_plan,
)
from fplt.ui.data import DashboardStore
from fplt.ui.renderers import get_renderer

_LOCAL_TZ = ZoneInfo("Asia/Hong_Kong")


def render(store: DashboardStore | None = None) -> None:
    """Lay out the This-week page (read-only; local files only).

    Args:
        store: Optional adapter (tests inject a fixture-backed instance).
    """
    source = store if store is not None else DashboardStore.from_settings()
    st.title("This week")
    st.caption(f"Read-only · data_dir=`{source.data_dir}` · localhost only")

    pointers = source.official_lines()
    if pointers.is_empty():
        st.info(
            "No official pointers under `official/`. Run `fplt data pull --official`."
        )
        return

    # Discover applied line for the latest T-90m GW (if any) for defaults.
    latest_gw = int(pointers["gw"].max())  # type: ignore[arg-type]
    seasons = pointers.filter(pl.col("gw") == latest_gw)["season"].to_list()
    season_guess = str(seasons[0]) if seasons else str(pointers["season"][0])
    applied = source.applied_line(season=season_guess, gw=latest_gw, checkpoint="T-90m")
    selection = default_official_selection(pointers, applied_line=applied)
    if selection is None:
        st.warning("Could not pick an official pointer.")
        return

    gw_options = sorted({int(g) for g in pointers["gw"].to_list()}, reverse=True)
    gw = st.selectbox("Gameweek", gw_options, index=gw_options.index(selection.gw))
    at_gw = pointers.filter(pl.col("gw") == gw)
    cp_options = sorted({str(c) for c in at_gw["checkpoint"].to_list()})
    cp_default = (
        selection.checkpoint if selection.checkpoint in cp_options else cp_options[0]
    )
    checkpoint = st.selectbox(
        "Checkpoint", cp_options, index=cp_options.index(cp_default)
    )
    at_cp = at_gw.filter(pl.col("checkpoint") == checkpoint)
    line_options = sorted({str(line) for line in at_cp["line"].to_list()})
    line_default = selection.line if selection.line in line_options else line_options[0]
    if applied and applied in line_options:
        line_default = applied
    line = st.selectbox("Line", line_options, index=line_options.index(line_default))
    row = at_cp.filter(pl.col("line") == line).to_dicts()[0]
    selection = OfficialSelection(
        season=str(row["season"]),
        gw=int(row["gw"]),
        checkpoint=str(row["checkpoint"]),
        line=str(row["line"]),
        role=str(row["role"]),
        predict_run_id=str(row["predict_run_id"]),
        plan_run_id=str(row["plan_run_id"]),
    )

    predict = source.run_manifest(selection.predict_run_id)
    plan_manifest = source.run_manifest(selection.plan_run_id)
    curated = source.curated_tables(getattr(predict, "build_id", None))
    deadline = _deadline(curated["gameweeks"], selection.season, selection.gw)
    header = header_fields(
        selection=selection,
        predict=predict,
        plan=plan_manifest,
        deadline_utc=deadline,
    )
    _render_header(header)
    _render_run_timing(source, selection)

    predictions = source.load_table("player_fixture", run_id=selection.predict_run_id)
    plan = source.load_table("plan", run_id=selection.plan_run_id)
    panel = plan_panel(
        plan,
        players=curated["players"],
        teams=curated["teams"],
        status=curated["player_status"],
    )
    _render_plan(panel)

    horizon_max = int(getattr(predict, "horizon", 1) or 1)
    horizon = st.slider("Horizon (gameweeks)", 1, horizon_max, min(3, horizon_max))
    squad = squad_keys_from_plan(plan)
    my_squad = st.checkbox("My squad only", value=False)
    table = players_table(
        predictions,
        players=curated["players"],
        teams=curated["teams"],
        status=curated["player_status"],
        as_of_gw=selection.gw,
        horizon=horizon,
        squad_keys=squad,
        my_squad_only=my_squad,
    )
    st.subheader("Players")
    if table.is_empty():
        st.write("No predictions for this run.")
        return
    get_renderer("table")
    st.dataframe(table, hide_index=True)  # pyright: ignore[reportUnknownMemberType]
    st.download_button(
        "Download CSV",
        data=table.write_csv(),
        file_name=f"this_week_{selection.season}_gw{selection.gw}.csv",
        mime="text/csv",
    )

    keys = table["player_key"].to_list()
    names = {
        int(r["player_key"]): str(r["web_name"])
        for r in table.select("player_key", "web_name").to_dicts()
    }
    label_options = [f"{names[int(k)]} ({k})" for k in keys]
    pick = st.selectbox("Player drill-down", label_options)
    player_key = int(keys[label_options.index(pick)])

    minutes = source.load_table("minutes", run_id=selection.predict_run_id)
    scoreline = source.load_table("scoreline", run_id=selection.predict_run_id)
    involvement = source.load_table("involvement", run_id=selection.predict_run_id)
    pmf = source.load_table("points_pmf_gw", run_id=selection.predict_run_id)
    drill = player_drilldown(
        player_key,
        predictions,
        players=curated["players"],
        fixtures=curated["fixtures"],
        teams=curated["teams"],
        season=selection.season,
        minutes=minutes if not minutes.is_empty() else None,
        scoreline=scoreline if not scoreline.is_empty() else None,
        involvement=involvement if not involvement.is_empty() else None,
        points_pmf_gw=pmf if not pmf.is_empty() else None,
        as_of_gw=selection.gw,
    )
    st.subheader(f"Drill-down · {names[player_key]}")
    if not drill.has_components:
        st.caption("No sub-model detail for this run.")
    st.dataframe(drill.fixtures, hide_index=True)  # pyright: ignore[reportUnknownMemberType]
    st.markdown("**Points split**")
    st.dataframe(drill.points_split, hide_index=True)  # pyright: ignore[reportUnknownMemberType]
    if not drill.points_pmf.is_empty():
        st.markdown("**Points distribution (next GW)**")
        fig = get_renderer("points_pmf_bar")(drill.points_pmf)
        st.plotly_chart(fig, use_container_width=True)  # pyright: ignore[reportUnknownMemberType]


def _render_run_timing(source: DashboardStore, selection: OfficialSelection) -> None:
    """Render the Run timing panel (stacked bar, sub-phases, transfers, trend)."""
    st.subheader("Run timing")
    ops = source.ops_timing(
        season=selection.season, gw=selection.gw, checkpoint=selection.checkpoint
    )
    trend = source.ops_timing_history(
        season=selection.season, checkpoint=selection.checkpoint
    )
    panel = run_timing_panel(
        ops_record=ops,
        remote_records=source.remote_timing_records(),
        trend_records=trend,
    )
    _draw_timing_panel(panel)


def _draw_timing_panel(panel: RunTimingPanel) -> None:
    """Lay out timing widgets from a shaped ``RunTimingPanel``."""
    if not panel.present:
        st.caption(
            "No timing for this checkpoint (older runs may lack ops_timing.json)."
        )
        if not panel.trend.is_empty() and panel.trend.height >= 2:
            st.markdown("**Trend across pulled gameweeks**")
            fig = get_renderer("timing_trend")(panel.trend)
            st.plotly_chart(fig, use_container_width=True)  # pyright: ignore[reportUnknownMemberType]
        return
    st.caption(f"kind `{panel.kind}` · total_s **{panel.total_s:.1f}s**")
    if not panel.top_phases.is_empty():
        fig = get_renderer("timing_stacked_bar")(panel.top_phases)
        st.plotly_chart(fig, use_container_width=True)  # pyright: ignore[reportUnknownMemberType]
        st.dataframe(panel.top_phases, hide_index=True)  # pyright: ignore[reportUnknownMemberType]
    with st.expander("Sub-phases (fit / plan / MILP)", expanded=False):
        if panel.sub_phases.is_empty():
            st.write("No nested sub-phases recorded.")
        else:
            st.dataframe(panel.sub_phases, hide_index=True)  # pyright: ignore[reportUnknownMemberType]
    if not panel.transfers.is_empty():
        st.markdown("**R2 transfers**")
        st.dataframe(panel.transfers, hide_index=True)  # pyright: ignore[reportUnknownMemberType]
    if not panel.trend.is_empty() and panel.trend.height >= 2:
        st.markdown("**Trend across pulled gameweeks**")
        fig = get_renderer("timing_trend")(panel.trend)
        st.plotly_chart(fig, use_container_width=True)  # pyright: ignore[reportUnknownMemberType]


def _render_header(header: dict[str, object]) -> None:
    deadline = header.get("deadline_utc")
    local = ""
    if isinstance(deadline, datetime):
        utc = deadline if deadline.tzinfo else deadline.replace(tzinfo=UTC)
        local = utc.astimezone(_LOCAL_TZ).strftime("%Y-%m-%d %H:%M %Z")
        deadline = utc.strftime("%Y-%m-%d %H:%M UTC")
    st.markdown(
        f"**{header['season']} GW{header['gw']}** · {header['line']} · "
        f"{header['checkpoint']} · deadline {deadline} / {local}"
    )
    st.caption(
        f"predict `{header['predict_run_id']}` · plan `{header['plan_run_id']}` · "
        f"code `{header.get('code_sha')}` · build `{header.get('build_id')}` · "
        f"finished `{header.get('finished_at')}`"
    )


def _render_plan(panel: object) -> None:
    st.subheader("Plan")
    chip = getattr(panel, "chip", None)
    st.write(
        f"Hit: {getattr(panel, 'hit', 0):.0f} · "
        f"Chip: {chip or '—'} · "
        f"Expected points (XI + captain − hit): "
        f"{getattr(panel, 'expected_points', 0):.2f}"
    )
    transfers = getattr(panel, "transfers")  # noqa: B009 - panel typed as object
    if transfers is not None and not transfers.is_empty():
        st.markdown("**Transfers**")
        st.dataframe(transfers, hide_index=True)  # pyright: ignore[reportUnknownMemberType]
    else:
        st.write("No transfers.")
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            f"**XI** (C: {getattr(panel, 'captain_key', None)}, "
            f"VC: {getattr(panel, 'vice_key', None)})"
        )
        st.dataframe(getattr(panel, "xi"), hide_index=True)  # noqa: B009  # pyright: ignore[reportUnknownMemberType]
    with cols[1]:
        st.markdown("**Bench**")
        st.dataframe(getattr(panel, "bench"), hide_index=True)  # noqa: B009  # pyright: ignore[reportUnknownMemberType]


def _deadline(gameweeks: pl.DataFrame, season: str, gw: int) -> datetime | None:
    if gameweeks.is_empty():
        return None
    match = gameweeks.filter((pl.col("season") == season) & (pl.col("gw") == gw)).sort(
        "available_at"
    )
    if match.is_empty():
        return None
    value = match["deadline_time"][-1]
    return value if isinstance(value, datetime) else None
