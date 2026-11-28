"""Overview page: recent runs, official lines, plan summary, quarantine."""

from __future__ import annotations

import json

import streamlit as st

from fplt.ui.data import DashboardStore
from fplt.ui.renderers import get_renderer


def render(store: DashboardStore | None = None) -> None:
    """Lay out the overview sections (read-only).

    Args:
        store: Optional adapter (tests inject a fixture-backed instance).
    """
    source = store if store is not None else DashboardStore.from_settings()
    st.title("fplt overview")
    st.caption(f"Read-only · data_dir=`{source.data_dir}` · localhost only · no writes")

    status = source.status_payload()
    if status is None:
        st.info("No `status/status.json` yet (ops-10). Quarantine still shown below.")
    else:
        st.subheader("Status")
        st.code(json.dumps(status, indent=2, default=str), language="json")

    table = get_renderer("table")

    st.subheader("Official lines")
    official = source.official_lines()
    if official.is_empty():
        st.write("No official pointers under `official/` yet (Stage B12).")
    else:
        _show_table(table(official))

    st.subheader("Latest plan summary")
    plan = source.latest_plan_summary()
    if plan.is_empty():
        st.write("No plan runs found under `runs/`.")
    else:
        _show_table(table(plan))

    st.subheader("Recent runs")
    runs = source.runs_catalog(limit=50)
    if runs.is_empty():
        st.write("No runs under `runs/` yet.")
    else:
        _show_table(table(runs))

    st.subheader("Quarantine")
    quarantine = source.quarantine_summary()
    if quarantine.is_empty():
        st.write("No active quarantine rows.")
    else:
        _show_table(table(quarantine))


def _show_table(frame: object) -> None:
    """Display a polars frame; Streamlit stubs are incomplete under pyright."""
    st.dataframe(frame, hide_index=True)  # pyright: ignore[reportUnknownMemberType]
