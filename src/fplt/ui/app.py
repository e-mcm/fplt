"""Streamlit entry point: ``uv run streamlit run src/fplt/ui/app.py``.

Bind to localhost only (``fplt ui`` sets ``--server.address localhost``).
The UI never writes; it reads through ``DashboardStore``.
"""

from __future__ import annotations

import streamlit as st

from fplt.ui.pages import overview, this_week


def main() -> None:
    """Configure the page and run navigation (This week + Overview)."""
    st.set_page_config(page_title="fplt", layout="wide")
    pages = [
        st.Page(
            this_week.render,
            title="This week",
            url_path="this-week",
            default=True,
        ),
        st.Page(overview.render, title="Overview", url_path="overview"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":  # pragma: no cover — streamlit entry point
    main()
