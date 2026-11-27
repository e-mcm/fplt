"""Streamlit entry point: ``uv run streamlit run src/fplt/ui/app.py``.

Bind to localhost only (``fplt ui`` sets ``--server.address localhost``).
The UI never writes; it reads through ``DashboardStore``.
"""

from __future__ import annotations

import streamlit as st

from fplt.ui.pages import overview


def main() -> None:
    """Configure the page and run the overview navigation."""
    st.set_page_config(page_title="fplt", layout="wide")
    pages = [
        st.Page(overview.render, title="Overview", default=True),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":  # pragma: no cover — streamlit entry point
    main()
