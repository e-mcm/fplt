"""Cover streamlit entry ``fplt.ui.app.main`` without launching a server."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_ui_app_main_configures_navigation(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() sets page config and runs the overview navigation."""
    fake_st = MagicMock()
    page = object()
    fake_st.Page = MagicMock(return_value=page)
    nav = MagicMock()
    fake_st.navigation = MagicMock(return_value=nav)

    monkeypatch.setitem(__import__("sys").modules, "streamlit", fake_st)
    # Re-import under test after stubbing streamlit.
    import importlib

    import fplt.ui.app as app

    importlib.reload(app)
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "overview", SimpleNamespace(render=lambda: None))

    app.main()

    fake_st.set_page_config.assert_called_once_with(page_title="fplt", layout="wide")
    fake_st.Page.assert_called_once()
    fake_st.navigation.assert_called_once_with([page])
    nav.run.assert_called_once_with()
