"""AppTest smoke: overview page renders against a fixture data dir."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.data.quarantine import append_quarantine
from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.manifests import AsOfDeadline
from fplt.settings import get_settings
from fplt.store.runs import RunWriter

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_TS = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
_APP = Path(__file__).resolve().parents[2] / "src" / "fplt" / "ui" / "app.py"


def _seed_fixture(data_dir: Path) -> None:
    deadline = AsOfDeadline(season="2025-26", gw=12, deadline_utc=_TS)
    plan = pl.DataFrame(
        {
            "model_id": ["deadbeef"],
            "as_of_season": ["2025-26"],
            "as_of_gw": [12],
            "target_gw": [12],
            "step": [0],
            "player_key": [101],
            "position": ["MID"],
            "team_key": [3],
            "in_xi": [True],
            "captain": [True],
            "vice": [False],
            "bought": [False],
            "sold": [False],
            "xpts": [5.0],
            "hit": [0.0],
        }
    )
    with RunWriter(
        data_dir,
        kind="adhoc",
        mode="live",
        run_type="plan",
        spec_hash="a" * 64,
        spec_label="demo",
        model_id="deadbeef",
        as_of=[deadline],
        horizon=3,
        plan_horizon=3,
        code_sha="abc1234",
        lockfile_hash="b" * 64,
        host="test",
        started_at=_TS,
        run_id="r_20261126T120000Z_smoke1",
    ) as writer:
        writer.write_parquet("plans/plan.parquet", plan)
    append_quarantine(
        data_dir,
        pl.DataFrame(
            {
                "season": ["2025-26"],
                "source": ["understat"],
                "capture_path": ["understat/season=2025-26/captured_at=x"],
                "captured_at": [_TS],
                "gate": ["row_drop"],
                "detail": ["smoke"],
                "previous_capture_path": [None],
                "available_at": [_TS],
                "schema_version": [SCHEMA_VERSION],
            }
        ),
    )
    status = data_dir / "status" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"ok": True}), encoding="utf-8")


def test_overview_app_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_fixture(tmp_path)
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        # Overview is no longer the default nav page; exercise it directly.
        def _overview_only() -> None:
            from fplt.ui.pages.overview import render

            render()

        app = AppTest.from_function(_overview_only, default_timeout=10)
        app.run()
        assert not app.exception
        titles = [item.value for item in app.title]
        assert any("overview" in str(value).lower() for value in titles)
        # Fixture plan + quarantine + status should surface as dataframes / code.
        assert len(app.dataframe) >= 2
    finally:
        get_settings.cache_clear()


def test_app_navigation_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        app = AppTest.from_file(str(_APP), default_timeout=10)
        app.run()
        assert not app.exception
        titles = [str(item.value).lower() for item in app.title]
        assert any("this week" in value for value in titles)
    finally:
        get_settings.cache_clear()


def test_ui_app_main_configures_navigation(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() sets page config and runs This-week + Overview navigation.

    Patch ``app.st`` in place — do not swap ``sys.modules['streamlit']`` or
    ``importlib.reload`` the app module, which left a mocked Streamlit bound
    for later AppTest runs under random order (#306).
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import fplt.ui.app as app

    fake_st = MagicMock()
    page = object()
    fake_st.Page = MagicMock(return_value=page)
    nav = MagicMock()
    fake_st.navigation = MagicMock(return_value=nav)
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "overview", SimpleNamespace(render=lambda: None))
    monkeypatch.setattr(app, "this_week", SimpleNamespace(render=lambda: None))
    app.main()
    fake_st.set_page_config.assert_called_once()
    assert fake_st.Page.call_count == 2
    fake_st.navigation.assert_called_once()
    nav.run.assert_called_once()
