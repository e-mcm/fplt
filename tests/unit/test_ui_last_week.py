"""AppTest smoke: Last-week page renders from a fixture post_gw run."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.manifests import AsOfDeadline
from fplt.settings import get_settings
from fplt.store.runs import RunWriter

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_TS = datetime(2026, 11, 28, 18, 0, tzinfo=UTC)
_SEASON = "2025-26"
_MODEL = "deadbeef"
_APP = Path(__file__).resolve().parents[2] / "src" / "fplt" / "ui" / "app.py"
_RUN = "r_20261128T180000Z_postgw"


def _scorecard() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_id": [_MODEL],
            "season": [_SEASON],
            "gw": [12],
            "gw_kind": ["normal"],
            "plan_ev": [55.0],
            "realised_points": [61],
            "realised_minus_plan_ev": [6.0],
            "pmf_percentile": [0.64],
            "fpl_average": [None],
            "fpl_highest": [None],
            "rank_change": [None],
            "season_points_to_date": [61],
            "points_at_rank": [None],
            "percentile": [None],
            "realised_ex_chips": [61],
            "chip": [None],
        }
    )


def _attribution() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_id": [_MODEL],
            "season": [_SEASON],
            "gw": [12],
            "gw_kind": ["normal"],
            "plan_ev": [55.0],
            "realised_points": [61],
            "luck_vs_model": [6.0],
            "captaincy_regret": [2],
            "xi_bench_regret": [0],
            "transfer_regret": [0],
            "chip_timing_regret": [None],
            "hindsight_best": [70],
            "regret": [9],
            "hit": [0.0],
            "hit_payback_to_date": [0.0],
            "chip_value": [None],
            "intervention_delta": [None],
        }
    )


def _std() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_id": [_MODEL],
            "season": [_SEASON],
            "through_gw": [12],
            "gameweeks": [1],
            "season_points": [61],
            "plan_ev_sum": [55.0],
            "regret_sum": [9],
            "captaincy_regret_sum": [2],
            "hits_sum": [0.0],
            "hit_payback_to_date": [0.0],
            "points_at_rank": [None],
            "percentile": [None],
        }
    )


def _misses() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_id": [_MODEL],
            "season": [_SEASON],
            "gw": [12],
            "player_key": [100],
            "fixture_id": [1],
            "xpts": [8.0],
            "actual_pts": [2],
            "abs_error": [6.0],
            "signed_error": [-6.0],
            "reason": ["benched"],
            "predictable_at_T90": [True],
            "rank": [1],
        }
    )


def _forecast_scores() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_id": [_MODEL, _MODEL],
            "season": [_SEASON, _SEASON],
            "gw": [12, 12],
            "player_key": [100, 200],
            "fixture_id": [1, 2],
            "position": ["MID", "FWD"],
            "price_band": ["mid", "mid"],
            "minutes_bucket": ["60+", "60+"],
            "y": [2, 10],
            "mu_hat": [8.0, 6.0],
            "rps": [0.4, 0.2],
            "log_score": [2.0, 1.0],
            "rps_clim": [0.5, 0.5],
            "pit_u": [0.2, 0.8],
            "p_played": [0.9, 0.85],
            "p_ge60": [0.7, 0.6],
            "p_goal": [0.2, 0.3],
            "p_assist": [0.15, 0.2],
            "p_cs": [0.1, 0.05],
            "p_haul": [0.1, 0.2],
            "p_blank": [0.3, 0.2],
            "y_played": [1, 1],
            "y_ge60": [0, 1],
            "y_goal": [0, 1],
            "y_assist": [0, 0],
            "y_cs": [0, 0],
            "y_haul": [0, 1],
            "y_blank": [1, 0],
            "selected_lines": ["primary", ""],
            "schema_version": [SCHEMA_VERSION, SCHEMA_VERSION],
        }
    )


def _forecast_agg() -> pl.DataFrame:
    rows = [
        {
            "model_id": _MODEL,
            "season": _SEASON,
            "gw": None,
            "slice": "all",
            "metric": metric,
            "value": value,
            "ci_lo": 0.1,
            "ci_hi": 0.9,
            "n": 2,
            "schema_version": SCHEMA_VERSION,
        }
        for metric, value in (
            ("mean_rps", 0.3),
            ("mean_log_score", 1.5),
            ("rps_skill_clim", 0.1),
            ("pit_coverage_50", 0.5),
            ("pit_coverage_80", 0.8),
            ("pit_coverage_90", 0.9),
        )
    ]
    rows.extend(
        [
            {
                "model_id": _MODEL,
                "season": _SEASON,
                "gw": None,
                "slice": f"bin={lo:.1f}-{hi:.1f}",
                "metric": "pit_hist",
                "value": 0.1,
                "ci_lo": None,
                "ci_hi": None,
                "n": 2,
                "schema_version": SCHEMA_VERSION,
            }
            for lo, hi in ((0.0, 0.1), (0.1, 0.2))
        ]
    )
    rows.append(
        {
            "model_id": _MODEL,
            "season": _SEASON,
            "gw": None,
            "slice": "line=primary",
            "metric": "selection_bias",
            "value": 0.2,
            "ci_lo": 0.0,
            "ci_hi": 0.4,
            "n": 1,
            "schema_version": SCHEMA_VERSION,
        }
    )
    return pl.DataFrame(rows)


def _seed(data_dir: Path) -> None:
    deadline = AsOfDeadline(season=_SEASON, gw=12, deadline_utc=_TS)
    with RunWriter(
        data_dir,
        kind="official",
        mode="live",
        run_type="eval",
        spec_hash="a" * 64,
        spec_label="demo",
        model_id=_MODEL,
        as_of=[deadline],
        horizon=1,
        purpose="post_gw",
        tags={"gw": f"{_SEASON}-12", "purpose": "post_gw"},
        code_sha="abc1234",
        lockfile_hash="b" * 64,
        host="test",
        started_at=_TS,
        run_id=_RUN,
        line="primary",
        auto_build_id=False,
    ) as writer:
        writer.write_parquet("scorecard.parquet", _scorecard())
        writer.write_parquet("attribution.parquet", _attribution())
        writer.write_parquet("season_to_date.parquet", _std())
        writer.write_parquet("biggest_misses.parquet", _misses())
        writer.write_parquet("forecast_scores.parquet", _forecast_scores())
        writer.write_parquet("forecast_score_agg.parquet", _forecast_agg())


def test_last_week_app_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(tmp_path)
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:

        def _last_week_only() -> None:
            from fplt.ui.pages.last_week import render

            render()

        app = AppTest.from_function(_last_week_only, default_timeout=15)
        app.run()
        assert not app.exception, app.exception
        titles = [str(item.value).lower() for item in app.title]
        assert any("last week" in value for value in titles)
        headers = [str(item.value).lower() for item in app.header]
        assert any("season to date" in value for value in headers)
        assert any("this gameweek" in value for value in headers)
        sub = [str(item.value).lower() for item in app.subheader]
        assert any("lines" in value for value in sub)
        assert any("forecast skill" in value for value in sub)
        assert any("pit" in value for value in sub)
        assert any("reliability" in value for value in sub)
        assert any("selection bias" in value for value in sub)
        assert any("scorecard" in value for value in sub)
        assert any("attribution" in value for value in sub)
        assert any("biggest misses" in value for value in sub)
        assert len(app.dataframe) >= 4
    finally:
        get_settings.cache_clear()


def test_last_week_empty_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:

        def _last_week_only() -> None:
            from fplt.ui.pages.last_week import render

            render()

        app = AppTest.from_function(_last_week_only, default_timeout=10)
        app.run()
        assert not app.exception
        infos = [str(item.value).lower() for item in app.info]
        assert any("post-gw" in value or "post-gw analysis" in value for value in infos)
    finally:
        get_settings.cache_clear()


def test_last_week_empty_tables_show_captions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deadline = AsOfDeadline(season=_SEASON, gw=12, deadline_utc=_TS)
    with RunWriter(
        tmp_path,
        kind="official",
        mode="live",
        run_type="eval",
        spec_hash="a" * 64,
        spec_label="demo",
        model_id=_MODEL,
        as_of=[deadline],
        horizon=1,
        purpose="post_gw",
        code_sha="abc1234",
        lockfile_hash="b" * 64,
        host="test",
        started_at=_TS,
        run_id="r_20261128T190000Z_empty",
        line="primary",
        auto_build_id=False,
    ) as writer:
        # Manifest only — load_table returns empty for missing parquets.
        _ = writer

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:

        def _last_week_only() -> None:
            from fplt.ui.pages.last_week import render

            render()

        app = AppTest.from_function(_last_week_only, default_timeout=15)
        app.run()
        assert not app.exception, app.exception
        captions = [str(item.value).lower() for item in app.caption]
        joined = " ".join(captions)
        assert "no scorecard" in joined or "no forecast" in joined or "no" in joined
    finally:
        get_settings.cache_clear()


def test_last_week_selection_none_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path)
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        import fplt.ui.pages.last_week as page

        monkeypatch.setattr(page, "selection_from_catalog", lambda *a, **k: None)

        def _last_week_only() -> None:
            from fplt.ui.pages.last_week import render

            render()

        app = AppTest.from_function(_last_week_only, default_timeout=10)
        app.run()
        assert not app.exception, app.exception
        warnings = [str(item.value).lower() for item in app.warning]
        assert any("could not resolve" in value for value in warnings)
    finally:
        get_settings.cache_clear()


def test_prior_forecast_scores_empty_when_no_files(tmp_path: Path) -> None:
    from fplt.ui.data import DashboardStore
    from fplt.ui.pages.last_week import _prior_forecast_scores

    store = DashboardStore(tmp_path)
    assert _prior_forecast_scores(store, season=_SEASON, through_gw=1).is_empty()
