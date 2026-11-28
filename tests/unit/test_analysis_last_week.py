"""Unit tests for ``fplt.analysis.last_week`` on synthetic post_gw frames."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from fplt.analysis.last_week import (
    default_post_gw_selection,
    header_fields,
    lines_season_to_date,
    pit_histogram,
    post_gw_catalog,
    reliability_diagram,
    season_panel,
    selection_bias_table,
    selection_from_catalog,
    skill_table,
    week_detail,
)
from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.manifests import AsOfDeadline, RunManifest

_TS = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)
_SEASON = "2025-26"
_MODEL = "deadbeef"


def _manifest(
    *,
    run_id: str,
    gw: int,
    purpose: str = "post_gw",
    line: str | None = "primary",
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        kind="official",
        mode="live",
        run_type="eval",
        purpose=purpose,
        spec_hash="a" * 64,
        spec_label="demo",
        model_id=_MODEL,
        as_of=[AsOfDeadline(season=_SEASON, gw=gw, deadline_utc=_TS)],
        horizon=1,
        code_sha="abc1234",
        lockfile_hash="b" * 64,
        host="test",
        started_at=_TS,
        finished_at=_TS,
        status="succeeded",
        line=line,
    )


def _scorecard(*, gw: int, pts: int, ex: int | None, chip: str | None = None) -> dict:
    return {
        "model_id": _MODEL,
        "season": _SEASON,
        "gw": gw,
        "gw_kind": "normal",
        "plan_ev": float(pts) - 1.0,
        "realised_points": pts,
        "realised_minus_plan_ev": 1.0,
        "pmf_percentile": 0.5,
        "fpl_average": None,
        "fpl_highest": None,
        "rank_change": None,
        "season_points_to_date": pts,
        "points_at_rank": None,
        "percentile": None,
        "realised_ex_chips": ex,
        "chip": chip,
    }


def _agg_row(
    *,
    metric: str,
    value: float,
    slice_name: str = "all",
    ci_lo: float | None = 0.1,
    ci_hi: float | None = 0.9,
    gw: int | None = None,
) -> dict:
    return {
        "model_id": _MODEL,
        "season": _SEASON,
        "gw": gw,
        "slice": slice_name,
        "metric": metric,
        "value": value,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n": 10,
        "schema_version": SCHEMA_VERSION,
    }


def test_post_gw_catalog_filters_and_orders() -> None:
    catalog = post_gw_catalog(
        [
            _manifest(run_id="r_old", gw=1),
            _manifest(run_id="r_new", gw=2),
            _manifest(run_id="r_pred", gw=2, purpose="predict"),
        ]
    )
    assert catalog["run_id"].to_list() == ["r_new", "r_old"]
    assert default_post_gw_selection(catalog) is not None
    assert default_post_gw_selection(catalog).gw == 2  # type: ignore[union-attr]
    assert selection_from_catalog(catalog, season=_SEASON, gw=1).run_id == "r_old"  # type: ignore[union-attr]
    assert selection_from_catalog(catalog, season=_SEASON, gw=99) is None
    assert post_gw_catalog([]).is_empty()
    assert default_post_gw_selection(pl.DataFrame()) is None


def test_lines_season_to_date_chips_and_ex_chips() -> None:
    cards = pl.DataFrame(
        [
            _scorecard(gw=1, pts=50, ex=50),
            _scorecard(gw=2, pts=60, ex=55, chip="bboost"),
        ]
    )
    rolled = lines_season_to_date(cards, season=_SEASON, through_gw=2)
    assert rolled.height == 1
    assert int(rolled["points_with_chips"][0]) == 110
    assert int(rolled["points_ex_chips"][0]) == 105
    assert int(rolled["gameweeks"][0]) == 2
    assert lines_season_to_date(pl.DataFrame(), season=_SEASON, through_gw=1).is_empty()
    assert lines_season_to_date(cards, season="other", through_gw=2).is_empty()


def test_lines_prefers_season_to_date_plan_ev() -> None:
    cards = pl.DataFrame([_scorecard(gw=1, pts=50, ex=50)])
    std = pl.DataFrame(
        {
            "model_id": [_MODEL],
            "season": [_SEASON],
            "through_gw": [1],
            "gameweeks": [1],
            "season_points": [50],
            "plan_ev_sum": [99.0],
            "regret_sum": [0],
            "captaincy_regret_sum": [0],
            "hits_sum": [0.0],
            "hit_payback_to_date": [0.0],
            "points_at_rank": [None],
            "percentile": [None],
        }
    )
    rolled = lines_season_to_date(
        cards, season=_SEASON, through_gw=1, season_to_date=std
    )
    assert float(rolled["plan_ev_sum"][0]) == 99.0


def test_skill_pit_selection_and_reliability() -> None:
    agg = pl.DataFrame(
        [
            _agg_row(metric="mean_rps", value=0.4),
            _agg_row(metric="mean_log_score", value=2.1),
            _agg_row(metric="rps_skill_clim", value=0.05),
            _agg_row(metric="pit_coverage_50", value=0.5),
            _agg_row(metric="pit_coverage_80", value=0.8),
            _agg_row(metric="pit_coverage_90", value=0.9),
            _agg_row(
                metric="pit_hist",
                value=0.1,
                slice_name="bin=0.0-0.1",
                ci_lo=None,
                ci_hi=None,
            ),
            _agg_row(
                metric="pit_hist",
                value=0.2,
                slice_name="bin=0.1-0.2",
                ci_lo=None,
                ci_hi=None,
            ),
            _agg_row(
                metric="selection_bias",
                value=0.3,
                slice_name="line=primary",
            ),
            _agg_row(
                metric="selection_bias_delta",
                value=0.1,
                slice_name="line=primary",
            ),
        ]
    )
    skill = skill_table(agg)
    assert skill["metric"].to_list()[:2] == ["mean_rps", "mean_log_score"]
    assert skill_table(pl.DataFrame()).is_empty()

    pit = pit_histogram(agg)
    assert pit.height == 2
    assert float(pit["bin_lo"][0]) == 0.0
    assert pit_histogram(pl.DataFrame()).is_empty()
    assert pit_histogram(
        pl.DataFrame([_agg_row(metric="mean_rps", value=0.1)])
    ).is_empty()

    bias = selection_bias_table(agg)
    assert bias.height == 2
    assert bias["line"].to_list() == ["primary", "primary"]
    assert selection_bias_table(pl.DataFrame()).is_empty()

    scores = pl.DataFrame(
        {
            "model_id": [_MODEL, _MODEL],
            "season": [_SEASON, _SEASON],
            "gw": [1, 1],
            "player_key": [1, 2],
            "fixture_id": [10, 11],
            "position": ["MID", "FWD"],
            "price_band": ["mid", "mid"],
            "minutes_bucket": ["60+", "60+"],
            "y": [5, 2],
            "mu_hat": [4.0, 3.0],
            "rps": [0.1, 0.2],
            "log_score": [1.0, 1.5],
            "rps_clim": [0.3, 0.3],
            "pit_u": [0.4, 0.6],
            "p_played": [0.9, 0.2],
            "p_ge60": [0.8, 0.1],
            "p_goal": [0.3, 0.05],
            "p_assist": [0.2, 0.05],
            "p_cs": [0.1, 0.1],
            "p_haul": [0.05, 0.02],
            "p_blank": [0.4, 0.7],
            "y_played": [1, 0],
            "y_ge60": [1, 0],
            "y_goal": [1, 0],
            "y_assist": [0, 0],
            "y_cs": [0, 0],
            "y_haul": [0, 0],
            "y_blank": [0, 1],
            "selected_lines": ["primary", ""],
            "schema_version": [SCHEMA_VERSION, SCHEMA_VERSION],
        }
    )
    rel = reliability_diagram(scores, n_bins=5)
    assert not rel.is_empty()
    assert set(rel["event"].to_list()) <= {
        "played",
        "ge60",
        "goal",
        "assist",
        "cs",
        "haul",
        "blank",
    }
    assert reliability_diagram(pl.DataFrame()).is_empty()
    assert reliability_diagram(scores, n_bins=0).is_empty()


def test_season_and_week_panels_and_header() -> None:
    cards = pl.DataFrame([_scorecard(gw=1, pts=50, ex=50)])
    std = pl.DataFrame(
        {
            "model_id": [_MODEL],
            "season": [_SEASON],
            "through_gw": [1],
            "gameweeks": [1],
            "season_points": [50],
            "plan_ev_sum": [49.0],
            "regret_sum": [0],
            "captaincy_regret_sum": [0],
            "hits_sum": [0.0],
            "hit_payback_to_date": [0.0],
            "points_at_rank": [None],
            "percentile": [None],
        }
    )
    agg = pl.DataFrame([_agg_row(metric="mean_rps", value=0.4)])
    panel = season_panel(
        scorecards=cards,
        season_to_date=std,
        forecast_agg=agg,
        forecast_scores=pl.DataFrame(),
        season=_SEASON,
        through_gw=1,
    )
    assert not panel.lines.is_empty()
    assert not panel.skill.is_empty()
    detail = week_detail(
        scorecard=cards,
        attribution=pl.DataFrame(),
        biggest_misses=pl.DataFrame(),
    )
    assert detail.scorecard.height == 1
    sel = default_post_gw_selection(post_gw_catalog([_manifest(run_id="r1", gw=1)]))
    assert sel is not None
    fields = header_fields(sel, finished_at="2026-11-28T12:00:00Z")
    assert fields["gw"] == 1
    assert fields["run_id"] == "r1"


def test_parse_bin_slice_fallback() -> None:
    from fplt.analysis import last_week as mod

    assert mod._parse_bin_slice("bin=not-a-bin") == (0.0, 1.0)
    assert mod._parse_bin_slice("bin=0.5-0.6") == (0.5, 0.6)


def test_skill_prefers_null_gw_std_rows() -> None:
    agg = pl.DataFrame(
        [
            _agg_row(metric="mean_rps", value=0.9, gw=1),
            _agg_row(metric="mean_rps", value=0.4, gw=None),
        ]
    )
    skill = skill_table(agg)
    assert skill.height == 1
    assert float(skill["value"][0]) == pytest.approx(0.4)


def test_skill_empty_when_slice_has_no_headline_metrics() -> None:
    agg = pl.DataFrame([_agg_row(metric="mse_mean", value=1.0)])
    assert skill_table(agg).is_empty()


def test_reliability_skips_missing_and_null_event_cols() -> None:
    bare = pl.DataFrame(
        {
            "model_id": [_MODEL],
            "season": [_SEASON],
            "gw": [1],
            "player_key": [1],
            "fixture_id": [1],
            "y": [1],
            "mu_hat": [1.0],
        }
    )
    assert reliability_diagram(bare).is_empty()
    nulls = pl.DataFrame(
        {
            "p_played": [None, None],
            "y_played": [None, None],
            "p_ge60": [None, None],
            "y_ge60": [None, None],
            "p_goal": [None, None],
            "y_goal": [None, None],
            "p_assist": [None, None],
            "y_assist": [None, None],
            "p_cs": [None, None],
            "y_cs": [None, None],
            "p_haul": [None, None],
            "y_haul": [None, None],
            "p_blank": [None, None],
            "y_blank": [None, None],
        }
    )
    assert reliability_diagram(nulls).is_empty()


def test_catalog_finished_at_naive_and_none() -> None:
    naive = RunManifest(
        run_id="r_naive",
        kind="official",
        mode="live",
        run_type="eval",
        purpose="post_gw",
        spec_hash="a" * 64,
        spec_label="demo",
        model_id=_MODEL,
        as_of=[AsOfDeadline(season=_SEASON, gw=1, deadline_utc=_TS)],
        horizon=1,
        code_sha="abc1234",
        lockfile_hash="b" * 64,
        host="test",
        started_at=_TS.replace(tzinfo=None),
        finished_at=_TS.replace(tzinfo=None),
        status="succeeded",
        line="primary",
    )
    catalog = post_gw_catalog([naive])
    assert catalog["finished_at"][0] is not None
    running_like = _manifest(run_id="r_done", gw=1)
    # finished_at required for succeeded; cover header without stamp.
    sel = default_post_gw_selection(post_gw_catalog([running_like]))
    assert sel is not None
    assert header_fields(sel)["finished_at"] == "—"
