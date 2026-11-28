"""Unit tests for ``fplt.analysis.last_week`` on synthetic post_gw frames."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from fplt.analysis.last_week import (
    benchmarks_table,
    default_post_gw_selection,
    header_fields,
    lines_season_to_date,
    ops_review_table,
    pit_histogram,
    post_gw_catalog,
    rank_context,
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


def _scorecard(
    *,
    gw: int,
    pts: int,
    ex: int | None,
    chip: str | None = None,
    fpl_average: float | None = None,
    fpl_highest: float | None = None,
    rank_change: float | None = None,
) -> dict:
    return {
        "model_id": _MODEL,
        "season": _SEASON,
        "gw": gw,
        "gw_kind": "normal",
        "plan_ev": float(pts) - 1.0,
        "realised_points": pts,
        "realised_minus_plan_ev": 1.0,
        "pmf_percentile": 0.5,
        "fpl_average": fpl_average,
        "fpl_highest": fpl_highest,
        "rank_change": rank_change,
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
    assert detail.rank_context.height == 3
    assert not detail.ops_review.is_empty()
    sel = default_post_gw_selection(post_gw_catalog([_manifest(run_id="r1", gw=1)]))
    assert sel is not None
    fields = header_fields(sel, finished_at="2026-11-28T12:00:00Z")
    assert fields["gw"] == 1
    assert fields["run_id"] == "r1"


def test_rank_context_populated_and_missing() -> None:
    filled = pl.DataFrame(
        [
            _scorecard(
                gw=12,
                pts=61,
                ex=61,
                fpl_average=45.0,
                fpl_highest=120.0,
                rank_change=1500.0,
            )
        ]
    )
    ctx = rank_context(filled)
    assert ctx.height == 3
    by_field = {str(r["field"]): r for r in ctx.iter_rows(named=True)}
    assert float(by_field["fpl_average"]["value"]) == pytest.approx(45.0)
    assert float(by_field["rank_change"]["value"]) == pytest.approx(1500.0)
    assert by_field["rank_change"]["note"] == "positive = climbed"
    empty = rank_context(pl.DataFrame())
    assert empty.height == 3
    assert all(r["note"] == "no scorecard" for r in empty.iter_rows(named=True))
    missing = rank_context(pl.DataFrame([_scorecard(gw=1, pts=10, ex=10)]))
    assert all(r["value"] is None for r in missing.iter_rows(named=True))


def test_benchmarks_same_fixtures_and_openfpl_guard() -> None:
    from datetime import UTC, datetime

    stamp = datetime(2025, 11, 1, tzinfo=UTC)
    primary = pl.DataFrame(
        {
            "player_key": [1, 2],
            "fixture_id": [10, 11],
            "xpts": [6.0, 4.0],
        }
    )
    outcomes = pl.DataFrame(
        {
            "player_key": [1, 2],
            "fixture_id": [10, 11],
            "pts": [8, 2],
        }
    )
    benches = pl.DataFrame(
        {
            "source": ["ep_next", "ep_next", "openfpl", "openfpl"],
            "as_of_season": [_SEASON] * 4,
            "as_of_gw": [12] * 4,
            "target_season": [_SEASON] * 4,
            "target_gw": [12] * 4,
            "player_key": [1, 2, 1, 2],
            "fixture_id": [10, 11, 10, 11],
            "xpts": [5.0, 5.0, 7.0, 3.0],
            "available_at": [stamp] * 4,
            "quality_flag": [None, None, None, None],
        }
    )
    table = benchmarks_table(primary, benches, outcomes, season=_SEASON, gw=12)
    assert set(table["arm"].to_list()) == {"primary", "ep_next", "openfpl"}
    primary_row = table.filter(pl.col("arm") == "primary").row(0, named=True)
    assert int(primary_row["n"]) == 2
    assert primary_row["rmse"] is not None
    early = benchmarks_table(primary, benches, outcomes, season="2022-23", gw=12)
    openfpl = early.filter(pl.col("arm") == "openfpl").row(0, named=True)
    assert int(openfpl["n"]) == 0
    assert "season guard" in str(openfpl["note"])
    assert (
        benchmarks_table(
            pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), season=_SEASON, gw=1
        ).filter(pl.col("arm") == "primary")["note"][0]
        == "no primary predictions"
    )


def test_ops_review_degrades_without_inputs() -> None:
    empty = ops_review_table(
        applied=None, freshness=[], timings=[], alarms=[], last_error=None
    )
    sections = {str(r["section"]) for r in empty.iter_rows(named=True)}
    assert {"applied", "freshness", "timing", "alarm", "failure"} <= sections
    applied = ops_review_table(
        applied={
            "checkpoint": "T-90m",
            "line": "v4_h3",
            "plan_run_id": "r_plan",
            "submitted_at": "2026-11-28T12:00:00Z",
        },
        freshness=[{"key": "fpl_api", "value": "2026-11-28T10:00:00Z"}],
        timings=[{"key": "T-90m", "value": "12.5s", "detail": "ops"}],
        alarms=[{"key": "retry", "value": "alarm", "detail": "2026-11-28T11:00:00Z"}],
        last_error="boom",
    )
    assert applied.filter(pl.col("section") == "applied").height == 4
    assert (
        applied.filter(
            (pl.col("section") == "failure") & (pl.col("key") == "last_error")
        )["value"][0]
        == "boom"
    )


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


def test_benchmarks_edge_paths_and_helpers() -> None:
    from datetime import UTC, datetime

    from fplt.analysis import last_week as mod

    stamp = datetime(2025, 11, 1, tzinfo=UTC)
    assert benchmarks_table(
        pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), season="", gw=1
    ).is_empty()
    # Source present in archive but empty after filter → no rows for source.
    benches = pl.DataFrame(
        {
            "source": ["ep_next"],
            "as_of_season": [_SEASON],
            "as_of_gw": [12],
            "target_season": [_SEASON],
            "target_gw": [12],
            "player_key": [99],
            "fixture_id": [99],
            "xpts": [1.0],
            "available_at": [stamp],
            "quality_flag": [None],
        }
    )
    # Primary keys exclude the only bench row → scored overlap empty after join.
    primary = pl.DataFrame({"player_key": [1], "fixture_id": [10], "xpts": [5.0]})
    outcomes = pl.DataFrame({"player_key": [1], "fixture_id": [10], "pts": [4]})
    table = benchmarks_table(primary, benches, outcomes, season=_SEASON, gw=12)
    ep = table.filter(pl.col("arm") == "ep_next").row(0, named=True)
    assert int(ep["n"]) == 0
    assert ep["note"] in {"no scored overlap", "no rows for source"}

    # actual_pts alias + missing pts columns + no overlap join.
    aliased = mod._outcome_pts(
        pl.DataFrame({"player_key": [1], "fixture_id": [1], "actual_pts": [3]})
    )
    assert "pts" in aliased.columns
    assert mod._outcome_pts(
        pl.DataFrame({"player_key": [1], "fixture_id": [1], "noise": [1]})
    ).is_empty()
    arm = mod._arm_on_fixtures(
        pl.DataFrame({"player_key": [1], "fixture_id": [1], "xpts": [2.0]}),
        pl.DataFrame({"player_key": [2], "fixture_id": [2], "pts": [1]}),
        arm="ep_next",
    )
    assert arm["note"] == "no scored overlap"
    assert mod._arm_on_fixtures(
        pl.DataFrame({"player_key": [1]}),  # missing columns
        pl.DataFrame({"player_key": [1], "fixture_id": [1], "pts": [1]}),
        arm="primary",
    )["note"] in {"no predictions", "no primary predictions", "no outcomes"}
    assert (
        mod._arm_on_fixtures(
            pl.DataFrame({"player_key": [1], "fixture_id": [1], "xpts": [1.0]}),
            pl.DataFrame(),
            arm="ep_next",
        )["note"]
        == "no outcomes"
    )
    assert mod._season_before("not-a-season", "2024-25") is True
    assert mod._as_str(None) is None
    assert mod._as_str(3) == "3"
    # week_detail without season/gw and empty scorecard uses empty benchmarks.
    detail = week_detail(
        scorecard=pl.DataFrame(),
        attribution=pl.DataFrame(),
        biggest_misses=pl.DataFrame(),
    )
    assert detail.benchmarks.is_empty()
