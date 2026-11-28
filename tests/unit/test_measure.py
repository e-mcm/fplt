"""Toolkit 4 / experiments step 3: Measure interface + scorer adapters."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.backtest.forecast_scores import aggregate_forecast_scores
from fplt.backtest.measure import (
    ADAPTERS,
    ComponentEvalMeasure,
    DecisionsBoardMeasure,
    ForecastScoresMeasure,
    MinutesMeasurementMeasure,
    OtherPointsMeasurementMeasure,
    SimulatedSeasonMeasure,
    empty_measure_metrics,
    filter_slices,
    get_measure,
    validate_measure_metrics,
)
from fplt.backtest.minutes_measurement import summarise_minutes_detail
from fplt.backtest.other_points_measurement import (
    captain_regret_by_gw,
    summarise_other_points_detail,
)
from fplt.backtest.simulated_season import mean_policy_totals
from fplt.optimise.realised import summarise_decisions
from fplt.schemas.component_eval import COMPONENT_EVAL_COLUMNS
from fplt.schemas.forecast_scores import (
    FORECAST_SCORE_AGG_COLUMNS,
    SCHEMA_VERSION,
)
from fplt.schemas.measure import MEASURE_METRICS_COLUMNS, MeasureMetricsSchema
from fplt.schemas.minutes_measurement import MINUTES_MEASUREMENT_SUMMARY_COLUMNS
from fplt.schemas.other_points_measurement import (
    OTHER_POINTS_MEASUREMENT_SUMMARY_COLUMNS,
)

_SEASON = "2024-25"
_EMPTY = pl.DataFrame()


def test_empty_measure_metrics_validates() -> None:
    frame = empty_measure_metrics()
    assert frame.is_empty()
    assert list(frame.columns) == list(MEASURE_METRICS_COLUMNS)
    MeasureMetricsSchema.validate(frame)


def test_get_measure_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown Measure"):
        get_measure("not_a_measure")
    assert set(ADAPTERS) == {
        "component_eval",
        "minutes_measurement",
        "other_points_measurement",
        "forecast_scores",
        "decisions_board",
        "simulated_season",
    }


def test_filter_slices_none_and_empty() -> None:
    rows = validate_measure_metrics(
        pl.DataFrame(
            {
                "arm": ["a", "a"],
                "window": ["primary", "primary"],
                "metric": ["m", "m"],
                "value": [1.0, 2.0],
                "se": [None, None],
                "n": [1, 1],
                "stage": ["prediction", "prediction"],
                "slice": ["all", "position=MID"],
                "schema_version": [SCHEMA_VERSION, SCHEMA_VERSION],
            }
        )
    )
    assert filter_slices(rows, None).height == 2
    assert filter_slices(rows, []).is_empty()
    kept = filter_slices(rows, ["all"])
    assert kept.height == 1
    assert kept["value"][0] == pytest.approx(1.0)


def test_component_eval_adapter_matches_summary_and_slices() -> None:
    summary = pl.DataFrame(
        {
            "component": ["minutes", "minutes", "score"],
            "metric": ["brier_played", "brier_60", "rps"],
            "model_name": ["emp_v2", "emp_v2", "dc"],
            "slice": ["overall", "source=fpl_api", "overall"],
            "value": [0.12, 0.34, 0.56],
            "n": [10, 8, 12],
        },
        schema=COMPONENT_EVAL_COLUMNS,
    )
    measure = ComponentEvalMeasure()
    out = measure.score("components", summary, _EMPTY)
    assert out.height == 3
    by_key = {(r["arm"], r["metric"], r["slice"]): r for r in out.iter_rows(named=True)}
    for row in summary.iter_rows(named=True):
        got = by_key[(row["model_name"], row["metric"], row["slice"])]
        assert got["value"] == pytest.approx(float(row["value"]))
        assert got["n"] == int(row["n"])
        assert got["stage"] == "components"
    sliced = measure.score(
        "components", {"summary": summary}, _EMPTY, slices=["overall"]
    )
    assert sliced.height == 2
    assert set(sliced["slice"].to_list()) == {"overall"}
    empty = measure.score(
        "components", pl.DataFrame(schema=COMPONENT_EVAL_COLUMNS), _EMPTY
    )
    assert empty.is_empty()


def test_minutes_adapter_matches_summarise_and_empty() -> None:
    detail = _minutes_detail()
    expected = summarise_minutes_detail(detail, model_name="emp_v2")
    measure = MinutesMeasurementMeasure(model_name="emp_v2")
    out = measure.score("components", detail, _EMPTY)
    _assert_summary_values_match(expected, out, arm_col="model_name")
    only_pool = measure.score("components", detail, _EMPTY, slices=["decision_pool"])
    assert set(only_pool["slice"].to_list()) == {"decision_pool"}
    assert (
        only_pool.height == expected.filter(pl.col("slice") == "decision_pool").height
    )
    assert measure.score("components", detail.clear(), _EMPTY).is_empty()


def test_other_points_adapter_matches_summarise_and_slices() -> None:
    detail = _other_detail()
    captain = captain_regret_by_gw(detail)
    expected = summarise_other_points_detail(detail, captain, model_name="dc_v4")
    measure = OtherPointsMeasurementMeasure(model_name="dc_v4")
    out = measure.score(
        "components",
        {"detail": detail, "captain_regret": captain},
        _EMPTY,
    )
    _assert_summary_values_match(expected, out, arm_col="model_name")
    sliced = measure.score(
        "components", detail, _EMPTY, slices=["all", "decision_pool"]
    )
    assert set(sliced["slice"].to_list()) <= {"all", "decision_pool"}
    assert measure.score(
        "components",
        pl.DataFrame(schema=detail.schema),
        _EMPTY,
    ).is_empty()


def test_forecast_scores_adapter_matches_agg() -> None:
    agg = pl.DataFrame(
        {
            "model_id": ["m1", "m1", "m1"],
            "season": [_SEASON, _SEASON, _SEASON],
            "gw": [None, None, None],
            "slice": ["all", "all", "position=MID"],
            "metric": ["mean_rps", "mean_log_score", "mean_rps"],
            "value": [0.2, 1.5, 0.25],
            "ci_lo": [0.1, 1.0, None],
            "ci_hi": [0.3, 2.0, None],
            "n": [100, 100, 40],
            "schema_version": [SCHEMA_VERSION] * 3,
        },
        schema=FORECAST_SCORE_AGG_COLUMNS,
    )
    # Round-trip through aggregate path: passing agg frame must preserve values.
    measure = ForecastScoresMeasure()
    out = measure.score("prediction", {"agg": agg}, _EMPTY)
    assert out.height == agg.height
    for row in agg.iter_rows(named=True):
        match = out.filter(
            (pl.col("arm") == row["model_id"])
            & (pl.col("metric") == row["metric"])
            & (pl.col("slice") == row["slice"])
        )
        assert match.height == 1
        assert match["value"][0] == pytest.approx(float(row["value"]))
        assert match["n"][0] == int(row["n"])
    # Direct frame detection via ci_lo.
    direct = measure.score("prediction", agg, _EMPTY, slices=["all"])
    assert set(direct["slice"].to_list()) == {"all"}
    assert measure.score(
        "prediction", pl.DataFrame(schema=FORECAST_SCORE_AGG_COLUMNS), _EMPTY
    ).is_empty()


def test_forecast_adapter_aggregate_path_matches_scorer() -> None:
    """Adapter aggregation equals calling ``aggregate_forecast_scores``."""
    scores = pl.DataFrame(
        {
            "model_id": ["m1", "m1"],
            "season": [_SEASON, _SEASON],
            "gw": [1, 2],
            "player_key": [10, 10],
            "fixture_id": [1, 2],
            "position": ["MID", "MID"],
            "price_band": ["5.0-6.9", "5.0-6.9"],
            "minutes_bucket": ["60+", "60+"],
            "y": [5, 7],
            "mu_hat": [4.0, 6.0],
            "rps": [0.1, 0.2],
            "log_score": [1.0, 1.2],
            "rps_clim": [0.3, 0.4],
            "pit_u": [0.4, 0.6],
            "p_played": [0.9, 0.9],
            "p_ge60": [0.8, 0.8],
            "p_goal": [0.2, 0.3],
            "p_assist": [0.1, 0.1],
            "p_cs": [0.0, 0.0],
            "p_haul": [0.05, 0.1],
            "p_blank": [0.2, 0.1],
            "y_played": [1, 1],
            "y_ge60": [1, 1],
            "y_goal": [0, 1],
            "y_assist": [0, 0],
            "y_cs": [0, 0],
            "y_haul": [0, 0],
            "y_blank": [0, 0],
            "selected_lines": ["primary", ""],
            "schema_version": [SCHEMA_VERSION, SCHEMA_VERSION],
        }
    )
    expected = aggregate_forecast_scores(scores, lines=("primary",))
    measure = ForecastScoresMeasure()
    # Direct scores frame (no ci_lo) aggregates the same as {"scores": …}.
    out = measure.score("prediction", scores, _EMPTY)
    mapped = measure.score("prediction", {"scores": scores}, _EMPTY)
    assert out.sort("metric", "slice").equals(mapped.sort("metric", "slice"))
    # Compare overlapping metric×slice keys (adapter drops ci columns).
    for row in expected.iter_rows(named=True):
        match = out.filter(
            (pl.col("arm") == row["model_id"])
            & (pl.col("metric") == row["metric"])
            & (pl.col("slice") == row["slice"])
        )
        assert match.height == 1
        assert match["value"][0] == pytest.approx(float(row["value"]))
        assert match["n"][0] == int(row["n"])


def test_decisions_board_adapter_matches_summarise() -> None:
    frame = pl.concat(
        [
            _decision_rows("a", gw=1, net_xi=10, hit=0),
            _decision_rows("a", gw=2, net_xi=12, hit=4),
            _decision_rows("b", gw=1, net_xi=20, hit=0),
        ]
    )
    expected = summarise_decisions(frame)
    measure = DecisionsBoardMeasure()
    out = measure.score("decision", frame, _EMPTY, slices=["all"])
    for row in expected.iter_rows(named=True):
        match = out.filter(
            (pl.col("arm") == row["model_id"])
            & (pl.col("metric") == "season_points")
            & (pl.col("slice") == "all")
        )
        assert match.height == 1
        assert match["value"][0] == pytest.approx(float(row["total_points"]))
        assert match["n"][0] == int(row["gameweeks"])
    season_slice = measure.score(
        "decision", frame, _EMPTY, slices=[f"season={_SEASON}"]
    )
    assert set(season_slice["slice"].to_list()) == {f"season={_SEASON}"}
    assert measure.score("decision", frame.clear(), _EMPTY).is_empty()


def test_simulated_season_adapter_matches_mean_totals() -> None:
    paths = pl.DataFrame(
        {
            "path_id": [0, 0, 1, 1],
            "policy_id": ["ctrl", "treat", "ctrl", "treat"],
            "season": [_SEASON] * 4,
            "total_points": [100, 110, 90, 130],
            "gameweeks": [2, 2, 2, 2],
            "attainable_points": [120, 120, 140, 140],
            "share_of_attainable": [100 / 120, 110 / 120, 90 / 140, 130 / 140],
        }
    )
    means = mean_policy_totals(paths)
    measure = SimulatedSeasonMeasure()
    out = measure.score("season", paths, _EMPTY)
    for policy_id, mean_pts in means.items():
        match = out.filter(
            (pl.col("arm") == policy_id)
            & (pl.col("metric") == "season_points")
            & (pl.col("slice") == "all")
        )
        assert match.height == 1
        assert match["value"][0] == pytest.approx(mean_pts)
        assert match["n"][0] == 2
    assert "share_of_attainable" in out["metric"].to_list()
    assert measure.score("season", paths.clear(), _EMPTY).is_empty()
    sliced_away = measure.score("season", paths, _EMPTY, slices=["position=MID"])
    assert sliced_away.is_empty()


def test_adapters_are_measure_protocol() -> None:
    from fplt.backtest.measure import Measure

    for adapter in ADAPTERS.values():
        assert isinstance(adapter, Measure)
        assert adapter.name in ADAPTERS


def _assert_summary_values_match(
    summary: pl.DataFrame,
    metrics: pl.DataFrame,
    *,
    arm_col: str,
) -> None:
    """Every scorer summary row appears with the same value/n on the Measure table."""
    assert metrics.height == summary.height
    for row in summary.iter_rows(named=True):
        arm = str(row[arm_col]) if arm_col in row else str(row["model_name"])
        match = metrics.filter(
            (pl.col("arm") == arm)
            & (pl.col("metric") == row["metric"])
            & (pl.col("slice") == row["slice"])
        )
        assert match.height == 1, row
        if row["value"] is None:
            assert match["value"][0] is None
        else:
            assert match["value"][0] == pytest.approx(float(row["value"]))
        assert match["n"][0] == int(row["n"])


def _minutes_detail() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [_SEASON, _SEASON, _SEASON],
            "gw": [3, 3, 3],
            "player_key": [10, 11, 12],
            "fixture_id": [3, 3, 3],
            "position": ["MID", "FWD", "DEF"],
            "source": ["fpl_api", "fpl_api", "fpl_api"],
            "regime": ["rotation", "early_sub", "availability"],
            "fpl_flag": ["a", "a", "d"],
            "returning_from_absence": [False, False, True],
            "p_played": [0.9, 0.8, 0.2],
            "p_60": [0.7, 0.6, 0.1],
            "p_start": [0.7, 0.6, 0.1],
            "played": [0.0, 1.0, 0.0],
            "is_60": [0.0, 1.0, 0.0],
            "started": [0.0, 1.0, 0.0],
            "xpts_weight": [5.0, 8.0, 4.0],
            "status": ["a", "a", "d"],
            "in_decision_pool": [True, True, False],
        }
    )


def _other_detail() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [_SEASON, _SEASON, _SEASON],
            "gw": [3, 3, 3],
            "player_key": [10, 11, 12],
            "fixture_id": [3, 3, 3],
            "position": ["MID", "FWD", "DEF"],
            "minutes": [90, 90, 90],
            "minutes_bucket": ["60+", "60+", "60+"],
            "xpts": [8.0, 6.0, 4.0],
            "p_haul": [0.2, 0.1, 0.05],
            "actual_pts": [2, 12, 3],
            "hauled": [0.0, 1.0, 0.0],
            "actual_other": [0.0, 3.0, 0.0],
            "predicted_other": [0.5, 0.5, 0.5],
            "other_residual": [-0.5, 2.5, -0.5],
            "actual_bonus": [0.0, 3.0, 0.0],
            "actual_saves": [0.0, 0.0, 0.0],
            "actual_defcon": [0.0, 0.0, 0.0],
            "team_goals": [1, 2, 0],
            "team_goals_band": ["1", "2+", "0"],
            "clean_sheet": [0, 0, 1],
            "player_scored": [0, 1, 0],
            "in_decision_pool": [True, True, False],
        }
    )


def _decision_rows(model_id: str, *, gw: int, net_xi: int, hit: float) -> pl.DataFrame:
    """Build a valid 15 with captain scoring so net matches net_xi - hit."""
    pts = [0] * 11
    mins = [0] * 11
    pts[0] = net_xi // 2
    mins[0] = 90
    remainder = net_xi - 2 * (net_xi // 2)
    if remainder:
        pts[1] = remainder
        mins[1] = 90
    positions = [
        "GK",
        "DEF",
        "DEF",
        "DEF",
        "MID",
        "MID",
        "MID",
        "MID",
        "FWD",
        "FWD",
        "FWD",
    ]
    rows: list[dict[str, object]] = []
    for index, (p, m, pos) in enumerate(zip(pts, mins, positions, strict=True)):
        rows.append(
            {
                "model_id": model_id,
                "season": _SEASON,
                "gw": gw,
                "player_key": index + 1,
                "position": pos,
                "in_xi": True,
                "captain": index == 0,
                "vice": index == 1,
                "bought": False,
                "sold": False,
                "xpts": 0.0,
                "actual_pts": p,
                "actual_minutes": m,
                "hit": hit,
                "free_transfers": 1,
                "team_key": (index % 7) + 1,
                "price": 50,
            }
        )
    rows.extend(
        [
            {
                "model_id": model_id,
                "season": _SEASON,
                "gw": gw,
                "player_key": key,
                "position": pos,
                "in_xi": False,
                "captain": False,
                "vice": False,
                "bought": False,
                "sold": False,
                "xpts": 0.0,
                "actual_pts": 0,
                "actual_minutes": 0,
                "hit": hit,
                "free_transfers": 1,
                "team_key": 8 + i,
                "price": 40,
            }
            for i, (key, pos) in enumerate(
                [(12, "GK"), (13, "DEF"), (14, "DEF"), (15, "MID")]
            )
        ]
    )
    return pl.DataFrame(rows)


def test_measure_mapping_errors_and_bad_summary() -> None:
    from fplt.backtest import measure as measure_mod
    from fplt.backtest.forecast_scores import empty_forecast_scores

    measure = ComponentEvalMeasure()
    with pytest.raises(KeyError, match="summary"):
        measure.score(
            "components",
            {"wrong": pl.DataFrame(schema=COMPONENT_EVAL_COLUMNS)},
            _EMPTY,
        )
    with pytest.raises(ValueError, match="missing columns"):
        measure.score(
            "components",
            pl.DataFrame({"model_name": ["a"], "metric": ["m"]}),
            _EMPTY,
        )
    forecast = ForecastScoresMeasure()
    with pytest.raises(KeyError, match="scores"):
        forecast.score("prediction", {"nope": _EMPTY}, _EMPTY)
    with pytest.raises(KeyError, match="detail"):
        MinutesMeasurementMeasure().score(
            "components",
            {
                "a": _minutes_detail(),
                "b": _minutes_detail(),
            },
            _EMPTY,
        )
    # Single-key mapping with key=None resolves; multi-key needs an explicit key.
    only = pl.DataFrame(schema=COMPONENT_EVAL_COLUMNS)
    assert measure_mod._as_frame({"summary": only}, None).is_empty()
    with pytest.raises(TypeError, match="explicit key"):
        measure_mod._as_frame({"a": only, "b": only}, None)
    # Empty scores → empty agg → empty Measure table.
    assert forecast.score(
        "prediction", {"scores": empty_forecast_scores()}, _EMPTY
    ).is_empty()
    # Direct scores frame (no ci_lo) goes through aggregate_forecast_scores.
    scores = empty_forecast_scores()
    assert forecast.score("prediction", scores, _EMPTY).is_empty()


def test_other_points_mapping_without_captain_and_null_values() -> None:
    detail = _other_detail()
    measure = OtherPointsMeasurementMeasure(model_name="dc_v4")
    out = measure.score("components", {"detail": detail}, _EMPTY)
    expected = summarise_other_points_detail(
        detail, captain_regret_by_gw(detail), model_name="dc_v4"
    )
    _assert_summary_values_match(expected, out, arm_col="model_name")
    # Null scorer values are preserved on the Measure table.
    summary = pl.DataFrame(
        {
            "component": ["minutes"],
            "metric": ["brier_played"],
            "model_name": ["emp"],
            "slice": ["overall"],
            "value": [None],
            "n": [0],
        },
        schema=COMPONENT_EVAL_COLUMNS,
    )
    null_out = ComponentEvalMeasure().score("components", summary, _EMPTY)
    assert null_out.height == 1
    assert null_out["value"][0] is None


def test_decisions_empty_summary_and_zero_gameweeks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fplt.backtest import measure as measure_mod

    frame = _decision_rows("a", gw=1, net_xi=10, hit=0)
    measure = DecisionsBoardMeasure()

    monkeypatch.setattr(
        measure_mod,
        "summarise_decisions",
        lambda _frame: pl.DataFrame(
            schema={
                "model_id": pl.Utf8(),
                "season": pl.Utf8(),
                "total_points": pl.Int64(),
                "gameweeks": pl.Int64(),
                "per_gw": pl.List(pl.Int64()),
            }
        ),
    )
    assert measure.score("decision", frame, _EMPTY).is_empty()

    monkeypatch.setattr(
        measure_mod,
        "summarise_decisions",
        lambda _frame: pl.DataFrame(
            {
                "model_id": ["a"],
                "season": [_SEASON],
                "total_points": [0],
                "gameweeks": [0],
                "per_gw": [[]],
            }
        ),
    )
    out = measure.score("decision", {"decisions": frame}, _EMPTY, slices=["all"])
    assert out.height == 1
    assert out["metric"][0] == "season_points"
    assert "mean_points_per_gw" not in out["metric"].to_list()


def test_simulated_season_without_share() -> None:
    paths = pl.DataFrame(
        {
            "path_id": [0, 1],
            "policy_id": ["ctrl", "ctrl"],
            "season": [_SEASON, _SEASON],
            "total_points": [100, 90],
            "gameweeks": [2, 2],
            "attainable_points": [None, None],
            "share_of_attainable": [None, None],
        }
    )
    out = SimulatedSeasonMeasure().score("season", {"paths": paths}, _EMPTY)
    assert out.height == 1
    assert out["metric"][0] == "season_points"
    assert out["value"][0] == pytest.approx(95.0)


def test_summary_helpers_empty_schemas() -> None:
    """Public summarise wrappers still validate empty detail (no rewrite)."""
    empty_minutes = summarise_minutes_detail(
        pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "gw": pl.Int64(),
                "player_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "position": pl.Utf8(),
                "source": pl.Utf8(),
                "regime": pl.Utf8(),
                "fpl_flag": pl.Utf8(),
                "returning_from_absence": pl.Boolean(),
                "p_played": pl.Float64(),
                "p_60": pl.Float64(),
                "p_start": pl.Float64(),
                "played": pl.Float64(),
                "is_60": pl.Float64(),
                "started": pl.Float64(),
                "xpts_weight": pl.Float64(),
                "status": pl.Utf8(),
                "in_decision_pool": pl.Boolean(),
            }
        ),
        model_name="x",
    )
    assert empty_minutes.is_empty()
    assert list(empty_minutes.columns) == list(MINUTES_MEASUREMENT_SUMMARY_COLUMNS)
    empty_other = summarise_other_points_detail(
        pl.DataFrame(schema=_other_detail().schema),
        pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "gw": pl.Int64(),
                "chosen_player_key": pl.Int64(),
                "oracle_player_key": pl.Int64(),
                "chosen_xpts": pl.Float64(),
                "chosen_actual": pl.Float64(),
                "oracle_actual": pl.Float64(),
                "regret": pl.Float64(),
                "n_pool": pl.Int64(),
            }
        ),
        model_name="x",
    )
    assert empty_other.is_empty()
    assert list(empty_other.columns) == list(OTHER_POINTS_MEASUREMENT_SUMMARY_COLUMNS)
