"""Biggest misses: reason tags + predictable_at_T90 PIT (#394)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.automation.biggest_misses import (
    TOP_N,
    compute_biggest_misses,
    empty_biggest_misses,
    t90_cutoff,
    tag_miss_reason,
)
from fplt.automation.post_gw import (
    _biggest_misses,
    _official_predictions,
    _optional_table,
    format_post_gw_ntfy,
    format_post_gw_summary,
    run_post_gw,
)
from fplt.automation.production import OfficialLine, ProductionOps
from fplt.data.predicted_lineups import ROLE_OUT, ROLE_START
from fplt.data.store import DataStore
from fplt.models.pipeline_spec import live_pipeline_defaults
from fplt.models.spec import load_model_specs
from fplt.schemas.manifests import BuildManifest, BuildTableEntry
from fplt.store.official import official_checkpoint_dir
from fplt.store.runs import run_dir
from fplt.types import GWKey
from tests.unit.test_decision_scoring import _decision_gw

_TS = datetime(2024, 9, 1, 12, 0, tzinfo=UTC)
_DEADLINE = datetime(2024, 8, 16, 11, 0, tzinfo=UTC)
_T90 = t90_cutoff(_DEADLINE)
_SEASON = "2024-25"
_SPEC = "rolling_mean_v1"
_MODEL = load_model_specs()[_SPEC].model_id
_KEY = GWKey(_SEASON, 1)


def _ops() -> ProductionOps:
    pipeline = live_pipeline_defaults(model=_SPEC, horizon=3)
    return ProductionOps(
        models=[_SPEC],
        plan_horizon=3,
        pipeline=pipeline,
        lines=[
            OfficialLine(
                line="primary",
                spec=_SPEC,
                horizon=3,
                role="primary",
                squad_mode="real",
            )
        ],
    )


def _build() -> BuildManifest:
    return BuildManifest(
        build_id="b" * 64,
        created_at=_TS,
        staging_version=1,
        code_sha="abc1234",
        raw_manifest_hash="r" * 64,
        raw_cutoff=_TS,
        tables={
            "gameweeks": BuildTableEntry(sha="g" * 64, rows=1, schema_hash="s" * 64),
            "player_fixture_outcomes": BuildTableEntry(
                sha="o" * 64, rows=1, schema_hash="s" * 64
            ),
            "fixture_outcomes": BuildTableEntry(
                sha="f" * 64, rows=1, schema_hash="s" * 64
            ),
        },
    )


def _predictions(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.lit(_MODEL).alias("model_id"),
        pl.lit(_SEASON).alias("target_season"),
        pl.lit(1).cast(pl.Int64).alias("target_gw"),
    )


def _outcomes(rows: list[dict[str, object]]) -> pl.DataFrame:
    base = {
        "season": _SEASON,
        "gw": 1,
        "available_at": _TS,
        "schema_version": 1,
        "source": "test",
        "goals": 0,
        "assists": 0,
        "clean_sheets": 0,
    }
    return pl.DataFrame([{**base, **row} for row in rows])


def _stats(rows: list[dict[str, object]]) -> pl.DataFrame:
    base = {
        "season": _SEASON,
        "gw": 1,
        "available_at": _TS,
        "source": "test",
        "total_points": 0,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 0,
        "saves": 0,
        "bonus": 0,
        "yellow_cards": 0,
        "own_goals": 0,
        "penalties_saved": 0,
        "influence": 0.0,
        "creativity": 0.0,
        "threat": 0.0,
        "bps": 0,
        "clearances_blocks_interceptions": 0,
        "tackles": 0,
        "recoveries": 0,
        "defensive_contribution": 0,
    }
    return pl.DataFrame([{**base, **row} for row in rows])


def _fixtures(
    *, n: int = 10, team_keys: list[tuple[int, int]] | None = None
) -> pl.DataFrame:
    schema = {
        "season": pl.Utf8(),
        "fixture_id": pl.Int64(),
        "gw": pl.Int64(),
        "team_h_key": pl.Int64(),
        "team_a_key": pl.Int64(),
        "kickoff_time": pl.Datetime(time_zone="UTC"),
        "team_h_score": pl.Int64(),
        "team_a_score": pl.Int64(),
        "finished": pl.Boolean(),
        "available_at": pl.Datetime(time_zone="UTC"),
        "source": pl.Utf8(),
    }
    pairs = team_keys if team_keys is not None else [(1, 2)] * n
    if not pairs:
        return pl.DataFrame(schema=schema)
    rows = []
    for i, (h, a) in enumerate(pairs, start=1):
        rows.append(
            {
                "season": _SEASON,
                "fixture_id": i,
                "gw": 1,
                "team_h_key": h,
                "team_a_key": a,
                "kickoff_time": _DEADLINE,
                "team_h_score": 1,
                "team_a_score": 0,
                "finished": True,
                "available_at": _TS,
                "source": "test",
            }
        )
    return pl.DataFrame(rows)


def test_tag_miss_reason_priority() -> None:
    assert (
        tag_miss_reason(
            minutes=30,
            starts=1,
            red_cards=1,
            penalties_missed=1,
            status="i",
            injury_out=True,
            n_player_fixtures=2,
            n_team_fixtures=1,
        )
        == "red_card"
    )
    assert (
        tag_miss_reason(
            minutes=90,
            starts=1,
            red_cards=0,
            penalties_missed=1,
            status="a",
            injury_out=False,
            n_player_fixtures=1,
            n_team_fixtures=1,
        )
        == "penalty"
    )
    assert (
        tag_miss_reason(
            minutes=0,
            starts=0,
            red_cards=0,
            penalties_missed=0,
            status="a",
            injury_out=False,
            n_player_fixtures=1,
            n_team_fixtures=0,
        )
        == "blank"
    )
    assert (
        tag_miss_reason(
            minutes=0,
            starts=0,
            red_cards=0,
            penalties_missed=0,
            status="i",
            injury_out=False,
            n_player_fixtures=1,
            n_team_fixtures=1,
        )
        == "injury"
    )
    assert (
        tag_miss_reason(
            minutes=0,
            starts=0,
            red_cards=0,
            penalties_missed=0,
            status="a",
            injury_out=False,
            n_player_fixtures=1,
            n_team_fixtures=1,
        )
        == "benched"
    )
    assert (
        tag_miss_reason(
            minutes=45,
            starts=1,
            red_cards=0,
            penalties_missed=0,
            status="a",
            injury_out=False,
            n_player_fixtures=1,
            n_team_fixtures=1,
        )
        == "subbed_early"
    )
    assert (
        tag_miss_reason(
            minutes=90,
            starts=1,
            red_cards=0,
            penalties_missed=0,
            status="a",
            injury_out=False,
            n_player_fixtures=2,
            n_team_fixtures=2,
        )
        == "dgw"
    )
    assert (
        tag_miss_reason(
            minutes=90,
            starts=1,
            red_cards=0,
            penalties_missed=0,
            status="a",
            injury_out=False,
            n_player_fixtures=1,
            n_team_fixtures=1,
        )
        == "other"
    )


def test_top_15_by_abs_error_and_reasons() -> None:
    preds = _predictions(
        [{"player_key": 100 + i, "fixture_id": 1, "xpts": float(i)} for i in range(20)]
    )
    # actual=0 → abs_error = xpts; largest xpts rank first.
    outs = _outcomes(
        [
            {
                "player_key": 100 + i,
                "fixture_id": 1,
                "team_key": 1,
                "pts": 0,
                "minutes": 0 if i >= 18 else 90,
            }
            for i in range(20)
        ]
    )
    stats = _stats(
        [
            {
                "player_key": 100 + i,
                "fixture_id": 1,
                "minutes": 0 if i >= 18 else 90,
                "starts": 0 if i >= 18 else 1,
                "red_cards": 0,
                "penalties_missed": 0,
            }
            for i in range(20)
        ]
    )
    frame = compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=preds,
        outcomes=outs,
        stats=stats,
        fixtures=_fixtures(n=10),
        player_status=pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "player_key": pl.Int64(),
                "status": pl.Utf8(),
                "chance_of_playing": pl.Float64(),
                "chance_of_playing_this_round": pl.Float64(),
                "available_at": pl.Datetime(time_zone="UTC"),
            }
        ),
        predicted_lineups=pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "player_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "role": pl.Utf8(),
                "available_at": pl.Datetime(time_zone="UTC"),
            }
        ),
        deadline=_DEADLINE,
    )
    assert frame.height == TOP_N
    assert frame["rank"].to_list() == list(range(1, TOP_N + 1))
    assert frame["player_key"].to_list() == list(range(119, 104, -1))
    assert frame.filter(pl.col("player_key") == 119)["reason"][0] == "benched"
    assert frame.filter(pl.col("player_key") == 110)["reason"][0] == "other"
    assert frame["predictable_at_T90"].to_list() == [False] * TOP_N


def test_predictable_at_t90_uses_only_pre_cutoff_rows() -> None:
    preds = _predictions([{"player_key": 1, "fixture_id": 1, "xpts": 8.0}])
    outs = _outcomes(
        [{"player_key": 1, "fixture_id": 1, "team_key": 1, "pts": 0, "minutes": 0}]
    )
    stats = _stats(
        [
            {
                "player_key": 1,
                "fixture_id": 1,
                "minutes": 0,
                "starts": 0,
                "red_cards": 0,
                "penalties_missed": 0,
            }
        ]
    )
    # Late injury news after T−90m must NOT set the flag.
    late_only = pl.DataFrame(
        [
            {
                "season": _SEASON,
                "player_key": 1,
                "status": "i",
                "news": "Knee",
                "news_added": _DEADLINE,
                "price": 100,
                "ep_next": 0.0,
                "chance_of_playing": 0.0,
                "chance_of_playing_this_round": 0.0,
                "penalties_order": None,
                "corners_and_indirect_freekicks_order": None,
                "direct_freekicks_order": None,
                "available_at": _T90 + timedelta(minutes=1),
                "source": "test",
            }
        ]
    )
    late = compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=preds,
        outcomes=outs,
        stats=stats,
        fixtures=_fixtures(n=1, team_keys=[(1, 2)]),
        player_status=late_only,
        predicted_lineups=pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "team_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "player_key": pl.Int64(),
                "element_id": pl.Int64(),
                "role": pl.Utf8(),
                "confidence": pl.Float64(),
                "available_at": pl.Datetime(time_zone="UTC"),
                "source": pl.Utf8(),
            }
        ),
        deadline=_DEADLINE,
    )
    assert late.height == 1
    assert late["predictable_at_T90"][0] is False

    # Same injury visible at/before T−90m → flag True (overprediction + out).
    early = late_only.with_columns(pl.lit(_T90).alias("available_at"))
    flagged = compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=preds,
        outcomes=outs,
        stats=stats,
        fixtures=_fixtures(n=1, team_keys=[(1, 2)]),
        player_status=early,
        predicted_lineups=pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "team_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "player_key": pl.Int64(),
                "element_id": pl.Int64(),
                "role": pl.Utf8(),
                "confidence": pl.Float64(),
                "available_at": pl.Datetime(time_zone="UTC"),
                "source": pl.Utf8(),
            }
        ),
        deadline=_DEADLINE,
    )
    assert flagged["predictable_at_T90"][0] is True
    assert flagged["reason"][0] == "injury"


def test_predictable_at_t90_from_predicted_lineups() -> None:
    preds = _predictions([{"player_key": 2, "fixture_id": 1, "xpts": 7.0}])
    outs = _outcomes(
        [{"player_key": 2, "fixture_id": 1, "team_key": 1, "pts": 0, "minutes": 0}]
    )
    stats = _stats(
        [
            {
                "player_key": 2,
                "fixture_id": 1,
                "minutes": 0,
                "starts": 0,
                "red_cards": 0,
                "penalties_missed": 0,
            }
        ]
    )
    lineups = pl.DataFrame(
        [
            {
                "season": _SEASON,
                "team_key": 1,
                "fixture_id": 1,
                "player_key": 2,
                "element_id": 2,
                "role": ROLE_OUT,
                "confidence": 0.9,
                "available_at": _T90,
                "source": "test",
            }
        ]
    )
    frame = compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=preds,
        outcomes=outs,
        stats=stats,
        fixtures=_fixtures(n=1, team_keys=[(1, 2)]),
        player_status=pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "player_key": pl.Int64(),
                "status": pl.Utf8(),
                "news": pl.Utf8(),
                "news_added": pl.Datetime(time_zone="UTC"),
                "price": pl.Int64(),
                "ep_next": pl.Float64(),
                "chance_of_playing": pl.Float64(),
                "chance_of_playing_this_round": pl.Float64(),
                "penalties_order": pl.Int64(),
                "corners_and_indirect_freekicks_order": pl.Int64(),
                "direct_freekicks_order": pl.Int64(),
                "available_at": pl.Datetime(time_zone="UTC"),
                "source": pl.Utf8(),
            }
        ),
        predicted_lineups=lineups,
        deadline=_DEADLINE,
    )
    assert frame["predictable_at_T90"][0] is True

    # Post-cutoff lineup flip must not set the flag.
    late_lineups = lineups.with_columns(
        pl.lit(_T90 + timedelta(minutes=5)).alias("available_at")
    )
    late = compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=preds,
        outcomes=outs,
        stats=stats,
        fixtures=_fixtures(n=1, team_keys=[(1, 2)]),
        player_status=pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "player_key": pl.Int64(),
                "status": pl.Utf8(),
                "news": pl.Utf8(),
                "news_added": pl.Datetime(time_zone="UTC"),
                "price": pl.Int64(),
                "ep_next": pl.Float64(),
                "chance_of_playing": pl.Float64(),
                "chance_of_playing_this_round": pl.Float64(),
                "penalties_order": pl.Int64(),
                "corners_and_indirect_freekicks_order": pl.Int64(),
                "direct_freekicks_order": pl.Int64(),
                "available_at": pl.Datetime(time_zone="UTC"),
                "source": pl.Utf8(),
            }
        ),
        predicted_lineups=late_lineups,
        deadline=_DEADLINE,
    )
    assert late["predictable_at_T90"][0] is False


def test_underprediction_start_signal_is_predictable() -> None:
    preds = _predictions([{"player_key": 3, "fixture_id": 1, "xpts": 2.0}])
    outs = _outcomes(
        [{"player_key": 3, "fixture_id": 1, "team_key": 1, "pts": 12, "minutes": 90}]
    )
    stats = _stats(
        [
            {
                "player_key": 3,
                "fixture_id": 1,
                "minutes": 90,
                "starts": 1,
                "red_cards": 0,
                "penalties_missed": 0,
            }
        ]
    )
    lineups = pl.DataFrame(
        [
            {
                "season": _SEASON,
                "team_key": 1,
                "fixture_id": 1,
                "player_key": 3,
                "element_id": 3,
                "role": ROLE_START,
                "confidence": 0.8,
                "available_at": _T90,
                "source": "test",
            }
        ]
    )
    frame = compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=preds,
        outcomes=outs,
        stats=stats,
        fixtures=_fixtures(n=1, team_keys=[(1, 2)]),
        player_status=pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "player_key": pl.Int64(),
                "status": pl.Utf8(),
                "news": pl.Utf8(),
                "news_added": pl.Datetime(time_zone="UTC"),
                "price": pl.Int64(),
                "ep_next": pl.Float64(),
                "chance_of_playing": pl.Float64(),
                "chance_of_playing_this_round": pl.Float64(),
                "penalties_order": pl.Int64(),
                "corners_and_indirect_freekicks_order": pl.Int64(),
                "direct_freekicks_order": pl.Int64(),
                "available_at": pl.Datetime(time_zone="UTC"),
                "source": pl.Utf8(),
            }
        ),
        predicted_lineups=lineups,
        deadline=_DEADLINE,
    )
    assert frame["predictable_at_T90"][0] is True
    assert frame["reason"][0] == "other"


def test_dgw_and_blank_tags() -> None:
    # DGW: two fixtures for same player; tag dgw when full 90 both.
    preds = _predictions(
        [
            {"player_key": 5, "fixture_id": 1, "xpts": 6.0},
            {"player_key": 5, "fixture_id": 2, "xpts": 5.0},
        ]
    )
    outs = _outcomes(
        [
            {
                "player_key": 5,
                "fixture_id": 1,
                "team_key": 1,
                "pts": 2,
                "minutes": 90,
            },
            {
                "player_key": 5,
                "fixture_id": 2,
                "team_key": 1,
                "pts": 1,
                "minutes": 90,
            },
        ]
    )
    stats = _stats(
        [
            {
                "player_key": 5,
                "fixture_id": 1,
                "minutes": 90,
                "starts": 1,
                "red_cards": 0,
                "penalties_missed": 0,
            },
            {
                "player_key": 5,
                "fixture_id": 2,
                "minutes": 90,
                "starts": 1,
                "red_cards": 0,
                "penalties_missed": 0,
            },
        ]
    )
    empty_status = pl.DataFrame(
        schema={
            "season": pl.Utf8(),
            "player_key": pl.Int64(),
            "status": pl.Utf8(),
            "chance_of_playing": pl.Float64(),
            "chance_of_playing_this_round": pl.Float64(),
            "available_at": pl.Datetime(time_zone="UTC"),
        }
    )
    empty_lineups = pl.DataFrame(
        schema={
            "season": pl.Utf8(),
            "player_key": pl.Int64(),
            "fixture_id": pl.Int64(),
            "role": pl.Utf8(),
            "available_at": pl.Datetime(time_zone="UTC"),
        }
    )
    dgw = compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=preds,
        outcomes=outs,
        stats=stats,
        fixtures=_fixtures(n=2, team_keys=[(1, 2), (1, 3)]),
        player_status=empty_status,
        predicted_lineups=empty_lineups,
        deadline=_DEADLINE,
    )
    assert set(dgw["reason"].to_list()) == {"dgw"}

    blank = compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=_predictions([{"player_key": 9, "fixture_id": 99, "xpts": 4.0}]),
        outcomes=_outcomes(
            [
                {
                    "player_key": 9,
                    "fixture_id": 99,
                    "team_key": 99,
                    "pts": 0,
                    "minutes": 0,
                }
            ]
        ),
        stats=_stats(
            [
                {
                    "player_key": 9,
                    "fixture_id": 99,
                    "minutes": 0,
                    "starts": 0,
                    "red_cards": 0,
                    "penalties_missed": 0,
                }
            ]
        ),
        fixtures=_fixtures(n=0),
        player_status=empty_status,
        predicted_lineups=empty_lineups,
        deadline=_DEADLINE,
    )
    assert blank["reason"][0] == "blank"


def test_empty_inputs() -> None:
    empty = empty_biggest_misses()
    assert empty.is_empty()
    assert compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=pl.DataFrame(),
        outcomes=pl.DataFrame(),
        stats=pl.DataFrame(),
        fixtures=pl.DataFrame(),
        player_status=pl.DataFrame(),
        predicted_lineups=pl.DataFrame(),
        deadline=_DEADLINE,
    ).is_empty()


def _seed_predict_run(root: Path, *, player_key: int = 42) -> str:
    run_id = "p" * 16
    pred_dir = run_dir(root, run_id) / "predictions"
    pred_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "model_id": _MODEL,
                "as_of_season": _SEASON,
                "as_of_gw": 1,
                "target_season": _SEASON,
                "target_gw": 1,
                "player_key": player_key,
                "fixture_id": 1,
                "xpts": 9.0,
                "xpts_sd": 1.0,
                "p_played": 0.9,
                "p_60": 0.8,
                "p_goal": 0.3,
                "p_assist": 0.2,
                "p_cs": 0.1,
                "p_haul": 0.05,
            }
        ]
    ).write_parquet(pred_dir / "player_fixture.parquet")
    base = official_checkpoint_dir(root, season=_SEASON, gw=1, checkpoint="T-90m")
    base.mkdir(parents=True)
    (base / "line=primary.json").write_text(
        (
            f'{{"predict_run_id":"{run_id}","plan_run_id":"plan1",'
            f'"role":"primary","squad_mode":"real",'
            f'"written_at":"2024-08-16T09:30:00+00:00"}}'
        ),
        encoding="utf-8",
    )
    return run_id


def test_run_post_gw_writes_biggest_misses(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    store.write(
        "gameweeks",
        pl.DataFrame(
            [
                {
                    "season": _SEASON,
                    "gw": 1,
                    "deadline_time": _DEADLINE,
                    "finished": True,
                    "data_checked": True,
                    "available_at": _TS,
                    "source": "fpl_api",
                }
            ]
        ),
    )
    store.write(
        "fixtures",
        _fixtures(n=10),
    )
    store.write(
        "player_fixture_outcomes",
        _outcomes(
            [
                {
                    "player_key": 42,
                    "fixture_id": 1,
                    "team_key": 1,
                    "pts": 0,
                    "minutes": 0,
                }
            ]
        ),
    )
    store.write(
        "player_fixture_stats",
        _stats(
            [
                {
                    "player_key": 42,
                    "fixture_id": 1,
                    "minutes": 0,
                    "starts": 0,
                    "red_cards": 0,
                    "penalties_missed": 0,
                }
            ]
        ),
    )
    store.write(
        "player_status",
        pl.DataFrame(
            [
                {
                    "season": _SEASON,
                    "player_key": 42,
                    "status": "i",
                    "news": "Hamstring",
                    "news_added": _T90,
                    "price": 100,
                    "ep_next": 0.0,
                    "chance_of_playing": 0.0,
                    "chance_of_playing_this_round": 0.0,
                    "penalties_order": None,
                    "corners_and_indirect_freekicks_order": None,
                    "direct_freekicks_order": None,
                    "available_at": _T90,
                    "source": "test",
                }
            ]
        ),
    )
    _seed_predict_run(tmp_path, player_key=42)
    decisions = _decision_gw(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 8],
        captain_index=10,
        n_fixtures=10,
    ).with_columns(
        pl.lit(1).cast(pl.Int64).alias("gw"),
        pl.lit(_SEASON).alias("season"),
    )
    result = run_post_gw(
        tmp_path,
        _KEY,
        build=_build(),
        ops=_ops(),
        decisions=decisions,
        notify=False,
        push=False,
        rank_snapshot=None,
        n_fixtures=10,
    )
    assert result.skipped is False
    assert result.run_id is not None
    path = run_dir(tmp_path, result.run_id) / "biggest_misses.parquet"
    assert path.is_file()
    misses = pl.read_parquet(path)
    assert misses.height == 1
    assert misses["reason"][0] == "injury"
    assert misses["predictable_at_T90"][0] is True
    summary = (run_dir(tmp_path, result.run_id) / "summary.md").read_text(
        encoding="utf-8"
    )
    assert "## Biggest misses" in summary


def test_summary_and_ntfy_include_miss() -> None:
    scorecard = pl.DataFrame(
        [
            {
                "model_id": _MODEL,
                "season": _SEASON,
                "gw": 1,
                "gw_kind": "normal",
                "plan_ev": 50.0,
                "realised_points": 40,
                "realised_minus_plan_ev": -10.0,
                "pmf_percentile": 0.3,
                "fpl_average": None,
                "fpl_highest": None,
                "rank_change": None,
                "season_points_to_date": 40,
                "points_at_rank": None,
                "percentile": None,
                "realised_ex_chips": None,
                "chip": None,
            }
        ]
    )
    std = pl.DataFrame(
        [
            {
                "model_id": _MODEL,
                "through_gw": 1,
                "season_points": 40,
                "plan_ev_sum": 50.0,
            }
        ]
    )
    misses = pl.DataFrame(
        [
            {
                "model_id": _MODEL,
                "season": _SEASON,
                "gw": 1,
                "player_key": 7,
                "fixture_id": 1,
                "xpts": 9.0,
                "actual_pts": 0,
                "abs_error": 9.0,
                "signed_error": -9.0,
                "reason": "benched",
                "predictable_at_T90": True,
                "rank": 1,
            }
        ]
    )
    body = format_post_gw_summary(scorecard, std, pl.DataFrame(), misses)
    assert "player 7" in body
    assert "benched" in body
    line = format_post_gw_ntfy(scorecard, std, misses)
    assert "biggest miss: player 7 (-9, benched)" in line


@pytest.mark.parametrize(
    ("minutes", "starts", "expected"),
    [(0, 0, "benched"), (30, 1, "subbed_early"), (90, 1, "other")],
)
def test_reason_minutes_starts(minutes: int, starts: int, expected: str) -> None:
    assert (
        tag_miss_reason(
            minutes=minutes,
            starts=starts,
            red_cards=0,
            penalties_missed=0,
            status="a",
            injury_out=False,
            n_player_fixtures=1,
            n_team_fixtures=1,
        )
        == expected
    )


def _empty_status() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "season": pl.Utf8(),
            "player_key": pl.Int64(),
            "status": pl.Utf8(),
            "chance_of_playing": pl.Float64(),
            "chance_of_playing_this_round": pl.Float64(),
            "available_at": pl.Datetime(time_zone="UTC"),
        }
    )


def _empty_lineups() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "season": pl.Utf8(),
            "player_key": pl.Int64(),
            "fixture_id": pl.Int64(),
            "role": pl.Utf8(),
            "available_at": pl.Datetime(time_zone="UTC"),
        }
    )


def test_t90_cutoff_naive_deadline() -> None:
    naive = datetime(2024, 8, 16, 11, 0)  # noqa: DTZ001 — intentional naive
    assert t90_cutoff(naive) == _T90


def test_no_matching_outcomes_returns_empty() -> None:
    """Predictions exist but share no keys with outcomes → empty join."""
    frame = compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=_predictions([{"player_key": 1, "fixture_id": 1, "xpts": 5.0}]),
        outcomes=_outcomes(
            [
                {
                    "player_key": 99,
                    "fixture_id": 99,
                    "team_key": 1,
                    "pts": 0,
                    "minutes": 0,
                }
            ]
        ),
        stats=pl.DataFrame(),
        fixtures=_fixtures(n=1, team_keys=[(1, 2)]),
        player_status=_empty_status(),
        predicted_lineups=_empty_lineups(),
        deadline=_DEADLINE,
    )
    assert frame.is_empty()


def test_empty_and_wrong_season_stats() -> None:
    preds = _predictions([{"player_key": 1, "fixture_id": 1, "xpts": 6.0}])
    outs = _outcomes(
        [{"player_key": 1, "fixture_id": 1, "team_key": 1, "pts": 1, "minutes": 90}]
    )
    empty_stats = compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=preds,
        outcomes=outs,
        stats=pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "player_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "starts": pl.Int64(),
                "red_cards": pl.Int64(),
                "penalties_missed": pl.Int64(),
            }
        ),
        fixtures=_fixtures(n=1, team_keys=[(1, 2)]),
        player_status=_empty_status(),
        predicted_lineups=_empty_lineups(),
        deadline=_DEADLINE,
    )
    assert empty_stats.height == 1
    wrong = _stats(
        [
            {
                "player_key": 1,
                "fixture_id": 1,
                "minutes": 90,
                "starts": 1,
                "red_cards": 0,
                "penalties_missed": 0,
            }
        ]
    ).with_columns(pl.lit("2099-00").alias("season"))
    other_season = compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=preds,
        outcomes=outs,
        stats=wrong,
        fixtures=_fixtures(n=1, team_keys=[(1, 2)]),
        player_status=_empty_status(),
        predicted_lineups=_empty_lineups(),
        deadline=_DEADLINE,
    )
    assert other_season.height == 1


def test_injury_out_signal_and_filtered_empty() -> None:
    preds = _predictions([{"player_key": 8, "fixture_id": 1, "xpts": 7.0}])
    outs = _outcomes(
        [{"player_key": 8, "fixture_id": 1, "team_key": 1, "pts": 0, "minutes": 0}]
    )
    stats = _stats(
        [
            {
                "player_key": 8,
                "fixture_id": 1,
                "minutes": 0,
                "starts": 0,
                "red_cards": 0,
                "penalties_missed": 0,
            }
        ]
    )
    hit = pl.DataFrame(
        [
            {
                "season": _SEASON,
                "player_key": 8,
                "api_player_id": 1,
                "team_key": 1,
                "api_team_id": 1,
                "api_fixture_id": 1,
                "fixture_id": 1,
                "injury_type": "Knee",
                "reason": "Knee",
                "category": "out",
                "available_at": _T90,
                "source": "test",
            }
        ]
    )
    flagged = compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=preds,
        outcomes=outs,
        stats=stats,
        fixtures=_fixtures(n=1, team_keys=[(1, 2)]),
        player_status=_empty_status(),
        predicted_lineups=_empty_lineups(),
        deadline=_DEADLINE,
        player_injuries=hit,
    )
    assert flagged["reason"][0] == "injury"
    # Non-empty injuries that filter out (wrong category / after cutoff).
    miss = hit.with_columns(
        pl.lit("doubtful").alias("category"),
        pl.lit(_T90 + timedelta(minutes=5)).alias("available_at"),
    )
    skipped = compute_biggest_misses(
        model_id=_MODEL,
        key=_KEY,
        predictions=preds,
        outcomes=outs,
        stats=stats,
        fixtures=_fixtures(n=1, team_keys=[(1, 2)]),
        player_status=_empty_status(),
        predicted_lineups=_empty_lineups(),
        deadline=_DEADLINE,
        player_injuries=miss,
    )
    assert skipped["reason"][0] == "benched"


def test_official_predictions_bad_pointer_and_optional_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = official_checkpoint_dir(tmp_path, season=_SEASON, gw=1, checkpoint="T-90m")
    base.mkdir(parents=True)
    (base / "line=primary.json").write_text("{not-json", encoding="utf-8")
    assert _official_predictions(tmp_path, _KEY, line="primary").is_empty()

    (base / "line=primary.json").write_text('{"plan_run_id":"x"}', encoding="utf-8")
    assert _official_predictions(tmp_path, _KEY, line="primary").is_empty()

    store = DataStore(tmp_path)

    def boom(self: DataStore, name: str) -> pl.DataFrame:
        raise OSError(f"missing {name}")

    monkeypatch.setattr(DataStore, "read", boom)
    missing = _optional_table(store, "player_injuries")
    assert missing.is_empty()
    assert "player_key" in missing.columns


def test_biggest_misses_outcomes_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_predict_run(tmp_path, player_key=1)

    def boom(self: DataStore, name: str) -> pl.DataFrame:
        if name == "player_fixture_outcomes":
            raise KeyError(name)
        raise KeyError(name)

    monkeypatch.setattr(DataStore, "read", boom)
    frame = _biggest_misses(
        tmp_path,
        _KEY,
        model_id=_MODEL,
        line="primary",
        deadline=_DEADLINE,
    )
    assert frame.is_empty()
