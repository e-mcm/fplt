"""Post-GW analysis run: fingerprint skip-if-done, data_checked gate (#393)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request

import polars as pl
import pytest

from fplt.automation.post_gw import (
    format_post_gw_ntfy,
    format_post_gw_summary,
    has_pending_post_gw,
    pending_post_gw,
    post_gw_fingerprint,
    run_post_gw,
)
from fplt.automation.production import OfficialLine, ProductionOps
from fplt.data.store import DataStore
from fplt.models.pipeline_spec import live_pipeline_defaults
from fplt.models.spec import load_model_specs
from fplt.schemas.manifests import BuildManifest, BuildTableEntry
from fplt.settings import Settings
from fplt.store.fingerprint import fp_index_exists, write_fp_index
from fplt.store.runs import run_dir
from fplt.types import GWKey
from tests.unit.test_decision_scoring import _decision_gw

_TS = datetime(2024, 9, 1, 12, 0, tzinfo=UTC)
_DEADLINE = datetime(2024, 8, 16, 11, 0, tzinfo=UTC)
_SEASON = "2024-25"


def _ops() -> ProductionOps:
    pipeline = live_pipeline_defaults(model="rolling_mean_v1", horizon=3)
    return ProductionOps(
        models=["rolling_mean_v1"],
        plan_horizon=3,
        pipeline=pipeline,
        lines=[
            OfficialLine(
                line="primary",
                spec="rolling_mean_v1",
                horizon=3,
                role="primary",
                squad_mode="real",
            )
        ],
    )


def _build(*, outcomes_sha: str = "o" * 64) -> BuildManifest:
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
                sha=outcomes_sha, rows=1, schema_hash="s" * 64
            ),
            "fixture_outcomes": BuildTableEntry(
                sha="f" * 64, rows=1, schema_hash="s" * 64
            ),
        },
    )


def _write_gameweeks(
    root: Path,
    *,
    checked: list[int],
    unchecked: list[int] | None = None,
    n_fixtures_by_gw: dict[int, int] | None = None,
) -> DataStore:
    store = DataStore(root)
    rows: list[dict[str, object]] = []
    for gw in checked:
        rows.append(
            {
                "season": _SEASON,
                "gw": gw,
                "deadline_time": _DEADLINE,
                "finished": True,
                "data_checked": True,
                "available_at": _TS,
                "source": "fpl_api",
            }
        )
    for gw in unchecked or []:
        rows.append(
            {
                "season": _SEASON,
                "gw": gw,
                "deadline_time": _DEADLINE,
                "finished": False,
                "data_checked": False,
                "available_at": _TS,
                "source": "fpl_api",
            }
        )
    store.write("gameweeks", pl.DataFrame(rows))
    if n_fixtures_by_gw:
        fx_rows: list[dict[str, object]] = []
        fixture_id = 1
        for gw, count in n_fixtures_by_gw.items():
            for _ in range(count):
                fx_rows.append(
                    {
                        "season": _SEASON,
                        "fixture_id": fixture_id,
                        "gw": gw,
                        "team_h_key": 1,
                        "team_a_key": 2,
                        "kickoff_time": _DEADLINE,
                        "team_h_score": 1,
                        "team_a_score": 0,
                        "finished": True,
                        "available_at": _TS,
                        "source": "fpl_api",
                    }
                )
                fixture_id += 1
        if fx_rows:
            store.write("fixtures", pl.DataFrame(fx_rows))
    return store


def _decisions_for(
    *,
    gw: int,
    xi_pts: list[int] | None = None,
    n_fixtures: int | None = None,
    chip: str | None = None,
) -> pl.DataFrame:
    frame = _decision_gw(
        xi_pts=xi_pts or [2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 8],
        captain_index=10,
        n_fixtures=n_fixtures,
    ).with_columns(
        pl.lit(gw).cast(pl.Int64).alias("gw"),
        pl.lit(_SEASON).alias("season"),
    )
    if chip is not None:
        frame = frame.with_columns(pl.lit(chip).alias("chip"))
    return frame


def test_fingerprint_stable_then_changes_with_outcomes_sha(tmp_path: Path) -> None:
    spec = load_model_specs()["rolling_mean_v1"]
    key = GWKey(_SEASON, 1)
    first = post_gw_fingerprint(
        spec=spec, build=_build(outcomes_sha="a" * 64), target_gw=key, data_dir=tmp_path
    )
    again = post_gw_fingerprint(
        spec=spec, build=_build(outcomes_sha="a" * 64), target_gw=key, data_dir=tmp_path
    )
    assert first == again
    changed = post_gw_fingerprint(
        spec=spec, build=_build(outcomes_sha="c" * 64), target_gw=key, data_dir=tmp_path
    )
    assert changed != first


def test_no_write_before_data_checked(tmp_path: Path) -> None:
    store = _write_gameweeks(tmp_path, checked=[], unchecked=[1])
    assert not has_pending_post_gw(tmp_path, store, _build(), ops=_ops())
    with pytest.raises(ValueError, match="unchecked"):
        run_post_gw(
            tmp_path,
            GWKey(_SEASON, 1),
            build=_build(),
            ops=_ops(),
            decisions=_decisions_for(gw=1),
            notify=False,
            push=False,
            rank_snapshot=None,
        )


def test_writes_once_and_skips_on_re_tick(tmp_path: Path) -> None:
    store = _write_gameweeks(tmp_path, checked=[1], n_fixtures_by_gw={1: 10})
    decisions = _decisions_for(gw=1)
    settings = Settings(ntfy_url="https://ntfy.sh/test-post-gw")
    posts: list[Request] = []

    def fake_post(request: Request) -> object:
        posts.append(request)
        return object()

    first = run_post_gw(
        tmp_path,
        GWKey(_SEASON, 1),
        build=_build(),
        ops=_ops(),
        decisions=decisions,
        settings=settings,
        notify=True,
        post=fake_post,
        push=False,
        rank_snapshot=None,
        n_fixtures=10,
    )
    assert first.skipped is False
    assert first.run_id is not None
    assert fp_index_exists(tmp_path, first.fp)
    run_path = run_dir(tmp_path, first.run_id)
    assert (run_path / "scorecard.parquet").is_file()
    assert (run_path / "attribution.parquet").is_file()
    assert (run_path / "summary.md").is_file()
    assert (run_path / "_SUCCESS").is_file()
    summary = (run_path / "summary.md").read_text(encoding="utf-8")
    assert "## Season to date" in summary
    assert summary.index("## Season to date") < summary.index("## This gameweek")
    assert len(posts) == 1

    second = run_post_gw(
        tmp_path,
        GWKey(_SEASON, 1),
        build=_build(),
        ops=_ops(),
        decisions=decisions,
        settings=settings,
        notify=True,
        post=fake_post,
        push=False,
        rank_snapshot=None,
        n_fixtures=10,
    )
    assert second.skipped is True
    assert second.run_id is None
    assert second.fp == first.fp
    assert len(posts) == 1
    assert not has_pending_post_gw(tmp_path, store, _build(), ops=_ops())


def test_double_and_blank_gameweeks(tmp_path: Path) -> None:
    store = _write_gameweeks(
        tmp_path,
        checked=[1, 2],
        n_fixtures_by_gw={1: 12, 2: 0},
    )
    del store
    dgw = _decisions_for(
        gw=1,
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 14],
        n_fixtures=12,
    )
    bgw = _decisions_for(
        gw=2,
        xi_pts=[0] * 11,
        n_fixtures=0,
    ).with_columns(pl.lit(0).alias("actual_minutes"))
    # Blank: force zero minutes on all rows.
    bgw = bgw.with_columns(pl.lit(0).cast(pl.Int64).alias("actual_minutes"))

    double = run_post_gw(
        tmp_path,
        GWKey(_SEASON, 1),
        build=_build(),
        ops=_ops(),
        decisions=dgw,
        notify=False,
        push=False,
        rank_snapshot=None,
        n_fixtures=12,
    )
    blank = run_post_gw(
        tmp_path,
        GWKey(_SEASON, 2),
        build=_build(),
        ops=_ops(),
        decisions=bgw,
        notify=False,
        push=False,
        rank_snapshot=None,
        n_fixtures=0,
    )
    assert double.skipped is False
    assert blank.skipped is False
    d_score = pl.read_parquet(run_dir(tmp_path, double.run_id) / "scorecard.parquet")
    b_score = pl.read_parquet(run_dir(tmp_path, blank.run_id) / "scorecard.parquet")
    assert d_score["gw_kind"][0] == "double"
    assert b_score["gw_kind"][0] == "blank"
    assert int(b_score["realised_points"][0]) == 0


def test_pending_skips_when_by_fp_preseeded(tmp_path: Path) -> None:
    store = _write_gameweeks(tmp_path, checked=[1])
    spec = load_model_specs()["rolling_mean_v1"]
    key = GWKey(_SEASON, 1)
    fp = post_gw_fingerprint(
        spec=spec, build=_build(), target_gw=key, data_dir=tmp_path
    )
    write_fp_index(tmp_path, fp, "r_preseed")
    assert pending_post_gw(tmp_path, store, _build(), ops=_ops()) == []
    assert has_pending_post_gw(tmp_path, store, _build(), ops=_ops()) is False


def test_summary_and_ntfy_lead_with_season_points() -> None:
    scorecard = pl.DataFrame(
        [
            {
                "model_id": "m",
                "season": _SEASON,
                "gw": 3,
                "gw_kind": "normal",
                "plan_ev": 50.0,
                "realised_points": 61,
                "realised_minus_plan_ev": 11.0,
                "pmf_percentile": 0.64,
                "fpl_average": None,
                "fpl_highest": None,
                "rank_change": None,
                "season_points_to_date": 180,
                "points_at_rank": None,
                "percentile": None,
                "realised_ex_chips": 55,
                "chip": "bboost",
            }
        ]
    )
    std = pl.DataFrame(
        [
            {
                "model_id": "m",
                "season": _SEASON,
                "through_gw": 3,
                "gameweeks": 3,
                "season_points": 180,
                "plan_ev_sum": 150.0,
                "regret_sum": 12,
                "captaincy_regret_sum": 4,
                "hits_sum": 4.0,
                "hit_payback_to_date": 2.0,
                "points_at_rank": None,
                "percentile": None,
            }
        ]
    )
    attribution = pl.DataFrame(
        [
            {
                "model_id": "m",
                "season": _SEASON,
                "gw": 3,
                "gw_kind": "normal",
                "plan_ev": 50.0,
                "realised_points": 61,
                "luck_vs_model": 11.0,
                "captaincy_regret": 2,
                "xi_bench_regret": 1,
                "transfer_regret": 0,
                "chip_timing_regret": None,
                "hindsight_best": 64,
                "regret": 3,
                "hit": 0.0,
                "hit_payback_to_date": 2.0,
                "chip_value": 6.0,
                "intervention_delta": None,
            }
        ]
    )
    body = format_post_gw_summary(scorecard, std, attribution)
    assert body.index("## Season to date") < body.index("## This gameweek")
    line = format_post_gw_ntfy(scorecard, std)
    assert "GW3: 61 pts" in line
    assert "season 180" in line


def test_chip_ex_chips_on_scorecard(tmp_path: Path) -> None:
    _write_gameweeks(tmp_path, checked=[1], n_fixtures_by_gw={1: 10})
    # Bench boost: ex-chips should drop bench points.
    picks = _decisions_for(gw=1, chip="bboost").with_columns(
        pl.when(~pl.col("in_xi"))
        .then(pl.lit(3))
        .otherwise(pl.col("actual_pts"))
        .alias("actual_pts"),
        pl.when(~pl.col("in_xi"))
        .then(pl.lit(90))
        .otherwise(pl.col("actual_minutes"))
        .alias("actual_minutes"),
    )
    result = run_post_gw(
        tmp_path,
        GWKey(_SEASON, 1),
        build=_build(),
        ops=_ops(),
        decisions=picks,
        notify=False,
        push=False,
        rank_snapshot=None,
        n_fixtures=10,
    )
    score = pl.read_parquet(run_dir(tmp_path, result.run_id) / "scorecard.parquet")
    assert score["chip"][0] == "bboost"
    assert score["realised_ex_chips"][0] is not None
    assert int(score["realised_points"][0]) > int(score["realised_ex_chips"][0])
    attr = pl.read_parquet(run_dir(tmp_path, result.run_id) / "attribution.parquet")
    assert attr["chip_value"][0] is not None
    assert float(attr["chip_value"][0]) > 0


def test_format_empty_scorecard() -> None:
    empty = pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "season": pl.Utf8(),
            "gw": pl.Int64(),
            "gw_kind": pl.Utf8(),
            "plan_ev": pl.Float64(),
            "realised_points": pl.Int64(),
            "realised_minus_plan_ev": pl.Float64(),
            "pmf_percentile": pl.Float64(),
            "fpl_average": pl.Float64(),
            "fpl_highest": pl.Float64(),
            "rank_change": pl.Float64(),
            "season_points_to_date": pl.Int64(),
            "points_at_rank": pl.Float64(),
            "percentile": pl.Float64(),
            "realised_ex_chips": pl.Int64(),
            "chip": pl.Utf8(),
        }
    )
    assert format_post_gw_summary(empty, empty, empty).startswith("_No")
    assert format_post_gw_ntfy(empty, empty) == "post_gw: no scorecard"


def test_fingerprint_includes_pointer_digest(tmp_path: Path) -> None:
    from fplt.store.official import official_checkpoint_dir

    spec = load_model_specs()["rolling_mean_v1"]
    key = GWKey(_SEASON, 1)
    base = official_checkpoint_dir(tmp_path, season=_SEASON, gw=1, checkpoint="T-90m")
    base.mkdir(parents=True)
    (base / "line=primary.json").write_text('{"predict_run_id":"r1"}', encoding="utf-8")
    (base / "applied.json").write_text('{"line":"primary"}', encoding="utf-8")
    with_ptr = post_gw_fingerprint(
        spec=spec, build=_build(), target_gw=key, data_dir=tmp_path
    )
    empty = post_gw_fingerprint(
        spec=spec, build=_build(), target_gw=GWKey(_SEASON, 2), data_dir=tmp_path
    )
    assert with_ptr != empty


def test_pmf_percentile_from_official_pointer(tmp_path: Path) -> None:
    from fplt.store.official import official_checkpoint_dir
    from fplt.store.runs import run_dir

    _write_gameweeks(tmp_path, checked=[1], n_fixtures_by_gw={1: 10})
    predict_id = "r_pmf_predict"
    pred = run_dir(tmp_path, predict_id)
    (pred / "predictions").mkdir(parents=True)
    (pred / "_SUCCESS").write_text("", encoding="utf-8")
    rows = []
    for player_key in range(1, 12):
        for points_bin, probability in ((2, 0.5), (8, 0.5)):
            rows.append(
                {
                    "run_id": predict_id,
                    "as_of_season": _SEASON,
                    "as_of_gw": 1,
                    "deadline_utc": _DEADLINE,
                    "target_season": _SEASON,
                    "target_gw": 1,
                    "horizon": 1,
                    "player_key": player_key,
                    "team_key": 1,
                    "schema_version": 1,
                    "points_bin": points_bin,
                    "probability": probability,
                }
            )
    pl.DataFrame(rows).write_parquet(pred / "predictions" / "points_pmf_gw.parquet")
    base = official_checkpoint_dir(tmp_path, season=_SEASON, gw=1, checkpoint="T-90m")
    base.mkdir(parents=True)
    (base / "line=primary.json").write_text(
        f'{{"predict_run_id":"{predict_id}","plan_run_id":"r_plan"}}',
        encoding="utf-8",
    )
    result = run_post_gw(
        tmp_path,
        GWKey(_SEASON, 1),
        build=_build(),
        ops=_ops(),
        decisions=_decisions_for(gw=1),
        notify=False,
        push=False,
        rank_snapshot=None,
        n_fixtures=10,
    )
    score = pl.read_parquet(run_dir(tmp_path, result.run_id) / "scorecard.parquet")
    assert score["pmf_percentile"][0] is not None
    assert 0.0 <= float(score["pmf_percentile"][0]) <= 1.0


def test_freehit_ex_chips_null(tmp_path: Path) -> None:
    _write_gameweeks(tmp_path, checked=[1], n_fixtures_by_gw={1: 10})
    result = run_post_gw(
        tmp_path,
        GWKey(_SEASON, 1),
        build=_build(),
        ops=_ops(),
        decisions=_decisions_for(gw=1, chip="freehit"),
        notify=False,
        push=False,
        rank_snapshot=None,
        n_fixtures=10,
    )
    score = pl.read_parquet(run_dir(tmp_path, result.run_id) / "scorecard.parquet")
    assert score["chip"][0] == "freehit"
    assert score["realised_ex_chips"][0] is None


def test_run_pending_swallows_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.automation.post_gw import run_pending_post_gw

    store = _write_gameweeks(tmp_path, checked=[1])
    monkeypatch.setattr(
        "fplt.automation.post_gw.run_post_gw",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")),
    )
    results = run_pending_post_gw(
        tmp_path, store, _build(), ops=_ops(), notify=False, push=False
    )
    assert results == []


def test_missing_decisions_raises(tmp_path: Path) -> None:
    _write_gameweeks(tmp_path, checked=[1])
    with pytest.raises(ValueError, match="no decisions"):
        run_post_gw(
            tmp_path,
            GWKey(_SEASON, 1),
            build=_build(),
            ops=_ops(),
            decisions=pl.DataFrame(
                schema={
                    "model_id": pl.Utf8(),
                    "season": pl.Utf8(),
                    "gw": pl.Int64(),
                    "player_key": pl.Int64(),
                    "position": pl.Utf8(),
                    "in_xi": pl.Boolean(),
                    "captain": pl.Boolean(),
                    "vice": pl.Boolean(),
                    "bought": pl.Boolean(),
                    "sold": pl.Boolean(),
                    "xpts": pl.Float64(),
                    "actual_pts": pl.Int64(),
                    "actual_minutes": pl.Int64(),
                    "hit": pl.Float64(),
                    "free_transfers": pl.Int64(),
                    "team_key": pl.Int64(),
                    "price": pl.Int64(),
                }
            ),
            notify=False,
            push=False,
            rank_snapshot=None,
        )


def test_pending_empty_season(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    assert pending_post_gw(tmp_path, store, _build(), ops=_ops()) == []


def test_pending_dedupes_duplicate_gameweek_rows(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": [_SEASON, _SEASON],
                "gw": [1, 1],
                "deadline_time": [_DEADLINE, _DEADLINE],
                "finished": [True, True],
                "data_checked": [True, True],
                "available_at": [_TS, _TS],
                "source": ["fpl_api", "fpl_api"],
            }
        ),
    )
    pending = pending_post_gw(tmp_path, store, _build(), ops=_ops())
    assert len(pending) == 1


def test_format_with_empty_season_to_date() -> None:
    scorecard = pl.DataFrame(
        [
            {
                "model_id": "m",
                "season": _SEASON,
                "gw": 1,
                "gw_kind": "normal",
                "plan_ev": 10.0,
                "realised_points": 10,
                "realised_minus_plan_ev": 0.0,
                "pmf_percentile": None,
                "fpl_average": None,
                "fpl_highest": None,
                "rank_change": None,
                "season_points_to_date": 10,
                "points_at_rank": None,
                "percentile": None,
                "realised_ex_chips": None,
                "chip": None,
            }
        ]
    )
    empty_std = pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "season": pl.Utf8(),
            "through_gw": pl.Int64(),
            "gameweeks": pl.Int64(),
            "season_points": pl.Int64(),
            "plan_ev_sum": pl.Float64(),
            "regret_sum": pl.Int64(),
            "captaincy_regret_sum": pl.Int64(),
            "hits_sum": pl.Float64(),
            "hit_payback_to_date": pl.Float64(),
            "points_at_rank": pl.Float64(),
            "percentile": pl.Float64(),
        }
    )
    body = format_post_gw_summary(scorecard, empty_std, empty_std)
    assert "_No season-to-date rows._" in body
    assert "season 0" in format_post_gw_ntfy(scorecard, empty_std)


def test_matching_model_id_and_naive_deadline(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    naive = datetime(2024, 8, 16, 11, 0)  # no tz
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": [_SEASON],
                "gw": [1],
                "deadline_time": [naive],
                "finished": [True],
                "data_checked": [True],
                "available_at": [_TS],
                "source": ["fpl_api"],
            }
        ),
    )
    mid = load_model_specs()["rolling_mean_v1"].model_id
    decisions = _decisions_for(gw=1).with_columns(pl.lit(mid).alias("model_id"))
    result = run_post_gw(
        tmp_path,
        GWKey(_SEASON, 1),
        build=_build(),
        ops=_ops(),
        decisions=decisions,
        notify=False,
        push=False,
        rank_snapshot=None,
        n_fixtures=10,
    )
    assert result.skipped is False


def test_n_fixtures_empty_and_chip_blank_column(tmp_path: Path) -> None:
    from fplt.automation import post_gw as mod

    store = DataStore(tmp_path)
    assert mod._n_fixtures(store, GWKey(_SEASON, 1)) is None
    store.write(
        "fixtures",
        pl.DataFrame(
            {
                "season": [_SEASON],
                "fixture_id": [1],
                "gw": [2],
                "team_h_key": [1],
                "team_a_key": [2],
                "kickoff_time": [_DEADLINE],
                "team_h_score": [None],
                "team_a_score": [None],
                "finished": [False],
                "available_at": [_TS],
                "source": ["fpl_api"],
            }
        ),
    )
    assert mod._n_fixtures(store, GWKey(_SEASON, 1)) == 0
    picks = _decisions_for(gw=1).with_columns(
        pl.lit(None).cast(pl.Utf8()).alias("chip")
    )
    assert mod._chip(picks) is None
    picks2 = picks.with_columns(pl.lit("  ").alias("chip"))
    assert mod._chip(picks2) is None
    assert mod._chip(pl.DataFrame({"x": [1]})) is None


def test_pmf_edge_paths(tmp_path: Path) -> None:
    from fplt.automation import post_gw as mod
    from fplt.store.official import official_checkpoint_dir
    from fplt.store.runs import run_dir

    key = GWKey(_SEASON, 1)
    picks = _decisions_for(gw=1)
    assert (
        mod._pmf_pct(tmp_path, key, picks, realised=10, line="primary", model_id="m")
        is None
    )
    base = official_checkpoint_dir(tmp_path, season=_SEASON, gw=1, checkpoint="T-90m")
    base.mkdir(parents=True)
    (base / "line=primary.json").write_text("{not-json", encoding="utf-8")
    assert (
        mod._pmf_pct(tmp_path, key, picks, realised=10, line="primary", model_id="m")
        is None
    )
    (base / "line=primary.json").write_text(
        '{"predict_run_id":"r_missing"}', encoding="utf-8"
    )
    assert (
        mod._pmf_pct(tmp_path, key, picks, realised=10, line="primary", model_id="m")
        is None
    )
    pred = run_dir(tmp_path, "r_empty")
    (pred / "predictions").mkdir(parents=True)
    pl.DataFrame(
        {
            "run_id": ["r_empty"],
            "as_of_season": [_SEASON],
            "as_of_gw": [1],
            "deadline_utc": [_DEADLINE],
            "target_season": [_SEASON],
            "target_gw": [2],
            "horizon": [1],
            "player_key": [1],
            "team_key": [1],
            "schema_version": [1],
            "points_bin": [1],
            "probability": [1.0],
        }
    ).write_parquet(pred / "predictions" / "points_pmf_gw.parquet")
    (base / "line=primary.json").write_text(
        '{"predict_run_id":"r_empty"}', encoding="utf-8"
    )
    assert (
        mod._pmf_pct(tmp_path, key, picks, realised=10, line="primary", model_id="m")
        is None
    )
    # empty XI → percentile based on realised
    empty_xi = picks.with_columns(pl.lit(False).alias("in_xi"))
    pl.DataFrame(
        {
            "run_id": ["r_empty"],
            "as_of_season": [_SEASON],
            "as_of_gw": [1],
            "deadline_utc": [_DEADLINE],
            "target_season": [_SEASON],
            "target_gw": [1],
            "horizon": [1],
            "player_key": [999],
            "team_key": [1],
            "schema_version": [1],
            "points_bin": [1],
            "probability": [1.0],
        }
    ).write_parquet(pred / "predictions" / "points_pmf_gw.parquet")
    (base / "line=primary.json").write_text(
        '{"predict_run_id":"r_empty"}', encoding="utf-8"
    )
    assert (
        mod._pmf_pct(tmp_path, key, empty_xi, realised=0, line="primary", model_id="m")
        == 0.0
    )
    # XI players missing from PMF → totals stay 0
    assert (
        mod._pmf_pct(tmp_path, key, picks, realised=10, line="primary", model_id="m")
        == 1.0
    )
