"""Transfer plans: hits, banking, and the decision ledger."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.backtest.ledger import append_predictions, ledger_path
from fplt.cli import main
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.decisions import (
    DEFAULT_LIVE_CHIP_ARM,
    PlanExistsError,
    _require_usable_points,
    actual_rows_from_ledger,
    points_from_ledger,
    prediction_rows_from_ledger,
    run_decision_bakeoff,
    run_decisions,
    run_plan,
    simulate_decisions,
    write_decisions,
)
from fplt.optimise.plan import (
    DEFAULT_PLAN_HORIZON,
    covered_horizon,
    plan_squad,
    plan_table,
    render_report,
    write_plan,
)
from fplt.optimise.squad import select_squad
from fplt.optimise.state import (
    OwnedPlayer,
    SquadState,
    apply_entry_picks,
    load_squad_state,
)
from fplt.schemas.manifests import RunManifest
from fplt.settings import get_settings

_WHEN = datetime(2024, 8, 1, tzinfo=UTC)


def test_initial_plan_matches_the_single_gameweek_squad() -> None:
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    plan = plan_squad(pool.drop("xpts"), points, _state(), initial=True, horizon=1)
    squad = select_squad(pool)
    assert plan.points == pytest.approx(squad.total_xpts)
    assert plan.hits == (0.0,)
    assert plan.free_transfers == (0,)
    kept = plan.picks.filter(~pl.col("sold"))
    assert kept.height == 15
    assert kept.filter(pl.col("in_xi")).height == 11
    assert kept.filter(pl.col("captain")).height == 1
    assert kept.filter(pl.col("vice")).height == 1


def test_a_second_transfer_is_taken_only_when_it_beats_the_hit() -> None:
    worth_it = _swap_plan(6.0)
    assert worth_it.picks.filter(pl.col("bought")).height == 2
    assert worth_it.hits == (4.0,)
    declined = _swap_plan(2.0)
    assert declined.picks.filter(pl.col("bought")).height == 1
    assert declined.hits == (0.0,)


def test_a_free_transfer_is_banked_for_a_better_week() -> None:
    plan = _bank_plan()
    buys = [
        plan.picks.filter((pl.col("step") == step) & pl.col("bought")).height
        for step in (0, 1)
    ]
    assert buys == [0, 2]
    assert plan.hits == (0.0, 0.0)
    assert plan.free_transfers == (1, 2)
    report = render_report(plan, "2024-25", [1, 2])
    assert "GW 2" in report
    assert "Hit: 0" in report
    captain = int(
        plan.picks.filter((pl.col("step") == 0) & pl.col("captain"))["player_key"][0]
    )
    vice = int(
        plan.picks.filter((pl.col("step") == 0) & pl.col("vice"))["player_key"][0]
    )
    named = render_report(
        plan, "2024-25", [1, 2], names={captain: "Saka", vice: "Saliba"}
    )
    assert "Saka (C)" in named
    assert "Saliba (V)" in named
    assert "### Transfers out" in named
    assert "### Transfers in" in named
    assert "### Starting XI" in named
    assert "### Bench" in named
    assert "| Name | Pos | Team | Price | xPts |" in named
    blank = render_report(plan, "2024-25", [1, 2], names={captain: "  "})
    assert f"{captain} (C)" in blank


def test_render_report_tables_mark_armbands_and_empty_transfers() -> None:
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    plan = plan_squad(pool.drop("xpts"), points, _state(), initial=True, horizon=1)
    report = render_report(plan, "2024-25", [1], names={1: "Raya"})
    assert "### Transfers out" in report
    assert "_None_" in report
    assert "### Starting XI" in report
    assert "### Bench" in report
    assert "(C)" in report
    assert "(V)" in report
    assert report.count("| Name | Pos | Team | Price | xPts |") >= 2
    xi_block = report.split("### Starting XI")[1].split("### Bench")[0]
    assert "(C)" in xi_block
    assert "(V)" in xi_block
    assert "### Transfers out" not in xi_block


def test_bad_plans_are_rejected() -> None:
    pool = _pool().drop("xpts")
    points = _pool().select("player_key", pl.lit(0).alias("step"), "xpts")
    with pytest.raises(ValueError, match="no plan"):
        plan_squad(pool, points, _state(bank=0), initial=True, horizon=1)
    owned = [OwnedPlayer(player_key=1, selling_price=40)]
    with pytest.raises(ValueError, match="15"):
        plan_squad(pool, points, _state(owned=owned), initial=False, horizon=1)
    with pytest.raises(ValueError, match="outside"):
        plan_squad(
            pool,
            pl.DataFrame({"player_key": [1], "step": [3], "xpts": [1.0]}),
            _state(),
            initial=True,
            horizon=1,
        )


def test_orphan_prediction_rows_are_ignored() -> None:
    """xpts for players without a PIT price do not break the plan.

    Mid-season additions can appear in the prediction universe (fixtures)
    before ``player_status`` has a row, so they are not candidates.
    """
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    orphan = pl.DataFrame(
        {
            "player_key": pl.Series([999_999], dtype=pl.Int64),
            "step": pl.Series([0], dtype=points["step"].dtype),
            "xpts": pl.Series([100.0], dtype=pl.Float64),
        }
    )
    with_orphan = plan_squad(
        pool.drop("xpts"),
        pl.concat([points, orphan]),
        _state(),
        initial=True,
        horizon=1,
    )
    without = plan_squad(pool.drop("xpts"), points, _state(), initial=True, horizon=1)
    assert with_orphan.points == pytest.approx(without.points)
    kept = with_orphan.picks.filter(~pl.col("sold"))["player_key"].to_list()
    assert 999_999 not in kept


def test_entry_picks_replace_the_squad_and_the_bank() -> None:
    state = load_squad_state()
    assert state.bank == 1000
    assert state.ft_bank_cap == 5
    assert state.owned == []
    updated = apply_entry_picks(
        state,
        {
            "picks": [{"element": 10, "selling_price": 55}],
            "entry_history": {"bank": 12},
        },
        {10: 100},
    )
    assert updated.bank == 12
    assert updated.owned[0].player_key == 100
    assert updated.owned[0].purchase_price == 55
    assert updated.owned[0].selling_price == 55
    assert updated.free_transfers == state.free_transfers
    with pytest.raises(KeyError, match="element"):
        apply_entry_picks(
            state,
            {
                "picks": [{"element": 1, "selling_price": 1}],
                "entry_history": {"bank": 0},
            },
            {},
        )
    with pytest.raises(ValueError, match="picks"):
        apply_entry_picks(state, {}, {})


def test_a_plan_file_is_written_once(tmp_path: Path) -> None:
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    plan = plan_squad(pool.drop("xpts"), points, _state(), initial=True, horizon=1)
    frame = plan_table(plan, "model", "2024-25", 1, [1])
    path = write_plan(tmp_path, frame)
    assert path.parent.name == "as_of=2024-25_gw01_h01"
    with pytest.raises(FileExistsError):
        write_plan(path.parents[3], frame)
    longer = plan_squad(pool.drop("xpts"), points, _state(), initial=True, horizon=2)
    other = write_plan(tmp_path, plan_table(longer, "model", "2024-25", 1, [1, 2]))
    assert other.parent.name == "as_of=2024-25_gw01_h02"
    assert covered_horizon(11, 2) == 2
    assert covered_horizon(38, 2) == 1
    assert covered_horizon(1, 0) == 0
    assert covered_horizon(39, 1) == 0
    broken = frame.with_columns(pl.lit(2).alias("step"))
    with pytest.raises(ValueError, match="horizon step"):
        write_plan(tmp_path, broken)


def test_decision_bakeoff_rejects_inflated_mean_ppg(tmp_path: Path) -> None:
    pool = _pool().drop("xpts")
    candidates = pl.concat([pool.with_columns(pl.lit(gw).alias("gw")) for gw in (1,)])
    predictions = pl.DataFrame(
        {
            "as_of_gw": [1],
            "target_gw": [1],
            "player_key": [1],
            "xpts": [4.0],
        }
    )
    # Huge actuals → mean well above the sanity band after captain ×2.
    actuals = pl.DataFrame(
        {
            "gw": [1],
            "player_key": [1],
            "actual_pts": [80],
            "actual_minutes": [90],
        }
    )
    frame = simulate_decisions(
        candidates,
        predictions,
        actuals,
        _state(),
        model_id="rolling_mean",
        season="2024-25",
        gameweeks=[1],
        horizon=1,
    )
    from fplt.models.spec import load_model_specs

    spec = load_model_specs()["rolling_mean_v1"]
    tagged = frame.with_columns(pl.lit(spec.model_id).alias("model_id"))
    write_decisions(tmp_path, tagged)
    with pytest.raises(ValueError, match="mean pts/GW"):
        run_decision_bakeoff(
            tmp_path,
            ["rolling_mean_v1"],
            "2024-25",
            1,
            1,
            1,
            simulate=False,
            max_mean_ppg=90.0,
        )


def test_decision_bakeoff_summarises_existing_ledger(tmp_path: Path) -> None:
    pool = _pool().drop("xpts")
    candidates = pl.concat([pool.with_columns(pl.lit(gw).alias("gw")) for gw in (1, 2)])
    predictions = pl.DataFrame(
        {
            "as_of_gw": [1, 2],
            "target_gw": [1, 2],
            "player_key": [1, 1],
            "xpts": [4.0, 4.0],
        }
    )
    actuals = pl.DataFrame(
        {
            "gw": [1, 2],
            "player_key": [1, 1],
            "actual_pts": [3, 5],
            "actual_minutes": [90, 90],
        }
    )
    frame = simulate_decisions(
        candidates,
        predictions,
        actuals,
        _state(),
        model_id="rolling_mean",
        season="2024-25",
        gameweeks=[1, 2],
        horizon=1,
    )
    write_decisions(tmp_path, frame)
    from fplt.models.spec import load_model_specs

    spec = load_model_specs()["rolling_mean_v1"]
    tagged = frame.with_columns(pl.lit(spec.model_id).alias("model_id"))
    write_decisions(tmp_path, tagged)
    summary = run_decision_bakeoff(
        tmp_path,
        ["rolling_mean_v1"],
        "2024-25",
        1,
        2,
        1,
        simulate=False,
    )
    assert summary.height == 1
    assert summary["gameweeks"].to_list() == [2]
    with pytest.raises(ValueError, match="non-empty"):
        run_decision_bakeoff(tmp_path, [], "2024-25", 1, 2, 1, simulate=False)


def test_two_gameweeks_record_a_decision_row_set(tmp_path: Path) -> None:
    pool = _pool().drop("xpts")
    candidates = pl.concat([pool.with_columns(pl.lit(gw).alias("gw")) for gw in (1, 2)])
    predictions = pl.DataFrame(
        {
            "as_of_gw": [1, 1, 2],
            "target_gw": [1, 2, 2],
            "player_key": [1, 1, 1],
            "xpts": [1.0, 4.0, 4.0],
        }
    )
    actuals = pl.DataFrame(
        {
            "gw": [1, 2],
            "player_key": [1, 1],
            "actual_pts": [3, 5],
            "actual_minutes": [90, 90],
        }
    )
    frame = simulate_decisions(
        candidates,
        predictions,
        actuals,
        _state(),
        model_id="model",
        season="2024-25",
        gameweeks=[1, 2],
        horizon=1,
    )
    for gameweek in (1, 2):
        part = frame.filter(pl.col("gw") == gameweek)
        assert part.filter(~pl.col("sold")).height == 15
    assert frame.filter(pl.col("gw") == 2)["free_transfers"][0] == 1
    path = write_decisions(tmp_path, frame)
    again = frame.with_columns(pl.lit(9.0).alias("hit"))
    write_decisions(tmp_path, again)
    saved = pl.read_parquet(path)
    assert saved.height == frame.height
    assert saved["hit"].unique().to_list() == [9.0]


def test_ledger_points_keep_the_deadline_that_issued_them() -> None:
    ledger = pl.DataFrame(
        {
            "model_id": ["m", "m", "m"],
            "as_of_season": ["2024-25"] * 3,
            "as_of_gw": [1, 1, 2],
            "target_season": ["2024-25"] * 3,
            "target_gw": [1, 1, 1],
            "player_key": [7, 7, 7],
            "xpts": [1.0, 2.0, 9.0],
        }
    )
    points = points_from_ledger(ledger, "m", "2024-25", 1, [1])
    assert points["xpts"].to_list() == [3.0]
    assert points_from_ledger(ledger.clear(), "m", "2024-25", 1, [1]).is_empty()


def test_prediction_rows_do_not_double_count_appended_fixtures() -> None:
    """Re-running backtest appends the same fixture; xpts must stay once."""
    ledger = pl.DataFrame(
        {
            "model_id": ["m"] * 4,
            "as_of_season": ["2024-25"] * 4,
            "as_of_gw": [1, 1, 1, 1],
            "target_season": ["2024-25"] * 4,
            "target_gw": [1, 1, 1, 1],
            "player_key": [7, 7, 7, 7],
            "fixture_id": [100, 100, 201, 201],
            "xpts": [2.0, 2.1, 3.0, 3.2],
        }
    )
    out = prediction_rows_from_ledger(ledger, "m", "2024-25")
    assert out.height == 1
    # keep="last" per fixture, then sum distinct fixtures (DGW).
    assert out["xpts"].to_list() == [pytest.approx(2.1 + 3.2)]


def test_points_from_ledger_dedupes_fixture_before_sum() -> None:
    ledger = pl.DataFrame(
        {
            "model_id": ["m"] * 3,
            "as_of_season": ["2024-25"] * 3,
            "as_of_gw": [1, 1, 1],
            "target_season": ["2024-25"] * 3,
            "target_gw": [1, 1, 1],
            "player_key": [7, 7, 7],
            "fixture_id": [100, 100, 201],
            "xpts": [1.0, 9.0, 2.0],
        }
    )
    points = points_from_ledger(ledger, "m", "2024-25", 1, [1])
    assert points["xpts"].to_list() == [pytest.approx(11.0)]


def test_actual_rows_do_not_double_count_horizon_duplicates() -> None:
    """steps=2 writes the same fixture twice; realised pts must stay once."""
    ledger = pl.DataFrame(
        {
            "model_id": ["m"] * 4,
            "as_of_gw": [1, 2, 1, 2],
            "horizon": [2, 1, 2, 1],
            "target_season": ["2024-25"] * 4,
            "target_gw": [2, 2, 2, 2],
            "player_key": [7, 7, 8, 8],
            "fixture_id": [100, 100, 100, 100],
            "actual_pts": [6, 6, 3, 3],
            "actual_minutes": [90, 90, 75, 75],
        }
    )
    out = actual_rows_from_ledger(ledger, "m", "2024-25").sort("player_key")
    assert out["actual_pts"].to_list() == [6, 3]
    assert out["actual_minutes"].to_list() == [90, 75]


def test_actual_rows_still_sum_distinct_fixtures_in_a_dgw() -> None:
    ledger = pl.DataFrame(
        {
            "model_id": ["m"] * 4,
            "as_of_gw": [10, 10, 11, 11],
            "horizon": [2, 2, 1, 1],
            "target_season": ["2024-25"] * 4,
            "target_gw": [11, 11, 11, 11],
            "player_key": [7, 7, 7, 7],
            "fixture_id": [200, 201, 200, 201],
            "actual_pts": [5, 4, 5, 4],
            "actual_minutes": [90, 90, 90, 90],
        }
    )
    out = actual_rows_from_ledger(ledger, "m", "2024-25")
    assert out.height == 1
    assert out["actual_pts"].to_list() == [9]
    assert out["actual_minutes"].to_list() == [180]


def test_run_plan_resolves_squad_when_entry_id_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    _score(tmp_path)
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FPLT_ENTRY_ID", "42")
    get_settings.cache_clear()
    resolved = _state()
    calls: list[object] = []

    def fake_resolve(view, **kwargs):  # noqa: ANN001
        calls.append(kwargs)
        return resolved

    monkeypatch.setattr("fplt.optimise.decisions.resolve_squad_state", fake_resolve)
    try:
        solved, _path, _report = run_plan(tmp_path, "rolling_mean_v1", "2024-25", 1, 1)
    finally:
        get_settings.cache_clear()
    assert len(calls) == 1
    assert calls[0]["entry_json"] is None
    assert calls[0]["event"] == 1
    assert solved.picks.filter(~pl.col("sold")).height == 15


def test_run_plan_skips_resolve_when_squad_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    _score(tmp_path)
    monkeypatch.setenv("FPLT_ENTRY_ID", "42")
    get_settings.cache_clear()

    def boom(*_a, **_k):  # noqa: ANN001
        raise AssertionError("resolve_squad_state must not run when squad is set")

    monkeypatch.setattr("fplt.optimise.decisions.resolve_squad_state", boom)
    try:
        run_plan(
            tmp_path,
            "rolling_mean_v1",
            "2024-25",
            1,
            1,
            squad=_state(),
        )
    finally:
        get_settings.cache_clear()


def test_plan_command_writes_the_report_and_rejects_unknown_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _stage(tmp_path)
    _score(tmp_path)
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        main(
            [
                "plan",
                "--model",
                "rolling_mean_v1",
                "--season",
                "2024-25",
                "--gw",
                "1",
                "--horizon",
                "1",
            ]
        )
        printed = capsys.readouterr().out
        saved = list((tmp_path / "plans").rglob("report.md"))
        assert len(saved) == 1
        assert str(saved[0]) in printed
        text = saved[0].read_text(encoding="utf-8")
        assert "### Starting XI" in text
        assert "(C)" in text
        assert "| Name | Pos | Team | Price | xPts |" in text
        assert "| p" in text
        with pytest.raises(SystemExit) as raised:
            main(
                [
                    "plan",
                    "--model",
                    "rolling_mean_v1",
                    "--season",
                    "2024-25",
                    "--gw",
                    "1",
                    "--horizon",
                    "1",
                ]
            )
        assert raised.value.code == 2
        again = capsys.readouterr()
        assert "(C)" in again.out
        assert "### Starting XI" in again.out
        assert "plan already exists" in again.err
        with pytest.raises(PlanExistsError):
            run_plan(tmp_path, "rolling_mean_v1", "2024-25", 1, 1)
        path = run_decisions(tmp_path, "rolling_mean_v1", "2024-25", 1, 2, 1)
        decisions = pl.read_parquet(path)
        assert set(decisions["gw"].unique().to_list()) == {1, 2}
        with pytest.raises(SystemExit):
            main(["plan", "--model", "missing", "--season", "2024-25", "--gw", "1"])
    finally:
        get_settings.cache_clear()


def test_decisions_cli_default_horizon_is_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fplt decisions`` without ``--horizon`` uses ``DEFAULT_PLAN_HORIZON``."""
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    seen: dict[str, object] = {}

    def _bakeoff(*args: object, **kwargs: object) -> pl.DataFrame:
        seen["args"] = args
        seen["kwargs"] = kwargs
        return pl.DataFrame(
            {
                "model_id": ["x"],
                "season": ["2024-25"],
                "total_points": [0],
                "gameweeks": [1],
                "per_gw": [[0]],
            }
        )

    monkeypatch.setattr("fplt.cli.run_decision_bakeoff", _bakeoff)
    try:
        main(
            [
                "decisions",
                "--season",
                "2024-25",
                "--models",
                "rolling_mean_v1",
                "--summarise-only",
            ]
        )
    finally:
        get_settings.cache_clear()
    assert seen["args"][5] == DEFAULT_PLAN_HORIZON
    assert DEFAULT_PLAN_HORIZON == 3


def test_live_chip_arm_default_is_optimised() -> None:
    """Live plan / automation default matches the C6 board winner."""
    assert DEFAULT_LIVE_CHIP_ARM == "optimised"
    assert DEFAULT_PLAN_HORIZON == 3


def test_run_plan_defaults_to_optimised_chip_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_plan`` without ``chip_arm`` schedules windows then enables chips."""
    _stage(tmp_path)
    _score(tmp_path)
    seen: dict[str, object] = {}
    real = plan_squad

    def spy_plan(*args: object, **kwargs: object):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr("fplt.optimise.decisions.plan_squad", spy_plan)
    solved, _path, _report = run_plan(
        tmp_path, "rolling_mean_v1", "2024-25", 1, 1, squad=_state()
    )
    assert seen["season"] == "2024-25"
    assert seen["gameweeks"] == [1]
    assert seen["chip_windows"] is not None
    assert seen["chip_rules_season"] == "2024-25"
    assert len(solved.chips) == 1


def test_run_plan_chip_arm_off_skips_chip_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``chip_arm='off'`` keeps the pre-C6 chips-off ``plan_squad`` call."""
    _stage(tmp_path)
    _score(tmp_path)
    seen: dict[str, object] = {}
    real = plan_squad

    def spy_plan(*args: object, **kwargs: object):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr("fplt.optimise.decisions.plan_squad", spy_plan)
    run_plan(
        tmp_path,
        "rolling_mean_v1",
        "2024-25",
        1,
        1,
        squad=_state(),
        chip_arm="off",
    )
    assert seen["season"] is None
    assert seen["gameweeks"] is None
    assert seen["chip_windows"] is None
    plan_runs = [
        RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "runs").glob("run=*/manifest.json")
    ]
    assert len(plan_runs) == 1
    assert plan_runs[0].run_type == "plan"
    assert (tmp_path / "runs" / f"run={plan_runs[0].run_id}" / "_SUCCESS").exists()


def test_run_plan_forwards_force_play_to_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``force_play=False`` reaches ``_scheduled_windows`` for optimised plans."""
    _stage(tmp_path)
    _score(tmp_path)
    seen: dict[str, object] = {}

    def spy_windows(**kwargs: object):
        seen.update(kwargs)
        from fplt.optimise.chip_windows import ChipUseWindows

        return ChipUseWindows()

    monkeypatch.setattr("fplt.optimise.decisions._scheduled_windows", spy_windows)
    run_plan(
        tmp_path,
        "rolling_mean_v1",
        "2024-25",
        1,
        1,
        squad=_state(),
        force_play=False,
    )
    assert seen["force_play"] is False


def test_run_plan_defaults_to_mean_gw_delta_price_forecast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live ``run_plan`` passes a ``mean_gw_delta`` forecast into ``plan_squad``."""
    _stage(tmp_path)
    _score(tmp_path)
    seen: dict[str, object] = {}
    real = plan_squad

    def spy_plan(*args: object, **kwargs: object):
        seen["price_forecast"] = kwargs.get("price_forecast")
        return real(*args, **kwargs)

    monkeypatch.setattr("fplt.optimise.decisions.plan_squad", spy_plan)
    run_plan(
        tmp_path,
        "rolling_mean_v1",
        "2024-25",
        1,
        1,
        squad=_state(),
        chip_arm="off",
    )
    forecast = seen["price_forecast"]
    assert isinstance(forecast, pl.DataFrame)
    assert forecast.height > 0
    assert set(forecast["step"].to_list()) == {0}


def test_run_plan_flat_price_forecast_skips_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``price_forecast_method=None`` keeps as-of-only ``plan_squad`` cash."""
    _stage(tmp_path)
    _score(tmp_path)
    seen: dict[str, object] = {}
    real = plan_squad

    def spy_plan(*args: object, **kwargs: object):
        seen["price_forecast"] = kwargs.get("price_forecast")
        return real(*args, **kwargs)

    monkeypatch.setattr("fplt.optimise.decisions.plan_squad", spy_plan)
    run_plan(
        tmp_path,
        "rolling_mean_v1",
        "2024-25",
        1,
        1,
        squad=_state(),
        chip_arm="off",
        price_forecast_method=None,
    )
    assert seen["price_forecast"] is None


def test_run_plan_rejects_missing_or_blank_predictions(tmp_path: Path) -> None:
    _stage(tmp_path)
    with pytest.raises(ValueError, match="no predictions"):
        run_plan(tmp_path, "rolling_mean_v1", "2024-25", 1, 1, squad=_state())
    _score(tmp_path)
    with pytest.raises(ValueError, match="gameweek 2"):
        run_plan(tmp_path, "rolling_mean_v1", "2024-25", 1, 2, squad=_state())
    keys = [int(key) for key in _pool()["player_key"].to_list()]
    _score(tmp_path, scores=dict.fromkeys(keys, 0.0))
    with pytest.raises(ValueError, match="all zero"):
        run_plan(tmp_path, "rolling_mean_v1", "2024-25", 1, 1, squad=_state())
    _score(tmp_path, scores={keys[0]: 5.0})
    with pytest.raises(ValueError, match="missing predictions"):
        run_plan(tmp_path, "rolling_mean_v1", "2024-25", 1, 1, squad=_state())


def _state(**overrides: object) -> SquadState:
    values: dict[str, object] = {
        "bank": 1000,
        "free_transfers": 1,
        "ft_bank_cap": 2,
        "hit_cost": 4,
        "decay": 1.0,
        "bench_weight": 0.0,
        "owned": [],
    }
    values.update(overrides)
    return SquadState.model_validate(values)


def _pool() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    player = 1
    for club in range(1, 7):
        for position in ("GK", "DEF", "DEF", "MID", "FWD"):
            rows.append(
                {
                    "player_key": player,
                    "position": position,
                    "team_key": club,
                    "price": 40 + club,
                    "xpts": float(club + (1 if position == "DEF" else 0)),
                }
            )
            player += 1
    return pl.DataFrame(rows)


def _swap_plan(outsider: float):
    pool = _pool()
    selected = select_squad(pool).players
    templates = selected.head(2)
    mirrors = templates.with_columns((pl.col("player_key") + 1000).alias("player_key"))
    candidates = pl.concat(
        [
            pool.select("player_key", "position", "team_key", "price"),
            mirrors.select("player_key", "position", "team_key", "price"),
        ]
    )
    owned_keys = set(selected["player_key"].to_list())
    mirror_keys = set(mirrors["player_key"].to_list())
    points = pl.DataFrame(
        {
            "player_key": candidates["player_key"].to_list(),
            "step": [0] * candidates.height,
            "xpts": [
                outsider if key in mirror_keys else 1.0 if key in owned_keys else 0.0
                for key in candidates["player_key"].to_list()
            ],
        }
    )
    owned = [
        OwnedPlayer(player_key=int(key), selling_price=int(price))
        for key, price in selected.select("player_key", "price").iter_rows()
    ]
    bank = 1000 - int(selected["price"].sum())
    return plan_squad(
        candidates,
        points,
        _state(bank=bank, owned=owned),
        initial=False,
        horizon=1,
    )


def _bank_plan():
    pool = _pool()
    selected = select_squad(pool).players
    templates = selected.head(3)
    keys = templates["player_key"].to_list()
    mirrors = pl.concat(
        [
            templates.head(1).with_columns(
                (pl.col("player_key") + 1000).alias("player_key")
            ),
            templates.tail(2).with_columns(
                (pl.col("player_key") + 2000).alias("player_key")
            ),
        ]
    )
    candidates = pl.concat(
        [
            pool.select("player_key", "position", "team_key", "price"),
            mirrors.select("player_key", "position", "team_key", "price"),
        ]
    )
    owned_keys = set(selected["player_key"].to_list())
    small = {int(keys[0]) + 1000}
    big = {int(key) + 2000 for key in keys[1:]}
    rows: list[dict[str, object]] = []
    for key in candidates["player_key"].to_list():
        for step in (0, 1):
            if key in small:
                xpts = 1.5 if step == 0 else 1.0
            elif key in big:
                # Zero now, so bringing a future star in early sits a worse bench.
                xpts = 0.0 if step == 0 else 3.0
            elif key in owned_keys:
                xpts = 1.0
            else:
                xpts = 0.0
            rows.append({"player_key": key, "step": step, "xpts": xpts})
    owned = [
        OwnedPlayer(player_key=int(key), selling_price=int(price))
        for key, price in selected.select("player_key", "price").iter_rows()
    ]
    return plan_squad(
        candidates,
        pl.DataFrame(rows),
        _state(
            bank=1000 - int(selected["price"].sum()),
            owned=owned,
            bench_weight=0.005,
        ),
        initial=False,
        horizon=2,
    )


def test_an_empty_candidate_pool_skips_the_coverage_check() -> None:
    points = pl.DataFrame({"player_key": [1], "step": [0], "xpts": [1.0]})
    candidates = pl.DataFrame(schema={"player_key": pl.Int64()})
    _require_usable_points(points, candidates, [1], "rolling_mean_v1")


def _score(root: Path, *, scores: dict[int, float] | None = None) -> None:
    path = ledger_path(root)
    if path.exists():
        path.unlink()
    spec = load_model_specs()["rolling_mean_v1"]
    pool = _pool()
    if scores is not None:
        pool = pool.filter(pl.col("player_key").is_in(list(scores)))
    records: list[dict[str, object]] = []
    for row in pool.iter_rows(named=True):
        key = int(row["player_key"])
        records.append(
            {
                "model_id": spec.model_id,
                "run_kind": "live",
                "as_of_season": "2024-25",
                "as_of_gw": 1,
                "target_season": "2024-25",
                "target_gw": 1,
                "horizon": 1,
                "player_key": key,
                "position": row["position"],
                "fixture_id": key,
                "xpts": float(row["xpts"] if scores is None else scores[key]),
                "xpts_sd": None,
                "p_haul": None,
                "p_60": None,
                "actual_pts": None,
                "actual_minutes": None,
            }
        )
    frame = pl.DataFrame(records).with_columns(
        pl.col("xpts_sd").cast(pl.Float64()),
        pl.col("p_haul").cast(pl.Float64()),
        pl.col("p_60").cast(pl.Float64()),
        pl.col("actual_pts").cast(pl.Int64()),
        pl.col("actual_minutes").cast(pl.Int64()),
    )
    append_predictions(root, frame)


def _stage(root: Path) -> None:
    store = DataStore(root)
    players: list[dict[str, object]] = []
    status: list[dict[str, object]] = []
    for row in _pool().iter_rows(named=True):
        key = int(row["player_key"])
        players.append(
            {
                "season": "2024-25",
                "player_key": key,
                "element_id": key,
                "team_key": int(row["team_key"]),
                "position": row["position"],
                "web_name": f"p{key}",
                "available_at": _WHEN,
                "source": "test",
            }
        )
        status.append(
            {
                "season": "2024-25",
                "player_key": key,
                "status": "a",
                "news": "",
                "price": int(row["price"]),
                "ep_next": None,
                "chance_of_playing": None,
                "available_at": _WHEN,
                "source": "test",
            }
        )
    store.write("players", pl.DataFrame(players))
    store.write(
        "player_status",
        pl.DataFrame(status).with_columns(
            pl.col("ep_next").cast(pl.Float64()),
            pl.col("chance_of_playing").cast(pl.Float64()),
        ),
    )
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25", "2024-25"],
                "gw": [1, 2],
                "deadline_time": [
                    datetime(2024, 8, 16, 18, tzinfo=UTC),
                    datetime(2024, 8, 23, 18, tzinfo=UTC),
                ],
                "finished": [False, False],
                "data_checked": [False, False],
                "available_at": [_WHEN, _WHEN],
                "source": ["test", "test"],
            }
        ),
    )
    # Optimised live plans need fixtures for ``schedule_chip_windows``.
    pairs = [(1, 2), (3, 4), (5, 6)]
    fixture_rows: list[dict[str, object]] = []
    fixture_id = 1
    for gw in (1, 2):
        kickoff = datetime(2024, 8, 16 + 7 * (gw - 1), 14, tzinfo=UTC)
        for home, away in pairs:
            fixture_rows.append(
                {
                    "season": "2024-25",
                    "fixture_id": fixture_id,
                    "gw": gw,
                    "team_h_key": home,
                    "team_a_key": away,
                    "kickoff_time": kickoff,
                    "team_h_score": None,
                    "team_a_score": None,
                    "finished": False,
                    "available_at": _WHEN,
                    "source": "test",
                }
            )
            fixture_id += 1
    store.write(
        "fixtures",
        pl.DataFrame(fixture_rows).with_columns(
            pl.col("team_h_score").cast(pl.Int64()),
            pl.col("team_a_score").cast(pl.Int64()),
        ),
    )
