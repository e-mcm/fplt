"""Transfer plans: hits, banking, and the decision ledger."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.cli import main
from fplt.data.store import DataStore
from fplt.optimise.decisions import (
    points_from_ledger,
    run_decisions,
    run_plan,
    simulate_decisions,
    write_decisions,
)
from fplt.optimise.plan import plan_squad, plan_table, render_report, write_plan
from fplt.optimise.squad import select_squad
from fplt.optimise.state import (
    OwnedPlayer,
    SquadState,
    apply_entry_picks,
    load_squad_state,
)
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
    assert "hit 0" in report


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
    assert path.name == "part.parquet"
    with pytest.raises(FileExistsError):
        write_plan(path.parents[3], frame)


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
    actuals = pl.DataFrame({"gw": [1, 2], "player_key": [1, 1], "actual_pts": [3, 5]})
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


def test_plan_command_writes_the_report_and_rejects_unknown_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
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
        saved = list((tmp_path / "plans").rglob("report.md"))
        assert len(saved) == 1
        assert "captain" in saved[0].read_text(encoding="utf-8")
        with pytest.raises(FileExistsError):
            run_plan(tmp_path, "rolling_mean_v1", "2024-25", 1, 1)
        path = run_decisions(tmp_path, "rolling_mean_v1", "2024-25", 1, 2, 1)
        decisions = pl.read_parquet(path)
        assert set(decisions["gw"].unique().to_list()) == {1, 2}
        with pytest.raises(SystemExit):
            main(["plan", "--model", "missing", "--season", "2024-25", "--gw", "1"])
    finally:
        get_settings.cache_clear()


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
