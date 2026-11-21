"""resolve_squad_state: YAML, entry JSON, and live FPL team."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fplt.data.fpl_client import CurrentTeam
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.optimise.decisions import resolve_squad_state
from fplt.optimise.state import (
    OwnedPlayer,
    SquadState,
    load_squad_state,
    rebuild_squad_state,
)
from fplt.settings import Settings
from fplt.types import GWKey

_WHEN = datetime(2024, 8, 1, tzinfo=UTC)
_FIXTURES = Path(__file__).parents[1] / "fixtures" / "fpl_api"


def test_resolve_keeps_yaml_without_entry_id(tmp_path: Path) -> None:
    view = _view(tmp_path, element_ids=list(range(1, 16)))
    state = resolve_squad_state(view, settings=Settings(entry_id=None))
    yaml_state = load_squad_state()
    assert state.owned == []
    assert state.bank == yaml_state.bank
    assert state.free_transfers == yaml_state.free_transfers


def test_resolve_prefers_entry_json_over_live_entry(tmp_path: Path) -> None:
    view = _view(tmp_path, element_ids=[10])
    entry = tmp_path / "picks.json"
    entry.write_text(
        json.dumps(
            {
                "picks": [{"element": 10, "selling_price": 55}],
                "entry_history": {"bank": 12},
            }
        ),
        encoding="utf-8",
    )
    calls: list[object] = []

    def boom(**_: object) -> CurrentTeam:
        calls.append(True)
        raise AssertionError("live team must not load when entry_json is set")

    state = resolve_squad_state(
        view,
        entry_json=entry,
        settings=Settings(entry_id=99, fpl_session="cookie"),
        load_team=boom,
    )
    assert calls == []
    assert state.bank == 12
    assert state.owned[0].player_key == 10
    assert state.owned[0].selling_price == 55


def test_resolve_applies_my_team_fixture(tmp_path: Path) -> None:
    payload = json.loads((_FIXTURES / "my_team.json").read_text(encoding="utf-8"))
    elements = [int(row["element"]) for row in payload["picks"]]
    view = _view(tmp_path, element_ids=elements)
    team = CurrentTeam.from_my_team(42, payload)

    def load_team(**_: object) -> CurrentTeam:
        return team

    state = resolve_squad_state(
        view,
        event=3,
        settings=Settings(entry_id=42, fpl_session="cookie"),
        load_team=load_team,
    )
    assert state.bank == 21
    assert len(state.owned) == 15
    assert state.owned[0].player_key == elements[0]
    assert state.owned[0].selling_price == 46
    assert state.free_transfers == 1


def test_resolve_keeps_yaml_free_transfers_for_public_picks(tmp_path: Path) -> None:
    payload = json.loads((_FIXTURES / "entry_picks.json").read_text(encoding="utf-8"))
    # Fixture public picks lack selling_price; inject so apply_entry_picks works.
    for index, pick in enumerate(payload["picks"]):
        pick["selling_price"] = 40 + index
    elements = [int(row["element"]) for row in payload["picks"]]
    view = _view(tmp_path, element_ids=elements)
    team = CurrentTeam.from_entry_picks(1, 9, payload)
    yaml_state = load_squad_state()

    def load_team(**_: object) -> CurrentTeam:
        return team

    state = resolve_squad_state(
        view,
        settings=Settings(entry_id=1, fpl_session=None),
        load_team=load_team,
    )
    assert state.bank == int(payload["entry_history"]["bank"])
    assert len(state.owned) == 15
    assert state.free_transfers == yaml_state.free_transfers


def test_rebuild_folds_selling_prices_into_bank() -> None:
    state = SquadState(
        bank=21,
        free_transfers=2,
        ft_bank_cap=5,
        hit_cost=4,
        decay=0.9,
        bench_weight=0.0001,
        owned=[
            OwnedPlayer(player_key=1, selling_price=100),
            OwnedPlayer(player_key=2, selling_price=55),
        ],
    )
    rebuilt = rebuild_squad_state(state)
    assert rebuilt.owned == []
    assert rebuilt.bank == 176
    assert rebuilt.free_transfers == 2
    assert rebuild_squad_state(rebuilt) == rebuilt


def test_resolve_ignores_free_transfers_outside_cap(tmp_path: Path) -> None:
    payload = json.loads((_FIXTURES / "my_team.json").read_text(encoding="utf-8"))
    payload = {
        **payload,
        "transfers": {**payload["transfers"], "limit": 9},
    }
    elements = [int(row["element"]) for row in payload["picks"]]
    view = _view(tmp_path, element_ids=elements)
    team = CurrentTeam.from_my_team(42, payload)
    yaml_state = load_squad_state()

    state = resolve_squad_state(
        view,
        settings=Settings(entry_id=42, fpl_session="cookie"),
        load_team=lambda **_: team,
    )
    assert state.free_transfers == yaml_state.free_transfers


def _view(root: Path, *, element_ids: list[int]):
    store = DataStore(root)
    players: list[dict[str, object]] = []
    status: list[dict[str, object]] = []
    for index, element in enumerate(element_ids, start=1):
        players.append(
            {
                "season": "2024-25",
                "player_key": element,
                "element_id": element,
                "team_key": (index % 6) + 1,
                "position": ("GK", "DEF", "MID", "FWD")[index % 4],
                "web_name": f"p{element}",
                "available_at": _WHEN,
                "source": "test",
            }
        )
        status.append(
            {
                "season": "2024-25",
                "player_key": element,
                "status": "a",
                "news": "",
                "price": 40 + index,
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
                "season": ["2024-25"],
                "gw": [1],
                "deadline_time": [datetime(2024, 8, 16, 18, tzinfo=UTC)],
                "finished": [False],
                "data_checked": [False],
                "available_at": [_WHEN],
                "source": ["test"],
            }
        ),
    )
    return PITData(store).view(GWKey("2024-25", 1))
