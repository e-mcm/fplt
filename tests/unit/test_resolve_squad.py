"""resolve_squad_state: YAML, entry JSON, and live FPL team."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.data.fpl_client import CurrentTeam
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.optimise.decisions import PastGameweekSquadError, resolve_squad_state
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
    resolved = resolve_squad_state(view, settings=Settings(entry_id=None))
    yaml_state = load_squad_state()
    assert resolved.source == "yaml"
    assert resolved.state.owned == []
    assert resolved.state.bank == yaml_state.bank
    assert resolved.state.free_transfers == yaml_state.free_transfers


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

    resolved = resolve_squad_state(
        view,
        entry_json=entry,
        settings=Settings(entry_id=99, fpl_session="cookie"),
        load_team=boom,
    )
    assert calls == []
    assert resolved.source == "entry_json"
    assert resolved.state.bank == 12
    assert resolved.state.owned[0].player_key == 10
    assert resolved.state.owned[0].selling_price == 55


def test_resolve_applies_my_team_fixture(tmp_path: Path) -> None:
    payload = json.loads((_FIXTURES / "my_team.json").read_text(encoding="utf-8"))
    elements = [int(row["element"]) for row in payload["picks"]]
    view = _view(tmp_path, element_ids=elements)
    team = CurrentTeam.from_my_team(42, payload)

    def load_team(**_: object) -> CurrentTeam:
        return team

    def load_public(entry_id: int, pit_view: object, settings: object, **_: object):
        from fplt.optimise.public_squad import PublicSquadState
        from fplt.optimise.state import ChipInventory

        return PublicSquadState(
            entry_id=entry_id,
            team=team.model_copy(update={"source": "public"}),
            chips=ChipInventory(),
            last_finished=3,
        )

    resolved = resolve_squad_state(
        view,
        event=3,
        settings=Settings(entry_id=42, fpl_session="cookie"),
        load_team=load_team,
        load_public=load_public,
    )
    assert resolved.source == "my_team"
    assert resolved.state.bank == 21
    assert len(resolved.state.owned) == 15
    assert resolved.state.owned[0].player_key == elements[0]
    assert resolved.state.owned[0].selling_price == 46
    assert resolved.state.owned[0].purchase_price == 46
    assert resolved.state.free_transfers == 1


def test_resolve_no_token_uses_public_not_raw_picks(tmp_path: Path) -> None:
    """Regression: entry_id set, no token → public path (not ValueError)."""
    payload = json.loads((_FIXTURES / "entry_picks.json").read_text(encoding="utf-8"))
    elements = [int(row["element"]) for row in payload["picks"]]
    view = _view(tmp_path, element_ids=elements)
    priced = CurrentTeam.from_entry_picks(1, 9, payload).model_copy(
        update={
            "source": "public",
            "free_transfers": 2,
            "picks": [
                pick.model_copy(
                    update={
                        "selling_price": 40 + index,
                        "purchase_price": 40 + index,
                    }
                )
                for index, pick in enumerate(
                    CurrentTeam.from_entry_picks(1, 9, payload).picks
                )
            ],
        }
    )

    def load_public(entry_id: int, pit_view: object, settings: object, **_: object):
        from fplt.optimise.public_squad import PublicSquadState
        from fplt.optimise.state import ChipInventory

        return PublicSquadState(
            entry_id=entry_id,
            team=priced,
            chips=ChipInventory(),
            last_finished=9,
        )

    resolved = resolve_squad_state(
        view,
        settings=Settings(entry_id=1, fpl_session=None, fpl_api_token=None),
        load_public=load_public,
    )
    assert resolved.source == "public"
    assert resolved.state.bank == int(payload["entry_history"]["bank"])
    assert len(resolved.state.owned) == 15
    assert resolved.state.free_transfers == 2


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


def test_resolve_refuses_past_gw_from_live_public_squad(tmp_path: Path) -> None:
    """#381: as_of_gw <= last_finished must not silently use the live squad."""
    payload = json.loads((_FIXTURES / "entry_picks.json").read_text(encoding="utf-8"))
    elements = [int(row["element"]) for row in payload["picks"]]
    view = _view(tmp_path, element_ids=elements)
    priced = CurrentTeam.from_entry_picks(1, 12, payload).model_copy(
        update={"source": "public"}
    )
    public_calls: list[int] = []

    def load_public(entry_id: int, pit_view: object, settings: object, **_: object):
        from fplt.optimise.public_squad import PublicSquadState
        from fplt.optimise.state import ChipInventory

        public_calls.append(entry_id)
        return PublicSquadState(
            entry_id=entry_id,
            team=priced,
            chips=ChipInventory(),
            last_finished=12,
        )

    def boom_team(**_: object) -> CurrentTeam:
        raise AssertionError("must refuse before my-team when as_of_gw is past")

    with pytest.raises(PastGameweekSquadError, match="as_of_gw=1") as exc_info:
        resolve_squad_state(
            view,
            as_of_gw=1,
            settings=Settings(entry_id=1, fpl_session="cookie"),
            load_team=boom_team,
            load_public=load_public,
        )
    assert exc_info.value.as_of_gw == 1
    assert exc_info.value.last_finished == 12
    assert public_calls == [1]

    # Equal to last_finished is also past (post-deadline squad).
    with pytest.raises(PastGameweekSquadError, match="as_of_gw=12"):
        resolve_squad_state(
            view,
            as_of_gw=12,
            settings=Settings(entry_id=1, fpl_api_token=None, fpl_session=None),
            load_public=load_public,
        )


def test_resolve_allows_upcoming_gw_after_last_finished(tmp_path: Path) -> None:
    """#381: upcoming deadline (as_of_gw > last_finished) may use the live squad."""
    payload = json.loads((_FIXTURES / "my_team.json").read_text(encoding="utf-8"))
    elements = [int(row["element"]) for row in payload["picks"]]
    view = _view(tmp_path, element_ids=elements)
    team = CurrentTeam.from_my_team(42, payload)

    def load_public(entry_id: int, pit_view: object, settings: object, **_: object):
        from fplt.optimise.public_squad import PublicSquadState
        from fplt.optimise.state import ChipInventory

        return PublicSquadState(
            entry_id=entry_id,
            team=team.model_copy(update={"source": "public"}),
            chips=ChipInventory(),
            last_finished=12,
        )

    resolved = resolve_squad_state(
        view,
        as_of_gw=13,
        event=13,
        settings=Settings(entry_id=42, fpl_session="cookie"),
        load_team=lambda **_: team,
        load_public=load_public,
    )
    assert resolved.source == "my_team"
    assert len(resolved.state.owned) == 15


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

    def load_public(entry_id: int, pit_view: object, settings: object, **_: object):
        from fplt.optimise.public_squad import PublicSquadState
        from fplt.optimise.state import ChipInventory

        return PublicSquadState(
            entry_id=entry_id,
            team=team.model_copy(update={"source": "public", "free_transfers": 1}),
            chips=ChipInventory(),
            last_finished=3,
        )

    resolved = resolve_squad_state(
        view,
        settings=Settings(entry_id=42, fpl_session="cookie"),
        load_team=lambda **_: team,
        load_public=load_public,
    )
    assert resolved.state.free_transfers == yaml_state.free_transfers


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
