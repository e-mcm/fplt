"""Public squad derivation: selling prices, FT replay, token fallback."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fplt.data.fpl_client import CurrentTeam, FPLAPIError, FPLClient
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.optimise.decisions import resolve_squad_state
from fplt.optimise.prices import fpl_selling_price
from fplt.optimise.public_squad import (
    compare_public_to_my_team,
    public_squad_state,
    replay_free_transfers,
)
from fplt.scoring.chips import ChipRules
from fplt.settings import Settings
from fplt.types import GWKey

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "fpl_api"
_WHEN = datetime(2024, 8, 1, tzinfo=UTC)
_API = "https://fantasy.premierleague.com/api"


def _load(name: str) -> dict | list:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _view(
    root: Path,
    *,
    element_ids: list[int],
    prices: dict[int, int] | None = None,
    season: str = "2024-25",
    as_of_gw: int = 10,
) -> object:
    store = DataStore(root)
    price_map = prices or {}
    players: list[dict[str, object]] = []
    status: list[dict[str, object]] = []
    for index, element in enumerate(element_ids, start=1):
        price = int(price_map.get(element, 40 + index))
        players.append(
            {
                "season": season,
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
                "season": season,
                "player_key": element,
                "status": "a",
                "news": "",
                "price": price,
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
    gws = list(range(1, as_of_gw + 1))
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": [season] * len(gws),
                "gw": gws,
                "deadline_time": [
                    datetime(2024, 8, 1 + gw, 18, tzinfo=UTC) for gw in gws
                ],
                "finished": [gw < as_of_gw for gw in gws],
                "data_checked": [gw < as_of_gw for gw in gws],
                "available_at": [_WHEN] * len(gws),
                "source": ["test"] * len(gws),
            }
        ),
    )
    return PITData(store).view(GWKey(season, as_of_gw))


def _public_transport(
    *,
    entry: dict,
    picks: dict,
    history: dict,
    transfers: list,
    my_team: dict | None = None,
    my_team_status: int = 200,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/my-team/42/") or path.endswith("/my-team/1/"):
            if my_team is None:
                return httpx.Response(my_team_status, json={"detail": "forbidden"})
            return httpx.Response(my_team_status, json=my_team)
        if "/entry/" in path and path.endswith("/history/"):
            return httpx.Response(200, json=history)
        if "/entry/" in path and path.endswith("/transfers/"):
            return httpx.Response(200, json=transfers)
        if "/entry/" in path and "/event/" in path and path.endswith("/picks/"):
            return httpx.Response(200, json=picks)
        if "/entry/" in path and path.endswith("/"):
            return httpx.Response(200, json=entry)
        raise AssertionError(path)

    return httpx.MockTransport(handler)


def test_fpl_selling_price_worked_examples() -> None:
    # Bought 5.0 → now 5.3 → sell 5.1; now 4.8 → sell 4.8 (tenths).
    assert fpl_selling_price(50, 53) == 51
    assert fpl_selling_price(50, 48) == 48


@given(
    purchase=st.integers(min_value=0, max_value=200),
    current=st.integers(min_value=0, max_value=200),
)
@settings(max_examples=100)
def test_fpl_selling_price_properties(purchase: int, current: int) -> None:
    sell = fpl_selling_price(purchase, current)
    assert sell <= current
    assert sell >= min(current, purchase)
    # Monotone in ``now``: raising current never lowers sell.
    higher = fpl_selling_price(purchase, current + 1)
    assert higher >= sell


def test_replay_free_transfers_wildcard_freehit_and_cap() -> None:
    history = _load("entry_history.json")
    assert isinstance(history, dict)
    rules = ChipRules.for_season("2024-25")
    free = replay_free_transfers(
        list(history["current"]),
        list(history["chips"]),
        ft_bank_cap=5,
        rules=rules,
        started_event=1,
    )
    # See module notes in fixture: ends at 3 after GW9 (cap respected at GW8).
    assert free == 3


def test_owned_set_applies_pending_transfers(tmp_path: Path) -> None:
    entry = _load("entry.json")
    picks = _load("entry_picks.json")
    history = _load("entry_history_agree.json")
    transfers = _load("entry_transfers_pending.json")
    assert isinstance(entry, dict)
    assert isinstance(picks, dict)
    assert isinstance(history, dict)
    assert isinstance(transfers, list)
    entry = {**entry, "id": 42, "current_event": 10}
    elements = [int(row["element"]) for row in picks["picks"]] + [999]
    my = _load("my_team.json")
    assert isinstance(my, dict)
    prices = {int(row["element"]): int(row["purchase_price"]) for row in my["picks"]}
    prices[999] = 55
    view = _view(tmp_path, element_ids=elements, prices=prices)
    http = httpx.Client(
        transport=_public_transport(
            entry=entry,
            picks=picks,
            history=history,
            transfers=transfers,
        ),
        base_url=_API,
    )
    try:
        public = public_squad_state(
            42,
            view,
            Settings(entry_id=42),
            client=http,  # type: ignore[arg-type]
        )
    finally:
        http.close()
    owned = {pick.element for pick in public.team.picks}
    assert 999 in owned
    assert 368 not in owned
    assert public.team.bank == int(picks["entry_history"]["bank"]) + 60 - 55


def test_public_agrees_with_my_team_fixture(tmp_path: Path) -> None:
    entry = {**_load("entry.json"), "id": 42, "current_event": 10}  # type: ignore[misc]
    picks = _load("entry_picks.json")
    history = _load("entry_history_agree.json")
    transfers = _load("entry_transfers_agree.json")
    my = _load("my_team.json")
    assert isinstance(picks, dict)
    assert isinstance(history, dict)
    assert isinstance(transfers, list)
    assert isinstance(my, dict)
    elements = [int(row["element"]) for row in my["picks"]]
    prices = {int(row["element"]): int(row["purchase_price"]) for row in my["picks"]}
    view = _view(tmp_path, element_ids=elements, prices=prices)
    http = httpx.Client(
        transport=_public_transport(
            entry=entry,  # type: ignore[arg-type]
            picks=picks,
            history=history,
            transfers=transfers,
            my_team=my,
        ),
        base_url=_API,
    )
    try:
        public = public_squad_state(
            42,
            view,
            Settings(entry_id=42),
            client=http,  # type: ignore[arg-type]
        )
        live = CurrentTeam.from_my_team(42, my)
    finally:
        http.close()
    assert compare_public_to_my_team(public, live) == []
    assert public.team.free_transfers == live.free_transfers == 1
    assert public.team.bank == live.bank


def test_resolve_no_token_uses_public_path(tmp_path: Path) -> None:
    entry = {**_load("entry.json"), "id": 42, "current_event": 10}  # type: ignore[misc]
    picks = _load("entry_picks.json")
    history = _load("entry_history_agree.json")
    transfers = _load("entry_transfers_agree.json")
    my = _load("my_team.json")
    assert isinstance(picks, dict)
    assert isinstance(history, dict)
    assert isinstance(transfers, list)
    assert isinstance(my, dict)
    elements = [int(row["element"]) for row in my["picks"]]
    prices = {int(row["element"]): int(row["purchase_price"]) for row in my["picks"]}
    view = _view(tmp_path, element_ids=elements, prices=prices)
    http = httpx.Client(
        transport=_public_transport(
            entry=entry,  # type: ignore[arg-type]
            picks=picks,
            history=history,
            transfers=transfers,
        ),
        base_url=_API,
    )

    def load_public(entry_id: int, pit_view: object, settings: Settings, **_: object):
        return public_squad_state(
            entry_id,
            pit_view,
            settings,
            client=http,  # type: ignore[arg-type]
        )

    try:
        resolved = resolve_squad_state(
            view,  # type: ignore[arg-type]
            settings=Settings(entry_id=42, fpl_api_token=None, fpl_session=None),
            load_public=load_public,
        )
    finally:
        http.close()
    assert resolved.source == "public"
    assert len(resolved.state.owned) == 15
    assert resolved.state.bank == 21
    assert all(p.selling_price > 0 for p in resolved.state.owned)


def test_resolve_my_team_401_falls_back_to_public(tmp_path: Path) -> None:
    entry = {**_load("entry.json"), "id": 42, "current_event": 10}  # type: ignore[misc]
    picks = _load("entry_picks.json")
    history = _load("entry_history_agree.json")
    transfers = _load("entry_transfers_agree.json")
    my = _load("my_team.json")
    assert isinstance(picks, dict)
    assert isinstance(history, dict)
    assert isinstance(transfers, list)
    assert isinstance(my, dict)
    elements = [int(row["element"]) for row in my["picks"]]
    prices = {int(row["element"]): int(row["purchase_price"]) for row in my["picks"]}
    view = _view(tmp_path, element_ids=elements, prices=prices)
    http = httpx.Client(
        transport=_public_transport(
            entry=entry,  # type: ignore[arg-type]
            picks=picks,
            history=history,
            transfers=transfers,
            my_team_status=401,
        ),
        base_url=_API,
    )

    def load_team(**_: object) -> CurrentTeam:
        raise FPLAPIError(
            "FPL API forbidden for /my-team/42/; live picks need a fresh "
            "FPLT_FPL_API_TOKEN (uv run fplt auth set)"
        )

    def load_public(entry_id: int, pit_view: object, settings: Settings, **_: object):
        return public_squad_state(
            entry_id,
            pit_view,
            settings,
            client=http,  # type: ignore[arg-type]
        )

    try:
        resolved = resolve_squad_state(
            view,  # type: ignore[arg-type]
            settings=Settings(entry_id=42, fpl_api_token="eyJexpired"),
            load_team=load_team,
            load_public=load_public,
        )
    finally:
        http.close()
    assert resolved.source == "public"
    assert len(resolved.state.owned) == 15


def test_mismatch_logs_warning_uses_my_team(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    my = _load("my_team.json")
    assert isinstance(my, dict)
    elements = [int(row["element"]) for row in my["picks"]]
    view = _view(tmp_path, element_ids=elements)
    live = CurrentTeam.from_my_team(42, my)

    def load_team(**_: object) -> CurrentTeam:
        return live

    def load_public(entry_id: int, pit_view: object, settings: Settings, **_: object):
        # Deliberately disagree on bank.
        team = live.model_copy(update={"bank": live.bank + 7, "source": "public"})
        from fplt.optimise.public_squad import PublicSquadState
        from fplt.optimise.state import ChipInventory

        return PublicSquadState(
            entry_id=entry_id,
            team=team,
            chips=ChipInventory(),
            last_finished=9,
        )

    with caplog.at_level(logging.WARNING):
        resolved = resolve_squad_state(
            view,  # type: ignore[arg-type]
            settings=Settings(entry_id=42, fpl_api_token="eyJok", fpl_session=None),
            load_team=load_team,
            load_public=load_public,
        )
    assert resolved.source == "my_team"
    assert resolved.state.bank == live.bank
    assert any("mismatch" in record.message for record in caplog.records)


def test_current_team_falls_back_on_403() -> None:
    entry = {**_load("entry.json"), "current_event": 9}  # type: ignore[misc]
    picks = _load("entry_picks.json")
    assert isinstance(picks, dict)

    def handler(request: httpx.Request) -> httpx.Response:
        if "/my-team/" in request.url.path:
            return httpx.Response(403, json={"detail": "forbidden"})
        if request.url.path.endswith("/entry/1/"):
            return httpx.Response(200, json=entry)
        if "/event/9/picks/" in request.url.path:
            return httpx.Response(200, json=picks)
        raise AssertionError(request.url.path)

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url=_API)
    api = FPLClient(http, api_token="eyJstale", base_url=_API)
    try:
        team = api.current_team(1)
    finally:
        api.close()
        http.close()
    assert team.source == "entry_picks"
    assert team.event == 9


def test_apply_still_requires_token() -> None:
    api = FPLClient(api_token=None)
    try:
        with pytest.raises(FPLAPIError, match="FPLT_FPL_API_TOKEN"):
            api.submit_transfers(
                {"entry": 1, "event": 1, "transfers": [], "chip": None}
            )
        with pytest.raises(FPLAPIError, match="FPLT_FPL_API_TOKEN"):
            api.submit_lineup(1, {"picks": [], "chip": None})
    finally:
        api.close()


def test_replay_rejects_bad_cap_and_skips_pre_start() -> None:
    rules = ChipRules.for_season("2024-25")
    with pytest.raises(ValueError, match="ft_bank_cap"):
        replay_free_transfers([], [], ft_bank_cap=0, rules=rules)
    free = replay_free_transfers(
        [
            {"event": 1, "event_transfers": 0},
            {"event": 2, "event_transfers": 0},
        ],
        [],
        ft_bank_cap=5,
        rules=rules,
        started_event=2,
    )
    # GW1 skipped; GW2: 1→2
    assert free == 2


def test_replay_freehit_without_keep_resets() -> None:
    rules = ChipRules.for_season("2024-25").model_copy(
        update={"free_hit_keeps_free_transfers": False}
    )
    free = replay_free_transfers(
        [
            {"event": 1, "event_transfers": 0},
            {"event": 2, "event_transfers": 3},
        ],
        [{"name": "freehit", "event": 2}],
        ft_bank_cap=5,
        rules=rules,
    )
    assert free == 1


def test_replay_hits_when_moves_exceed_free() -> None:
    rules = ChipRules.for_season("2024-25")
    free = replay_free_transfers(
        [{"event": 1, "event_transfers": 3}],
        [],
        ft_bank_cap=5,
        rules=rules,
    )
    assert free == 1


def test_chips_from_history_and_aliases() -> None:
    from fplt.optimise.public_squad import (
        _normalize_chip,
        chips_from_history,
    )

    rules = ChipRules.for_season("2024-25")
    inv = chips_from_history(
        {
            "chips": [
                {"name": "wildcard", "event": 3},
                {"name": "Bench Boost", "event": 8},
            ]
        },
        rules,
    )
    assert inv.remaining["wildcard"] == rules.remaining_counts()["wildcard"] - 1
    assert inv.remaining["bboost"] == rules.remaining_counts()["bboost"] - 1
    assert inv.played[0].name == "wildcard"
    assert _normalize_chip("tc") == "3xc"
    assert _normalize_chip("fh") == "freehit"
    with pytest.raises(ValueError, match="unknown chip"):
        _normalize_chip("notachip")
    with pytest.raises(ValueError, match="exhausted"):
        chips_from_history(
            {
                "chips": [
                    {"name": "wildcard", "event": 3},
                    {"name": "wildcard", "event": 4},
                    {"name": "wildcard", "event": 5},
                ]
            },
            rules,
        )


def test_compare_mismatches_ft_owned_and_sell() -> None:
    from fplt.optimise.public_squad import PublicSquadState
    from fplt.optimise.state import ChipInventory

    base = CurrentTeam.from_my_team(42, _load("my_team.json"))  # type: ignore[arg-type]
    public = PublicSquadState(
        entry_id=42,
        team=base.model_copy(update={"source": "public", "free_transfers": 3}),
        chips=ChipInventory(),
        last_finished=9,
    )
    lines = compare_public_to_my_team(public, base)
    assert any("free_transfers" in line for line in lines)

    other_picks = list(base.picks)
    other_picks[0] = other_picks[0].model_copy(update={"selling_price": 99})
    sell_mismatch = PublicSquadState(
        entry_id=42,
        team=base.model_copy(update={"source": "public", "picks": other_picks}),
        chips=ChipInventory(),
        last_finished=9,
    )
    assert any(
        "selling_price" in line
        for line in compare_public_to_my_team(sell_mismatch, base)
    )

    owned_mismatch = PublicSquadState(
        entry_id=42,
        team=base.model_copy(
            update={
                "source": "public",
                "picks": base.picks[:14],
            }
        ),
        chips=ChipInventory(),
        last_finished=9,
    )
    assert any(
        "owned elements" in line
        for line in compare_public_to_my_team(owned_mismatch, base)
    )


def test_ft_after_pending_edges() -> None:
    from fplt.optimise.public_squad import _ft_after_pending

    assert _ft_after_pending(3, 0, 5) == 3
    assert _ft_after_pending(3, 2, 5) == 1
    assert _ft_after_pending(2, 5, 5) == 1


def test_owned_after_pending_errors() -> None:
    from fplt.optimise.public_squad import _owned_after_pending

    with pytest.raises(FPLAPIError, match="picks and entry_history"):
        _owned_after_pending({}, [])
    with pytest.raises(FPLAPIError, match="entry_history.bank"):
        _owned_after_pending({"picks": [], "entry_history": {}}, [])
    with pytest.raises(FPLAPIError, match="not in last-finished"):
        _owned_after_pending(
            {
                "picks": [{"element": 1, "position": 1}],
                "entry_history": {"bank": 10},
            },
            [
                {
                    "element_in": 2,
                    "element_out": 99,
                    "element_in_cost": 50,
                    "element_out_cost": 40,
                    "event": 10,
                    "time": "t",
                }
            ],
        )


def test_history_parse_errors() -> None:
    from fplt.optimise.public_squad import _history_chips, _history_current

    with pytest.raises(FPLAPIError, match="current"):
        _history_current({})
    assert _history_chips({}) == []
    with pytest.raises(FPLAPIError, match="chips"):
        _history_chips({"chips": "nope"})


def test_purchase_from_deadline_when_no_transfer_in(tmp_path: Path) -> None:
    """Players never transferred in use PIT price at started_event deadline."""
    entry = {**_load("entry.json"), "id": 42, "current_event": 10}  # type: ignore[misc]
    picks = _load("entry_picks.json")
    history = _load("entry_history_agree.json")
    assert isinstance(picks, dict)
    assert isinstance(history, dict)
    # No transfers → purchase from GW1 deadline prices.
    elements = [int(row["element"]) for row in picks["picks"]]
    prices = {el: 40 + i for i, el in enumerate(elements)}
    view = _view(tmp_path, element_ids=elements, prices=prices)
    http = httpx.Client(
        transport=_public_transport(
            entry=entry,  # type: ignore[arg-type]
            picks=picks,
            history=history,
            transfers=[],
        ),
        base_url=_API,
    )
    try:
        public = public_squad_state(
            42,
            view,
            Settings(entry_id=42),
            client=http,  # type: ignore[arg-type]
        )
    finally:
        http.close()
    assert public.team.picks[0].purchase_price == prices[elements[0]]


def test_resolve_token_present_but_entry_picks_uses_public(tmp_path: Path) -> None:
    payload = _load("entry_picks.json")
    assert isinstance(payload, dict)
    elements = [int(row["element"]) for row in payload["picks"]]
    view = _view(tmp_path, element_ids=elements)
    team = CurrentTeam.from_entry_picks(1, 9, payload)

    def load_team(**_: object) -> CurrentTeam:
        return team

    def load_public(entry_id: int, pit_view: object, settings: object, **_: object):
        from fplt.optimise.public_squad import PublicSquadState
        from fplt.optimise.state import ChipInventory

        priced = [
            pick.model_copy(update={"selling_price": 50, "purchase_price": 50})
            for pick in team.picks
        ]
        return PublicSquadState(
            entry_id=entry_id,
            team=team.model_copy(
                update={
                    "source": "public",
                    "picks": priced,
                    "free_transfers": 2,
                }
            ),
            chips=ChipInventory(),
            last_finished=9,
        )

    resolved = resolve_squad_state(
        view,  # type: ignore[arg-type]
        settings=Settings(entry_id=1, fpl_api_token="eyJok"),
        load_team=load_team,
        load_public=load_public,
    )
    assert resolved.source == "public"
    assert resolved.state.free_transfers == 2


def test_resolve_non_auth_error_propagates(tmp_path: Path) -> None:
    view = _view(tmp_path, element_ids=list(range(1, 16)))

    def load_team(**_: object) -> CurrentTeam:
        raise FPLAPIError("FPL API HTTP 500 for /my-team/1/: boom")

    with pytest.raises(FPLAPIError, match="HTTP 500"):
        resolve_squad_state(
            view,  # type: ignore[arg-type]
            settings=Settings(entry_id=1, fpl_api_token="eyJok"),
            load_team=load_team,
        )


def test_cross_check_skip_on_public_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    my = _load("my_team.json")
    assert isinstance(my, dict)
    elements = [int(row["element"]) for row in my["picks"]]
    view = _view(tmp_path, element_ids=elements)
    live = CurrentTeam.from_my_team(42, my)

    def load_team(**_: object) -> CurrentTeam:
        return live

    def load_public(*_a: object, **_k: object):
        raise FPLAPIError("public boom")

    with caplog.at_level(logging.WARNING):
        resolved = resolve_squad_state(
            view,  # type: ignore[arg-type]
            settings=Settings(entry_id=42, fpl_api_token="eyJok"),
            load_team=load_team,
            load_public=load_public,
        )
    assert resolved.source == "my_team"
    assert any("cross-check skipped" in r.message for r in caplog.records)


def test_entry_transfers_error_paths() -> None:
    cases = [
        (401, "forbidden"),
        (404, "not found"),
        (500, "HTTP 500"),
    ]
    for status, match in cases:

        def handler(request: httpx.Request, *, _status: int = status) -> httpx.Response:
            return httpx.Response(_status, text="err")

        http = httpx.Client(transport=httpx.MockTransport(handler), base_url=_API)
        api = FPLClient(http, base_url=_API)
        try:
            with pytest.raises(FPLAPIError, match=match):
                api.entry_transfers(1)
        finally:
            api.close()
            http.close()

    def bad_json(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    http = httpx.Client(transport=httpx.MockTransport(bad_json), base_url=_API)
    api = FPLClient(http, base_url=_API)
    try:
        with pytest.raises(FPLAPIError, match="non-JSON"):
            api.entry_transfers(1)
    finally:
        api.close()
        http.close()

    def not_list(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nope": True})

    http = httpx.Client(transport=httpx.MockTransport(not_list), base_url=_API)
    api = FPLClient(http, base_url=_API)
    try:
        with pytest.raises(FPLAPIError, match="expected list"):
            api.entry_transfers(1)
    finally:
        api.close()
        http.close()


def test_entry_history_ok() -> None:
    history = _load("entry_history.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/history/")
        return httpx.Response(200, json=history)

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url=_API)
    api = FPLClient(http, base_url=_API)
    try:
        body = api.entry_history(1)
    finally:
        api.close()
        http.close()
    assert "current" in body
    assert len(body["chips"]) == 2


def test_current_team_rethrows_non_auth_my_team_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url=_API)
    api = FPLClient(http, api_token="eyJok", base_url=_API)
    try:
        with pytest.raises(FPLAPIError, match="HTTP 500"):
            api.current_team(1)
    finally:
        api.close()
        http.close()
