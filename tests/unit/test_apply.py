"""Apply a saved plan to FPL transfer / lineup payloads."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl
import pytest

from fplt.cli import main
from fplt.data.fpl_client import CurrentTeam, FPLClient
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.apply import (
    ApplyError,
    ApplyPayload,
    ApplyResult,
    TransferMove,
    apply_plan,
    build_apply_payload,
    format_apply_preview,
)
from fplt.optimise.plan import plan_parquet, write_plan
from fplt.settings import Settings
from fplt.types import GWKey

_WHEN = datetime(2024, 8, 1, tzinfo=UTC)
_FIXTURES = Path(__file__).parents[1] / "fixtures" / "fpl_api"
_API = "https://fantasy.premierleague.com/api"
_MODEL = "rolling_mean_v1"
_SEASON = "2024-25"
_GW = 1
_HORIZON = 1


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _positions(count: int) -> list[str]:
    base = ["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    if count <= 15:
        return base[:count]
    return base + ["MID"] * (count - 15)


def _plan_rows(
    *,
    elements: list[int],
    bought: list[bool],
    sold: list[bool],
    in_xi: list[bool] | None = None,
    captain_at: int = 0,
    vice_at: int = 1,
) -> pl.DataFrame:
    spec = load_model_specs()[_MODEL]
    positions = _positions(len(elements))
    if in_xi is None:
        in_xi = [index < 11 for index in range(len(elements))]
    rows: list[dict[str, object]] = []
    for index, element in enumerate(elements):
        rows.append(
            {
                "model_id": spec.model_id,
                "as_of_season": _SEASON,
                "as_of_gw": _GW,
                "target_gw": _GW,
                "step": 0,
                "player_key": element,
                "position": positions[index],
                "team_key": (index % 6) + 1,
                "in_xi": in_xi[index],
                "captain": index == captain_at and not sold[index],
                "vice": index == vice_at and not sold[index],
                "bought": bought[index],
                "sold": sold[index],
                "xpts": float(index),
                "hit": 0.0,
            }
        )
    return pl.DataFrame(rows)


def _swap_plan(team: CurrentTeam, buy_key: int = 9001) -> pl.DataFrame:
    owned = [pick.element for pick in team.picks]
    sold_out = owned[0]
    kept = owned[1:] + [buy_key]
    return _plan_rows(
        elements=kept + [sold_out],
        bought=[key == buy_key for key in kept] + [False],
        sold=[False] * 15 + [True],
        in_xi=[True] * 11 + [False] * 4 + [False],
        captain_at=0,
        vice_at=1,
    )


def _view(root: Path, *, element_ids: list[int]) -> None:
    store = DataStore(root)
    players: list[dict[str, object]] = []
    status: list[dict[str, object]] = []
    for index, element in enumerate(element_ids, start=1):
        players.append(
            {
                "season": _SEASON,
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
                "season": _SEASON,
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
                "season": [_SEASON],
                "gw": [_GW],
                "deadline_time": [datetime(2024, 8, 16, 18, tzinfo=UTC)],
                "finished": [False],
                "data_checked": [False],
                "available_at": [_WHEN],
                "source": ["test"],
            }
        ),
    )
    PITData(store).view(GWKey(_SEASON, _GW))


def test_build_payload_pairs_transfers_and_lineup() -> None:
    team = CurrentTeam.from_my_team(42, _load("my_team.json"))
    owned = [pick.element for pick in team.picks]
    buy_key = 9001
    sold_out = owned[0]
    frame = _swap_plan(team, buy_key)
    element_by_key = {key: key for key in owned + [buy_key]}
    prices = {key: 50 + (key % 7) for key in owned + [buy_key]}
    payload = build_apply_payload(
        frame,
        team,
        element_by_key=element_by_key,
        purchase_price_by_key=prices,
        entry_id=42,
        event=_GW,
    )
    assert len(payload.transfers) == 1
    move = payload.transfers[0]
    assert move.element_out == sold_out
    assert move.element_in == buy_key
    assert move.selling_price == team.picks[0].selling_price
    assert move.purchase_price == prices[buy_key]
    body = payload.transfers_body()
    assert body["entry"] == 42
    assert body["transfers"][0]["element_in"] == buy_key
    assert len(payload.picks) == 15
    assert {pick["element"] for pick in payload.picks} == set(owned[1:] + [buy_key])
    assert sum(1 for pick in payload.picks if pick["is_captain"]) == 1


def test_refuses_initial_draft() -> None:
    team = CurrentTeam.from_my_team(42, _load("my_team.json"))
    elements = [pick.element for pick in team.picks]
    frame = _plan_rows(
        elements=elements,
        bought=[True] * 15,
        sold=[False] * 15,
    )
    with pytest.raises(ApplyError, match="initial draft"):
        build_apply_payload(
            frame,
            team,
            element_by_key={key: key for key in elements},
            purchase_price_by_key={key: 50 for key in elements},
            entry_id=42,
        )


def test_unequal_buys_and_sells_raise() -> None:
    team = CurrentTeam.from_my_team(42, _load("my_team.json"))
    owned = [pick.element for pick in team.picks]
    # Two buys, one sell — invalid for FPL pairing.
    kept = owned[1:] + [9001]
    frame = _plan_rows(
        elements=kept + [owned[0]],
        bought=[True, True] + [False] * 13 + [False],
        sold=[False] * 15 + [True],
        in_xi=[True] * 11 + [False] * 4 + [False],
    )
    with pytest.raises(ApplyError, match="unequal transfers"):
        build_apply_payload(
            frame,
            team,
            element_by_key={key: key for key in owned + [9001]},
            purchase_price_by_key={key: 50 for key in owned + [9001]},
            entry_id=42,
        )


def test_lineup_only_when_no_transfers() -> None:
    team = CurrentTeam.from_my_team(42, _load("my_team.json"))
    elements = [pick.element for pick in team.picks]
    frame = _plan_rows(
        elements=elements,
        bought=[False] * 15,
        sold=[False] * 15,
    )
    payload = build_apply_payload(
        frame,
        team,
        element_by_key={key: key for key in elements},
        purchase_price_by_key={key: 50 for key in elements},
        entry_id=42,
    )
    assert payload.transfers == ()
    assert len(payload.picks) == 15


def test_apply_plan_dry_run_and_confirm(tmp_path: Path) -> None:
    team = CurrentTeam.from_my_team(42, _load("my_team.json"))
    owned = [pick.element for pick in team.picks]
    buy_key = 9001
    _view(tmp_path, element_ids=owned + [buy_key])
    write_plan(tmp_path, _swap_plan(team, buy_key))

    posts: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content.decode())
            posts.append((request.url.path, body))
            if request.url.path.endswith("/transfers/"):
                return httpx.Response(200, content=b"")
            return httpx.Response(200, json={"ok": True})
        raise AssertionError(f"{request.method} {request.url.path}")

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url=_API)
    api = FPLClient(http, api_token="eyJtest", base_url=_API)
    settings = Settings(entry_id=42, fpl_api_token="eyJtest")
    try:
        dry = apply_plan(
            tmp_path,
            _MODEL,
            _SEASON,
            _GW,
            horizon=_HORIZON,
            settings=settings,
            client=api,
            team=team,
        )
        assert dry.mode == "dry_run"
        assert dry.posted_transfers is False
        assert posts == []
        preview = format_apply_preview(dry)
        assert "dry-run" in preview
        assert f"out element={owned[0]}" in preview

        confirmed = apply_plan(
            tmp_path,
            _MODEL,
            _SEASON,
            _GW,
            horizon=_HORIZON,
            confirm=True,
            settings=settings,
            client=api,
            team=team,
        )
    finally:
        api.close()
        http.close()

    assert confirmed.mode == "confirmed"
    assert confirmed.posted_transfers is True
    assert confirmed.posted_lineup is True
    assert any(path.endswith("/transfers/") for path, _ in posts)
    assert any("/my-team/42/" in path for path, _ in posts)


def test_apply_plan_transfers_only(tmp_path: Path) -> None:
    team = CurrentTeam.from_my_team(42, _load("my_team.json"))
    owned = [pick.element for pick in team.picks]
    _view(tmp_path, element_ids=owned + [9001])
    write_plan(tmp_path, _swap_plan(team))
    posts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request.url.path)
        if request.url.path.endswith("/transfers/"):
            return httpx.Response(200, content=b"")
        raise AssertionError(request.url.path)

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url=_API)
    api = FPLClient(http, api_token="eyJtest", base_url=_API)
    try:
        result = apply_plan(
            tmp_path,
            _MODEL,
            _SEASON,
            _GW,
            horizon=_HORIZON,
            confirm=True,
            transfers_only=True,
            settings=Settings(entry_id=42, fpl_api_token="eyJtest"),
            client=api,
            team=team,
        )
    finally:
        api.close()
        http.close()

    assert result.posted_transfers is True
    assert result.posted_lineup is False
    assert len(posts) == 1
    assert posts[0].endswith("/transfers/")


def test_confirm_requires_token() -> None:
    with pytest.raises(ApplyError, match="FPLT_FPL_API_TOKEN"):
        apply_plan(
            Path("/tmp"),
            _MODEL,
            _SEASON,
            _GW,
            confirm=True,
            settings=Settings(entry_id=42, fpl_api_token=None),
        )


def test_cli_apply_dry_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    canned = ApplyResult(
        payload=ApplyPayload(
            entry_id=42,
            event=1,
            transfers=(
                TransferMove(
                    element_in=2,
                    element_out=1,
                    purchase_price=50,
                    selling_price=45,
                    player_key_in=2,
                    player_key_out=1,
                ),
            ),
            picks=tuple(
                {
                    "element": index,
                    "position": index,
                    "is_captain": index == 1,
                    "is_vice_captain": index == 2,
                }
                for index in range(1, 16)
            ),
        ),
        posted_transfers=False,
        posted_lineup=False,
        mode="dry_run",
    )
    monkeypatch.setattr("fplt.cli.apply_plan", lambda *a, **k: canned)
    main(["apply", "--model", _MODEL, "--season", _SEASON, "--gw", "1"])
    out = capsys.readouterr().out
    assert "mode=dry_run" in out
    assert "dry-run: nothing posted" in out


def test_plan_parquet_path_matches_write(tmp_path: Path) -> None:
    team = CurrentTeam.from_my_team(42, _load("my_team.json"))
    elements = [pick.element for pick in team.picks]
    frame = _plan_rows(
        elements=elements,
        bought=[False] * 15,
        sold=[False] * 15,
    )
    path = write_plan(tmp_path, frame)
    spec = load_model_specs()[_MODEL]
    assert path == plan_parquet(tmp_path, spec.model_id, _SEASON, _GW, _HORIZON)
