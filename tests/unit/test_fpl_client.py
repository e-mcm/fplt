"""FPL API client: current team from my-team and public entry picks."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from fplt.data.fpl_client import (
    CurrentTeam,
    FPLAPIError,
    FPLClient,
    load_current_team,
)
from fplt.optimise.state import apply_entry_picks, load_squad_state
from fplt.settings import Settings, get_settings

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "fpl_api"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


_API = "https://fantasy.premierleague.com/api"


def _client(
    handler: httpx.MockTransport | None = None,
    *,
    session: str | None = None,
    api_token: str | None = None,
) -> tuple[FPLClient, httpx.Client]:
    transport = handler
    if transport is None:
        transport = httpx.MockTransport(lambda _r: httpx.Response(500))
    http = httpx.Client(transport=transport, base_url=_API)
    return FPLClient(http, session=session, api_token=api_token, base_url=_API), http


def test_my_team_loads_selling_prices_and_free_transfers() -> None:
    payload = _load("my_team.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/my-team/42/")
        assert "pl_profile=secret" in request.headers.get("cookie", "")
        return httpx.Response(200, json=payload)

    api, http = _client(httpx.MockTransport(handler), session="secret")
    try:
        team = api.current_team(42)
    finally:
        api.close()
        http.close()

    assert team.source == "my_team"
    assert team.entry_id == 42
    assert team.bank == 21
    assert team.value == 1022
    assert team.free_transfers == 1
    assert len(team.picks) == 15
    assert team.picks[0].selling_price == 46
    assert team.picks[0].element == payload["picks"][0]["element"]


def test_my_team_sends_bearer_token_without_cookie() -> None:
    payload = _load("my_team.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-api-authorization") == "Bearer eyJabc"
        assert "pl_profile" not in request.headers.get("cookie", "")
        return httpx.Response(200, json=payload)

    api, http = _client(httpx.MockTransport(handler), api_token="Bearer eyJabc")
    try:
        team = api.current_team(42)
    finally:
        api.close()
        http.close()

    assert team.source == "my_team"
    assert team.bank == 21


def test_picks_payload_feeds_apply_entry_picks() -> None:
    team = CurrentTeam.from_my_team(7, _load("my_team.json"))
    state = load_squad_state()
    element_to_key = {pick.element: pick.element * 10 for pick in team.picks}
    updated = apply_entry_picks(state, team.picks_payload(), element_to_key)
    assert updated.bank == 21
    assert len(updated.owned) == 15
    assert updated.owned[0].selling_price == team.picks[0].selling_price
    assert updated.owned[0].purchase_price == team.picks[0].purchase_price
    assert "purchase_price" in team.picks_payload()["picks"][0]
    assert updated.free_transfers == state.free_transfers


def test_public_entry_picks_for_finished_gameweek() -> None:
    entry = _load("entry.json")
    picks = _load("entry_picks.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/entry/1/"):
            return httpx.Response(200, json=entry)
        if request.url.path.endswith("/entry/1/event/9/picks/"):
            return httpx.Response(200, json=picks)
        raise AssertionError(request.url.path)

    api, http = _client(httpx.MockTransport(handler))
    try:
        team = api.current_team(1, event=9, prefer_my_team=False)
    finally:
        api.close()
        http.close()

    assert team.source == "entry_picks"
    assert team.event == 9
    assert team.bank == picks["entry_history"]["bank"]
    assert team.free_transfers is None
    assert team.picks[0].selling_price is None
    with pytest.raises(ValueError, match="selling_price"):
        team.picks_payload()


def test_resolves_current_event_when_event_omitted() -> None:
    entry = _load("entry.json")
    picks = _load("entry_picks.json")
    entry = {**entry, "current_event": 9}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/entry/1/"):
            return httpx.Response(200, json=entry)
        if request.url.path.endswith("/entry/1/event/9/picks/"):
            return httpx.Response(200, json=picks)
        raise AssertionError(request.url.path)

    api, http = _client(httpx.MockTransport(handler))
    try:
        team = api.current_team(1, prefer_my_team=False)
    finally:
        api.close()
        http.close()
    assert team.event == 9


def test_live_picks_forbidden_without_session() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, content=b"")

    api, http = _client(httpx.MockTransport(handler))
    try:
        with pytest.raises(FPLAPIError, match="forbidden|FPLT_FPL_API_TOKEN"):
            api.entry_picks(1, 10)
        with pytest.raises(FPLAPIError, match="FPLT_FPL_API_TOKEN"):
            api.my_team(1)
    finally:
        api.close()
        http.close()


def test_load_current_team_uses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("FPLT_ENTRY_ID", "42")
    monkeypatch.setenv("FPLT_FPL_SESSION", "cookie-value")
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/my-team/42/")
        return httpx.Response(200, json=_load("my_team.json"))

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://fantasy.premierleague.com/api",
    )
    try:
        team = load_current_team(client=http)
    finally:
        http.close()
        get_settings.cache_clear()

    assert team.entry_id == 42
    assert team.source == "my_team"


def test_load_current_team_uses_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.delenv("FPLT_FPL_SESSION", raising=False)
    monkeypatch.setenv("FPLT_ENTRY_ID", "42")
    monkeypatch.setenv("FPLT_FPL_API_TOKEN", '"Bearer eyJfrom-env"')
    get_settings.cache_clear()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-api-authorization") == "Bearer eyJfrom-env"
        return httpx.Response(200, json=_load("my_team.json"))

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://fantasy.premierleague.com/api",
    )
    try:
        team = load_current_team(client=http)
    finally:
        http.close()
        get_settings.cache_clear()

    assert team.source == "my_team"
    assert team.entry_id == 42


def test_blank_api_token_is_unset() -> None:
    assert Settings(fpl_api_token=None).fpl_api_token is None
    assert Settings(fpl_api_token="  ").fpl_api_token is None
    assert Settings(fpl_api_token="Bearer eyJabc").fpl_api_token == "eyJabc"


def test_pytest_ignores_repo_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "FPLT_ENTRY_ID=999\nFPLT_FPL_API_TOKEN=eyJfile\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FPLT_ENTRY_ID", raising=False)
    monkeypatch.delenv("FPLT_FPL_API_TOKEN", raising=False)
    loaded = Settings()
    assert loaded.entry_id is None
    assert loaded.fpl_api_token is None


def test_load_current_team_requires_entry_id() -> None:
    with pytest.raises(FPLAPIError, match="FPLT_ENTRY_ID"):
        load_current_team(settings=Settings(entry_id=None, fpl_session=None))


def test_malformed_my_team_payload() -> None:
    with pytest.raises(FPLAPIError, match="picks and transfers"):
        CurrentTeam.from_my_team(1, {"picks": []})
    with pytest.raises(FPLAPIError, match="transfers.bank"):
        CurrentTeam.from_my_team(1, {"picks": [], "transfers": {}})


def test_malformed_entry_picks_payload() -> None:
    with pytest.raises(FPLAPIError, match="picks and entry_history"):
        CurrentTeam.from_entry_picks(1, 1, {})
    with pytest.raises(FPLAPIError, match="entry_history.bank"):
        CurrentTeam.from_entry_picks(1, 1, {"picks": [], "entry_history": {}})


def test_http_errors_are_wrapped() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    api, http = _client(httpx.MockTransport(boom))
    try:
        with pytest.raises(FPLAPIError, match="HTTP 500"):
            api.entry(1)
    finally:
        api.close()
        http.close()

    def missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "nope"})

    api, http = _client(httpx.MockTransport(missing))
    try:
        with pytest.raises(FPLAPIError, match="not found"):
            api.entry(99)
    finally:
        api.close()
        http.close()


def test_submit_transfers_posts_empty_ok_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.method == "POST"
        assert request.url.path.endswith("/transfers/")
        assert request.headers.get("x-api-authorization") == "Bearer eyJwrite"
        return httpx.Response(200, content=b"")

    api, http = _client(httpx.MockTransport(handler), api_token="eyJwrite")
    try:
        api.submit_transfers(
            {
                "entry": 42,
                "event": 11,
                "transfers": [
                    {
                        "element_in": 2,
                        "element_out": 1,
                        "purchase_price": 50,
                        "selling_price": 45,
                    }
                ],
                "chip": None,
            }
        )
    finally:
        api.close()
        http.close()
    assert len(seen) == 1


def test_submit_lineup_requires_token() -> None:
    api, http = _client()
    try:
        with pytest.raises(FPLAPIError, match="FPLT_FPL_API_TOKEN"):
            api.submit_lineup(1, {"picks": [], "chip": None})
    finally:
        api.close()
        http.close()


def test_submit_lineup_returns_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/my-team/7/")
        return httpx.Response(200, json={"ok": True})

    api, http = _client(httpx.MockTransport(handler), api_token="eyJwrite")
    try:
        body = api.submit_lineup(
            7,
            {
                "chip": None,
                "picks": [
                    {
                        "element": 1,
                        "position": 1,
                        "is_captain": True,
                        "is_vice_captain": False,
                    }
                ],
            },
        )
    finally:
        api.close()
        http.close()
    assert body == {"ok": True}
