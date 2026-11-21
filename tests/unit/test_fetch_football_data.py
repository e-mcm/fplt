"""fetch_football_data helpers stay write-once and need no network."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import httpx

_PATH = Path(__file__).parents[2] / "scripts" / "fetch_football_data.py"
_SPEC = importlib.util.spec_from_file_location("fplt_fetch_football_data", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
fetch = importlib.util.module_from_spec(_SPEC)
sys.modules["fplt_fetch_football_data"] = fetch
_SPEC.loader.exec_module(fetch)


def test_e0_url_uses_season_token() -> None:
    assert fetch.e0_url("2024-25") == (
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
    )


def test_write_once_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "football_data" / "season=2024-25" / "E0.csv"
    assert fetch.write_once(path, "first") is True
    assert path.read_text(encoding="utf-8") == "first"
    assert fetch.write_once(path, "second") is False
    assert path.read_text(encoding="utf-8") == "first"


def test_fetch_season_skips_existing_without_http(tmp_path: Path) -> None:
    from fplt.data.football_data import e0_raw_path

    root = tmp_path / "raw"
    dest = e0_raw_path(root, "2024-25")
    dest.parent.mkdir(parents=True)
    dest.write_text("already", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = fetch.fetch_season(client, root, "2024-25")
    assert result.wrote is False
    assert dest.read_text(encoding="utf-8") == "already"


def test_fetch_season_writes_once(tmp_path: Path) -> None:
    root = tmp_path / "raw"

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/2425/E0.csv")
        return httpx.Response(200, text="Div,Date\nE0,01/01/2024\n")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = fetch.fetch_season(client, root, "2024-25")
    second = fetch.fetch_season(client, root, "2024-25")
    assert first.wrote is True
    assert second.wrote is False
    assert first.path.read_text(encoding="utf-8").startswith("Div,Date")


def test_fetch_season_follows_www_to_apex_redirect(tmp_path: Path) -> None:
    """www.football-data.co.uk 302s to the apex host before serving CSV."""
    root = tmp_path / "raw"
    apex = "https://football-data.co.uk/mmz4281/2425/E0.csv"
    body = "Div,Date\nE0,16/08/2024\n"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == fetch.e0_url("2024-25"):
            return httpx.Response(302, headers={"Location": apex})
        if url == apex:
            return httpx.Response(200, text=body)
        raise AssertionError(f"unexpected request {url}")

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    result = fetch.fetch_season(client, root, "2024-25")
    assert result.wrote is True
    assert result.path.read_text(encoding="utf-8") == body
