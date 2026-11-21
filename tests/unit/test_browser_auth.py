"""Local browser localStorage → fplt auth credentials."""

from __future__ import annotations

import base64
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fplt.auth import read_env_file
from fplt.browser_auth import (
    BrowserAuthError,
    BrowserProfile,
    auth_from_browser,
    discover_profiles,
    parse_oidc_user_value,
)
from fplt.cli import main


def _jwt(*, exp: int) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode("ascii").rstrip("=")
    payload = (
        base64.urlsafe_b64encode(json.dumps({"sub": "sso", "exp": exp}).encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )
    return f"{header}.{payload}.sig"


def test_parse_oidc_user_value_extracts_access_token() -> None:
    token = _jwt(exp=int(datetime(2099, 1, 1, tzinfo=UTC).timestamp()))
    raw = json.dumps(
        {
            "access_token": token,
            "refresh_token": "refresh-xyz",
            "expires_at": 1893456000,
        }
    )
    session = parse_oidc_user_value(raw, source="test")
    assert session.access_token == token
    assert session.refresh_token == "refresh-xyz"
    assert session.expires_at == 1893456000


def test_parse_oidc_user_value_requires_access_token() -> None:
    with pytest.raises(BrowserAuthError, match="access_token"):
        parse_oidc_user_value("{}", source="test")
    with pytest.raises(BrowserAuthError, match="invalid"):
        parse_oidc_user_value("not-json", source="test")


def test_discover_chromium_and_firefox_profiles(tmp_path: Path) -> None:
    chrome = tmp_path / ".config" / "google-chrome" / "Default" / "Local Storage"
    (chrome / "leveldb").mkdir(parents=True)
    (chrome / "leveldb" / "CURRENT").write_text("1", encoding="utf-8")
    ff = tmp_path / ".mozilla" / "firefox" / "abcd.default"
    ff.mkdir(parents=True)
    (ff / "webappsstore.sqlite").write_bytes(b"")
    profiles = discover_profiles(home=tmp_path)
    assert any(p.browser == "chrome" and p.label == "Default" for p in profiles)
    assert any(p.browser == "firefox" for p in profiles)
    only_ff = discover_profiles(browser="firefox", home=tmp_path)
    assert all(p.browser == "firefox" for p in only_ff)


def test_auth_from_browser_firefox_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_dir = tmp_path / ".mozilla" / "firefox" / "abcd.default"
    profile_dir.mkdir(parents=True)
    db_path = profile_dir / "webappsstore.sqlite"
    token = _jwt(exp=int(datetime(2099, 1, 1, tzinfo=UTC).timestamp()))
    payload = json.dumps(
        {
            "access_token": f"Bearer {token}",
            "refresh_token": "r1",
            "expires_at": 1893456000,
            "profile": {"iss": "https://account.premierleague.com/as"},
        }
    )
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE webappsstore2 ("
        "originKey TEXT, originAttributes TEXT, scope TEXT, "
        "key TEXT, value TEXT)"
    )
    conn.execute(
        "INSERT INTO webappsstore2 VALUES (?, '', '', ?, ?)",
        (
            "moc.eugaellreimerp.ysatnaf.|https|443",
            "oidc.user:https://account.premierleague.com/as:client",
            payload,
        ),
    )
    conn.commit()
    conn.close()

    env_path = tmp_path / "env"
    status = auth_from_browser(
        browser="firefox",
        entry_id=42,
        env_path=env_path,
        home=tmp_path,
    )
    assert status.entry_id == 42
    assert status.token is not None
    assert status.token.token == token
    values = read_env_file(env_path)
    assert values["FPLT_FPL_API_TOKEN"] == token
    assert values["FPLT_ENTRY_ID"] == "42"


def test_auth_from_browser_no_profiles(tmp_path: Path) -> None:
    with pytest.raises(BrowserAuthError, match="no local browser profiles"):
        auth_from_browser(home=tmp_path, env_path=tmp_path / "env")


def test_auth_from_browser_chromium_uses_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leveldb = (
        tmp_path / ".config" / "google-chrome" / "Default" / "Local Storage" / "leveldb"
    )
    leveldb.mkdir(parents=True)
    (leveldb / "CURRENT").write_text("1", encoding="utf-8")
    token = _jwt(exp=int(datetime(2099, 1, 1, tzinfo=UTC).timestamp()))
    payload = json.dumps({"access_token": token, "refresh_token": "r"})

    class _Rec:
        is_live = True
        value = payload
        script_key = "oidc.user:https://account.premierleague.com/as:x"
        leveldb_seq_number = 9

    class _Reader:
        def __init__(self, path: Path) -> None:
            self.path = path

        def __enter__(self) -> _Reader:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def records(self, **_: object) -> list[_Rec]:
            return [_Rec()]

    monkeypatch.setattr(
        "chromium_reader.localstorage.LocalStorageReader",
        _Reader,
    )
    env_path = tmp_path / "env"
    status = auth_from_browser(
        browser="chrome",
        entry_id=7,
        env_path=env_path,
        home=tmp_path,
        copy_locked=False,
    )
    assert status.token is not None
    assert status.token.token == token
    assert status.entry_id == 7


def test_cli_auth_from_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    token = _jwt(exp=int(datetime(2099, 1, 1, tzinfo=UTC).timestamp()))
    env_path = tmp_path / "env"
    monkeypatch.setenv("FPLT_ENV_FILE", str(env_path))

    def fake_auth_from_browser(**kwargs: object) -> object:
        from fplt.auth import set_credentials

        return set_credentials(token=token, entry_id=kwargs.get("entry_id"))

    monkeypatch.setattr("fplt.cli.auth_from_browser", fake_auth_from_browser)
    main(["auth", "from-browser", "--entry-id", "99", "--browser", "chrome"])
    out = capsys.readouterr().out
    assert "entry_id: 99" in out
    assert "token: set (ok)" in out


def test_extract_firefox_profile_helper_message() -> None:
    profile = BrowserProfile(
        browser="firefox",
        label="missing",
        path=Path("/no/such/webappsstore.sqlite"),
        kind="firefox",
    )
    from fplt.browser_auth import extract_oidc_session

    with pytest.raises(BrowserAuthError):
        extract_oidc_session(profile)
