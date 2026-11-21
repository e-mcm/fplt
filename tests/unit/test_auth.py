"""Local FPL auth storage and JWT expiry helpers."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fplt.auth import (
    clear_credentials,
    format_status,
    inspect_token,
    read_env_file,
    set_credentials,
    status,
    upsert_env_file,
)
from fplt.cli import main
from fplt.settings import Settings, get_settings, user_env_file


def _jwt(*, exp: int | None, sub: str = "user-1") -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode("ascii").rstrip("=")
    payload_obj: dict[str, object] = {"sub": sub}
    if exp is not None:
        payload_obj["exp"] = exp
    payload = (
        base64.urlsafe_b64encode(json.dumps(payload_obj).encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )
    return f"{header}.{payload}.sig"


def test_inspect_token_reads_exp_and_strips_bearer() -> None:
    exp = int(datetime(2099, 1, 1, tzinfo=UTC).timestamp())
    now = datetime(2026, 1, 1, tzinfo=UTC)
    info = inspect_token(f"Bearer {_jwt(exp=exp)}", now=now)
    assert info is not None
    assert info.expired is False
    assert info.subject == "user-1"
    assert info.expires_at == datetime(2099, 1, 1, tzinfo=UTC)


def test_inspect_token_marks_expired() -> None:
    exp = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
    info = inspect_token(_jwt(exp=exp), now=datetime(2026, 1, 1, tzinfo=UTC))
    assert info is not None
    assert info.expired is True


def test_inspect_token_blank_and_non_jwt() -> None:
    assert inspect_token(None) is None
    assert inspect_token("   ") is None
    info = inspect_token("not-a-jwt")
    assert info is not None
    assert info.expires_at is None
    assert info.expired is None


def test_upsert_env_file_preserves_other_keys(tmp_path: Path) -> None:
    path = tmp_path / "env"
    path.write_text("FPLT_NTFY_URL=https://ntfy.sh/x\n# keep\n", encoding="utf-8")
    upsert_env_file(path, {"FPLT_FPL_API_TOKEN": "eyJabc", "FPLT_ENTRY_ID": "42"})
    values = read_env_file(path)
    assert values["FPLT_NTFY_URL"] == "https://ntfy.sh/x"
    assert values["FPLT_FPL_API_TOKEN"] == "eyJabc"
    assert values["FPLT_ENTRY_ID"] == "42"
    assert path.stat().st_mode & 0o777 == 0o600
    text = path.read_text(encoding="utf-8")
    assert "# keep" in text


def test_set_and_clear_credentials(tmp_path: Path) -> None:
    path = tmp_path / "fplt" / "env"
    exp = int(datetime(2099, 6, 1, tzinfo=UTC).timestamp())
    token = _jwt(exp=exp)
    result = set_credentials(token=f"Bearer {token}", entry_id=99, env_path=path)
    assert result.entry_id == 99
    assert result.token is not None
    assert result.token.token == token
    cleared = clear_credentials(env_path=path, clear_entry_id=True)
    assert cleared.token is None
    assert cleared.entry_id is None


def test_set_credentials_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        set_credentials(token="  ")


def test_format_status_includes_never_paste_warning(tmp_path: Path) -> None:
    path = tmp_path / "env"
    auth = status(env_path=path)
    text = format_status(auth)
    assert "never paste" in text
    assert "unset" in text


def test_user_env_file_honours_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "custom.env"
    monkeypatch.setenv("FPLT_ENV_FILE", str(target))
    assert user_env_file() == target


def test_settings_loads_user_env_outside_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / "env"
    env_path.write_text(
        "FPLT_ENTRY_ID=77\nFPLT_FPL_API_TOKEN=Bearer eyJuser\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FPLT_ENV_FILE", str(env_path))
    monkeypatch.delenv("FPLT_ENTRY_ID", raising=False)
    monkeypatch.delenv("FPLT_FPL_API_TOKEN", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.chdir(tmp_path)
    loaded = Settings()
    assert loaded.entry_id == 77
    assert loaded.fpl_api_token == "eyJuser"


def test_settings_user_env_beats_repo_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = tmp_path / "user.env"
    user.write_text("FPLT_FPL_API_TOKEN=eyJuser\n", encoding="utf-8")
    (tmp_path / ".env").write_text("FPLT_FPL_API_TOKEN=eyJrepo\n", encoding="utf-8")
    monkeypatch.setenv("FPLT_ENV_FILE", str(user))
    monkeypatch.delenv("FPLT_FPL_API_TOKEN", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.chdir(tmp_path)
    assert Settings().fpl_api_token == "eyJuser"


def test_cli_auth_set_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_path = tmp_path / "env"
    monkeypatch.setenv("FPLT_ENV_FILE", str(env_path))
    get_settings.cache_clear()
    exp = int(datetime(2099, 1, 1, tzinfo=UTC).timestamp())
    token = _jwt(exp=exp)
    main(["auth", "set", "--token", token, "--entry-id", "123"])
    out = capsys.readouterr().out
    assert "entry_id: 123" in out
    assert "token: set (ok)" in out
    main(["auth", "status"])
    status_out = capsys.readouterr().out
    assert "entry_id: 123" in status_out
    main(["auth", "clear", "--also-entry-id"])
    cleared = capsys.readouterr().out
    assert "token: (unset)" in cleared
    get_settings.cache_clear()


def test_cli_auth_set_rejects_empty_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _EmptyStdin:
        def read(self) -> str:
            return ""

        def isatty(self) -> bool:
            return False

    monkeypatch.setattr("sys.stdin", _EmptyStdin())
    with pytest.raises(SystemExit) as exc:
        main(["auth", "set", "--token", "-"])
    assert exc.value.code == 2
    assert "empty" in capsys.readouterr().err


def test_load_current_team_warns_when_token_expired(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import httpx

    from fplt.data.fpl_client import FPLAPIError, load_current_team

    exp = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
    settings = Settings(entry_id=1, fpl_api_token=_jwt(exp=exp))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "forbidden"})

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://fantasy.premierleague.com/api",
    )
    try:
        with caplog.at_level("WARNING"), pytest.raises(FPLAPIError):
            load_current_team(settings=settings, client=http)
    finally:
        http.close()
    assert any("expired" in record.message for record in caplog.records)
