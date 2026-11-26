"""ops-01: FPL API token expiry monitor (check_token / auth check)."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.request import Request

import pytest

from fplt.auth import (
    FIX_COMMAND,
    MANUAL_APPLY_BANNER,
    check_token,
    format_auth_check,
    with_manual_apply_banner,
)
from fplt.cli import main
from fplt.settings import get_settings


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


_NOW = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
_DEADLINE = datetime(2026, 11, 28, 11, 30, tzinfo=UTC)


def test_check_token_valid_past_warn_and_deadline() -> None:
    exp = int((_DEADLINE + timedelta(days=5)).timestamp())
    result = check_token(
        raw=_jwt(exp=exp),
        now=_NOW,
        warn_days=3,
        next_deadline=_DEADLINE,
        deadline_aware=True,
    )
    assert result.level == "ok"
    assert result.reason == "ok"
    assert result.ok
    assert result.exit_code == 0
    assert result.manual_apply is False
    assert result.expires_at is not None
    assert "ok until" in result.message
    assert FIX_COMMAND not in result.message


def test_check_token_expiring_soon_warning() -> None:
    # After the deadline, but inside the 3-day warn window.
    exp_at = _NOW + timedelta(days=2, hours=1)
    result = check_token(
        raw=_jwt(exp=int(exp_at.timestamp())),
        now=_NOW,
        warn_days=3,
        next_deadline=_NOW + timedelta(hours=12),
        deadline_aware=True,
    )
    assert result.level == "warning"
    assert result.reason == "expiring_soon"
    assert result.exit_code == 1
    assert "in " in result.message
    assert FIX_COMMAND in result.message
    assert result.next_deadline is not None
    assert _jwt(exp=int(exp_at.timestamp())) not in result.message


def test_check_token_expires_before_deadline_urgent() -> None:
    exp_at = _DEADLINE - timedelta(hours=6)
    result = check_token(
        raw=_jwt(exp=int(exp_at.timestamp())),
        now=_NOW,
        warn_days=3,
        next_deadline=_DEADLINE,
        deadline_aware=True,
    )
    assert result.level == "urgent"
    assert result.reason == "expires_before_deadline"
    assert result.exit_code == 1
    assert _DEADLINE.isoformat() in result.message
    assert FIX_COMMAND in result.message
    assert result.manual_apply is False


def test_check_token_expired_sets_manual_apply_banner() -> None:
    exp_at = _NOW - timedelta(hours=1)
    result = check_token(
        raw=_jwt(exp=int(exp_at.timestamp())),
        now=_NOW,
        warn_days=3,
        next_deadline=_DEADLINE,
        deadline_aware=True,
    )
    assert result.level == "urgent"
    assert result.reason == "expired"
    assert result.manual_apply is True
    assert MANUAL_APPLY_BANNER in result.message
    assert FIX_COMMAND in result.message
    bannered = with_manual_apply_banner("plan body", result)
    assert bannered.startswith(f"*** {MANUAL_APPLY_BANNER} ***")
    assert "plan body" in bannered


def test_check_token_missing_urgent_unless_allow_missing() -> None:
    missing = check_token(
        raw=None,
        now=_NOW,
        next_deadline=_DEADLINE,
        allow_missing=False,
    )
    assert missing.level == "urgent"
    assert missing.reason == "missing"
    assert missing.manual_apply is True
    assert FIX_COMMAND in missing.message

    skipped = check_token(
        raw=None,
        now=_NOW,
        next_deadline=_DEADLINE,
        allow_missing=True,
    )
    assert skipped.level == "soft_skip"
    assert skipped.reason == "soft_skip_missing"
    assert skipped.ok
    assert skipped.exit_code == 0
    assert "soft-skip" in skipped.message


def test_check_token_malformed_urgent() -> None:
    result = check_token(
        raw="not-a-jwt",
        now=_NOW,
        next_deadline=_DEADLINE,
        deadline_aware=True,
    )
    assert result.level == "urgent"
    assert result.reason == "malformed"
    assert result.manual_apply is True
    assert FIX_COMMAND in result.message
    assert "not-a-jwt" not in result.message


def test_check_token_deadline_aware_off_skips_deadline_urgent() -> None:
    # Would be expires_before_deadline if aware; with aware off, only warn window.
    exp_at = _DEADLINE - timedelta(hours=1)
    result = check_token(
        raw=_jwt(exp=int(exp_at.timestamp())),
        now=_NOW,
        warn_days=3,
        next_deadline=_DEADLINE,
        deadline_aware=False,
    )
    assert result.level == "warning"
    assert result.reason == "expiring_soon"


def test_format_auth_check_never_embeds_token() -> None:
    token = _jwt(exp=int((_NOW + timedelta(hours=1)).timestamp()))
    result = check_token(raw=token, now=_NOW, warn_days=3)
    text = format_auth_check(result)
    assert token not in text
    assert "level:" in text
    assert FIX_COMMAND in text


def test_cli_auth_check_soft_skip_when_allow_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("FPLT_FPL_API_TOKEN", raising=False)
    get_settings.cache_clear()
    with pytest.raises(SystemExit) as exc:
        main(["auth", "check", "--allow-missing", "--no-notify"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "soft_skip" in out
    get_settings.cache_clear()


def test_cli_auth_check_exits_nonzero_when_expired(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exp = int((_NOW - timedelta(hours=2)).timestamp())
    token = _jwt(exp=exp)
    monkeypatch.setenv("FPLT_FPL_API_TOKEN", token)
    get_settings.cache_clear()
    # Freeze clock via monkeypatch on check_token's default by passing through
    # env; CLI uses datetime.now — patch check_token path by setting token
    # that is already expired relative to real now is flaky. Use a far-past exp.
    with pytest.raises(SystemExit) as exc:
        main(["auth", "check", "--no-notify", "--no-deadline-aware"])
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "urgent" in out
    assert MANUAL_APPLY_BANNER in out
    assert token not in out
    get_settings.cache_clear()


def test_cli_auth_check_notifies_on_urgent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FPLT_NTFY_URL", "https://ntfy.test/topic")
    monkeypatch.delenv("FPLT_FPL_API_TOKEN", raising=False)
    get_settings.cache_clear()
    posted: list[Request] = []

    def fake_urlopen(request: Request, timeout: float | None = None) -> object:
        posted.append(request)
        return object()

    monkeypatch.setattr(
        "fplt.automation.notify.urlopen",
        fake_urlopen,
    )
    with pytest.raises(SystemExit) as exc:
        main(["auth", "check"])
    assert exc.value.code == 1
    assert len(posted) == 1
    body = posted[0].data
    assert body is not None
    text = body.decode("utf-8")
    assert "missing" in text.lower() or "FPLT_FPL_API_TOKEN" in text
    assert "ops-01-token-expiry" in text
    assert "eyJ" not in text
    get_settings.cache_clear()


def test_with_manual_apply_banner_noop_when_ok() -> None:
    result = check_token(
        raw=_jwt(exp=int((_NOW + timedelta(days=10)).timestamp())),
        now=_NOW,
        warn_days=3,
    )
    assert with_manual_apply_banner("hello", result) == "hello"
