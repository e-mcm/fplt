"""SecretStr settings + log redaction (public Actions logs; #308)."""

from __future__ import annotations

import logging
from urllib.parse import quote

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from fplt.data.fpl_client import FPLAPIError, load_current_team
from fplt.settings import Settings, secret_value
from fplt.utils.logging import (
    SecretRedactionFilter,
    configure_logging,
    derived_secret_forms,
    redact_text,
    reset_logging_for_tests,
)

_FAKE_JWT = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJmYWtlIn0.sig"
_FAKE_API_KEY = "api-football-secret-key-001"
_FAKE_R2_SECRET = "r2-secret-access-key-xyz"
_FAKE_SESSION = "pl_profile_cookie_value"
_FAKE_NTFY = "https://ntfy.sh/fplt-secret-topic"
_FAKE_SMTP_PW = "smtp-password-value"
_FAKE_ACCESS_KEY = "AKIAFAKEACCESS"


def _settings_with_fakes() -> Settings:
    return Settings(
        entry_id=1,
        fpl_api_token=f"Bearer {_FAKE_JWT}",
        fpl_session=_FAKE_SESSION,
        api_football_key=_FAKE_API_KEY,
        r2_access_key_id=_FAKE_ACCESS_KEY,
        r2_secret_access_key=SecretStr(_FAKE_R2_SECRET),
        ntfy_url=_FAKE_NTFY,
        smtp_password=_FAKE_SMTP_PW,
    )


def _assert_no_secrets(blob: str) -> None:
    for secret in (
        _FAKE_JWT,
        _FAKE_API_KEY,
        _FAKE_R2_SECRET,
        _FAKE_SESSION,
        _FAKE_NTFY,
        _FAKE_SMTP_PW,
        _FAKE_ACCESS_KEY,
    ):
        assert secret not in blob, f"leaked {secret!r} in:\n{blob}"


def test_repr_settings_hides_secret_values() -> None:
    settings = _settings_with_fakes()
    text = repr(settings)
    _assert_no_secrets(text)
    assert "**********" in text or "SecretStr" in text


def test_validation_error_hides_secret_values() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings(
            fpl_api_token=_FAKE_JWT,
            entry_id="not-an-int",  # type: ignore[arg-type]
        )
    _assert_no_secrets(str(exc.value))


def test_secret_value_unwraps_and_blanks() -> None:
    assert secret_value(None) is None
    assert secret_value(SecretStr("")) is None
    assert secret_value(SecretStr("abc")) == "abc"
    assert secret_value("plain") == "plain"


def test_configured_secrets_lists_unwraped_values() -> None:
    secrets = _settings_with_fakes().configured_secrets()
    assert _FAKE_JWT in secrets
    assert _FAKE_API_KEY in secrets
    assert _FAKE_R2_SECRET in secrets
    assert _FAKE_NTFY in secrets


def test_derived_secret_forms_include_bearer_and_urlencoded() -> None:
    forms = derived_secret_forms(f"Bearer {_FAKE_JWT}")
    assert _FAKE_JWT in forms
    assert f"Bearer {_FAKE_JWT}" in forms
    assert quote(_FAKE_JWT, safe="") in forms


def test_redact_text_masks_headers_and_query_params() -> None:
    raw = (
        "Authorization: Bearer leaked-token "
        "x-api-authorization: Bearer other-token "
        "api-key: key-value "
        "GET https://example.com/x?token=query-tok&key=query-key&ok=1"
    )
    out = redact_text(raw)
    assert "leaked-token" not in out
    assert "other-token" not in out
    assert "key-value" not in out
    assert "query-tok" not in out
    assert "query-key" not in out
    assert "***" in out
    assert "ok=1" in out


def test_redact_text_masks_configured_secrets_and_derived_forms() -> None:
    text = (
        f"token={_FAKE_JWT} bearer=Bearer {_FAKE_JWT} "
        f"enc={quote(_FAKE_JWT, safe='')} key={_FAKE_API_KEY}"
    )
    out = redact_text(text, secrets=[_FAKE_JWT, _FAKE_API_KEY])
    assert _FAKE_JWT not in out
    assert _FAKE_API_KEY not in out
    assert "***" in out


def test_filter_redacts_exception_text() -> None:
    filt = SecretRedactionFilter(secrets=[_FAKE_API_KEY])
    record = logging.LogRecord(
        name="t",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="boom",
        args=(),
        exc_info=None,
    )
    try:
        raise RuntimeError(f"failed with {_FAKE_API_KEY}")
    except RuntimeError:
        import sys

        record.exc_info = sys.exc_info()
    assert filt.filter(record) is True
    assert record.exc_text is not None
    assert _FAKE_API_KEY not in record.exc_text
    assert "***" in record.exc_text


class _RecordingHandler(logging.Handler):
    """Capture formatted log output after handler filters run."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


def test_ops_path_debug_logs_never_leak_secrets() -> None:
    """Run the live-team ops path at DEBUG with fake secrets; logs stay clean."""
    reset_logging_for_tests()
    settings = _settings_with_fakes()
    configure_logging(level="DEBUG", secrets=settings.configured_secrets())
    from fplt.utils.logging import redaction_filter

    sink = _RecordingHandler()
    sink.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    filt = redaction_filter()
    assert filt is not None
    sink.addFilter(filt)
    root = logging.getLogger()
    root.addHandler(sink)

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("x-api-authorization", "")
        return httpx.Response(403, json={"detail": auth})

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://fantasy.premierleague.com/api",
    )
    logger = logging.getLogger("fplt.data.fpl_client")
    try:
        logger.debug(
            "probe Authorization: Bearer %s api-key: %s url=?token=%s&key=%s",
            _FAKE_JWT,
            _FAKE_API_KEY,
            _FAKE_JWT,
            _FAKE_API_KEY,
        )
        logger.debug("settings repr %s", repr(settings))
        logger.debug(
            "ntfy=%s smtp=%s session=%s r2=%s access=%s",
            _FAKE_NTFY,
            _FAKE_SMTP_PW,
            _FAKE_SESSION,
            _FAKE_R2_SECRET,
            _FAKE_ACCESS_KEY,
        )
        with pytest.raises(FPLAPIError):
            load_current_team(settings=settings, client=http)
    finally:
        http.close()
        root.removeHandler(sink)
        reset_logging_for_tests()

    _assert_no_secrets("\n".join(sink.lines))


def test_configure_logging_loads_secrets_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_logging_for_tests()
    settings = _settings_with_fakes()
    monkeypatch.setattr("fplt.settings.get_settings", lambda: settings)
    configure_logging(verbose=True)
    from fplt.utils.logging import redaction_filter

    sink = _RecordingHandler()
    sink.setFormatter(logging.Formatter("%(message)s"))
    filt = redaction_filter()
    assert filt is not None
    sink.addFilter(filt)
    root = logging.getLogger()
    root.addHandler(sink)
    logger = logging.getLogger("fplt.test.redact")
    try:
        logger.debug("jwt=%s", _FAKE_JWT)
        assert _FAKE_JWT not in "\n".join(sink.lines)
    finally:
        root.removeHandler(sink)
        reset_logging_for_tests()


def test_validator_accepts_secret_str_instance() -> None:
    token = SecretStr(f"Bearer {_FAKE_JWT}")
    settings = Settings(fpl_api_token=token)
    assert secret_value(settings.fpl_api_token) == _FAKE_JWT


def test_filter_handles_bad_args_and_existing_exc_text() -> None:
    filt = SecretRedactionFilter(secrets=[_FAKE_API_KEY])
    bad = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="token=%s %s",
        args=(_FAKE_API_KEY,),  # arity mismatch → getMessage raises
        exc_info=None,
    )
    assert filt.filter(bad) is True
    assert _FAKE_API_KEY not in str(bad.msg)

    pre = logging.LogRecord(
        name="t",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="already formatted",
        args=(),
        exc_info=None,
    )
    pre.exc_text = f"Traceback with {_FAKE_API_KEY}"
    assert filt.filter(pre) is True
    assert pre.exc_text is not None
    assert _FAKE_API_KEY not in pre.exc_text


def test_filter_format_exception_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filt = SecretRedactionFilter(secrets=[_FAKE_API_KEY])
    record = logging.LogRecord(
        name="t",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="boom",
        args=(),
        exc_info=None,
    )
    try:
        raise RuntimeError(_FAKE_API_KEY)
    except RuntimeError:
        import sys

        record.exc_info = sys.exc_info()

    def _boom(_exc_info: object) -> str:
        raise RuntimeError("formatter failed")

    monkeypatch.setattr(
        logging.Formatter,
        "formatException",
        _boom,
    )
    assert filt.filter(record) is True
    assert record.exc_text == "***" or record.exc_text == ""


def test_filter_secrets_property_roundtrip() -> None:
    filt = SecretRedactionFilter(secrets=[_FAKE_JWT, ""])
    assert filt.secrets == (_FAKE_JWT,)
    filt.set_secrets([_FAKE_API_KEY])
    assert filt.secrets == (_FAKE_API_KEY,)
