"""Empty env vars are unset for Settings and required settings fail clearly (#362)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from fplt.cli import main
from fplt.optimise.apply import apply_plan
from fplt.settings import (
    Settings,
    SettingsConfigError,
    get_settings,
    require_entry_id,
)


def test_empty_entry_id_and_smtp_port_are_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FPLT_ENTRY_ID", "")
    monkeypatch.setenv("FPLT_SMTP_PORT", "")
    monkeypatch.setenv("FPLT_SMTP_USE_TLS", "")
    get_settings.cache_clear()
    try:
        settings = Settings()
        assert settings.entry_id is None
        assert settings.smtp_port == 587
        assert settings.smtp_use_tls is True
    finally:
        get_settings.cache_clear()


def test_require_entry_id_names_env_and_github_secret() -> None:
    with pytest.raises(SettingsConfigError, match="FPLT_ENTRY_ID") as exc:
        require_entry_id(Settings(entry_id=None))
    assert "r2" in str(exc.value)
    assert require_entry_id(Settings(entry_id=42)) == 42


def test_apply_plan_unset_entry_id_raises_typed_error(tmp_path: Path) -> None:
    with pytest.raises(SettingsConfigError, match="FPLT_ENTRY_ID"):
        apply_plan(
            tmp_path,
            "rolling_mean_v1",
            "2024-25",
            1,
            settings=Settings(entry_id=None),
        )


def test_cli_configure_logging_validation_error_exits_2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid non-empty Settings env fails cleanly at logging startup (#362)."""
    from fplt.utils.logging import reset_logging_for_tests

    reset_logging_for_tests()
    monkeypatch.setenv("FPLT_ENTRY_ID", "not-an-int")
    get_settings.cache_clear()
    with pytest.raises(SystemExit) as exc:
        main(["status"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "entry_id" in err.lower() or "integer" in err.lower()
    assert "SettingsConfigError" not in err
    get_settings.cache_clear()
    reset_logging_for_tests()


def test_cli_apply_unset_entry_id_exits_2_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    from fplt.utils.logging import reset_logging_for_tests

    reset_logging_for_tests()
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("FPLT_ENTRY_ID", raising=False)
    get_settings.cache_clear()
    caplog.set_level(logging.ERROR)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "apply",
                "--model",
                "rolling_mean_v1",
                "--season",
                "2024-25",
                "--gw",
                "1",
            ]
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "FPLT_ENTRY_ID" in err
    # Config path must not dump a Python exception stack for the operator.
    assert "SettingsConfigError" not in err
    assert 'File "src/fplt' not in err
    assert any("FPLT_ENTRY_ID" in r.message for r in caplog.records)
    get_settings.cache_clear()
    reset_logging_for_tests()


def test_exit_config_error_survives_broken_handler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failing log handler must not dump SettingsConfigError to stderr (#425)."""
    from fplt.cli import _exit_config_error

    class _BrokenHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # Match StreamHandler-style failures: catch and handleError.
            try:
                raise OSError("stream closed")
            except OSError:
                self.handleError(record)

    root = logging.getLogger()
    broken = _BrokenHandler()
    root.addHandler(broken)
    previous = logging.raiseExceptions
    try:
        # Default True would print ``--- Logging error ---``; the helper must
        # suppress that so the exception class never reaches stderr.
        logging.raiseExceptions = True
        with pytest.raises(SystemExit) as exc:
            _exit_config_error(SettingsConfigError("FPLT_ENTRY_ID is not set"))
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "FPLT_ENTRY_ID" in err
        assert "SettingsConfigError" not in err
        assert "--- Logging error ---" not in err
    finally:
        root.removeHandler(broken)
        logging.raiseExceptions = previous
