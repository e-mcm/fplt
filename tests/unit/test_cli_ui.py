"""CLI ``fplt ui`` launches Streamlit on localhost."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from fplt.cli import _ui, main
from fplt.settings import get_settings


def test_ui_cli_invokes_streamlit_localhost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    calls: list[list[str]] = []

    def fake_run_command(argv: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(list(argv))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("fplt.cli.run_command", fake_run_command)
    try:
        with pytest.raises(SystemExit) as excinfo:
            main(["ui", "--port", "8502"])
        assert excinfo.value.code == 0
    finally:
        get_settings.cache_clear()
    assert len(calls) == 1
    argv = calls[0]
    assert argv[1:4] == ["-m", "streamlit", "run"]
    assert argv[argv.index("--server.address") + 1] == "localhost"
    assert argv[argv.index("--server.port") + 1] == "8502"
    assert Path(argv[4]).name == "app.py"


def test_ui_missing_streamlit_returns_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    code = _ui(SimpleNamespace(port=8501))
    assert code == 2
