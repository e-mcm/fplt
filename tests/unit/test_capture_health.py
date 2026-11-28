"""Unit tests for fplt.data.capture_health (no network)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fplt.data.capture_health import (
    DEFAULT_THRESHOLD,
    load_unavailable_state,
    load_unavailable_threshold,
    record_capture_outcome,
    status_path,
)

_NOW = datetime(2026, 11, 28, 4, 34, tzinfo=UTC)


def test_load_threshold_default_and_missing_file(tmp_path: Path) -> None:
    assert load_unavailable_threshold(tmp_path / "missing.yaml") == DEFAULT_THRESHOLD


def test_load_threshold_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "capture_health.yaml"
    path.write_text("unavailable_threshold: 5\n", encoding="utf-8")
    assert load_unavailable_threshold(path) == 5


def test_load_threshold_env_overrides_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "capture_health.yaml"
    path.write_text("unavailable_threshold: 5\n", encoding="utf-8")
    monkeypatch.setenv("FPLT_CAPTURE_UNAVAILABLE_THRESHOLD", "2")
    assert load_unavailable_threshold(path) == 2


def test_load_threshold_rejects_non_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_CAPTURE_UNAVAILABLE_THRESHOLD", "0")
    with pytest.raises(ValueError, match="must be >= 1"):
        load_unavailable_threshold(tmp_path / "x.yaml")


def test_missing_status_file_starts_at_one(tmp_path: Path) -> None:
    path = status_path(tmp_path, "understat")
    assert load_unavailable_state(path) is None
    result = record_capture_outcome(
        source="understat",
        status_dir=tmp_path,
        available=False,
        reason="maintenance HTML",
        threshold=3,
        now=_NOW,
        emit_warning=False,
    )
    assert result.consecutive_unavailable == 1
    assert result.exit_code == 0
    assert result.ping_heartbeat is False
    assert result.available is False
    state = load_unavailable_state(path)
    assert state is not None
    assert state["consecutive_unavailable"] == 1
    assert state["last_reason"] == "maintenance HTML"


def test_counter_increments_and_threshold_fails(tmp_path: Path) -> None:
    for expected in (1, 2):
        result = record_capture_outcome(
            source="understat",
            status_dir=tmp_path,
            available=False,
            reason="down",
            threshold=3,
            now=_NOW,
            emit_warning=False,
        )
        assert result.consecutive_unavailable == expected
        assert result.exit_code == 0
        assert result.ping_heartbeat is False
    result = record_capture_outcome(
        source="understat",
        status_dir=tmp_path,
        available=False,
        reason="down",
        threshold=3,
        now=_NOW,
        emit_warning=False,
    )
    assert result.consecutive_unavailable == 3
    assert result.exit_code == 1
    assert result.ping_heartbeat is False


def test_success_resets_counter(tmp_path: Path) -> None:
    record_capture_outcome(
        source="api_football",
        status_dir=tmp_path,
        available=False,
        reason="500",
        threshold=3,
        now=_NOW,
        emit_warning=False,
    )
    result = record_capture_outcome(
        source="api_football",
        status_dir=tmp_path,
        available=True,
        threshold=3,
        now=_NOW,
    )
    assert result.consecutive_unavailable == 0
    assert result.exit_code == 0
    assert result.ping_heartbeat is True
    assert result.available is True
    state = load_unavailable_state(status_path(tmp_path, "api_football"))
    assert state is not None
    assert state["consecutive_unavailable"] == 0
    assert state["available"] is True


def test_emit_warning_annotation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record_capture_outcome(
        source="football_data",
        status_dir=tmp_path,
        available=False,
        reason="connection reset",
        threshold=3,
        now=_NOW,
        emit_warning=True,
    )
    out = capsys.readouterr().out
    assert "::warning::" in out
    assert "football_data unavailable" in out
    assert "connection reset" in out


def test_github_outputs_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "github_output"
    out.write_text("", encoding="utf-8")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    record_capture_outcome(
        source="vaastav",
        status_dir=tmp_path / "status",
        available=False,
        reason="git clone failed",
        threshold=3,
        now=_NOW,
        emit_warning=False,
    )
    text = out.read_text(encoding="utf-8")
    assert "source_ok=false" in text
    assert "ping_heartbeat=false" in text


def test_status_file_roundtrip_json(tmp_path: Path) -> None:
    result = record_capture_outcome(
        source="understat",
        status_dir=tmp_path,
        available=False,
        reason="maintenance",
        threshold=3,
        now=_NOW,
        emit_warning=False,
    )
    raw = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert raw["source"] == "understat"
    assert raw["updated_at"] == "2026-11-28T04:34:00Z"


def test_invalid_source_name_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid capture source"):
        status_path(tmp_path, "../evil")


def test_repo_config_default_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    # configs/ops/capture_health.yaml ships with the issue default of 3.
    monkeypatch.delenv("FPLT_CAPTURE_UNAVAILABLE_THRESHOLD", raising=False)
    assert load_unavailable_threshold() == 3
