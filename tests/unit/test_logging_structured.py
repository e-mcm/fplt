"""Structured logging: ISO UTC, JSON, contextvars, run/ops tees (#310)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fplt.schemas.manifests import AsOfDeadline
from fplt.store.official import official_checkpoint_dir
from fplt.store.runs import RunWriter
from fplt.utils.logging import (
    CONTEXT_FIELDS,
    RUN_LOG_RELATIVE,
    ContextFilter,
    JsonFormatter,
    SecretRedactionFilter,
    attach_run_log,
    capture_ops_log,
    clear_log_context,
    configure_logging,
    detach_run_log,
    format_iso_utc,
    get_log_context,
    log_context,
    ops_log_path,
    reset_logging_for_tests,
    resolve_log_format,
    set_log_context,
    write_ops_log_once,
)


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    """Keep handlers and context from leaking across tests."""
    reset_logging_for_tests()
    clear_log_context()
    yield
    reset_logging_for_tests()
    clear_log_context()


_TS = datetime(2026, 11, 24, 12, 56, 1, tzinfo=UTC)


def _writer(tmp_path: Path, **overrides: object) -> RunWriter:
    kwargs: dict[str, object] = {
        "kind": "adhoc",
        "mode": "live",
        "run_type": "predict",
        "spec_hash": "a" * 64,
        "spec_label": "demo",
        "model_id": "deadbeef",
        "as_of": [AsOfDeadline(season="2026-27", gw=12, deadline_utc=_TS)],
        "horizon": 3,
        "code_sha": "abc1234",
        "code_dirty": False,
        "lockfile_hash": "b" * 64,
        "host": "test",
        "raw_manifest_hash": "c" * 64,
        "staged_digests": {},
        "started_at": _TS,
        "run_id": "r_20261124T125601Z_7f3a2c",
        "checkpoint": "T-90m",
        "line": "v4_h5",
        "experiment_id": "exp_demo",
    }
    kwargs.update(overrides)
    return RunWriter(tmp_path, **kwargs)  # type: ignore[arg-type]


def test_log_context_none_default_reads_as_empty() -> None:
    """ContextVar default is None (B039); readers treat that as {}."""
    reset_logging_for_tests()
    assert get_log_context() == {}
    filter_ = ContextFilter()
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hi",
        args=(),
        exc_info=None,
    )
    assert filter_.filter(record) is True
    assert record.run_id == ""  # type: ignore[attr-defined]


def test_format_iso_utc_has_date_and_z() -> None:
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hi",
        args=(),
        exc_info=None,
    )
    record.created = datetime(2026, 11, 28, 3, 4, 9, 123_000, tzinfo=UTC).timestamp()
    stamp = format_iso_utc(record)
    assert stamp == "2026-11-28T03:04:09.123Z"


def test_json_formatter_includes_context_fields() -> None:
    configure_logging(secrets=(), log_format="json")
    with log_context(run_id="r1", season="2025-26", gw=3, checkpoint="T-90m"):
        record = logging.LogRecord(
            name="fplt.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        for name in CONTEXT_FIELDS:
            setattr(record, name, get_log_context().get(name, "") or "")
        line = JsonFormatter().format(record)
    payload = json.loads(line)
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "fplt.test"
    assert payload["run_id"] == "r1"
    assert payload["season"] == "2025-26"
    assert payload["gw"] == 3
    assert payload["checkpoint"] == "T-90m"
    assert payload["ts"].endswith("Z")
    assert "T" in payload["ts"]


def test_resolve_log_format_env_and_explicit() -> None:
    assert resolve_log_format(environ={}) == "text"
    assert resolve_log_format(environ={"FPLT_LOG_FORMAT": "json"}) == "json"
    assert resolve_log_format("text", environ={"FPLT_LOG_FORMAT": "json"}) == "text"
    with pytest.raises(ValueError, match="FPLT_LOG_FORMAT"):
        resolve_log_format(environ={"FPLT_LOG_FORMAT": "yaml"})


def test_run_writer_sets_and_clears_context_and_writes_run_log(
    tmp_path: Path,
) -> None:
    configure_logging(secrets=("super-secret-token",), log_format="json")
    logger = logging.getLogger("fplt.test.run_writer")
    assert get_log_context() == {}
    with _writer(tmp_path) as writer:
        ctx = get_log_context()
        assert ctx["run_id"] == writer.run_id
        assert ctx["experiment_id"] == "exp_demo"
        assert ctx["checkpoint"] == "T-90m"
        assert ctx["line"] == "v4_h5"
        assert ctx["spec"] == "demo"
        assert ctx["season"] == "2026-27"
        assert ctx["gw"] == 12
        logger.info("inside run token=super-secret-token")
    assert get_log_context() == {}
    log_path = writer.path / RUN_LOG_RELATIVE
    assert log_path.is_file()
    lines = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines
    assert any(row.get("run_id") == writer.run_id for row in lines)
    joined = log_path.read_text(encoding="utf-8")
    assert "super-secret-token" not in joined
    assert "***" in joined


def test_ops_log_write_once_and_redaction(tmp_path: Path) -> None:
    configure_logging(secrets=("ops-secret-value",))
    checkpoint_dir = official_checkpoint_dir(
        tmp_path, season="2026-27", gw=12, checkpoint="T-90m"
    )
    with (
        log_context(checkpoint="T-90m", season="2026-27", gw=12),
        capture_ops_log() as buf,
    ):
        logging.getLogger("fplt.test.ops").info("tick with ops-secret-value")
    path = write_ops_log_once(checkpoint_dir, buf.lines)
    assert path == ops_log_path(checkpoint_dir)
    body = path.read_text(encoding="utf-8")
    assert "ops-secret-value" not in body
    assert "***" in body
    row = json.loads(body.strip().splitlines()[-1])
    assert row["checkpoint"] == "T-90m"
    assert row["season"] == "2026-27"
    assert row["gw"] == 12
    with pytest.raises(FileExistsError):
        write_ops_log_once(checkpoint_dir, ["{}"])


def test_text_format_uses_iso_utc(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(secrets=(), log_format="text")
    logging.getLogger("fplt.test.text").info("plain")
    err = capsys.readouterr().err
    # HH:MM:SS alone is no longer accepted; require date + Z.
    assert "T" in err
    assert "Z" in err
    assert "plain" in err


def test_attach_detach_run_log_roundtrip(tmp_path: Path) -> None:
    configure_logging(secrets=())
    dest = tmp_path / "logs" / "run.log.jsonl"
    handler = attach_run_log(dest)
    try:
        logging.getLogger("fplt.test.tee").info("teed")
    finally:
        detach_run_log(handler)
    assert dest.is_file()
    assert "teed" in dest.read_text(encoding="utf-8")


def test_set_log_context_rejects_unknown_field() -> None:
    with pytest.raises(TypeError, match="unknown"):
        set_log_context(not_a_field="x")  # type: ignore[arg-type]


def test_json_formatter_includes_exception() -> None:
    record = logging.LogRecord(
        name="fplt.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="boom",
        args=(),
        exc_info=None,
    )
    for name in CONTEXT_FIELDS:
        setattr(record, name, "")
    try:
        raise RuntimeError("fail")
    except RuntimeError:
        import sys

        record.exc_info = sys.exc_info()
    payload = json.loads(JsonFormatter().format(record))
    assert "RuntimeError" in payload["exception"]


def test_redaction_filter_still_on_stderr_handler() -> None:
    configure_logging(secrets=("sekrit",), log_format="json")
    root = logging.getLogger()
    redactors = [
        f
        for h in root.handlers
        for f in h.filters
        if isinstance(f, SecretRedactionFilter)
    ]
    assert redactors
    assert redactors[0].secrets == ("sekrit",)
