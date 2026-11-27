"""Process-wide logging setup for CLI and scripts.

Provides ISO-8601 UTC timestamps, optional JSON lines, a ``contextvars``
filter for run/ops identity fields, and a JSONL tee into run / official
checkpoint folders (issue #310). Builds on ``SecretRedactionFilter`` (#308).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, quote_plus

from fplt.settings import bearer_token

_configured = False
_redaction_filter: SecretRedactionFilter | None = None
_context_filter: ContextFilter | None = None
_HANDLER_NAME = "fplt.stderr"
_RUN_LOG_HANDLER_NAME = "fplt.run_log"
_OPS_LOG_HANDLER_NAME = "fplt.ops_log"
_LOG_FORMAT_ENV = "FPLT_LOG_FORMAT"

RUN_LOG_RELATIVE = "logs/run.log.jsonl"
OPS_LOG_FILENAME = "ops_log.jsonl"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
LogFormat = Literal["text", "json"]

CONTEXT_FIELDS: tuple[str, ...] = (
    "run_id",
    "experiment_id",
    "checkpoint",
    "line",
    "spec",
    "season",
    "gw",
)

_LogContext = dict[str, str | int | None]
_log_context: ContextVar[_LogContext] = ContextVar("fplt_log_context", default={})

_REDACTED = "***"

# Header names (and close variants) whose values must never appear in logs.
_HEADER_VALUE_RE = re.compile(
    r"(?i)(\b(?:authorization|x-api-authorization|api-key|x-apisports-key)"
    r"\b\s*[:=]\s*)(?:Bearer\s+)?([^\s,;\"']+)"
)
# Query params that commonly carry credentials.
_QUERY_SECRET_RE = re.compile(r"(?i)([?&](?:token|key)=)([^&\s\"']+)")


def derived_secret_forms(value: str) -> list[str]:
    """Return ``value`` plus common derived forms to redact.

    Includes with/without a ``Bearer `` prefix and URL-encoded variants so a
    log line cannot leak a GitHub-masked secret after local transformation.

    Args:
        value: Raw configured secret.

    Returns:
        Unique non-empty strings, longest first (for safe substring replace).
    """
    forms: set[str] = set()
    stripped = value.strip()
    if stripped:
        forms.add(stripped)
    normalised = bearer_token(value)
    if normalised:
        forms.add(normalised)
        forms.add(f"Bearer {normalised}")
        forms.add(f"bearer {normalised}")
    for form in list(forms):
        forms.add(quote(form, safe=""))
        forms.add(quote_plus(form))
    return sorted((f for f in forms if f), key=len, reverse=True)


def redact_text(text: str, *, secrets: Iterable[str] = ()) -> str:
    """Replace configured secrets and auth header / query values with ``***``.

    Args:
        text: Log message or exception text.
        secrets: Known secret strings (raw forms; derived forms are expanded).

    Returns:
        Redacted copy of ``text``.
    """
    out = text
    literals: list[str] = []
    for secret in secrets:
        literals.extend(derived_secret_forms(secret))
    # De-dupe while keeping longest-first order.
    seen: set[str] = set()
    ordered: list[str] = []
    for lit in sorted(literals, key=len, reverse=True):
        if lit not in seen:
            seen.add(lit)
            ordered.append(lit)
    for lit in ordered:
        if lit in out:
            out = out.replace(lit, _REDACTED)
    out = _HEADER_VALUE_RE.sub(rf"\1{_REDACTED}", out)
    out = _QUERY_SECRET_RE.sub(rf"\1{_REDACTED}", out)
    return out


class SecretRedactionFilter(logging.Filter):
    """Rewrite log records so secrets never reach handlers (incl. Actions)."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        """Create a filter with an initial secret set.

        Args:
            secrets: Credential strings to redact (plus derived forms).
        """
        super().__init__()
        self._secrets: tuple[str, ...] = ()
        self.set_secrets(secrets)

    def set_secrets(self, secrets: Iterable[str]) -> None:
        """Replace the set of secrets used for literal redaction.

        Args:
            secrets: Credential strings to redact.
        """
        self._secrets = tuple(s for s in secrets if s)

    @property
    def secrets(self) -> tuple[str, ...]:
        """Configured raw secrets (not derived forms)."""
        return self._secrets

    def redact(self, text: str) -> str:
        """Redact ``text`` with this filter's secrets and header/query rules.

        Args:
            text: Message or exception text.

        Returns:
            Redacted string.
        """
        return redact_text(text, secrets=self._secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        """Mutate ``record`` in place so every handler sees redacted text.

        Args:
            record: Incoming log record.

        Returns:
            Always ``True`` (records are kept, not dropped).
        """
        if record.args:
            try:
                message = record.getMessage()
            except Exception:  # noqa: BLE001 — never break logging on bad args
                message = str(record.msg)
            record.msg = self.redact(message)
            record.args = ()
        elif isinstance(record.msg, str):
            record.msg = self.redact(record.msg)
        if record.exc_info:
            try:
                exc_text = logging.Formatter().formatException(record.exc_info)
            except Exception:  # noqa: BLE001
                exc_text = ""
            record.exc_text = self.redact(exc_text)
        elif record.exc_text:
            record.exc_text = self.redact(record.exc_text)
        return True


def get_log_context() -> _LogContext:
    """Return a shallow copy of the current log context mapping."""
    return dict(_log_context.get())


def set_log_context(**fields: str | int | None) -> Token[_LogContext]:
    """Merge identity fields into the contextvars log context.

    Only keys in ``CONTEXT_FIELDS`` are accepted; unknown keys raise
    ``TypeError``. ``None`` values clear a previously set key.

    Args:
        **fields: Context keys to set (``run_id``, ``experiment_id``, …).

    Returns:
        Token for ``reset_log_context`` / ``Token.var.reset``.

    Raises:
        TypeError: An unknown field name was passed.
    """
    unknown = set(fields) - set(CONTEXT_FIELDS)
    if unknown:
        raise TypeError(f"unknown log context fields: {sorted(unknown)}")
    current = dict(_log_context.get())
    for key, value in fields.items():
        if value is None:
            current.pop(key, None)
        else:
            current[key] = value
    return _log_context.set(current)


def reset_log_context(token: Token[_LogContext]) -> None:
    """Restore the log context to the state before ``set_log_context``."""
    _log_context.reset(token)


def clear_log_context() -> Token[_LogContext]:
    """Clear all context fields; return a token to restore the prior map."""
    return _log_context.set({})


@contextmanager
def log_context(**fields: str | int | None) -> Iterator[None]:
    """Temporarily bind log context fields for the current task.

    Args:
        **fields: Context keys to set for the duration of the block.
    """
    token = set_log_context(**fields)
    try:
        yield
    finally:
        reset_log_context(token)


class ContextFilter(logging.Filter):
    """Inject ``CONTEXT_FIELDS`` from contextvars onto every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach context attributes (empty string when unset).

        Args:
            record: Incoming log record.

        Returns:
            Always ``True``.
        """
        ctx = _log_context.get()
        for name in CONTEXT_FIELDS:
            value = ctx.get(name)
            setattr(record, name, "" if value is None else value)
        return True


def format_iso_utc(record: logging.LogRecord) -> str:
    """Format ``record.created`` as ISO-8601 UTC with millisecond precision."""
    stamp = datetime.fromtimestamp(record.created, tz=UTC)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stamp.microsecond // 1000:03d}Z"


class IsoUtcFormatter(logging.Formatter):
    """Text formatter with ISO-8601 UTC timestamps and optional context."""

    def __init__(self) -> None:
        """Build the operator-facing stderr text format."""
        super().__init__(
            fmt=("%(asctime)s %(levelname)s [%(name)s]%(context)s %(message)s"),
        )

    def formatTime(  # noqa: N802 — logging.Formatter API
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        """Return ISO-8601 UTC; ``datefmt`` is ignored."""
        del datefmt
        return format_iso_utc(record)

    def format(self, record: logging.LogRecord) -> str:
        """Render one text line, appending non-empty context fields."""
        parts: list[str] = []
        for name in CONTEXT_FIELDS:
            value = getattr(record, name, "")
            if value != "" and value is not None:
                parts.append(f"{name}={value}")
        record.context = f" {{{' '.join(parts)}}}" if parts else ""
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """One JSON object per line with level, logger, message, and context."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize ``record`` as a single JSON object string."""
        payload: dict[str, Any] = {
            "ts": format_iso_utc(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name in CONTEXT_FIELDS:
            value = getattr(record, name, "")
            if value != "" and value is not None:
                payload[name] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exception"] = record.exc_text
        return json.dumps(payload, ensure_ascii=False, default=str)


def resolve_log_format(
    log_format: LogFormat | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> LogFormat:
    """Resolve text vs JSON from an explicit arg or ``FPLT_LOG_FORMAT``.

    Args:
        log_format: Explicit choice; wins when set.
        environ: Environment mapping; defaults to ``os.environ``.

    Returns:
        ``\"text\"`` or ``\"json\"`` (default ``text``).

    Raises:
        ValueError: ``FPLT_LOG_FORMAT`` is set to an unknown value.
    """
    if log_format is not None:
        return log_format
    env = os.environ if environ is None else environ
    raw = env.get(_LOG_FORMAT_ENV, "").strip().lower()
    if not raw:
        return "text"
    if raw == "text" or raw == "json":
        return raw
    raise ValueError(f"{_LOG_FORMAT_ENV}={raw!r} is invalid; expected 'text' or 'json'")


def _make_formatter(log_format: LogFormat) -> logging.Formatter:
    if log_format == "json":
        return JsonFormatter()
    return IsoUtcFormatter()


def _ensure_context_defaults(record: logging.LogRecord) -> None:
    for name in CONTEXT_FIELDS:
        if not hasattr(record, name):
            setattr(record, name, "")


class JsonlFileHandler(logging.Handler):
    """Append INFO+ records as JSON lines to a file (write-once open)."""

    def __init__(self, path: Path, *, level: int = logging.INFO) -> None:
        """Open ``path`` for append and install a ``JsonFormatter``.

        Args:
            path: Destination JSONL file (parent dirs are created).
            level: Minimum level to persist (default INFO).
        """
        super().__init__(level=level)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self.setFormatter(JsonFormatter())
        self.addFilter(ContextFilter())
        # Redact even when configure_logging has not run yet.
        redactor = redaction_filter()
        self.addFilter(redactor if redactor is not None else SecretRedactionFilter())

    def emit(self, record: logging.LogRecord) -> None:
        """Write one JSON line; never raise into the caller."""
        try:
            _ensure_context_defaults(record)
            line = self.format(record)
            self._file.write(line + "\n")
            self._file.flush()
        except Exception:  # noqa: BLE001 — logging must not break the app
            self.handleError(record)

    def close(self) -> None:
        """Flush and close the underlying file."""
        try:
            self._file.flush()
            self._file.close()
        finally:
            super().close()


class JsonlBufferHandler(logging.Handler):
    """Collect INFO+ JSON lines in memory for a later write-once flush."""

    def __init__(self, *, level: int = logging.INFO) -> None:
        """Create an empty buffer handler with JSON formatting."""
        super().__init__(level=level)
        self.lines: list[str] = []
        self.setFormatter(JsonFormatter())
        self.addFilter(ContextFilter())
        redactor = redaction_filter()
        self.addFilter(redactor if redactor is not None else SecretRedactionFilter())

    def emit(self, record: logging.LogRecord) -> None:
        """Append one formatted JSON line to ``lines``."""
        try:
            _ensure_context_defaults(record)
            self.lines.append(self.format(record))
        except Exception:  # noqa: BLE001
            self.handleError(record)


def attach_run_log(path: Path) -> JsonlFileHandler:
    """Tee root INFO+ records into ``path`` as JSONL.

    Args:
        path: Usually ``runs/run=<id>/logs/run.log.jsonl``.

    Returns:
        The attached handler (pass to ``detach_run_log``).
    """
    root = logging.getLogger()
    handler = JsonlFileHandler(path)
    handler.set_name(_RUN_LOG_HANDLER_NAME)
    root.addHandler(handler)
    return handler


def detach_run_log(handler: logging.Handler | None) -> None:
    """Remove and close a handler previously returned by ``attach_run_log``."""
    if handler is None:
        return
    root = logging.getLogger()
    root.removeHandler(handler)
    handler.close()


def ops_log_path(checkpoint_dir: Path) -> Path:
    """Return ``checkpoint_dir/ops_log.jsonl`` under an official checkpoint."""
    return Path(checkpoint_dir) / OPS_LOG_FILENAME


def write_ops_log_once(checkpoint_dir: Path, lines: Sequence[str]) -> Path:
    """Write ``ops_log.jsonl`` once under an official checkpoint directory.

    Args:
        checkpoint_dir: ``official/season=…/gw=…/checkpoint=…``.
        lines: Already-redacted JSON lines (no trailing newlines required).

    Returns:
        Path written.

    Raises:
        FileExistsError: The ops log already exists (write-once).
    """
    dest = ops_log_path(checkpoint_dir)
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = ""
    if lines:
        body = "\n".join(lines) + "\n"
    dest.write_text(body, encoding="utf-8")
    return dest


@contextmanager
def capture_ops_log() -> Iterator[JsonlBufferHandler]:
    """Attach a buffering JSONL handler for the duration of an ops checkpoint.

    Yields:
        The buffer handler; call ``write_ops_log_once`` with ``handler.lines``.
    """
    root = logging.getLogger()
    handler = JsonlBufferHandler()
    handler.set_name(_OPS_LOG_HANDLER_NAME)
    root.addHandler(handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        handler.close()


def configure_logging(
    *,
    verbose: bool = False,
    quiet: bool = False,
    level: LogLevel | None = None,
    secrets: Iterable[str] | None = None,
    log_format: LogFormat | None = None,
) -> None:
    """Configure the root logger for fplt operators.

    Default is ``INFO`` (progress for plan / backtest / compare / API).
    ``verbose`` selects ``DEBUG``; ``quiet`` selects ``WARNING``. An explicit
    ``level`` wins over both flags. Idempotent for the process: later calls
    only change the level / format (and refresh the redaction secret set).

    A ``SecretRedactionFilter`` on the stderr handler replaces configured
    secrets and ``Authorization`` / query-token values with ``***`` before
    emit (public Actions logs). A ``ContextFilter`` injects run/ops identity
    fields from contextvars.

    Args:
        verbose: Log DEBUG (HTTP paths, per-gameweek walk-forward steps).
        quiet: Log WARNING and above only.
        level: Explicit level name; overrides ``verbose`` / ``quiet``.
        secrets: Optional explicit secret list. When omitted, loads from
            ``get_settings().configured_secrets()``.
        log_format: ``text`` (default) or ``json``. When omitted, reads
            ``FPLT_LOG_FORMAT`` (workflows set ``json``).
    """
    global _configured, _redaction_filter, _context_filter
    if level is not None:
        chosen = getattr(logging, level)
    elif quiet:
        chosen = logging.WARNING
    elif verbose:
        chosen = logging.DEBUG
    else:
        chosen = logging.INFO

    fmt = resolve_log_format(log_format)
    root = logging.getLogger()
    if not _configured:
        _redaction_filter = SecretRedactionFilter()
        _context_filter = ContextFilter()
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name(_HANDLER_NAME)
        handler.addFilter(_context_filter)
        handler.addFilter(_redaction_filter)
        handler.setFormatter(_make_formatter(fmt))
        root.addHandler(handler)
        _configured = True
    else:
        for handler in root.handlers:
            if handler.get_name() == _HANDLER_NAME:
                handler.setFormatter(_make_formatter(fmt))
    assert _redaction_filter is not None
    if secrets is not None:
        _redaction_filter.set_secrets(secrets)
    else:
        from fplt.settings import get_settings

        _redaction_filter.set_secrets(get_settings().configured_secrets())
    root.setLevel(chosen)
    # Keep httpx chatter down unless the operator asked for DEBUG.
    logging.getLogger("httpx").setLevel(
        logging.DEBUG if chosen <= logging.DEBUG else logging.WARNING
    )
    logging.getLogger("httpcore").setLevel(
        logging.DEBUG if chosen <= logging.DEBUG else logging.WARNING
    )


def redaction_filter() -> SecretRedactionFilter | None:
    """Return the process redaction filter, if ``configure_logging`` has run."""
    return _redaction_filter


def reset_logging_for_tests() -> None:
    """Remove fplt handlers and clear context (leave pytest handlers)."""
    global _configured, _redaction_filter, _context_filter
    root = logging.getLogger()
    for handler in list(root.handlers):
        name = handler.get_name()
        if name in {_HANDLER_NAME, _RUN_LOG_HANDLER_NAME, _OPS_LOG_HANDLER_NAME}:
            root.removeHandler(handler)
            handler.close()
    _redaction_filter = None
    _context_filter = None
    _configured = False
    _log_context.set({})
