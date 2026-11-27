"""Process-wide logging setup for CLI and scripts."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterable
from typing import Literal
from urllib.parse import quote, quote_plus

from fplt.settings import bearer_token

_configured = False
_redaction_filter: SecretRedactionFilter | None = None
_HANDLER_NAME = "fplt.stderr"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

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


def configure_logging(
    *,
    verbose: bool = False,
    quiet: bool = False,
    level: LogLevel | None = None,
    secrets: Iterable[str] | None = None,
) -> None:
    """Configure the root logger for fplt operators.

    Default is ``INFO`` (progress for plan / backtest / compare / API).
    ``verbose`` selects ``DEBUG``; ``quiet`` selects ``WARNING``. An explicit
    ``level`` wins over both flags. Idempotent for the process: later calls
    only change the level (and refresh the redaction secret set).

    A ``SecretRedactionFilter`` on the root logger replaces configured
    secrets and ``Authorization`` / query-token values with ``***`` before
    any handler (including public Actions logs) emits the record.

    Args:
        verbose: Log DEBUG (HTTP paths, per-gameweek walk-forward steps).
        quiet: Log WARNING and above only.
        level: Explicit level name; overrides ``verbose`` / ``quiet``.
        secrets: Optional explicit secret list. When omitted, loads from
            ``get_settings().configured_secrets()``.
    """
    global _configured, _redaction_filter
    if level is not None:
        chosen = getattr(logging, level)
    elif quiet:
        chosen = logging.WARNING
    elif verbose:
        chosen = logging.DEBUG
    else:
        chosen = logging.INFO

    root = logging.getLogger()
    if not _configured:
        _redaction_filter = SecretRedactionFilter()
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name(_HANDLER_NAME)
        handler.addFilter(_redaction_filter)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
        _configured = True
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
    """Remove fplt's stderr handler (leave pytest / third-party handlers)."""
    global _configured, _redaction_filter
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == _HANDLER_NAME:
            root.removeHandler(handler)
            handler.close()
    _redaction_filter = None
    _configured = False
