"""Local FPL live-team credentials outside the repo.

Fantasy Premier League authenticates ``/api/my-team/{id}/`` with a Bearer JWT
from the ``x-api-authorization`` header (typically ~8 hours). There is no
supported password login for third-party tools. This module stores a
DevTools-pasted token (and optional entry id) under ``~/.config/fplt/env`` so
operators never commit secrets into the checkout.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from fplt.settings import bearer_token, user_env_file

logger = logging.getLogger(__name__)

_TOKEN_KEY = "FPLT_FPL_API_TOKEN"
_ENTRY_KEY = "FPLT_ENTRY_ID"
_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


@dataclass(frozen=True, slots=True)
class TokenInfo:
    """Decoded access-token metadata (unverified JWT claims)."""

    token: str
    expires_at: datetime | None
    expired: bool | None
    subject: str | None


@dataclass(frozen=True, slots=True)
class AuthStatus:
    """Snapshot of local auth storage for CLI display."""

    env_path: Path
    env_exists: bool
    entry_id: int | None
    token: TokenInfo | None
    source_hint: str


def default_env_path() -> Path:
    """Return the user env file path (``FPLT_ENV_FILE`` or ``~/.config/fplt/env``)."""
    return user_env_file()


def inspect_token(raw: str | None, *, now: datetime | None = None) -> TokenInfo | None:
    """Normalise a Bearer token and read ``exp`` / ``sub`` without verifying.

    Args:
        raw: Token as pasted from DevTools (optional ``Bearer `` prefix).
        now: Clock for expiry checks; defaults to UTC now.

    Returns:
        ``TokenInfo``, or ``None`` when ``raw`` is blank.
    """
    token = bearer_token(raw)
    if token is None:
        return None
    claims = _jwt_claims(token)
    expires_at: datetime | None = None
    expired: bool | None = None
    subject: str | None = None
    if claims is not None:
        sub = claims.get("sub")
        if isinstance(sub, str) and sub:
            subject = sub
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            expires_at = datetime.fromtimestamp(float(exp), tz=UTC)
            clock = now if now is not None else datetime.now(tz=UTC)
            expired = expires_at <= clock
    return TokenInfo(
        token=token,
        expires_at=expires_at,
        expired=expired,
        subject=subject,
    )


def read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines from an env file.

    Args:
        path: File to read.

    Returns:
        Mapping of keys to raw values (quotes are kept as written).
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _LINE_RE.match(stripped)
        if match is None:
            continue
        values[match.group(1)] = match.group(2)
    return values


def upsert_env_file(
    path: Path,
    updates: dict[str, str | None],
    *,
    mode: int = 0o600,
) -> Path:
    """Create or update keys in an env file and set restrictive permissions.

    Args:
        path: Target env file (created with parents when missing).
        updates: Keys to set; ``None`` removes the key.
        mode: File mode after write (default ``0o600``).

    Returns:
        The path written.
    """
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: list[str] = []
    if path.is_file():
        existing_lines = path.read_text(encoding="utf-8").splitlines()

    seen: set[str] = set()
    out: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        match = (
            _LINE_RE.match(stripped)
            if stripped and not stripped.startswith("#")
            else None
        )
        if match is None:
            out.append(line)
            continue
        key = match.group(1)
        if key not in updates:
            out.append(line)
            continue
        seen.add(key)
        value = updates[key]
        if value is None:
            continue
        out.append(f"{key}={value}")

    for key, value in updates.items():
        if key in seen or value is None:
            continue
        out.append(f"{key}={value}")

    text = "\n".join(out)
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    # Best-effort: ensure directory is not world-writable.
    try:
        path.parent.chmod(stat.S_IRWXU)
    except OSError:
        logger.debug("could not chmod %s", path.parent, exc_info=True)
    return path


def set_credentials(
    *,
    token: str,
    entry_id: int | None = None,
    env_path: Path | None = None,
) -> AuthStatus:
    """Persist an API token (and optional entry id) outside the repo.

    Args:
        token: Bearer value from ``x-api-authorization`` (prefix optional).
        entry_id: Manager entry id to store as ``FPLT_ENTRY_ID``.
        env_path: Override for the env file path.

    Returns:
        Status after the write.

    Raises:
        ValueError: Token is blank after normalisation.
    """
    info = inspect_token(token)
    if info is None:
        raise ValueError("token is empty; paste the x-api-authorization Bearer value")
    path = env_path if env_path is not None else default_env_path()
    updates: dict[str, str | None] = {_TOKEN_KEY: info.token}
    if entry_id is not None:
        updates[_ENTRY_KEY] = str(entry_id)
    upsert_env_file(path, updates)
    logger.info("wrote FPL credentials to %s", path)
    return status(env_path=path)


def clear_credentials(
    *,
    env_path: Path | None = None,
    clear_entry_id: bool = False,
) -> AuthStatus:
    """Remove the stored API token from the user env file.

    Args:
        env_path: Override for the env file path.
        clear_entry_id: Also remove ``FPLT_ENTRY_ID``.

    Returns:
        Status after the clear.
    """
    path = env_path if env_path is not None else default_env_path()
    updates: dict[str, str | None] = {_TOKEN_KEY: None}
    if clear_entry_id:
        updates[_ENTRY_KEY] = None
    if path.is_file():
        upsert_env_file(path, updates)
    logger.info("cleared FPL API token in %s", path)
    return status(env_path=path)


def status(*, env_path: Path | None = None, now: datetime | None = None) -> AuthStatus:
    """Read local auth storage and summarise token expiry.

    Args:
        env_path: Override for the env file path.
        now: Clock for expiry checks.

    Returns:
        ``AuthStatus`` for CLI / library callers.
    """
    path = env_path if env_path is not None else default_env_path()
    values = read_env_file(path)
    entry_raw = values.get(_ENTRY_KEY)
    entry_id: int | None = None
    if entry_raw is not None and entry_raw.strip():
        try:
            entry_id = int(entry_raw.strip().strip('"').strip("'"))
        except ValueError:
            entry_id = None
    token = inspect_token(values.get(_TOKEN_KEY), now=now)
    if token is None:
        hint = "no token in user env; run: uv run fplt auth set"
    elif token.expired is True:
        hint = "token expired; paste a fresh x-api-authorization Bearer value"
    elif token.expires_at is not None:
        hint = f"token valid until {token.expires_at.isoformat()}"
    else:
        hint = "token stored (expiry claim missing; FPL usually issues ~8h JWTs)"
    return AuthStatus(
        env_path=path,
        env_exists=path.is_file(),
        entry_id=entry_id,
        token=token,
        source_hint=hint,
    )


def format_status(auth: AuthStatus) -> str:
    """Render a human-readable auth status block.

    Args:
        auth: Status from ``status``.

    Returns:
        Multi-line text for stdout (no trailing newline requirement beyond lines).
    """
    lines = [
        f"env_file: {auth.env_path}",
        f"exists: {str(auth.env_exists).lower()}",
        f"entry_id: {auth.entry_id if auth.entry_id is not None else '(unset)'}",
    ]
    if auth.token is None:
        lines.append("token: (unset)")
    else:
        exp = (
            auth.token.expires_at.isoformat()
            if auth.token.expires_at is not None
            else "(unknown)"
        )
        if auth.token.expired is True:
            state = "expired"
        elif auth.token.expired is False:
            state = "ok"
        else:
            state = "unknown"
        lines.append(f"token: set ({state})")
        lines.append(f"expires_at: {exp}")
        if auth.token.subject:
            lines.append(f"sub: {auth.token.subject}")
    lines.append(f"hint: {auth.source_hint}")
    lines.append("never paste FPLT_FPL_API_TOKEN into the repo or commit .env")
    return "\n".join(lines) + "\n"


def _jwt_claims(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    payload = parts[1]
    pad = "=" * (-len(payload) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload + pad)
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return cast(dict[str, Any], data)
