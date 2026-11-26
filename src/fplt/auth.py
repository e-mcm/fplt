"""Local FPL live-team credentials outside the repo.

Fantasy Premier League authenticates ``/api/my-team/{id}/`` with a Bearer JWT
from the ``x-api-authorization`` header (typically ~8 hours). There is no
supported password login for third-party tools. Prefer
``fplt auth from-browser`` (reads local browser ``oidc.user`` localStorage) or
``fplt auth set``; both write under ``~/.config/fplt/env`` so operators never
commit secrets into the checkout.

``check_token`` / ``fplt auth check`` (ops-01) watch JWT ``exp`` so a dead
token is noticed before a deadline run, without failing planning itself.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from fplt.settings import bearer_token, user_env_file

logger = logging.getLogger(__name__)

_TOKEN_KEY = "FPLT_FPL_API_TOKEN"
_ENTRY_KEY = "FPLT_ENTRY_ID"
_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")

# Operator fix shown in alerts (never include the token value itself).
FIX_COMMAND = "uv run fplt auth set --token …"
MANUAL_APPLY_BANNER = "apply manually: token expired"

AuthCheckLevel = Literal["ok", "soft_skip", "warning", "urgent"]
AuthCheckReason = Literal[
    "ok",
    "missing",
    "malformed",
    "expired",
    "expires_before_deadline",
    "expiring_soon",
    "soft_skip_missing",
]


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


@dataclass(frozen=True, slots=True)
class AuthCheckResult:
    """Outcome of ``check_token`` for CLI exit codes and ops notifications.

    Attributes:
        level: ``ok`` / ``soft_skip`` (exit 0) or ``warning`` / ``urgent``
            (exit 1). Soft-skip is for unconfigured CI/crons.
        reason: Stable machine-readable cause.
        message: Human body for stdout / ntfy (never contains the token).
        expires_at: JWT ``exp`` when known.
        next_deadline: Staged deadline used for the deadline-aware check.
        time_to_expiry: ``expires_at - now`` when both are known.
        manual_apply: True when planning should carry the manual-apply banner.
    """

    level: AuthCheckLevel
    reason: AuthCheckReason
    message: str
    expires_at: datetime | None = None
    next_deadline: datetime | None = None
    time_to_expiry: timedelta | None = None
    manual_apply: bool = False

    @property
    def ok(self) -> bool:
        """True when the CLI / cron should exit zero."""
        return self.level in ("ok", "soft_skip")

    @property
    def exit_code(self) -> int:
        """Process exit code: 0 for ok/soft_skip, 1 otherwise."""
        return 0 if self.ok else 1


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


def resolve_token_raw(*, env_path: Path | None = None) -> str | None:
    """Resolve the live-team token from process env, then the user env file.

    Process env wins so GitHub Actions secrets (``FPLT_FPL_API_TOKEN``) are
    visible without writing ``~/.config/fplt/env`` on the runner.

    Args:
        env_path: Override for the env file path.

    Returns:
        Normalised Bearer value, or ``None`` when unset.
    """
    from_env = bearer_token(os.environ.get(_TOKEN_KEY))
    if from_env is not None:
        return from_env
    auth = status(env_path=env_path)
    if auth.token is None:
        return None
    return auth.token.token


def check_token(
    *,
    raw: str | None = None,
    now: datetime | None = None,
    warn_days: float = 3.0,
    next_deadline: datetime | None = None,
    deadline_aware: bool = True,
    allow_missing: bool = False,
    env_path: Path | None = None,
) -> AuthCheckResult:
    """Classify FPL API token expiry for ops monitors (ops-01).

    Args:
        raw: Token to inspect. When ``None``, resolves via ``resolve_token_raw``.
        now: Clock for expiry checks; defaults to UTC now.
        warn_days: Emit a warning when ``exp`` falls within this many days
            (and after the next deadline when deadline-aware).
        next_deadline: Upcoming ``gameweeks.deadline_time``. When ``None`` and
            ``deadline_aware`` is True, only the warn-days window applies.
        deadline_aware: When True and ``next_deadline`` is set, treat expiry
            before that deadline as urgent.
        allow_missing: Soft-skip (exit 0, no alert) when the token is unset so
            CI/crons stay green before secrets are configured.
        env_path: Override for the env file when resolving ``raw``.

    Returns:
        ``AuthCheckResult``. Messages never include the token value.
    """
    if warn_days < 0:
        raise ValueError("warn_days must be >= 0")
    moment = now if now is not None else datetime.now(tz=UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    deadline = next_deadline
    if deadline is not None and deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)

    token_raw = raw if raw is not None else resolve_token_raw(env_path=env_path)
    info = inspect_token(token_raw, now=moment)
    if info is None:
        if allow_missing:
            return AuthCheckResult(
                level="soft_skip",
                reason="soft_skip_missing",
                message=(
                    "auth check soft-skip: FPLT_FPL_API_TOKEN unset "
                    "(configure the secret or run "
                    f"{FIX_COMMAND}; planning continues on YAML squad)"
                ),
                next_deadline=deadline,
            )
        return AuthCheckResult(
            level="urgent",
            reason="missing",
            message=(
                "FPLT_FPL_API_TOKEN is missing. Live squad / apply need a JWT. "
                f"Fix: {FIX_COMMAND}" + _deadline_suffix(deadline)
            ),
            next_deadline=deadline,
            manual_apply=True,
        )

    if info.expires_at is None:
        # Present but not a JWT with exp — treat as unparseable for ops.
        return AuthCheckResult(
            level="urgent",
            reason="malformed",
            message=(
                "FPLT_FPL_API_TOKEN is set but has no usable exp claim "
                "(malformed or non-JWT). Paste a fresh x-api-authorization "
                f"Bearer value. Fix: {FIX_COMMAND}" + _deadline_suffix(deadline)
            ),
            next_deadline=deadline,
            manual_apply=True,
        )

    expires_at = info.expires_at
    ttl = expires_at - moment
    warn_horizon = timedelta(days=warn_days)

    if info.expired is True or ttl <= timedelta(0):
        return AuthCheckResult(
            level="urgent",
            reason="expired",
            message=(
                f"{MANUAL_APPLY_BANNER}. Token expired at "
                f"{expires_at.isoformat()} "
                f"({_format_duration(-ttl)} ago). Planning continues; "
                f"apply by hand until refreshed. Fix: {FIX_COMMAND}"
                + _deadline_suffix(deadline)
            ),
            expires_at=expires_at,
            next_deadline=deadline,
            time_to_expiry=ttl,
            manual_apply=True,
        )

    if deadline_aware and deadline is not None and expires_at <= deadline:
        return AuthCheckResult(
            level="urgent",
            reason="expires_before_deadline",
            message=(
                "FPL API token expires at "
                f"{expires_at.isoformat()} "
                f"(in {_format_duration(ttl)}), before the next deadline "
                f"{deadline.isoformat()}. Refresh before then. "
                f"Fix: {FIX_COMMAND}"
            ),
            expires_at=expires_at,
            next_deadline=deadline,
            time_to_expiry=ttl,
            manual_apply=False,
        )

    if ttl <= warn_horizon:
        return AuthCheckResult(
            level="warning",
            reason="expiring_soon",
            message=(
                "FPL API token expires at "
                f"{expires_at.isoformat()} "
                f"(in {_format_duration(ttl)}; warn window "
                f"{warn_days:g}d). Fix: {FIX_COMMAND}" + _deadline_suffix(deadline)
            ),
            expires_at=expires_at,
            next_deadline=deadline,
            time_to_expiry=ttl,
            manual_apply=False,
        )

    return AuthCheckResult(
        level="ok",
        reason="ok",
        message=(
            f"FPL API token ok until {expires_at.isoformat()} "
            f"(in {_format_duration(ttl)})" + _deadline_suffix(deadline)
        ),
        expires_at=expires_at,
        next_deadline=deadline,
        time_to_expiry=ttl,
        manual_apply=False,
    )


def format_auth_check(result: AuthCheckResult) -> str:
    """Render ``check_token`` for CLI stdout.

    Args:
        result: Outcome from ``check_token``.

    Returns:
        Multi-line text ending with a newline.
    """
    lines = [
        f"level: {result.level}",
        f"reason: {result.reason}",
        f"message: {result.message}",
    ]
    if result.expires_at is not None:
        lines.append(f"expires_at: {result.expires_at.isoformat()}")
    if result.time_to_expiry is not None:
        lines.append(f"time_to_expiry: {_format_duration(result.time_to_expiry)}")
    if result.next_deadline is not None:
        lines.append(f"next_deadline: {result.next_deadline.isoformat()}")
    if result.manual_apply:
        lines.append(f"banner: {MANUAL_APPLY_BANNER}")
    lines.append(f"fix: {FIX_COMMAND}")
    return "\n".join(lines) + "\n"


def with_manual_apply_banner(body: str, result: AuthCheckResult) -> str:
    """Prepend the manual-apply banner when the token is already dead.

    Args:
        body: Plan report or notification body.
        result: Auth check from the same ops run.

    Returns:
        ``body``, optionally prefixed with the banner line.
    """
    if not result.manual_apply:
        return body
    banner = f"*** {MANUAL_APPLY_BANNER} ***\n{result.message}\nFix: {FIX_COMMAND}"
    text = body.strip()
    if not text:
        return banner + "\n"
    if MANUAL_APPLY_BANNER in text:
        return text + "\n"
    return f"{banner}\n\n{text}\n"


def _deadline_suffix(deadline: datetime | None) -> str:
    if deadline is None:
        return ""
    return f" Next deadline: {deadline.isoformat()}."


def _format_duration(delta: timedelta) -> str:
    """Format a signed duration for alerts (no token material)."""
    total = int(delta.total_seconds())
    sign = "-" if total < 0 else ""
    total = abs(total)
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{sign}{days}d{hours}h"
    if hours:
        return f"{sign}{hours}h{minutes}m"
    return f"{sign}{minutes}m"


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
