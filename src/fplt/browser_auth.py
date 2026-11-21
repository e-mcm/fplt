"""Read FPL OIDC tokens from a local browser profile (no password login).

Fantasy Premier League keeps the live ``access_token`` in **localStorage**
under keys starting with ``oidc.user:`` (PingOne). Legacy ``pl_profile``
cookies are not enough for ``/api/my-team/``. This module only reads on-disk
browser storage on the local machine and never uploads cookies or tokens.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from fplt.auth import AuthStatus, inspect_token, set_credentials

logger = logging.getLogger(__name__)

FPL_ORIGIN = "https://fantasy.premierleague.com"
_OIDC_KEY_PREFIX = "oidc.user:"
BrowserName = Literal[
    "auto",
    "chrome",
    "chromium",
    "edge",
    "brave",
    "firefox",
]


class BrowserAuthError(RuntimeError):
    """Local browser storage could not yield a usable FPL access token."""


@dataclass(frozen=True, slots=True)
class OidcSession:
    """Parsed ``oidc.user:*`` localStorage payload."""

    access_token: str
    refresh_token: str | None
    expires_at: int | None
    storage_key: str
    source: str


@dataclass(frozen=True, slots=True)
class BrowserProfile:
    """One discovered browser profile that may hold FPL localStorage."""

    browser: str
    label: str
    path: Path
    kind: Literal["chromium", "firefox"]


def discover_profiles(
    *,
    browser: BrowserName = "auto",
    home: Path | None = None,
) -> list[BrowserProfile]:
    """List local browser profiles that can hold FPL localStorage.

    Args:
        browser: Restrict to one browser, or ``auto`` for all supported.
        home: Override ``Path.home()`` (tests).

    Returns:
        Profiles in a stable search order (Chrome → Chromium → Edge → Brave →
        Firefox). Missing install paths are skipped.
    """
    root = home if home is not None else Path.home()
    found: list[BrowserProfile] = []
    want = None if browser == "auto" else browser

    for name, user_data in _chromium_user_data_dirs(root):
        if want is not None and want != name:
            continue
        if not user_data.is_dir():
            continue
        for profile_dir in _chromium_profile_dirs(user_data):
            leveldb = profile_dir / "Local Storage" / "leveldb"
            if leveldb.is_dir():
                found.append(
                    BrowserProfile(
                        browser=name,
                        label=profile_dir.name,
                        path=leveldb,
                        kind="chromium",
                    )
                )

    if want in (None, "firefox"):
        for profile_dir in _firefox_profile_dirs(root):
            store = profile_dir / "webappsstore.sqlite"
            if store.is_file():
                found.append(
                    BrowserProfile(
                        browser="firefox",
                        label=profile_dir.name,
                        path=store,
                        kind="firefox",
                    )
                )
    return found


def extract_oidc_session(
    profile: BrowserProfile,
    *,
    copy_locked: bool = True,
) -> OidcSession:
    """Read the newest ``oidc.user:*`` session from one profile.

    Args:
        profile: Discovered profile from ``discover_profiles``.
        copy_locked: For Chromium, copy LevelDB to a temp dir first so a
            running browser's lock is less likely to block the read.

    Returns:
        Parsed OIDC session with ``access_token``.

    Raises:
        BrowserAuthError: Storage missing, locked, or has no FPL session.
    """
    if profile.kind == "chromium":
        return _extract_chromium(profile, copy_locked=copy_locked)
    return _extract_firefox(profile)


def auth_from_browser(
    *,
    browser: str = "auto",
    profile: str | None = None,
    entry_id: int | None = None,
    env_path: Path | None = None,
    home: Path | None = None,
    copy_locked: bool = True,
) -> AuthStatus:
    """Pull an FPL access token from local browser storage into user env.

    Args:
        browser: Browser to search (``auto`` tries all).
        profile: Optional profile directory name (e.g. ``Default``,
            ``Profile 1``, or a Firefox profile id).
        entry_id: Optional ``FPLT_ENTRY_ID`` to store alongside the token.
        env_path: Override ``~/.config/fplt/env``.
        home: Override home directory (tests).
        copy_locked: Copy Chromium LevelDB before reading.

    Returns:
        ``AuthStatus`` after writing credentials.

    Raises:
        BrowserAuthError: No usable session found.
        ValueError: Token payload is empty after parse.
    """
    allowed: set[str] = {
        "auto",
        "chrome",
        "chromium",
        "edge",
        "brave",
        "firefox",
    }
    if browser not in allowed:
        raise BrowserAuthError(
            f"unsupported browser {browser!r}; choose one of {sorted(allowed)}"
        )
    browser_name = cast(BrowserName, browser)
    profiles = discover_profiles(browser=browser_name, home=home)
    if profile is not None:
        profiles = [p for p in profiles if p.label == profile]
    if not profiles:
        raise BrowserAuthError(
            _no_profiles_message(browser=browser_name, profile=profile)
        )

    errors: list[str] = []
    for candidate in profiles:
        try:
            session = extract_oidc_session(candidate, copy_locked=copy_locked)
        except BrowserAuthError as exc:
            errors.append(f"{candidate.browser}/{candidate.label}: {exc}")
            continue
        info = inspect_token(session.access_token)
        if info is not None and info.expired is True:
            logger.warning(
                "browser token for %s/%s looks expired (exp=%s); "
                "log into FPL in that browser and retry",
                candidate.browser,
                candidate.label,
                info.expires_at,
            )
        logger.info(
            "using FPL oidc session from %s/%s (%s)",
            candidate.browser,
            candidate.label,
            session.source,
        )
        return set_credentials(
            token=session.access_token,
            entry_id=entry_id,
            env_path=env_path,
        )

    detail = "; ".join(errors) if errors else "no profiles yielded a session"
    raise BrowserAuthError(
        "no FPL oidc.user session in local browser storage "
        f"({detail}). Log into https://fantasy.premierleague.com in Chrome "
        "or Firefox, close the browser if the DB is locked, then retry. "
        "Safari is not supported. Legacy pl_profile cookies are not enough."
    )


def parse_oidc_user_value(raw: str, *, source: str) -> OidcSession:
    """Parse an ``oidc.user:*`` localStorage JSON value.

    Args:
        raw: JSON string from browser storage.
        source: Human-readable origin for errors (browser/profile/key).

    Returns:
        ``OidcSession``.

    Raises:
        BrowserAuthError: JSON missing ``access_token``.
    """
    try:
        loaded: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BrowserAuthError(f"invalid oidc.user JSON at {source}") from exc
    if not isinstance(loaded, dict):
        raise BrowserAuthError(f"oidc.user JSON must be an object at {source}")
    data = cast(dict[str, object], loaded)
    token_obj = data.get("access_token")
    if not isinstance(token_obj, str) or not token_obj.strip():
        raise BrowserAuthError(f"oidc.user missing access_token at {source}")
    refresh_obj = data.get("refresh_token")
    expires_obj = data.get("expires_at")
    return OidcSession(
        access_token=token_obj.strip(),
        refresh_token=(refresh_obj.strip() if isinstance(refresh_obj, str) else None),
        expires_at=(
            int(expires_obj) if isinstance(expires_obj, (int, float)) else None
        ),
        storage_key=source,
        source=source,
    )


def _extract_chromium(profile: BrowserProfile, *, copy_locked: bool) -> OidcSession:
    from chromium_reader.localstorage import LocalStorageReader

    leveldb = profile.path
    read_path = leveldb
    tmp: Path | None = None
    try:
        if copy_locked:
            tmp = Path(tempfile.mkdtemp(prefix="fplt-ls-"))
            try:
                shutil.copytree(leveldb, tmp / "leveldb", dirs_exist_ok=True)
                read_path = tmp / "leveldb"
            except OSError as exc:
                logger.warning(
                    "could not copy Chromium LevelDB (%s); reading in place",
                    exc,
                )
                read_path = leveldb
        try:
            with LocalStorageReader(read_path) as reader:
                records = list(
                    reader.records(
                        host=re.compile(r"fantasy\.premierleague\.com"),
                        script_key=re.compile(r"^oidc\.user:"),
                    )
                )
        except OSError as exc:
            raise BrowserAuthError(
                f"cannot read Chromium localStorage at {leveldb} "
                f"(is the browser open / DB locked?): {exc}"
            ) from exc
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)

    live = [rec for rec in records if rec.is_live and rec.value]
    if not live:
        raise BrowserAuthError(
            f"no oidc.user localStorage for {FPL_ORIGIN} in "
            f"{profile.browser}/{profile.label}"
        )
    live.sort(key=lambda rec: rec.leveldb_seq_number, reverse=True)
    best = live[0]
    assert best.value is not None
    return parse_oidc_user_value(
        best.value,
        source=f"{profile.browser}/{profile.label}:{best.script_key}",
    )


def _extract_firefox(profile: BrowserProfile) -> OidcSession:
    store = profile.path
    tmp_copy: Path | None = None
    db_path = store
    try:
        # Firefox often locks the sqlite while running; copy first.
        fd, name = tempfile.mkstemp(prefix="fplt-ff-", suffix=".sqlite")
        os.close(fd)
        tmp_copy = Path(name)
        try:
            shutil.copy2(store, tmp_copy)
            db_path = tmp_copy
        except OSError as exc:
            logger.warning("could not copy Firefox webappsstore (%s)", exc)
            db_path = store
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise BrowserAuthError(
                f"cannot open Firefox webappsstore at {store} "
                f"(close Firefox if locked): {exc}"
            ) from exc
        try:
            rows = conn.execute(
                "SELECT originKey, key, value FROM webappsstore2 WHERE key LIKE ?",
                (f"{_OIDC_KEY_PREFIX}%",),
            ).fetchall()
        except sqlite3.Error as exc:
            raise BrowserAuthError(f"Firefox webappsstore query failed: {exc}") from exc
        finally:
            conn.close()
    finally:
        if tmp_copy is not None:
            tmp_copy.unlink(missing_ok=True)

    sessions: list[OidcSession] = []
    fallback: list[OidcSession] = []
    for origin_key, key, value in rows:
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if "access_token" not in value:
            continue
        origin = origin_key if isinstance(origin_key, str) else ""
        try:
            session = parse_oidc_user_value(
                value,
                source=f"{profile.browser}/{profile.label}:{key}",
            )
        except BrowserAuthError:
            continue
        # Firefox originKey reverses the host (…ysatnaf… for fantasy).
        if (
            "ysatnaf" in origin.lower()
            or "premierleague" in origin.lower()
            or "premierleague" in value.lower()
            or "account.premierleague.com" in value
        ):
            sessions.append(session)
        else:
            fallback.append(session)
    chosen = sessions or fallback
    if not chosen:
        raise BrowserAuthError(
            f"no oidc.user localStorage in Firefox profile {profile.label}"
        )
    return chosen[0]


def _chromium_user_data_dirs(home: Path) -> list[tuple[str, Path]]:
    linux = home / ".config"
    mac = home / "Library" / "Application Support"
    return [
        ("chrome", linux / "google-chrome"),
        ("chrome", mac / "Google" / "Chrome"),
        ("chromium", linux / "chromium"),
        ("chromium", mac / "Chromium"),
        ("edge", linux / "microsoft-edge"),
        ("edge", mac / "Microsoft Edge"),
        ("brave", linux / "BraveSoftware" / "Brave-Browser"),
        ("brave", mac / "BraveSoftware" / "Brave-Browser"),
        ("chrome", mac / "Arc" / "User Data"),
    ]


def _chromium_profile_dirs(user_data: Path) -> list[Path]:
    names = ("Default", "Profile 1", "Profile 2", "Profile 3", "Profile 4")
    found = [user_data / name for name in names if (user_data / name).is_dir()]
    # Also pick up oddly named profiles that have Local Storage.
    for child in sorted(user_data.iterdir()):
        if not child.is_dir() or child in found:
            continue
        if (child / "Local Storage" / "leveldb").is_dir():
            found.append(child)
    return found


def _firefox_profile_dirs(home: Path) -> list[Path]:
    roots = [
        home / ".mozilla" / "firefox",
        home / "Library" / "Application Support" / "Firefox" / "Profiles",
    ]
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "webappsstore.sqlite").is_file():
                found.append(child)
    return found


def _no_profiles_message(*, browser: BrowserName, profile: str | None) -> str:
    bits = [
        "no local browser profiles found",
        f"(browser={browser}",
    ]
    if profile:
        bits.append(f", profile={profile}")
    bits.append(
        "). Supported: Chrome/Chromium/Edge/Brave (macOS+Linux) and Firefox. "
        "Safari is not supported. Close the browser if its storage is locked."
    )
    return "".join(bits)
