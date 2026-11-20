"""Official Fantasy Premier League API client for the user's current team.

Public ``entry/{id}/event/{gw}/picks`` covers finished gameweeks only. The
live squad (selling prices, bank, free transfers) comes from authenticated
``my-team/{id}``, which needs the ``pl_profile`` session cookie. Set
``FPLT_ENTRY_ID`` and, for the live team, ``FPLT_FPL_SESSION``. Free-transfer
and chip overrides still live in ``configs/squad_state.yaml`` when the cookie
is unavailable.
"""

from __future__ import annotations

from typing import Any, Literal, Self, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field

from fplt.settings import Settings, get_settings

API_ROOT = "https://fantasy.premierleague.com/api"


class FPLAPIError(RuntimeError):
    """The FPL API returned an error or an unexpected payload."""


class EntryPick(BaseModel):
    """One player in an entry's fifteen."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    element: int
    position: int = Field(ge=1, le=15)
    selling_price: int | None = Field(default=None, ge=0)
    purchase_price: int | None = Field(default=None, ge=0)
    multiplier: int = 0
    is_captain: bool = False
    is_vice_captain: bool = False
    element_type: int | None = None


class CurrentTeam(BaseModel):
    """Typed current squad ready for the optimiser boundary.

    ``picks_payload`` matches ``entry/{id}/event/{gw}/picks`` closely enough
    for ``fplt.optimise.state.apply_entry_picks``. Selling prices are present
    when the source is ``my_team``; public finished-GW picks omit them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_id: int
    event: int | None = Field(default=None, ge=1, le=38)
    bank: int = Field(ge=0)
    value: int | None = Field(default=None, ge=0)
    free_transfers: int | None = Field(default=None, ge=0)
    picks: list[EntryPick]
    active_chip: str | None = None
    source: Literal["my_team", "entry_picks"]

    def picks_payload(self) -> dict[str, Any]:
        """Build the dict shape expected by ``apply_entry_picks``.

        Returns:
            A payload with ``picks`` and ``entry_history.bank``.

        Raises:
            ValueError: Any pick is missing ``selling_price``.
        """
        owned: list[dict[str, Any]] = []
        for pick in self.picks:
            if pick.selling_price is None:
                raise ValueError(
                    "selling_price missing; fetch via my-team with FPLT_FPL_SESSION"
                )
            owned.append(
                {
                    "element": pick.element,
                    "position": pick.position,
                    "selling_price": pick.selling_price,
                    "multiplier": pick.multiplier,
                    "is_captain": pick.is_captain,
                    "is_vice_captain": pick.is_vice_captain,
                }
            )
        return {
            "active_chip": self.active_chip,
            "entry_history": {
                "event": self.event,
                "bank": self.bank,
                "value": self.value,
            },
            "picks": owned,
        }

    @classmethod
    def from_my_team(cls, entry_id: int, payload: dict[str, Any]) -> Self:
        """Parse an authenticated ``my-team/{id}`` body.

        Args:
            entry_id: FPL entry id the payload belongs to.
            payload: JSON body from ``GET /api/my-team/{id}/``.

        Returns:
            A ``CurrentTeam`` with selling prices and free transfers.

        Raises:
            FPLAPIError: Required fields are missing or malformed.
        """
        picks_raw = payload.get("picks")
        transfers_raw = payload.get("transfers")
        if not isinstance(picks_raw, list) or not isinstance(transfers_raw, dict):
            raise FPLAPIError("my-team payload needs picks and transfers")
        transfers = cast(dict[str, Any], transfers_raw)
        if "bank" not in transfers:
            raise FPLAPIError("my-team payload needs transfers.bank")
        picks = [EntryPick.model_validate(row) for row in cast(list[Any], picks_raw)]
        if any(pick.selling_price is None for pick in picks):
            raise FPLAPIError("my-team picks need selling_price")
        return cls(
            entry_id=entry_id,
            event=None,
            bank=int(transfers["bank"]),
            value=_optional_int(transfers.get("value")),
            free_transfers=_optional_int(transfers.get("limit")),
            picks=picks,
            active_chip=_active_chip(payload),
            source="my_team",
        )

    @classmethod
    def from_entry_picks(
        cls,
        entry_id: int,
        event: int,
        payload: dict[str, Any],
    ) -> Self:
        """Parse a public ``entry/{id}/event/{gw}/picks`` body.

        Args:
            entry_id: FPL entry id.
            event: Gameweek the picks belong to.
            payload: JSON body from the picks endpoint.

        Returns:
            A ``CurrentTeam``. Selling prices are usually absent on public
            responses for other managers' finished gameweeks.

        Raises:
            FPLAPIError: Required fields are missing or malformed.
        """
        picks_raw = payload.get("picks")
        history_raw = payload.get("entry_history")
        if not isinstance(picks_raw, list) or not isinstance(history_raw, dict):
            raise FPLAPIError("entry picks payload needs picks and entry_history")
        history = cast(dict[str, Any], history_raw)
        if "bank" not in history:
            raise FPLAPIError("entry picks payload needs entry_history.bank")
        picks = [EntryPick.model_validate(row) for row in cast(list[Any], picks_raw)]
        chip = payload.get("active_chip")
        return cls(
            entry_id=entry_id,
            event=event,
            bank=int(history["bank"]),
            value=_optional_int(history.get("value")),
            free_transfers=None,
            picks=picks,
            active_chip=str(chip) if chip is not None else None,
            source="entry_picks",
        )


class FPLClient:
    """HTTP client for FPL entry and my-team endpoints.

    Args:
        client: Optional shared ``httpx.Client``. When omitted, one is created
            with a 30s timeout and closed by ``close`` / context manager.
        session: Value of the ``pl_profile`` cookie. Required for ``my_team``.
        base_url: API root; override only in tests.
    """

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        session: str | None = None,
        base_url: str = API_ROOT,
    ) -> None:
        self._owns_client = client is None
        headers = {"User-Agent": "fplt"}
        cookies = {"pl_profile": session} if session else None
        self._client = client or httpx.Client(
            timeout=30.0,
            headers=headers,
            cookies=cookies,
        )
        if client is not None and session:
            self._client.cookies.set("pl_profile", session)
        self._session = session
        self._base_url = base_url.rstrip("/")

    def close(self) -> None:
        """Close the underlying HTTP client when this instance created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close on exit."""
        self.close()

    def entry(self, entry_id: int) -> dict[str, Any]:
        """Fetch public entry summary including ``current_event``.

        Args:
            entry_id: FPL manager entry id.

        Returns:
            Parsed JSON object.
        """
        return self._get_json(f"/entry/{entry_id}/")

    def entry_picks(self, entry_id: int, event: int) -> dict[str, Any]:
        """Fetch public picks for a finished gameweek.

        Live (current) gameweek picks for other managers return HTTP 403 until
        the deadline; use ``my_team`` for the live squad.

        Args:
            entry_id: FPL manager entry id.
            event: Gameweek number.

        Returns:
            Parsed JSON object with ``picks`` and ``entry_history``.
        """
        return self._get_json(f"/entry/{entry_id}/event/{event}/picks/")

    def my_team(self, entry_id: int) -> dict[str, Any]:
        """Fetch the authenticated live squad for ``entry_id``.

        Args:
            entry_id: FPL manager entry id (must match the session).

        Returns:
            Parsed JSON object with ``picks``, ``transfers``, and ``chips``.

        Raises:
            FPLAPIError: No session cookie was configured, or the API refused
                the request.
        """
        if not self._session:
            raise FPLAPIError(
                "my-team needs FPLT_FPL_SESSION (pl_profile cookie from "
                "fantasy.premierleague.com)"
            )
        return self._get_json(f"/my-team/{entry_id}/")

    def current_team(
        self,
        entry_id: int,
        *,
        event: int | None = None,
        prefer_my_team: bool = True,
    ) -> CurrentTeam:
        """Load the current (or historical) team for an entry.

        Prefers authenticated ``my-team`` when a session cookie is set. Falls
        back to public ``entry/.../picks`` for a finished ``event`` (resolved
        from the entry summary when omitted).

        Args:
            entry_id: FPL manager entry id.
            event: Gameweek for the public picks path. Ignored when
                ``my-team`` succeeds.
            prefer_my_team: Try ``my-team`` first when a session is present.

        Returns:
            Typed ``CurrentTeam``.

        Raises:
            FPLAPIError: Neither path can load a squad.
        """
        if prefer_my_team and self._session:
            return CurrentTeam.from_my_team(entry_id, self.my_team(entry_id))
        gw = event
        if gw is None:
            summary = self.entry(entry_id)
            current = summary.get("current_event")
            if current is None:
                raise FPLAPIError("entry summary has no current_event")
            gw = int(current)
        picks = self.entry_picks(entry_id, gw)
        return CurrentTeam.from_entry_picks(entry_id, gw, picks)

    def _get_json(self, path: str) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = self._client.get(url)
        except httpx.HTTPError as exc:
            raise FPLAPIError(f"request failed for {path}") from exc
        if response.status_code == 403:
            raise FPLAPIError(
                f"FPL API forbidden for {path}; live picks need FPLT_FPL_SESSION"
            )
        if response.status_code == 404:
            raise FPLAPIError(f"FPL API not found: {path}")
        if response.status_code >= 400:
            raise FPLAPIError(f"FPL API HTTP {response.status_code} for {path}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise FPLAPIError(f"FPL API returned non-JSON for {path}") from exc
        if not isinstance(payload, dict):
            raise FPLAPIError(f"FPL API expected object for {path}")
        return cast(dict[str, Any], payload)


def load_current_team(
    entry_id: int | None = None,
    *,
    event: int | None = None,
    settings: Settings | None = None,
    client: httpx.Client | None = None,
) -> CurrentTeam:
    """Load the configured entry's current team.

    Args:
        entry_id: Override ``FPLT_ENTRY_ID``.
        event: Optional finished gameweek for the public picks fallback.
        settings: Process settings; defaults to ``get_settings()``.
        client: Optional shared ``httpx.Client`` (for tests).

    Returns:
        Typed ``CurrentTeam``.

    Raises:
        FPLAPIError: ``entry_id`` is unset or the API call fails.
    """
    cfg = settings if settings is not None else get_settings()
    resolved = entry_id if entry_id is not None else cfg.entry_id
    if resolved is None:
        raise FPLAPIError("set FPLT_ENTRY_ID or pass entry_id")
    api = FPLClient(client, session=cfg.fpl_session)
    try:
        return api.current_team(resolved, event=event)
    finally:
        api.close()


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(cast(Any, value))


def _active_chip(payload: dict[str, Any]) -> str | None:
    chip = payload.get("active_chip")
    if isinstance(chip, str):
        return chip
    return None
