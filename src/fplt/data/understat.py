"""Stage Understat match rosters. The caller supplies the id map and the clock."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import polars as pl

from fplt.data.store import DataStore


def stage_understat(
    store: DataStore,
    matches: list[dict[str, Any]],
    season: str,
    available_at: datetime,
    player_keys: dict[int, int],
    team_keys: dict[int, int],
) -> pl.DataFrame:
    """Append player-match xG and xA.

    Args:
        store: Destination for ``understat_player_matches``.
        matches: Understat match rosters. Each has ``id``, ``team_h``,
            ``team_a``, and ``h`` / ``a`` player lists with ``id``, ``xg``,
            ``xa``, and ``time``.
        season: Season label stored on every row.
        available_at: When this roster became visible. Later deadlines can
            see it; earlier ones cannot.
        player_keys: Understat player id to ``player_key``. Missing ids are
            stored with a null ``player_key``.
        team_keys: Understat team id to ``team_key``.

    Returns:
        The staged table after the append.

    Raises:
        KeyError: A side's team id has no ``team_key``.
    """
    rows: list[dict[str, object]] = []
    for match in matches:
        match_id = int(match["id"])
        for side, team_field in (("h", "team_h"), ("a", "team_a")):
            team_id = int(match[team_field])
            if team_id not in team_keys:
                raise KeyError(f"understat team {team_id} has no team_key")
            players = match[side]
            if not isinstance(players, list):
                raise ValueError(f"match {match_id} side {side} must be a list")
            for player in cast(list[dict[str, Any]], players):
                understat_id = int(player["id"])
                rows.append(
                    {
                        "season": season,
                        "match_id": match_id,
                        "understat_player_id": understat_id,
                        "player_key": player_keys.get(understat_id),
                        "team_key": team_keys[team_id],
                        "xg": float(player["xg"]),
                        "xa": float(player["xa"]),
                        "minutes": int(player["time"]),
                        "available_at": available_at,
                        "source": "understat",
                    }
                )
    frame = pl.DataFrame(rows) if rows else _empty()
    return store.write(
        "understat_player_matches",
        frame.with_columns(pl.col("player_key").cast(pl.Int64())),
    )


def _empty() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "season": pl.Utf8(),
            "match_id": pl.Int64(),
            "understat_player_id": pl.Int64(),
            "player_key": pl.Int64(),
            "team_key": pl.Int64(),
            "xg": pl.Float64(),
            "xa": pl.Float64(),
            "minutes": pl.Int64(),
            "available_at": pl.Datetime(time_zone="UTC"),
            "source": pl.Utf8(),
        }
    )
