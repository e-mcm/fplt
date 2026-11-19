"""Player-fixture rows a predictor is asked to score."""

from __future__ import annotations

import polars as pl

from fplt.data.pit import PITView
from fplt.types import GWKey

_COLUMNS: dict[str, pl.DataType] = {
    "season": pl.Utf8(),
    "gw": pl.Int64(),
    "player_key": pl.Int64(),
    "fixture_id": pl.Int64(),
    "source": pl.Utf8(),
}


def player_fixtures(view: PITView, horizon: list[GWKey]) -> pl.DataFrame:
    """Players who have a fixture in ``horizon``.

    A blank gameweek contributes no rows. A double gameweek contributes one
    row per fixture. ``source`` is the fixture's source, so a caller can tell
    a live snapshot from a vaastav import.

    Args:
        view: Point-in-time tables visible at one deadline.
        horizon: Gameweeks to predict.

    Returns:
        Columns ``season``, ``gw``, ``player_key``, ``fixture_id``, ``source``.
    """
    empty = pl.DataFrame(schema=_COLUMNS)
    if not horizon:
        return empty
    fixtures = view.latest("fixtures", by=["season", "fixture_id"]).collect()
    players = view.latest("players", by=["season", "player_key"]).collect()
    if fixtures.is_empty() or players.is_empty():
        return empty
    wanted = pl.DataFrame(
        {
            "season": [key.season for key in horizon],
            "gw": [key.gw for key in horizon],
        }
    ).unique()
    selected = fixtures.join(wanted, on=["season", "gw"], how="inner")
    if selected.is_empty():
        return empty
    roster = players.select("season", "player_key", "team_key")
    home = selected.join(
        roster,
        left_on=["season", "team_h_key"],
        right_on=["season", "team_key"],
        how="inner",
    )
    away = selected.join(
        roster,
        left_on=["season", "team_a_key"],
        right_on=["season", "team_key"],
        how="inner",
    )
    rows = pl.concat([home, away], how="diagonal_relaxed")
    return rows.select("season", "gw", "player_key", "fixture_id", "source").unique(
        subset=["season", "fixture_id", "player_key"]
    )
