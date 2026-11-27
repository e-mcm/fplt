"""Pandera schemas for curated actuals (ledger join side).

Outcomes live in content-addressed curated builds and are joined to run
predictions at read time — never written into prediction rows
(``docs/design.md`` §7.2; data model §3.6). Producers: ``fplt.data.outcomes``.
"""

from __future__ import annotations

from datetime import datetime

import pandera.polars as pa


class PlayerFixtureOutcomesSchema(pa.DataFrameModel):
    """One player's realised FPL stats for one fixture.

    Join key with run predictions: ``(season, fixture_id, player_key)``
    (predictions use ``target_season`` for the season column).
    """

    season: str
    gw: int = pa.Field(ge=1, le=38)
    fixture_id: int
    player_key: int
    team_key: int
    pts: int
    minutes: int = pa.Field(ge=0)
    goals: int = pa.Field(ge=0)
    assists: int = pa.Field(ge=0)
    clean_sheets: int = pa.Field(ge=0)
    available_at: datetime
    schema_version: int = pa.Field(ge=1)
    source: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["season", "fixture_id", "player_key"]  # noqa: RUF012


class FixtureOutcomesSchema(pa.DataFrameModel):
    """Final scoreline for one fixture once the gameweek is data-checked."""

    season: str
    gw: int = pa.Field(ge=1, le=38)
    fixture_id: int
    team_h_key: int
    team_a_key: int
    team_h_score: int = pa.Field(ge=0)
    team_a_score: int = pa.Field(ge=0)
    available_at: datetime
    schema_version: int = pa.Field(ge=1)
    source: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["season", "fixture_id"]  # noqa: RUF012
