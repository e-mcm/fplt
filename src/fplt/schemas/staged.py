"""Pandera schemas for staged tables. Every table carries ``available_at``."""

from __future__ import annotations

from datetime import datetime

import pandera.polars as pa
import polars as pl


class PlayersSchema(pa.DataFrameModel):
    """One row per player per season, from the first snapshot that contained them."""

    season: str
    player_key: int
    element_id: int
    team_key: int
    position: str
    web_name: str
    available_at: datetime
    source: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class TeamsSchema(pa.DataFrameModel):
    season: str
    team_key: int
    team_id: int
    name: str
    short_name: str
    available_at: datetime
    source: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class GameweeksSchema(pa.DataFrameModel):
    season: str
    gw: int = pa.Field(ge=1, le=38)
    deadline_time: datetime
    finished: bool
    data_checked: bool
    available_at: datetime
    source: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class FixturesSchema(pa.DataFrameModel):
    season: str
    fixture_id: int
    gw: int = pa.Field(ge=1, le=38)
    team_h_key: int
    team_a_key: int
    kickoff_time: datetime
    team_h_score: int = pa.Field(nullable=True, ge=0)
    team_a_score: int = pa.Field(nullable=True, ge=0)
    finished: bool
    available_at: datetime
    source: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class PlayerStatusSchema(pa.DataFrameModel):
    season: str
    player_key: int
    status: str
    news: str
    price: int = pa.Field(ge=0)
    ep_next: float = pa.Field(nullable=True)
    chance_of_playing: float = pa.Field(nullable=True, ge=0, le=100)
    penalties_order: int = pa.Field(nullable=True, ge=1)
    available_at: datetime
    source: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class UnderstatPlayerMatchSchema(pa.DataFrameModel):
    """One player's Understat match row.

    ``player_key`` is null when the Understat id has not been mapped onto an
    FPL element code. ``minutes`` is the Understat ``time`` field. ``shots``,
    ``key_passes``, ``xg_chain``, and ``xg_buildup`` come from the roster when
    present (else 0) so OpenFPL-style samples can be built without a second
    scrape.
    """

    season: str
    match_id: int
    understat_player_id: int
    player_key: int = pa.Field(nullable=True)
    team_key: int
    xg: float = pa.Field(ge=0)
    xa: float = pa.Field(ge=0)
    shots: int = pa.Field(ge=0)
    key_passes: int = pa.Field(ge=0)
    xg_chain: float = pa.Field(ge=0)
    xg_buildup: float = pa.Field(ge=0)
    minutes: int = pa.Field(ge=0)
    available_at: datetime
    source: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class UnderstatTeamMatchSchema(pa.DataFrameModel):
    """One team's Understat match row (Deep / PPDA / xG).

    Sourced from the league-page ``teamsData`` history (not ``getMatchData``).
    ``match_id`` is joined from the season dates list on kickoff + side.
    """

    season: str
    match_id: int
    team_key: int
    deep: float = pa.Field(ge=0)
    deep_allowed: float = pa.Field(ge=0)
    ppda_att: float = pa.Field(ge=0)
    ppda_def: float = pa.Field(ge=0)
    ppda_allowed_att: float = pa.Field(ge=0)
    ppda_allowed_def: float = pa.Field(ge=0)
    xg: float = pa.Field(ge=0)
    xga: float = pa.Field(ge=0)
    available_at: datetime
    source: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class PlayerFixtureStatsSchema(pa.DataFrameModel):
    season: str
    gw: int = pa.Field(ge=1, le=38)
    player_key: int
    fixture_id: int
    minutes: int = pa.Field(ge=0)
    total_points: int
    goals_scored: int = pa.Field(ge=0)
    assists: int = pa.Field(ge=0)
    clean_sheets: int = pa.Field(ge=0)
    goals_conceded: int = pa.Field(ge=0)
    saves: int = pa.Field(ge=0)
    bonus: int = pa.Field(ge=0)
    yellow_cards: int = pa.Field(ge=0)
    red_cards: int = pa.Field(ge=0)
    own_goals: int = pa.Field(ge=0)
    penalties_saved: int = pa.Field(ge=0)
    penalties_missed: int = pa.Field(ge=0)
    starts: int = pa.Field(ge=0)
    influence: float = pa.Field(ge=0)
    creativity: float = pa.Field(ge=0)
    threat: float = pa.Field(ge=0)
    bps: int
    clearances_blocks_interceptions: int = pa.Field(ge=0)
    tackles: int = pa.Field(ge=0)
    recoveries: int = pa.Field(ge=0)
    defensive_contribution: int = pa.Field(ge=0)
    available_at: datetime
    source: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class MatchOddsSchema(pa.DataFrameModel):
    """Closing bookmaker 1X2 (+ O/U 2.5) for one EPL match.

    Sourced from football-data.co.uk E0 CSVs. ``odds_kind`` is always
    ``closing_proxy``: closing lines are pre-kickoff, **not** FPL deadline.
    ``available_at`` is therefore kickoff (UTC). Do not treat these rows as
    deadline-safe PIT. ``fixture_id`` is null until joined to staged FPL
    fixtures on ``(season, team_h_key, team_a_key, kickoff date)``.
    """

    season: str
    fixture_id: int = pa.Field(nullable=True)
    team_h_key: int
    team_a_key: int
    kickoff_time: datetime
    home_odds: float = pa.Field(gt=1.0)
    draw_odds: float = pa.Field(gt=1.0)
    away_odds: float = pa.Field(gt=1.0)
    over_25_odds: float = pa.Field(nullable=True, gt=1.0)
    under_25_odds: float = pa.Field(nullable=True, gt=1.0)
    odds_kind: str
    available_at: datetime
    source: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


STAGED_SCHEMAS: dict[str, type[pa.DataFrameModel]] = {
    "players": PlayersSchema,
    "teams": TeamsSchema,
    "gameweeks": GameweeksSchema,
    "fixtures": FixturesSchema,
    "player_status": PlayerStatusSchema,
    "player_fixture_stats": PlayerFixtureStatsSchema,
    "understat_player_matches": UnderstatPlayerMatchSchema,
    "understat_team_matches": UnderstatTeamMatchSchema,
    "match_odds": MatchOddsSchema,
}

TABLE_KEYS: dict[str, list[str]] = {
    "players": ["season", "player_key", "available_at"],
    "teams": ["season", "team_key", "available_at"],
    "gameweeks": ["season", "gw", "available_at"],
    "fixtures": ["season", "fixture_id", "available_at"],
    "player_status": ["season", "player_key", "available_at"],
    "player_fixture_stats": ["season", "player_key", "fixture_id", "available_at"],
    "understat_player_matches": [
        "season",
        "match_id",
        "understat_player_id",
        "available_at",
    ],
    "understat_team_matches": ["season", "match_id", "team_key", "available_at"],
    "match_odds": ["season", "team_h_key", "team_a_key", "kickoff_time"],
}

POLARS_SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "players": {
        "season": pl.Utf8(),
        "player_key": pl.Int64(),
        "element_id": pl.Int64(),
        "team_key": pl.Int64(),
        "position": pl.Utf8(),
        "web_name": pl.Utf8(),
        "available_at": pl.Datetime(time_zone="UTC"),
        "source": pl.Utf8(),
    },
    "teams": {
        "season": pl.Utf8(),
        "team_key": pl.Int64(),
        "team_id": pl.Int64(),
        "name": pl.Utf8(),
        "short_name": pl.Utf8(),
        "available_at": pl.Datetime(time_zone="UTC"),
        "source": pl.Utf8(),
    },
    "gameweeks": {
        "season": pl.Utf8(),
        "gw": pl.Int64(),
        "deadline_time": pl.Datetime(time_zone="UTC"),
        "finished": pl.Boolean(),
        "data_checked": pl.Boolean(),
        "available_at": pl.Datetime(time_zone="UTC"),
        "source": pl.Utf8(),
    },
    "fixtures": {
        "season": pl.Utf8(),
        "fixture_id": pl.Int64(),
        "gw": pl.Int64(),
        "team_h_key": pl.Int64(),
        "team_a_key": pl.Int64(),
        "kickoff_time": pl.Datetime(time_zone="UTC"),
        "team_h_score": pl.Int64(),
        "team_a_score": pl.Int64(),
        "finished": pl.Boolean(),
        "available_at": pl.Datetime(time_zone="UTC"),
        "source": pl.Utf8(),
    },
    "player_status": {
        "season": pl.Utf8(),
        "player_key": pl.Int64(),
        "status": pl.Utf8(),
        "news": pl.Utf8(),
        "price": pl.Int64(),
        "ep_next": pl.Float64(),
        "chance_of_playing": pl.Float64(),
        "penalties_order": pl.Int64(),
        "available_at": pl.Datetime(time_zone="UTC"),
        "source": pl.Utf8(),
    },
    "player_fixture_stats": {
        "season": pl.Utf8(),
        "gw": pl.Int64(),
        "player_key": pl.Int64(),
        "fixture_id": pl.Int64(),
        "minutes": pl.Int64(),
        "total_points": pl.Int64(),
        "goals_scored": pl.Int64(),
        "assists": pl.Int64(),
        "clean_sheets": pl.Int64(),
        "goals_conceded": pl.Int64(),
        "saves": pl.Int64(),
        "bonus": pl.Int64(),
        "yellow_cards": pl.Int64(),
        "red_cards": pl.Int64(),
        "own_goals": pl.Int64(),
        "penalties_saved": pl.Int64(),
        "penalties_missed": pl.Int64(),
        "starts": pl.Int64(),
        "influence": pl.Float64(),
        "creativity": pl.Float64(),
        "threat": pl.Float64(),
        "bps": pl.Int64(),
        "clearances_blocks_interceptions": pl.Int64(),
        "tackles": pl.Int64(),
        "recoveries": pl.Int64(),
        "defensive_contribution": pl.Int64(),
        "available_at": pl.Datetime(time_zone="UTC"),
        "source": pl.Utf8(),
    },
    "understat_player_matches": {
        "season": pl.Utf8(),
        "match_id": pl.Int64(),
        "understat_player_id": pl.Int64(),
        "player_key": pl.Int64(),
        "team_key": pl.Int64(),
        "xg": pl.Float64(),
        "xa": pl.Float64(),
        "shots": pl.Int64(),
        "key_passes": pl.Int64(),
        "xg_chain": pl.Float64(),
        "xg_buildup": pl.Float64(),
        "minutes": pl.Int64(),
        "available_at": pl.Datetime(time_zone="UTC"),
        "source": pl.Utf8(),
    },
    "understat_team_matches": {
        "season": pl.Utf8(),
        "match_id": pl.Int64(),
        "team_key": pl.Int64(),
        "deep": pl.Float64(),
        "deep_allowed": pl.Float64(),
        "ppda_att": pl.Float64(),
        "ppda_def": pl.Float64(),
        "ppda_allowed_att": pl.Float64(),
        "ppda_allowed_def": pl.Float64(),
        "xg": pl.Float64(),
        "xga": pl.Float64(),
        "available_at": pl.Datetime(time_zone="UTC"),
        "source": pl.Utf8(),
    },
    "match_odds": {
        "season": pl.Utf8(),
        "fixture_id": pl.Int64(),
        "team_h_key": pl.Int64(),
        "team_a_key": pl.Int64(),
        "kickoff_time": pl.Datetime(time_zone="UTC"),
        "home_odds": pl.Float64(),
        "draw_odds": pl.Float64(),
        "away_odds": pl.Float64(),
        "over_25_odds": pl.Float64(),
        "under_25_odds": pl.Float64(),
        "odds_kind": pl.Utf8(),
        "available_at": pl.Datetime(time_zone="UTC"),
        "source": pl.Utf8(),
    },
}
