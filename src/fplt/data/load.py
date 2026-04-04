"""Data loading for FPL CSV datasets (vaastav/Fantasy-Premier-League)."""

from pathlib import Path

import pandas as pd

# Columns present in every season — safe to rely on
CORE_GW_COLS = [
    "name",
    "element",
    "fixture",
    "round",
    "GW",
    "kickoff_time",
    "was_home",
    "opponent_team",
    "team_h_score",
    "team_a_score",
    "minutes",
    "total_points",
    "value",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "ict_index",
    "influence",
    "creativity",
    "threat",
]

# Columns added later — included when available
EXTENDED_GW_COLS = [
    "position",
    "team",
    "xP",
    "starts",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]

ALL_SEASONS = [
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
]


def _data_dir() -> Path:
    """Resolve the data/ directory relative to the package root."""
    return Path(__file__).parents[3] / "data"


def load_gameweeks(season: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load merged gameweek data for a single season.

    Args:
        season: Season string e.g. "2024-25".
        data_dir: Path to data directory. Defaults to the repo data/ folder.

    Returns:
        DataFrame with one row per player per gameweek.
    """
    root = data_dir or _data_dir()
    path = root / season / "gws" / "merged_gw.csv"
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")

    keep = [c for c in CORE_GW_COLS + EXTENDED_GW_COLS if c in df.columns]
    df = df[keep].copy()
    df["season"] = season
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)

    return df


def load_fixtures(season: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load fixture list for a season.

    Args:
        season: Season string e.g. "2024-25".
        data_dir: Path to data directory.

    Returns:
        DataFrame with one row per fixture.
    """
    root = data_dir or _data_dir()
    path = root / season / "fixtures.csv"
    df = pd.read_csv(path, low_memory=False)
    df["season"] = season

    keep = [
        "id",
        "event",
        "team_h",
        "team_a",
        "team_h_score",
        "team_a_score",
        "team_h_difficulty",
        "team_a_difficulty",
        "kickoff_time",
        "finished",
        "season",
    ]
    return df[[c for c in keep if c in df.columns]]


def load_teams(season: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load team info and strength ratings for a season.

    Args:
        season: Season string e.g. "2024-25".
        data_dir: Path to data directory.

    Returns:
        DataFrame with one row per team.
    """
    root = data_dir or _data_dir()
    path = root / season / "teams.csv"
    df = pd.read_csv(path, low_memory=False)
    df["season"] = season

    keep = [
        "id",
        "name",
        "short_name",
        "strength",
        "strength_overall_home",
        "strength_overall_away",
        "strength_attack_home",
        "strength_attack_away",
        "strength_defence_home",
        "strength_defence_away",
        "season",
    ]
    return df[[c for c in keep if c in df.columns]]


def load_all_seasons(
    seasons: list[str] | None = None,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load and concatenate gameweek data across multiple seasons.

    Args:
        seasons: List of season strings. Defaults to all available seasons.
        data_dir: Path to data directory.

    Returns:
        DataFrame with all seasons combined, sorted by season and GW.
    """
    seasons = seasons or ALL_SEASONS
    frames: list[pd.DataFrame] = []

    for season in seasons:
        root = data_dir or _data_dir()
        path = root / season / "gws" / "merged_gw.csv"
        if not path.exists():
            continue
        frames.append(load_gameweeks(season, data_dir))

    if not frames:
        return pd.DataFrame(columns=CORE_GW_COLS + ["season"])

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["season", "GW", "name"]).reset_index(drop=True)
    return df
