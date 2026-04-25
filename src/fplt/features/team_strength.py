"""Fixture-level feature builder for the Dixon-Coles match score model.

Dixon-Coles learns per-team attack/defence parameters directly from observed
scores, so no hand-crafted team features are needed for training.  This module
provides the function that assembles one clean row per fixture from historical
data, which is what the model trains on.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from fplt.data.load import ALL_SEASONS, load_fixtures, load_gameweeks
from fplt.schemas import FixtureRecordSchema


@pa.check_types
def build_fixture_records(
    seasons: list[str] | None = None,
    data_dir: Path | None = None,
) -> DataFrame[FixtureRecordSchema]:
    """Build a fixture-level training DataFrame for Dixon-Coles.

    Deduplicates gameweek player rows to one row per fixture (two teams).
    Uses `load_fixtures` where available; falls back to deduplication from
    the gameweek data so all seasons work uniformly.

    Args:
        seasons: Seasons to include. Defaults to all available.
        data_dir: Path to data root. Defaults to repo data/ directory.

    Returns:
        DataFrame with columns:
            fixture_id, season, team_h, team_a,
            team_h_score, team_a_score, kickoff_time
        Only rows where both scores are present (finished fixtures) are kept.
    """
    seasons = seasons or ALL_SEASONS
    frames: list[pd.DataFrame] = []

    for season in seasons:
        # Try fixtures.csv first — it has explicit team IDs
        try:
            fix = load_fixtures(season, data_dir)
            if {"team_h", "team_a", "team_h_score", "team_a_score"}.issubset(
                fix.columns
            ):
                fix = fix.dropna(subset=["team_h_score", "team_a_score"])
                fix = fix.rename(columns={"id": "fixture_id"})
                fix["season"] = season
                cols = [
                    "fixture_id",
                    "season",
                    "team_h",
                    "team_a",
                    "team_h_score",
                    "team_a_score",
                    "kickoff_time",
                ]
                frames.append(fix[[c for c in cols if c in fix.columns]])
                continue
        except FileNotFoundError:
            pass

        # Fall back: deduplicate from player-level gameweek data
        try:
            gw = load_gameweeks(season, data_dir)
        except FileNotFoundError:
            continue

        if not {"fixture", "team_h_score", "team_a_score"}.issubset(gw.columns):
            continue

        dedup = (
            gw.dropna(subset=["team_h_score", "team_a_score"])
            .drop_duplicates("fixture")[
                ["fixture", "team_h_score", "team_a_score", "kickoff_time"]
            ]
            .rename(columns={"fixture": "fixture_id"})
        )
        # Recover team_h / team_a from was_home flag
        if {"was_home", "opponent_team", "team"}.issubset(gw.columns):
            home_rows = gw[gw["was_home"]].drop_duplicates("fixture")[
                ["fixture", "team", "opponent_team"]
            ]
            dedup = dedup.merge(
                home_rows.rename(
                    columns={
                        "fixture": "fixture_id",
                        "team": "team_h",
                        "opponent_team": "team_a",
                    }
                ),
                on="fixture_id",
                how="left",
            )
        dedup["season"] = season
        frames.append(dedup)

    if not frames:
        return pd.DataFrame(  # type: ignore[return-value]
            columns=[
                "fixture_id",
                "season",
                "team_h",
                "team_a",
                "team_h_score",
                "team_a_score",
                "kickoff_time",
            ]
        )

    out = pd.concat(frames, ignore_index=True)
    out["team_h_score"] = out["team_h_score"].astype(int)
    out["team_a_score"] = out["team_a_score"].astype(int)
    out["kickoff_time"] = pd.to_datetime(out["kickoff_time"], utc=True)
    return out.reset_index(drop=True)  # type: ignore[return-value]
