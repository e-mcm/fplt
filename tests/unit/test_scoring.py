"""Season scoring tables reconstruct official points."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fplt.scoring import MatchStats, Rules
from fplt.scoring.rules import PositionName

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "scoring"
_SAMPLE = _FIXTURES / "2026-27-gw-sample.json"
_SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27")
# Recorded players whose action counts clear the 2025/26 defensive threshold.
_DEFENSIVE = {87, 98, 194}


def _sample() -> list[dict[str, object]]:
    loaded = json.loads(_SAMPLE.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    return loaded


def _stats(row: dict[str, object]) -> MatchStats:
    fields = set(MatchStats.model_fields)
    return MatchStats.model_validate({key: row[key] for key in row if key in fields})


def test_recorded_sample_matches_official_totals() -> None:
    rules = Rules.for_season("2026-27")
    rows = _sample()
    assert rows
    for row in rows:
        scored = rules.score(_stats(row))
        assert scored == row["total_points"], row["element_id"]


def test_older_seasons_drop_only_the_defensive_award() -> None:
    current = Rules.for_season("2026-27")
    previous = Rules.for_season("2024-25")
    for row in _sample():
        stats = _stats(row)
        gap = current.score(stats) - previous.score(stats)
        assert gap == (2 if row["element_id"] in _DEFENSIVE else 0)


@pytest.mark.parametrize("season", _SEASONS)
def test_every_trained_season_loads(season: str) -> None:
    rules = Rules.for_season(season)
    assert rules.appearance.long == 2
    awarded = bool(rules.defensive_contributions)
    assert awarded == (season >= "2025-26")


def test_missing_season_raises() -> None:
    with pytest.raises(FileNotFoundError, match="1999-00"):
        Rules.for_season("1999-00")


def test_season_must_be_a_file_name() -> None:
    with pytest.raises(ValueError, match="file name"):
        Rules.for_season("../2024-25")


def test_appearance_thresholds() -> None:
    rules = Rules.for_season("2024-25")
    assert rules.score(MatchStats(position="MID", minutes=0)) == 0
    assert rules.score(MatchStats(position="MID", minutes=59)) == 1
    assert rules.score(MatchStats(position="MID", minutes=60)) == 2


@pytest.mark.parametrize(
    ("position", "goal", "sheet"),
    [("GK", 6, 4), ("DEF", 6, 4), ("MID", 5, 1), ("FWD", 4, 0)],
)
def test_goals_and_clean_sheets_by_position(
    position: PositionName, goal: int, sheet: int
) -> None:
    rules = Rules.for_season("2024-25")
    base = rules.score(MatchStats(position=position, minutes=90))
    scored = rules.score(
        MatchStats(position=position, minutes=90, goals_scored=1, clean_sheets=1)
    )
    assert scored - base == goal + sheet


def test_goals_conceded_and_saves_are_positional() -> None:
    rules = Rules.for_season("2024-25")
    keeper = MatchStats(position="GK", minutes=90, goals_conceded=3, saves=5)
    midfielder = MatchStats(position="MID", minutes=90, goals_conceded=3, saves=5)
    assert rules.score(keeper) == 2 - 1 + 1
    assert rules.score(midfielder) == 2


def test_cards_penalties_own_goal_and_bonus() -> None:
    rules = Rules.for_season("2024-25")
    stats = MatchStats(
        position="FWD",
        minutes=90,
        assists=1,
        penalties_saved=1,
        penalties_missed=1,
        yellow_cards=1,
        red_cards=1,
        own_goals=1,
        bonus=3,
    )
    assert rules.score(stats) == 2 + 3 + 5 - 2 - 1 - 3 - 2 + 3


def test_defensive_contribution_thresholds() -> None:
    rules = Rules.for_season("2025-26")
    short = MatchStats(
        position="DEF",
        minutes=90,
        clearances_blocks_interceptions=7,
        tackles=2,
        recoveries=20,
    )
    reached = short.model_copy(update={"tackles": 3})
    assert rules.score(reached) - rules.score(short) == 2
    midfield = MatchStats(
        position="MID",
        minutes=90,
        clearances_blocks_interceptions=4,
        tackles=4,
        recoveries=3,
    )
    tipped = midfield.model_copy(update={"recoveries": 4})
    assert rules.score(midfield) == 2
    assert rules.score(tipped) == 4
    keeper = MatchStats(position="GK", minutes=90, tackles=20, recoveries=20)
    assert rules.score(keeper) == 2


def test_incomplete_table_and_overlapping_contributions_are_rejected() -> None:
    loaded = Rules.for_season("2024-25").model_dump()
    loaded["goals"].pop("FWD")
    with pytest.raises(ValidationError, match="goals"):
        Rules.model_validate(loaded)
    loaded = Rules.for_season("2025-26").model_dump()
    loaded["defensive_contributions"][1]["positions"] = ["DEF", "MID", "FWD"]
    with pytest.raises(ValidationError, match="overlap"):
        Rules.model_validate(loaded)
