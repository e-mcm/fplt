"""football-data E0 odds stage into match_odds without network."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.data.football_data import (
    PrematchCoverageError,
    attach_fixture_ids,
    check_prematch_deadline_coverage,
    default_team_map_path,
    e0_raw_path,
    estimated_prematch_available_at,
    ingest_football_data,
    iter_raw_e0,
    load_team_names,
    parse_e0_csv,
    parse_kickoff,
    prematch_board_odds,
    require_prematch_deadline_coverage,
    season_label_from_token,
    season_token,
    stage_match_odds,
)
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.schemas.staged import MatchOddsSchema
from fplt.types import GWKey

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "football_data"
_CSV = _FIXTURES / "E0.csv"
_MAP = _FIXTURES / "team_names.yaml"
_TEAMS = {
    "Arsenal": 3,
    "Tottenham": 6,
    "Man United": 1,
    "Fulham": 54,
    "Liverpool": 14,
    "Ipswich": 40,
}


def test_season_token_helpers() -> None:
    assert season_token("2024-25") == "2425"
    assert season_label_from_token("2425") == "2024-25"
    with pytest.raises(ValueError, match="four digits"):
        season_label_from_token("24")


def test_load_team_names_reads_yaml() -> None:
    assert load_team_names(_MAP) == _TEAMS
    assert "Arsenal" in load_team_names(default_team_map_path())


def test_parse_kickoff_london_to_utc() -> None:
    # 15:00 BST on 17 Aug 2024 → 14:00 UTC
    assert parse_kickoff("17/08/2024", "15:00") == datetime(
        2024, 8, 17, 14, 0, tzinfo=UTC
    )
    # Missing time → 15:00 local default
    assert parse_kickoff("18/08/2024", None) == datetime(2024, 8, 18, 14, 0, tzinfo=UTC)


def test_estimated_prematch_available_at_weekend_and_midweek() -> None:
    # Saturday kickoff → preceding Friday 17:00 BST = 16:00 UTC
    saturday = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)
    assert estimated_prematch_available_at(saturday) == datetime(
        2024, 8, 16, 16, 0, tzinfo=UTC
    )
    # Wednesday kickoff → Tuesday 17:00 BST = 16:00 UTC
    wednesday = datetime(2024, 8, 21, 18, 30, tzinfo=UTC)
    assert estimated_prematch_available_at(wednesday) == datetime(
        2024, 8, 20, 16, 0, tzinfo=UTC
    )
    # Monday kickoff → Tuesday 17:00 (after KO; coverage gate drops it)
    monday = datetime(2024, 8, 19, 19, 0, tzinfo=UTC)
    assert estimated_prematch_available_at(monday) == datetime(
        2024, 8, 20, 16, 0, tzinfo=UTC
    )


def test_parse_e0_csv_emits_closing_and_prematch() -> None:
    frame = parse_e0_csv(_CSV, "2024-25", _TEAMS)
    assert frame.height == 6
    MatchOddsSchema.validate(frame)
    kinds = set(frame["odds_kind"].unique())
    assert kinds == {"closing_proxy", "prematch_scheduled"}

    closing = frame.filter(
        (pl.col("team_h_key") == 3) & (pl.col("odds_kind") == "closing_proxy")
    ).row(0, named=True)
    assert closing["home_odds"] == pytest.approx(1.60)
    assert closing["draw_odds"] == pytest.approx(4.10)
    assert closing["away_odds"] == pytest.approx(5.50)
    assert closing["over_25_odds"] == pytest.approx(1.65)
    assert closing["under_25_odds"] == pytest.approx(2.30)
    assert closing["source"] == "football_data"
    assert closing["available_at"] == closing["kickoff_time"]
    assert closing["available_at"] == datetime(2024, 8, 17, 14, 0, tzinfo=UTC)

    prematch = frame.filter(
        (pl.col("team_h_key") == 3) & (pl.col("odds_kind") == "prematch_scheduled")
    ).row(0, named=True)
    # Non-closing AvgH/D/A — not the closing AvgCH values.
    assert prematch["home_odds"] == pytest.approx(1.55)
    assert prematch["draw_odds"] == pytest.approx(4.20)
    assert prematch["away_odds"] == pytest.approx(6.00)
    assert prematch["over_25_odds"] == pytest.approx(1.70)
    assert prematch["under_25_odds"] == pytest.approx(2.20)
    assert prematch["available_at"] == estimated_prematch_available_at(
        prematch["kickoff_time"]
    )
    assert prematch["available_at"] == datetime(2024, 8, 16, 16, 0, tzinfo=UTC)

    liverpool_closing = frame.filter(
        (pl.col("team_h_key") == 14) & (pl.col("odds_kind") == "closing_proxy")
    ).row(0, named=True)
    assert liverpool_closing["over_25_odds"] is None
    assert liverpool_closing["under_25_odds"] is None
    liverpool_prematch = frame.filter(
        (pl.col("team_h_key") == 14) & (pl.col("odds_kind") == "prematch_scheduled")
    ).row(0, named=True)
    assert liverpool_prematch["over_25_odds"] == pytest.approx(1.40)
    assert liverpool_prematch["under_25_odds"] == pytest.approx(3.00)


def test_unknown_team_is_rejected() -> None:
    with pytest.raises(KeyError, match="Arsenal"):
        parse_e0_csv(_CSV, "2024-25", {"Tottenham": 6})


def test_stage_and_pit_separates_kinds_by_available_at(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    _seed_deadlines(store)
    staged = stage_match_odds(store, _CSV, "2024-25", _TEAMS, join_fixtures=False)
    assert staged.height == 6
    assert store.read("match_odds").height == 6
    # GW1 deadline 16 Aug 17:00 UTC is after Fri 16:00 UTC collection →
    # prematch visible; closing still stamped at kickoff → hidden.
    early = PITData(store).view(GWKey("2024-25", 1)).table("match_odds").collect()
    assert early.height == 3
    assert set(early["odds_kind"].unique()) == {"prematch_scheduled"}
    late = PITData(store).view(GWKey("2024-25", 2)).table("match_odds").collect()
    assert late.height == 6
    assert set(late["odds_kind"].unique()) == {
        "closing_proxy",
        "prematch_scheduled",
    }


def test_attach_fixture_ids_on_date_and_teams(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    kickoff = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)
    store.write(
        "fixtures",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "fixture_id": [99],
                "gw": [1],
                "team_h_key": [3],
                "team_a_key": [6],
                "kickoff_time": [kickoff],
                "team_h_score": [None],
                "team_a_score": [None],
                "finished": [False],
                "available_at": [datetime(2024, 8, 1, tzinfo=UTC)],
                "source": ["fpl_api"],
            }
        ),
    )
    odds = parse_e0_csv(_CSV, "2024-25", _TEAMS)
    joined = attach_fixture_ids(odds, store.read("fixtures"))
    arsenal = joined.filter(pl.col("team_h_key") == 3)
    assert arsenal.height == 2
    assert set(arsenal["fixture_id"].to_list()) == {99}
    others = joined.filter(pl.col("team_h_key") != 3)
    assert others["fixture_id"].null_count() == others.height


def test_ingest_keeps_both_kinds_on_unique_key(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    dest = e0_raw_path(raw, "2024-25")
    dest.parent.mkdir(parents=True)
    dest.write_text(_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    store = DataStore(tmp_path / "data")
    assert ingest_football_data(store, raw, _TEAMS, join_fixtures=False) == 1
    first = store.read("match_odds")
    assert first.height == 6
    assert set(first["odds_kind"].unique()) == {
        "closing_proxy",
        "prematch_scheduled",
    }
    assert ingest_football_data(store, raw, _TEAMS, join_fixtures=False) == 1
    assert store.read("match_odds").height == 6
    assert ingest_football_data(store, raw, _TEAMS, seasons=["2023-24"]) == 0


def test_iter_raw_e0_orders_captured_at_after_legacy(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    legacy = e0_raw_path(raw, "2024-25")
    early = e0_raw_path(raw, "2024-25", captured_at="2025-10-01")
    late = e0_raw_path(raw, "2024-25", captured_at="2025-11-01")
    for path in (legacy, early, late):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Div,Date\n", encoding="utf-8")
    paths = [path for path, season in iter_raw_e0(raw)]
    assert paths == [legacy, early, late]
    assert iter_raw_e0(raw, seasons=["2023-24"]) == []


def test_prematch_deadline_coverage_drops_early_deadlines() -> None:
    odds = parse_e0_csv(_CSV, "2024-25", _TEAMS)
    id_map = pl.DataFrame(
        {
            "team_h_key": [3, 1, 14],
            "fixture_id": [101, 102, 103],
        }
    )
    odds = (
        odds.drop("fixture_id")
        .join(id_map, on="team_h_key", how="left")
        .select(list(odds.columns))
    )
    kickoffs = {
        3: datetime(2024, 8, 17, 14, 0, tzinfo=UTC),
        1: datetime(2024, 8, 17, 16, 30, tzinfo=UTC),
        14: datetime(2024, 8, 18, 14, 0, tzinfo=UTC),
    }

    # Deadline before Friday collection → all weekend fixtures drop.
    early_deadline = datetime(2024, 8, 16, 10, 0, tzinfo=UTC)
    fixtures = pl.DataFrame(
        {
            "season": ["2024-25"] * 3,
            "fixture_id": [101, 102, 103],
            "team_h_key": [3, 1, 14],
            "kickoff_time": [kickoffs[3], kickoffs[1], kickoffs[14]],
            "deadline_time": [early_deadline] * 3,
        }
    )
    board = prematch_board_odds(odds, fixtures)
    assert board.height == 0
    result = check_prematch_deadline_coverage(fixtures, odds, min_coverage=1.0)
    assert result.passed is False
    assert result.numerator == 0
    assert result.denominator == 3
    assert set(result.dropped_fixture_ids) == {101, 102, 103}
    assert "prematch_scheduled deadline coverage failed" in result.message
    with pytest.raises(PrematchCoverageError, match="dropped"):
        require_prematch_deadline_coverage(fixtures, odds, min_coverage=1.0)

    # Deadline after Friday collection → all three pass.
    late_deadline = datetime(2024, 8, 16, 18, 0, tzinfo=UTC)
    fixtures_ok = fixtures.with_columns(pl.lit(late_deadline).alias("deadline_time"))
    board_ok = prematch_board_odds(odds, fixtures_ok)
    assert board_ok.height == 3
    assert set(board_ok["odds_kind"].unique()) == {"prematch_scheduled"}
    ok = require_prematch_deadline_coverage(fixtures_ok, odds, min_coverage=1.0)
    assert ok.passed is True
    assert ok.numerator == 3


def test_repo_team_map_covers_fixture_names() -> None:
    repo = load_team_names(default_team_map_path())
    for name in _TEAMS:
        assert name in repo


def _seed_deadlines(store: DataStore) -> None:
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25", "2024-25"],
                "gw": [1, 2],
                "deadline_time": [
                    datetime(2024, 8, 16, 17, 0, tzinfo=UTC),
                    datetime(2024, 8, 20, 17, 0, tzinfo=UTC),
                ],
                "finished": [False, False],
                "data_checked": [False, False],
                "available_at": [
                    datetime(2024, 8, 1, tzinfo=UTC),
                    datetime(2024, 8, 1, tzinfo=UTC),
                ],
                "source": ["fpl_api", "fpl_api"],
            }
        ),
    )
