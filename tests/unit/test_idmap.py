"""Fuzzy id-map matcher (ops-07): accents, nicknames, same-name clubs, threshold."""

from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from fplt.ops.idmap import (
    MatchCandidate,
    SourcePlayer,
    TargetPlayer,
    _token_subset_match,
    _understat_roster_players,
    collect_api_football_unmapped,
    collect_understat_unmapped,
    format_candidates_report,
    high_confidence_map_entries,
    name_similarity,
    normalize_name,
    suggest_matches,
    write_candidates_yaml,
)


def _target(
    player_key: int,
    *,
    web_name: str,
    full_name: str | None = None,
    team_key: int = 3,
    position: str = "MID",
    dob: date | None = None,
) -> TargetPlayer:
    return TargetPlayer(
        player_key=player_key,
        web_name=web_name,
        full_name=full_name or web_name,
        team_key=team_key,
        position=position,
        dob=dob,
    )


def test_normalize_name_strips_accents() -> None:
    assert normalize_name("José Mourinho") == "jose mourinho"
    assert normalize_name("Škriniar") == "skriniar"
    assert name_similarity("José", "Jose") == 1.0


def test_nickname_matches_full_name() -> None:
    unmapped = [SourcePlayer(source_id=1, name="Son", team_key=6)]
    targets = [
        _target(1001, web_name="Son", full_name="Heung-Min Son", team_key=6),
        _target(1002, web_name="Kane", full_name="Harry Kane", team_key=6),
    ]
    hits = suggest_matches(unmapped, targets, min_score=0.75, high_confidence=0.92)
    assert len(hits) == 1
    assert hits[0].player_key == 1001
    assert hits[0].score >= 0.9


def test_same_name_different_clubs_needs_team() -> None:
    unmapped = [SourcePlayer(source_id=9, name="J. Smith", team_key=14)]
    targets = [
        _target(2001, web_name="J. Smith", full_name="John Smith", team_key=3),
        _target(2002, web_name="J. Smith", full_name="John Smith", team_key=14),
    ]
    hits = suggest_matches(unmapped, targets, min_score=0.7, high_confidence=0.92)
    assert len(hits) == 1
    assert hits[0].player_key == 2002
    assert hits[0].high_confidence is True

    ambiguous = [SourcePlayer(source_id=10, name="J. Smith", team_key=None)]
    hits2 = suggest_matches(ambiguous, targets, min_score=0.7, high_confidence=0.92)
    assert len(hits2) == 1
    assert hits2[0].high_confidence is False


def test_below_threshold_never_suggested() -> None:
    unmapped = [SourcePlayer(source_id=3, name="Completely Unrelated", team_key=3)]
    targets = [_target(3001, web_name="Salah", full_name="Mohamed Salah", team_key=14)]
    hits = suggest_matches(unmapped, targets, min_score=0.75, high_confidence=0.92)
    assert hits == []


def test_high_confidence_map_entries_and_yaml(tmp_path: Path) -> None:
    unmapped = [
        SourcePlayer(source_id=42, name="Bukayo Saka", team_key=3, position="MID")
    ]
    targets = [
        _target(
            4001,
            web_name="Saka",
            full_name="Bukayo Saka",
            team_key=3,
            position="MID",
        )
    ]
    hits = suggest_matches(unmapped, targets, min_score=0.5, high_confidence=0.9)
    assert hits
    assert hits[0].high_confidence
    assert high_confidence_map_entries(hits) == {42: 4001}

    out = tmp_path / "candidates.yaml"
    write_candidates_yaml(out, hits, source="understat")
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert loaded["source"] == "understat"
    assert loaded["high_confidence_players"] == {"42": 4001}
    assert "nothing is auto-merged" in loaded["note"]


def test_name_similarity_empty_after_normalize() -> None:
    assert name_similarity("!!!", "Saka") == 0.0
    assert name_similarity("Saka", "   ") == 0.0
    assert name_similarity("", "") == 0.0


def test_suggest_matches_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="min_score"):
        suggest_matches([], [], min_score=-0.1)
    with pytest.raises(ValueError, match="min_score"):
        suggest_matches([], [], min_score=1.1)
    with pytest.raises(ValueError, match="high_confidence"):
        suggest_matches([], [], min_score=0.8, high_confidence=0.7)
    with pytest.raises(ValueError, match="high_confidence"):
        suggest_matches([], [], min_score=0.5, high_confidence=1.1)


def test_suggest_matches_skips_blank_target_labels() -> None:
    unmapped = [SourcePlayer(source_id=1, name="Saka", team_key=3)]
    targets = [
        _target(1, web_name="!!!", full_name="!!!", team_key=3),
        _target(2, web_name="Saka", full_name="Bukayo Saka", team_key=3),
    ]
    hits = suggest_matches(unmapped, targets, min_score=0.7, high_confidence=0.95)
    assert len(hits) == 1
    assert hits[0].player_key == 2


def test_format_candidates_report_empty_and_high() -> None:
    empty = format_candidates_report([], source="understat")
    assert empty == "idmap suggest (understat): no candidates above threshold"

    review_only = [
        MatchCandidate(
            source_id=1,
            source_name="A",
            player_key=10,
            target_name="B",
            score=0.8,
            high_confidence=False,
            reason="name=0.800",
        )
    ]
    text = format_candidates_report(review_only, source="api_football")
    assert "1 candidate(s)" in text
    assert "[review]" in text
    assert "high-confidence players map fragment" not in text

    high = [
        MatchCandidate(
            source_id=9,
            source_name="Saka",
            player_key=4001,
            target_name="Bukayo Saka",
            score=0.99,
            high_confidence=True,
            reason="name=0.990+team",
        )
    ]
    text_high = format_candidates_report(high, source="understat")
    assert "[HIGH]" in text_high
    assert "high-confidence players map fragment:" in text_high
    assert "9: 4001" in text_high


def test_token_subset_match_rejects_empty_sides() -> None:
    assert _token_subset_match("", "saka") is False
    assert _token_subset_match("saka", "") is False
    assert _token_subset_match("bukayo saka", "saka") is True


def test_same_name_multi_club_demotes_high_score() -> None:
    # team_key=None avoids the team_mismatch penalty so the exact-name score
    # stays at the high-confidence floor and hits the multi-club demotion.
    unmapped = [SourcePlayer(source_id=5, name="John Smith", team_key=None)]
    targets = [
        _target(1, web_name="John Smith", full_name="John Smith", team_key=3),
        _target(2, web_name="John Smith", full_name="John Smith", team_key=14),
    ]
    hits = suggest_matches(unmapped, targets, min_score=0.7, high_confidence=0.92)
    assert len(hits) == 1
    assert hits[0].high_confidence is False
    assert "same-name multi-club" in hits[0].reason
    assert hits[0].score < 0.92


def test_same_name_multi_club_below_high_skips_demotion_body() -> None:
    # Wrong team applies the mismatch penalty so score stays below
    # high_confidence; the multi-club guard still runs but does not demote.
    unmapped = [SourcePlayer(source_id=5, name="John Smith", team_key=99)]
    targets = [
        _target(1, web_name="John Smith", full_name="John Smith", team_key=3),
        _target(2, web_name="John Smith", full_name="John Smith", team_key=14),
    ]
    hits = suggest_matches(unmapped, targets, min_score=0.7, high_confidence=0.92)
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(0.85)
    assert "same-name multi-club" not in hits[0].reason
    assert hits[0].high_confidence is False


def test_same_name_multi_club_demotion_can_drop_below_min_score() -> None:
    unmapped = [SourcePlayer(source_id=5, name="John Smith", team_key=None)]
    targets = [
        _target(1, web_name="John Smith", full_name="John Smith", team_key=3),
        _target(2, web_name="John Smith", full_name="John Smith", team_key=14),
    ]
    hits = suggest_matches(unmapped, targets, min_score=0.95, high_confidence=0.95)
    assert hits == []


def test_score_pair_dob_match_and_mismatch() -> None:
    dob = date(1998, 5, 1)
    unmapped_ok = [
        SourcePlayer(
            source_id=1,
            name="Exact Name",
            team_key=3,
            position="MID",
            dob=dob,
        )
    ]
    targets = [
        _target(
            10,
            web_name="Exact Name",
            full_name="Exact Name",
            team_key=3,
            position="MID",
            dob=dob,
        )
    ]
    hits = suggest_matches(unmapped_ok, targets, min_score=0.5, high_confidence=0.9)
    assert hits
    assert "dob" in hits[0].reason
    assert "dob_mismatch" not in hits[0].reason

    unmapped_bad = [
        SourcePlayer(
            source_id=2,
            name="Exact Name",
            team_key=3,
            dob=date(1990, 1, 1),
        )
    ]
    hits_bad = suggest_matches(
        unmapped_bad, targets, min_score=0.5, high_confidence=0.9
    )
    assert hits_bad
    assert "dob_mismatch" in hits_bad[0].reason


def test_collect_understat_unmapped(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    match_dir = raw / "understat" / "season=2024-25" / "match_id=42"
    match_dir.mkdir(parents=True)
    payload = {
        "rosters": {
            "h": {
                "1": {
                    "player_id": "501",
                    "team_id": "83",
                    "player": "Home One",
                },
                "2": {
                    "player_id": "502",
                    "team_id": "83",
                    "player": "Already Mapped",
                },
            },
            "a": [
                {"player_id": "999", "team_id": "87", "player": "Away One"},
                {"team_id": "87", "player": "Missing Id"},
            ],
            "x": "not-a-side-list",
        }
    }
    with gzip.open(match_dir / "match.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    found = collect_understat_unmapped(
        raw,
        "2024-25",
        player_keys={502: 100},
        team_keys={83: 3, 87: 14},
    )
    by_id = {row.source_id: row for row in found}
    assert set(by_id) == {501, 999}
    assert by_id[501].name == "Home One"
    assert by_id[501].team_key == 3
    assert by_id[999].team_key == 14


def test_collect_api_football_unmapped(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    injury_dir = (
        raw
        / "api_football"
        / "season=2024-25"
        / "injuries"
        / "captured_at=2024-08-10T00-00-00Z"
    )
    injury_dir.mkdir(parents=True)
    payload = {
        "response": [
            {
                "player": {"id": 276, "name": "Mapped Player"},
                "team": {"id": 34, "name": "Newcastle"},
            },
            {
                "player": {"id": 999, "name": "Unmapped With Team"},
                "team": {"id": 40, "name": "Liverpool"},
            },
            {
                "player": {"id": 1000},
                "team": {"name": "No Id"},
            },
            {"player": "not-a-dict"},
            "skip-me",
            {
                "player": {"name": "No Player Id"},
                "team": {"id": 34},
            },
        ]
    }
    with gzip.open(injury_dir / "injuries.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    # Non-list response should be skipped entirely.
    bad_dir = (
        raw
        / "api_football"
        / "season=2024-25"
        / "injuries"
        / "captured_at=2024-08-11T00-00-00Z"
    )
    bad_dir.mkdir(parents=True)
    with gzip.open(bad_dir / "injuries.json.gz", "wt", encoding="utf-8") as handle:
        json.dump({"response": {"oops": True}}, handle)

    found = collect_api_football_unmapped(
        raw,
        "2024-25",
        player_keys={276: 100},
        team_keys={40: 14},
    )
    by_id = {row.source_id: row for row in found}
    assert set(by_id) == {999, 1000}
    assert by_id[999].name == "Unmapped With Team"
    assert by_id[999].team_key == 14
    assert by_id[1000].name == "id:1000"
    assert by_id[1000].team_key is None


def test_understat_roster_players_normalised_and_roster_shapes() -> None:
    roster_rows = _understat_roster_players(
        {
            "rosters": {
                "h": None,
                "a": {"1": {"player_id": "7", "player": "Away", "team_id": "2"}},
            }
        }
    )
    assert roster_rows == [(7, "Away", 2)]

    # Side that is neither dict nor list is skipped; missing name falls back.
    skipped = _understat_roster_players(
        {
            "rosters": {
                "h": 123,
                "a": [{"player_id": "8", "team_id": "9"}],
            }
        }
    )
    assert skipped == [(8, "id:8", 9)]

    normalised = _understat_roster_players(
        {
            "team_h": 83,
            "team_a": 87,
            "h": [{"id": 1, "player": "Home"}],
            "a": [{"id": 2, "name": "Away"}],
        }
    )
    assert normalised == [(1, "Home", 83), (2, "Away", 87)]

    # No rosters and non-list sides → empty.
    assert _understat_roster_players({"h": {}, "a": None}) == []
