"""Fuzzy id-map matcher (ops-07): accents, nicknames, same-name clubs, threshold."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from fplt.ops.idmap import (
    SourcePlayer,
    TargetPlayer,
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
    assert hits and hits[0].high_confidence
    assert high_confidence_map_entries(hits) == {42: 4001}

    out = tmp_path / "candidates.yaml"
    write_candidates_yaml(out, hits, source="understat")
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert loaded["source"] == "understat"
    assert loaded["high_confidence_players"] == {"42": 4001}
    assert "nothing is auto-merged" in loaded["note"]
