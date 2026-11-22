"""Hardened availability gate: this/next chance and null-with-news."""

from __future__ import annotations

from datetime import UTC, datetime

from fplt.models.components.minutes.common import (
    PlayerAvailability,
    chance,
    gate_for,
)


def test_prefers_this_round_over_next() -> None:
    assert chance("a", 100.0, 0.5, chance_this=25.0) == 0.25


def test_null_chance_with_news_is_not_full_availability() -> None:
    assert chance("a", None, 0.5, news="Knock - 75% chance of playing") == 0.5
    stamped = datetime(2024, 8, 10, tzinfo=UTC)
    assert chance("a", None, 0.5, news_added=stamped) == 0.5


def test_available_without_news_stays_full() -> None:
    assert chance("a", None, 0.5) == 1.0
    assert chance("a", None, 0.5, news="") == 1.0


def test_unavailable_and_doubtful_unchanged() -> None:
    assert chance("i", None, 0.5) == 0.0
    assert chance("d", None, 0.4) == 0.4
    assert chance("d", 80.0, 0.4) == 0.8


def test_gate_for_uses_this_round() -> None:
    tags = PlayerAvailability(
        status="a",
        chance_next=100.0,
        chance_this=50.0,
        news="",
        news_added=None,
    )
    assert gate_for(tags, 0.5) == 0.5
