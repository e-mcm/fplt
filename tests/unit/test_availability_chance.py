"""Hardened availability gate: this/next chance and null-with-news."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fplt.models.components.minutes.common import (
    PlayerAvailability,
    PlayerInjury,
    availability_gate,
    chance,
    chance_at_step,
    gate_at_step,
    gate_for,
    horizon_step,
    injury_multiplier_at_step,
    recover_toward_one,
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
    assert chance("s", None, 0.5) == 0.0
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


def test_use_availability_gate_false_ignores_fpl_tags() -> None:
    tags = PlayerAvailability(
        status="i",
        chance_next=0.0,
        chance_this=0.0,
        news="Injured",
        news_added=None,
    )
    assert availability_gate(tags, 0.5, use_availability_gate=False) == 1.0
    # Injury overlay can still lower the ungated baseline.
    out = PlayerInjury(category="out", injury_type="Suspension", reason="")
    assert (
        availability_gate(
            tags,
            0.5,
            injury=out,
            use_injury_overlay=True,
            use_availability_gate=False,
        )
        == 0.0
    )


def test_horizon_step_clamps_past_and_matches_offsets() -> None:
    assert horizon_step(8, 8) == 0
    assert horizon_step(8, 10) == 2
    assert horizon_step(8, 5) == 0


def test_recover_toward_one_holds_anchor_then_blends() -> None:
    assert recover_toward_one(0.25, step=0, recovery_games=3.0) == 0.25
    assert recover_toward_one(0.25, step=1, recovery_games=3.0) == 0.25
    assert recover_toward_one(0.25, step=2, recovery_games=3.0) == pytest.approx(
        0.25 + 0.75 / 3.0
    )
    assert recover_toward_one(0.25, step=4, recovery_games=3.0) == 1.0
    assert recover_toward_one(0.0, step=3, recovery_games=0.0) == 1.0


def test_chance_at_step_uses_this_then_next_then_recovers() -> None:
    # This-round 25%, next-round 75% — far steps recover from next toward 1.
    assert (
        chance_at_step("a", 75.0, 0.5, chance_this=25.0, step=0, recovery_games=3.0)
        == 0.25
    )
    assert (
        chance_at_step("a", 75.0, 0.5, chance_this=25.0, step=1, recovery_games=3.0)
        == 0.75
    )
    assert chance_at_step(
        "a", 75.0, 0.5, chance_this=25.0, step=4, recovery_games=3.0
    ) == pytest.approx(1.0)


def test_suspended_this_round_clears_on_next_when_chance_next_full() -> None:
    tags = PlayerAvailability(
        status="s",
        chance_next=100.0,
        chance_this=0.0,
        news="Suspended",
        news_added=None,
    )
    assert gate_at_step(tags, 0.5, step=0) == 0.0
    assert gate_at_step(tags, 0.5, step=1) == 1.0
    assert gate_at_step(tags, 0.5, step=3) == 1.0


def test_injury_overlay_recovers_for_player_level_not_fixture_scoped() -> None:
    player_out = PlayerInjury(category="out", injury_type="Knee", reason="ACL")
    ban = PlayerInjury(
        category="out", injury_type="Suspension", reason="Red", fixture_id=99
    )
    assert injury_multiplier_at_step(player_out, 0.5, step=0) == 0.0
    assert injury_multiplier_at_step(player_out, 0.5, step=1) == 0.0
    assert injury_multiplier_at_step(
        player_out, 0.5, step=4, recovery_games=3.0
    ) == pytest.approx(1.0)
    # Fixture-scoped bans stay hard on that fixture (no recovery).
    assert injury_multiplier_at_step(ban, 0.5, step=4, recovery_games=3.0) == 0.0


def test_availability_gate_horizon_recovers_zero_this_round() -> None:
    tags = PlayerAvailability(
        status="a",
        chance_next=0.0,
        chance_this=0.0,
        news="",
        news_added=None,
    )
    assert availability_gate(tags, 0.5, step=0, recovery_games=3.0) == 0.0
    assert availability_gate(tags, 0.5, step=1, recovery_games=3.0) == 0.0
    assert availability_gate(tags, 0.5, step=4, recovery_games=3.0) == pytest.approx(
        1.0
    )
