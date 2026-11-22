"""Chip-use windows and rule-based landmark assignment."""

from __future__ import annotations

import polars as pl
import pytest
from pydantic import ValidationError

from fplt.optimise.chip_windows import (
    ChipUseWindows,
    chip_arm_model_id,
    gameweek_landmarks,
    is_counterfactual_rules,
    parse_chip_arm_model_id,
    rule_based_chip_windows,
    rules_label,
    team_fixture_counts,
)
from fplt.scoring.chips import ChipRules


def test_forced_must_subset_allowed_and_not_collide() -> None:
    with pytest.raises(ValidationError, match="subset"):
        ChipUseWindows(
            allowed={"bboost": frozenset({10})},
            forced={"bboost": frozenset({10, 11})},
        )
    with pytest.raises(ValidationError, match="no_chip"):
        ChipUseWindows(
            forced={"bboost": frozenset({10})},
            no_chip=frozenset({10}),
        )
    with pytest.raises(ValidationError, match="one forced"):
        ChipUseWindows(
            forced={"bboost": frozenset({10}), "3xc": frozenset({10})},
        )


def test_allows_and_requires() -> None:
    windows = ChipUseWindows(
        allowed={"bboost": frozenset({8, 9})},
        forced={"bboost": frozenset({8})},
        no_chip=frozenset({12}),
    )
    assert windows.allows("bboost", 8)
    assert windows.requires("bboost", 8)
    assert not windows.allows("bboost", 10)
    assert not windows.allows("bboost", 12)
    assert windows.allows("3xc", 9)


def test_team_fixture_counts_and_landmarks() -> None:
    fixtures = pl.DataFrame(
        {
            "season": ["2024-25"] * 5,
            "gw": [10, 10, 10, 11, 11],
            "team_h_key": [1, 1, 3, 1, 2],
            "team_a_key": [2, 3, 4, 2, 3],
        }
    )
    counts = team_fixture_counts(fixtures, season="2024-25")
    assert (
        counts.filter((pl.col("gw") == 10) & (pl.col("team_key") == 1))["n_fixtures"][0]
        == 2
    )
    landmarks = gameweek_landmarks(counts)
    gw10 = landmarks.filter(pl.col("gw") == 10).row(0, named=True)
    assert gw10["n_double_teams"] >= 1
    gw11 = landmarks.filter(pl.col("gw") == 11).row(0, named=True)
    assert gw11["n_blank_teams"] >= 1


def test_rule_based_assigns_one_chip_per_landmark() -> None:
    fixtures = _landmark_fixtures()
    rules = ChipRules.for_season("2025-26")
    predictions = pl.DataFrame(
        {
            "gw": [10, 10, 28, 28],
            "player_key": [1, 2, 1, 2],
            "xpts": [20.0, 5.0, 12.0, 18.0],
        }
    )
    windows = rule_based_chip_windows(
        fixtures,
        season="2024-25",
        rules=rules,
        predictions=predictions,
        gameweeks=list(range(1, 39)),
    )
    assert windows.forced["bboost"]
    assert windows.forced["3xc"]
    assert windows.forced["freehit"]
    assert windows.forced["wildcard"]
    used = [gw for weeks in windows.forced.values() for gw in weeks]
    assert len(used) == len(set(used))
    for chip, weeks in windows.forced.items():
        assert weeks <= windows.allowed[chip]


def test_rules_label_marks_counterfactual() -> None:
    assert rules_label("2025-26", "2025-26") == "2025-26"
    assert rules_label("2024-25", "2025-26") == "2024-25-under-2025/26-rules"
    assert is_counterfactual_rules("2024-25", "2025-26")
    assert not is_counterfactual_rules("2025-26", "2025-26")


def test_chip_arm_model_id_round_trip() -> None:
    encoded = chip_arm_model_id("abc123", "rule_based")
    base, arm = parse_chip_arm_model_id(encoded)
    assert base == "abc123"
    assert arm == "rule_based"
    assert parse_chip_arm_model_id("plain") == ("plain", None)


def _landmark_fixtures() -> pl.DataFrame:
    """Two doubles + one blank per half so BB/TC/FH/WC can all assign."""
    rows: list[dict[str, object]] = []
    # First half: GW10 biggest double, GW12 smaller double, GW11 blank-heavy.
    rows.extend(
        [
            {"season": "2024-25", "gw": 10, "team_h_key": 1, "team_a_key": 2},
            {"season": "2024-25", "gw": 10, "team_h_key": 1, "team_a_key": 3},
            {"season": "2024-25", "gw": 10, "team_h_key": 4, "team_a_key": 5},
            {"season": "2024-25", "gw": 10, "team_h_key": 2, "team_a_key": 4},
            {"season": "2024-25", "gw": 11, "team_h_key": 2, "team_a_key": 3},
            {"season": "2024-25", "gw": 12, "team_h_key": 1, "team_a_key": 2},
            {"season": "2024-25", "gw": 12, "team_h_key": 1, "team_a_key": 3},
        ]
    )
    # Second half mirrors.
    rows.extend(
        [
            {"season": "2024-25", "gw": 28, "team_h_key": 1, "team_a_key": 2},
            {"season": "2024-25", "gw": 28, "team_h_key": 1, "team_a_key": 3},
            {"season": "2024-25", "gw": 28, "team_h_key": 4, "team_a_key": 5},
            {"season": "2024-25", "gw": 28, "team_h_key": 2, "team_a_key": 4},
            {"season": "2024-25", "gw": 29, "team_h_key": 2, "team_a_key": 3},
            {"season": "2024-25", "gw": 30, "team_h_key": 1, "team_a_key": 2},
            {"season": "2024-25", "gw": 30, "team_h_key": 1, "team_a_key": 3},
        ]
    )
    return pl.DataFrame(rows)
