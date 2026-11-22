"""Per-season chip rules load and validate."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fplt.scoring import ChipRules
from fplt.scoring.chips import ChipGrant, ChipWindow

_SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27")
_TWO_SET = frozenset({"2025-26", "2026-27"})
_CHIP_NAMES = frozenset({"wildcard", "freehit", "bboost", "3xc"})


@pytest.mark.parametrize("season", _SEASONS)
def test_every_trained_season_loads(season: str) -> None:
    rules = ChipRules.for_season(season)
    assert rules.one_chip_per_gameweek is True
    assert rules.free_hit_keeps_free_transfers is True
    counts = rules.remaining_counts()
    assert set(counts) == _CHIP_NAMES
    if season in _TWO_SET:
        assert counts == {
            "wildcard": 2,
            "freehit": 2,
            "bboost": 2,
            "3xc": 2,
        }
        first = next(window for window in rules.windows if window.id == "first")
        assert first.stop_event == 19
        assert first.carries_over is False
        assert len(rules.grants_for_window("first")) == 4
        assert len(rules.grants_for_window("second")) == 4
    else:
        assert counts == {
            "wildcard": 2,
            "freehit": 1,
            "bboost": 1,
            "3xc": 1,
        }


def test_missing_season_raises() -> None:
    with pytest.raises(FileNotFoundError, match="1999-00"):
        ChipRules.for_season("1999-00")


def test_season_must_be_a_file_name() -> None:
    with pytest.raises(ValueError, match="file name"):
        ChipRules.for_season("../2025-26")


def test_unknown_window_on_grant_is_rejected() -> None:
    loaded = ChipRules.for_season("2025-26").model_dump()
    loaded["grants"][0]["window"] = "missing"
    with pytest.raises(ValidationError, match="unknown window"):
        ChipRules.model_validate(loaded)


def test_duplicate_window_ids_are_rejected() -> None:
    loaded = ChipRules.for_season("2025-26").model_dump()
    loaded["windows"].append(dict(loaded["windows"][0]))
    with pytest.raises(ValidationError, match="unique"):
        ChipRules.model_validate(loaded)


def test_unordered_window_is_rejected() -> None:
    with pytest.raises(ValidationError, match="start_event"):
        ChipWindow(id="bad", start_event=20, stop_event=19, carries_over=False)


def test_grants_for_unknown_window_raises() -> None:
    rules = ChipRules.for_season("2025-26")
    with pytest.raises(KeyError, match="unknown chip window"):
        rules.grants_for_window("nope")


def test_yaml_files_parse_as_mappings() -> None:
    root = Path(__file__).resolve().parents[2] / "configs" / "chips"
    for season in _SEASONS:
        loaded = yaml.safe_load((root / f"{season}.yaml").read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        ChipRules.model_validate(loaded)


def test_chip_grant_names_match_fpl_api() -> None:
    grant = ChipGrant(name="3xc", window="first")
    assert grant.name == "3xc"
