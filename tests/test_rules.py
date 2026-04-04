import pytest
from fplt.rules import score_components, score_dataframe
import pandas as pd


def test_no_minutes_scores_zero():
    assert score_components(3, minutes=0) == 0.0


def test_sub_appearance():
    pts = score_components(3, minutes=45)
    assert pts == 1.0


def test_full_appearance():
    pts = score_components(3, minutes=60)
    assert pts == 2.0


def test_fwd_goal():
    pts = score_components(3, minutes=90, goals_scored=1)
    assert pts == 2 + 4  # appearance + goal


def test_mid_goal():
    pts = score_components(2, minutes=90, goals_scored=1)
    assert pts == 2 + 5


def test_def_goal():
    pts = score_components(1, minutes=90, goals_scored=1)
    assert pts == 2 + 6


def test_gk_goal():
    pts = score_components(0, minutes=90, goals_scored=1)
    assert pts == 2 + 6


def test_assist():
    pts = score_components(3, minutes=90, assists=1)
    assert pts == 2 + 3


def test_gk_clean_sheet():
    pts = score_components(0, minutes=90, clean_sheets=1)
    assert pts == 2 + 4


def test_def_clean_sheet():
    pts = score_components(1, minutes=90, clean_sheets=1)
    assert pts == 2 + 4


def test_mid_clean_sheet():
    pts = score_components(2, minutes=90, clean_sheets=1)
    assert pts == 2 + 1


def test_fwd_clean_sheet():
    pts = score_components(3, minutes=90, clean_sheets=1)
    assert pts == 2 + 0


def test_gk_save_bonus():
    pts = score_components(0, minutes=90, saves=6)
    assert pts == 2 + 2  # 6 saves = 2 bonus pts


def test_yellow_card():
    pts = score_components(3, minutes=90, yellow_cards=1)
    assert pts == 2 - 1


def test_red_card():
    pts = score_components(3, minutes=90, red_cards=1)
    assert pts == 2 - 3


def test_own_goal():
    pts = score_components(3, minutes=90, own_goals=1)
    assert pts == 2 - 2


def test_penalty_miss():
    pts = score_components(3, minutes=90, penalties_missed=1)
    assert pts == 2 - 2


def test_penalty_save():
    pts = score_components(0, minutes=90, penalties_saved=1)
    assert pts == 2 + 5


def test_bonus():
    pts = score_components(3, minutes=90, bonus=3)
    assert pts == 2 + 3


def test_score_dataframe():
    df = pd.DataFrame([
        {"position_enc": 3, "minutes": 90, "goals_scored": 1, "assists": 1},
        {"position_enc": 0, "minutes": 90, "clean_sheets": 1, "saves": 3},
    ])
    result = score_dataframe(df)
    assert result.iloc[0] == 2 + 4 + 3   # FWD: appearance + goal + assist
    assert result.iloc[1] == 2 + 4 + 1   # GK:  appearance + clean sheet + 1 save bonus
