"""Unit tests for ``fplt.analysis.this_week`` on synthetic frames."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.analysis.this_week import (
    default_official_selection,
    latest_rows,
    plan_panel,
    player_drilldown,
    players_table,
    squad_keys_from_plan,
)
from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.manifests import OfficialPointer

_TS = datetime(2026, 11, 27, 11, 30, tzinfo=UTC)
_SEASON = "2026-27"
_RUN = "r_20261127T120000Z_pred01"


def _pointers_frame() -> pl.DataFrame:
    from fplt.analysis.overview import official_pointers_frame

    pointer = OfficialPointer(
        predict_run_id=_RUN,
        plan_run_id="r_plan",
        role="primary",
        squad_mode="real",
        written_at=_TS,
    )
    candidate = OfficialPointer(
        predict_run_id=_RUN,
        plan_run_id="r_plan_h5",
        role="candidate",
        squad_mode="virtual",
        written_at=_TS,
    )
    return official_pointers_frame(
        [
            (_SEASON, 12, "T-90m", "v4_h3", pointer),
            (_SEASON, 13, "T-90m", "v4_h3", pointer),
            (_SEASON, 13, "T-90m", "v4_h5", candidate),
            (_SEASON, 13, "T-24h", "v4_h3", pointer),
        ]
    )


def _predictions() -> pl.DataFrame:
    # Player 100: DGW in gw13 (two fixtures), blank gw14, single gw15.
    # Player 999: only in predictions (missing from curated players).
    return pl.DataFrame(
        {
            "model_id": ["deadbeef"] * 5,
            "as_of_season": [_SEASON] * 5,
            "as_of_gw": [13] * 5,
            "target_season": [_SEASON] * 5,
            "target_gw": [13, 13, 15, 13, 15],
            "player_key": [100, 100, 100, 999, 999],
            "fixture_id": [1, 2, 3, 1, 3],
            "xpts": [4.0, 3.0, 5.0, 2.0, 1.0],
            "xpts_sd": [1.0] * 5,
            "p_played": [0.9, 0.8, 0.85, 0.5, 0.4],
            "p_60": [0.7, 0.6, 0.65, 0.3, 0.2],
            "p_goal": [0.3, 0.2, 0.25, 0.1, 0.05],
            "p_assist": [0.2, 0.15, 0.18, 0.05, 0.02],
            "p_cs": [0.4, 0.35, 0.3, 0.2, 0.1],
            "p_haul": [0.1] * 5,
        }
    )


def _players() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [_SEASON],
            "player_key": [100],
            "element_id": [10],
            "team_key": [3],
            "position": ["MID"],
            "web_name": ["Saka"],
            "available_at": [_TS],
            "source": ["test"],
        }
    )


def _teams() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [_SEASON, _SEASON],
            "team_key": [3, 7],
            "team_id": [1, 2],
            "name": ["Arsenal", "Spurs"],
            "short_name": ["ARS", "TOT"],
            "available_at": [_TS, _TS],
            "source": ["test", "test"],
        }
    )


def _status() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [_SEASON],
            "player_key": [100],
            "status": ["a"],
            "news": [""],
            "news_added": [None],
            "price": [105],
            "ep_next": [6.0],
            "chance_of_playing": [100.0],
            "chance_of_playing_this_round": [100.0],
            "penalties_order": [1],
            "corners_and_indirect_freekicks_order": [None],
            "direct_freekicks_order": [None],
            "available_at": [_TS],
            "source": ["test"],
        }
    )


def _fixtures() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [_SEASON, _SEASON, _SEASON],
            "fixture_id": [1, 2, 3],
            "gw": [13, 13, 15],
            "team_h_key": [3, 7, 3],
            "team_a_key": [7, 3, 7],
            "kickoff_time": [_TS, _TS, _TS],
            "team_h_score": [None, None, None],
            "team_a_score": [None, None, None],
            "finished": [False, False, False],
            "available_at": [_TS, _TS, _TS],
            "source": ["test", "test", "test"],
        }
    )


def _plan() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_id": ["deadbeef"] * 3,
            "as_of_season": [_SEASON] * 3,
            "as_of_gw": [13] * 3,
            "target_gw": [13] * 3,
            "step": [0, 0, 0],
            "player_key": [100, 200, 300],
            "position": ["MID", "FWD", "DEF"],
            "team_key": [3, 7, 3],
            "in_xi": [True, True, False],
            "captain": [True, False, False],
            "vice": [False, True, False],
            "bought": [False, True, False],
            "sold": [False, False, True],
            "xpts": [7.0, 5.0, 2.0],
            "hit": [4.0, 4.0, 4.0],
        }
    )


def _component_minutes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": [_RUN, _RUN],
            "as_of_season": [_SEASON, _SEASON],
            "as_of_gw": [13, 13],
            "deadline_utc": [_TS, _TS],
            "target_season": [_SEASON, _SEASON],
            "target_gw": [13, 13],
            "horizon": [0, 0],
            "fixture_id": [1, 2],
            "player_key": [100, 100],
            "team_key": [3, 3],
            "schema_version": [SCHEMA_VERSION, SCHEMA_VERSION],
            "xmins": [80.0, 70.0],
            "p_played": [0.95, 0.9],
            "p_60": [0.8, 0.7],
        }
    )


def _component_scoreline() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": [_RUN, _RUN],
            "as_of_season": [_SEASON, _SEASON],
            "as_of_gw": [13, 13],
            "deadline_utc": [_TS, _TS],
            "target_season": [_SEASON, _SEASON],
            "target_gw": [13, 13],
            "horizon": [0, 0],
            "fixture_id": [1, 2],
            "team_key": [3, 3],
            "schema_version": [SCHEMA_VERSION, SCHEMA_VERSION],
            "opponent_team_key": [7, 7],
            "is_home": [True, False],
            "team_eg": [1.8, 1.2],
            "opp_eg": [0.9, 1.5],
            "p_cs": [0.45, 0.25],
        }
    )


def _component_involvement() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": [_RUN, _RUN],
            "as_of_season": [_SEASON, _SEASON],
            "as_of_gw": [13, 13],
            "deadline_utc": [_TS, _TS],
            "target_season": [_SEASON, _SEASON],
            "target_gw": [13, 13],
            "horizon": [0, 0],
            "fixture_id": [1, 2],
            "player_key": [100, 100],
            "team_key": [3, 3],
            "schema_version": [SCHEMA_VERSION, SCHEMA_VERSION],
            "goal_weight": [0.3, 0.25],
            "assist_weight": [0.2, 0.22],
            "exp_goals": [0.54, 0.3],
            "exp_assists": [0.36, 0.26],
            "exp_goals_method": ["team_eg_x_weight", "team_eg_x_weight"],
        }
    )


def test_default_selection_applied_then_primary_t90m() -> None:
    pointers = _pointers_frame()
    # Applied line v4_h5 at T-90m → prefer that over primary.
    sel = default_official_selection(pointers, applied_line="v4_h5")
    assert sel is not None
    assert sel.line == "v4_h5"
    assert sel.checkpoint == "T-90m"
    assert sel.gw == 13
    # No applied → newest primary pointer (same written_at → T-90m > T-24h).
    sel2 = default_official_selection(pointers, applied_line=None)
    assert sel2 is not None
    assert sel2.line == "v4_h3"
    assert sel2.role == "primary"
    assert sel2.gw == 13
    assert sel2.checkpoint == "T-90m"
    # preferred_checkpoint still scopes when callers ask for T-90m only.
    sel3 = default_official_selection(
        pointers, applied_line=None, preferred_checkpoint="T-90m"
    )
    assert sel3 is not None
    assert sel3.checkpoint == "T-90m"


def test_default_selection_prefers_newest_manual_checkpoint() -> None:
    """Without applied.json, newest written_at checkpoint wins (#364)."""
    from fplt.analysis.overview import official_pointers_frame

    older = OfficialPointer(
        predict_run_id=_RUN,
        plan_run_id="r_plan",
        role="primary",
        squad_mode="real",
        written_at=datetime(2026, 11, 27, 11, 30, tzinfo=UTC),
    )
    newer = OfficialPointer(
        predict_run_id=_RUN,
        plan_run_id="r_plan_manual",
        role="primary",
        squad_mode="real",
        written_at=datetime(2026, 11, 28, 7, 30, 15, tzinfo=UTC),
    )
    pointers = official_pointers_frame(
        [
            (_SEASON, 13, "T-90m", "v4_h3", older),
            (_SEASON, 13, "manual-20261128T073015Z", "v4_h3", newer),
        ]
    )
    sel = default_official_selection(pointers, applied_line=None)
    assert sel is not None
    assert sel.checkpoint == "manual-20261128T073015Z"
    assert sel.plan_run_id == "r_plan_manual"
    # Applied line still wins even when an older checkpoint.
    sel_applied = default_official_selection(pointers, applied_line="v4_h3")
    assert sel_applied is not None
    assert sel_applied.checkpoint == "manual-20261128T073015Z"


def test_default_selection_empty() -> None:
    assert default_official_selection(pl.DataFrame()) is None


def test_applied_for_gw_prefers_t90m_then_none(tmp_path: Path) -> None:
    """``_applied_for_gw`` scans checkpoints; returns None when none applied."""
    from fplt.ui.data import DashboardStore
    from fplt.ui.pages.this_week import _applied_for_gw

    pointers = _pointers_frame()
    store = DashboardStore(tmp_path)
    assert _applied_for_gw(store, season=_SEASON, gw=13, pointers=pointers) is None


def test_players_table_horizon_double_and_blank() -> None:
    table = players_table(
        _predictions(),
        players=_players(),
        teams=_teams(),
        status=_status(),
        as_of_gw=13,
        horizon=3,  # gw13,14,15 — 14 blank
        squad_keys=[100],
    )
    assert "xpts_gw13" in table.columns
    assert "xpts_gw14" in table.columns
    assert "xpts_gw15" in table.columns
    saka = table.filter(pl.col("player_key") == 100).to_dicts()[0]
    assert saka["xpts_gw13"] == pytest.approx(7.0)  # 4+3 DGW
    assert saka["xpts_gw14"] == pytest.approx(0.0)  # blank
    assert saka["xpts_gw15"] == pytest.approx(5.0)
    assert saka["total_xpts"] == pytest.approx(12.0)
    assert saka["web_name"] == "Saka"
    assert saka["team"] == "ARS"
    assert saka["in_squad"] is True


def test_players_table_missing_curated_placeholder() -> None:
    table = players_table(
        _predictions(),
        players=_players(),
        teams=_teams(),
        status=_status(),
        as_of_gw=13,
        horizon=1,
    )
    orphan = table.filter(pl.col("player_key") == 999).to_dicts()[0]
    assert orphan["web_name"] == "Unknown"
    assert orphan["team"] == "?"


def test_players_table_my_squad_only() -> None:
    table = players_table(
        _predictions(),
        players=_players(),
        teams=_teams(),
        status=_status(),
        as_of_gw=13,
        horizon=1,
        squad_keys=[100],
        my_squad_only=True,
    )
    assert table["player_key"].to_list() == [100]


def test_plan_panel_and_squad_keys() -> None:
    players = pl.concat(
        [
            _players(),
            pl.DataFrame(
                {
                    "season": [_SEASON, _SEASON],
                    "player_key": [200, 300],
                    "element_id": [20, 30],
                    "team_key": [7, 3],
                    "position": ["FWD", "DEF"],
                    "web_name": ["Richarlison", "Saliba"],
                    "available_at": [_TS, _TS],
                    "source": ["test", "test"],
                }
            ),
        ]
    )
    panel = plan_panel(
        _plan(),
        players=players,
        teams=_teams(),
        status=_status(),
        chip="wildcard",
    )
    assert panel.hit == 4.0
    assert panel.chip == "wildcard"
    assert panel.captain_key == 100
    assert panel.vice_key == 200
    assert panel.transfers.height == 1
    assert panel.transfers["delta_xpts"][0] == pytest.approx(3.0)  # 5-2
    assert set(panel.xi["player_key"].to_list()) == {100, 200}
    assert panel.bench["player_key"].to_list() == []  # sold DEF not on bench
    assert set(squad_keys_from_plan(_plan())) == {100, 200}

    # Chip comes from PlanSchema.chip when not passed (Free Hit / WC plans).
    freehit = _plan().with_columns(pl.lit("freehit").alias("chip"))
    from_frame = plan_panel(freehit, players=players, teams=_teams(), status=_status())
    assert from_frame.chip == "freehit"
    assert from_frame.transfers.height == 1
    legacy = plan_panel(_plan(), players=players, teams=_teams(), status=_status())
    assert legacy.chip is None


def test_drilldown_with_and_without_components() -> None:
    pred = _predictions()
    bare = player_drilldown(
        100,
        pred,
        players=_players(),
        fixtures=_fixtures(),
        teams=_teams(),
        season=_SEASON,
        as_of_gw=13,
    )
    assert bare.has_components is False
    assert bare.fixtures.height >= 1
    assert bare.fixtures["xmins"].null_count() == bare.fixtures.height

    rich = player_drilldown(
        100,
        pred,
        players=_players(),
        fixtures=_fixtures(),
        teams=_teams(),
        season=_SEASON,
        minutes=_component_minutes(),
        scoreline=_component_scoreline(),
        involvement=_component_involvement(),
        as_of_gw=13,
    )
    assert rich.has_components is True
    gw13 = rich.fixtures.filter(pl.col("target_gw") == 13)
    assert gw13.height == 2
    assert float(gw13.filter(pl.col("fixture_id") == 1)["xmins"][0]) == 80.0
    assert float(gw13.filter(pl.col("fixture_id") == 1)["exp_goals"][0]) == 0.54
    assert gw13.filter(pl.col("fixture_id") == 1)["opponent"][0] == "TOT"


def test_points_split_sums_to_xpts() -> None:
    drill = player_drilldown(
        100,
        _predictions(),
        players=_players(),
        fixtures=_fixtures(),
        teams=_teams(),
        season=_SEASON,
        as_of_gw=13,
    )
    assert not drill.points_split.is_empty()
    for row in drill.points_split.to_dicts():
        parts = [
            row["appearance_xpts"],
            row["goals_xpts"],
            row["assists_xpts"],
            row["cs_xpts"],
            row["residual_xpts"],
        ]
        if any(p is None for p in parts):
            continue
        assert sum(float(p) for p in parts) == pytest.approx(float(row["xpts"]))


def test_default_selection_falls_back_without_t90m() -> None:
    from fplt.analysis.overview import official_pointers_frame

    pointer = OfficialPointer(
        predict_run_id=_RUN,
        plan_run_id="r_plan",
        role="candidate",
        squad_mode="virtual",
        written_at=_TS,
    )
    pointers = official_pointers_frame([(_SEASON, 10, "T-24h", "v4_h5", pointer)])
    sel = default_official_selection(pointers, applied_line=None)
    assert sel is not None
    assert sel.checkpoint == "T-24h"
    assert sel.line == "v4_h5"


def test_plan_panel_empty_and_no_step0() -> None:
    empty = plan_panel(
        pl.DataFrame(),
        players=_players(),
        teams=_teams(),
        status=_status(),
    )
    assert empty.transfers.is_empty()
    assert empty.expected_points == 0.0
    no_step = _plan().with_columns(pl.lit(1).alias("step"))
    panel = plan_panel(no_step, players=_players(), teams=_teams(), status=_status())
    assert panel.xi.is_empty()
    assert squad_keys_from_plan(pl.DataFrame()) == []


def test_players_table_filters_and_empty() -> None:
    with pytest.raises(ValueError, match="horizon"):
        players_table(
            _predictions(),
            players=_players(),
            teams=_teams(),
            status=_status(),
            as_of_gw=13,
            horizon=0,
        )
    empty = players_table(
        pl.DataFrame(
            schema={
                "player_key": pl.Int64(),
                "target_gw": pl.Int64(),
                "xpts": pl.Float64(),
                "p_played": pl.Float64(),
            }
        ),
        players=_players(),
        teams=_teams(),
        status=_status(),
        as_of_gw=13,
        horizon=2,
    )
    assert empty.is_empty()
    assert "xpts_gw13" in empty.columns
    filtered = players_table(
        _predictions(),
        players=_players(),
        teams=_teams(),
        status=_status(),
        as_of_gw=13,
        horizon=1,
        positions=["MID"],
        team_keys=[3],
        price_min=100,
        price_max=110,
        min_p_played=0.5,
    )
    assert filtered["player_key"].to_list() == [100]


def test_drilldown_pmf_and_missing_player_meta() -> None:
    from fplt.analysis.this_week import header_fields

    pmf = pl.DataFrame(
        {
            "player_key": [100, 100],
            "target_gw": [13, 13],
            "points_bin": [2, 8],
            "probability": [0.4, 0.6],
        }
    )
    drill = player_drilldown(
        100,
        _predictions(),
        players=pl.DataFrame(),
        fixtures=pl.DataFrame(),
        teams=pl.DataFrame(),
        season=_SEASON,
        points_pmf_gw=pmf,
    )
    assert drill.points_pmf.height == 2
    assert drill.has_components is False
    # Unknown fixture id → null opponent; scoreline still optional.
    pred = _predictions().filter(pl.col("player_key") == 100).head(1)
    drill2 = player_drilldown(
        100,
        pred,
        players=_players(),
        fixtures=_fixtures().head(0),
        teams=_teams(),
        season=_SEASON,
        minutes=_component_minutes(),
        scoreline=_component_scoreline(),
        involvement=_component_involvement(),
        as_of_gw=13,
    )
    assert drill2.fixtures["opponent"].null_count() == 1
    sel = default_official_selection(_pointers_frame())
    assert sel is not None
    header = header_fields(selection=sel, predict=None, plan=None, deadline_utc=_TS)
    assert header["predict_run_id"] == sel.predict_run_id
    assert header["code_sha"] is None


def test_player_labels_without_teams_or_status() -> None:
    table = players_table(
        _predictions(),
        players=_players(),
        teams=pl.DataFrame(),
        status=pl.DataFrame(),
        as_of_gw=13,
        horizon=1,
    )
    assert table.filter(pl.col("player_key") == 100)["web_name"][0] == "Saka"
    assert table.filter(pl.col("player_key") == 100)["price"][0] is None


def _versioned(
    frame: pl.DataFrame, *, seasons: list[str], captures: int
) -> pl.DataFrame:
    """Repeat each row per season x capture, like PIT-versioned curated tables."""
    parts: list[pl.DataFrame] = []
    for season in seasons:
        for i in range(captures):
            parts.append(
                frame.with_columns(
                    pl.lit(season).alias("season"),
                    (pl.col("available_at") + timedelta(minutes=15 * i)).alias(
                        "available_at"
                    ),
                )
            )
    return pl.concat(parts)


def _all_players() -> pl.DataFrame:
    return pl.concat(
        [
            _players(),
            pl.DataFrame(
                {
                    "season": [_SEASON, _SEASON],
                    "player_key": [200, 300],
                    "element_id": [20, 30],
                    "team_key": [7, 3],
                    "position": ["FWD", "DEF"],
                    "web_name": ["Richarlison", "Saliba"],
                    "available_at": [_TS, _TS],
                    "source": ["test", "test"],
                }
            ),
        ]
    )


def test_plan_panel_not_multiplied_by_versioned_curated_rows() -> None:
    """Regression #379: one row per capture per season fanned out the XI.

    The live preview showed repeated players and expected points around
    60,000 because labels joined raw PIT-versioned players/teams.
    """
    seasons = ["2024-25", "2025-26", _SEASON]
    players = _versioned(_all_players(), seasons=seasons, captures=50)
    teams = _versioned(_teams(), seasons=seasons, captures=50)
    status = _versioned(_status(), seasons=seasons, captures=50)
    clean = plan_panel(
        _plan(), players=_all_players(), teams=_teams(), status=_status()
    )
    panel = plan_panel(
        _plan(), players=players, teams=teams, status=status, season=_SEASON
    )
    assert panel.xi["player_key"].to_list() == clean.xi["player_key"].to_list()
    assert panel.xi["player_key"].is_unique().all()
    # 7 + 5 + captain 7 - hit 4
    assert panel.expected_points == pytest.approx(15.0)
    assert panel.expected_points == pytest.approx(clean.expected_points)
    assert panel.transfers.height == 1
    # Default (no season) still collapses to the latest row per player.
    default = plan_panel(_plan(), players=players, teams=teams, status=status)
    assert default.expected_points == pytest.approx(15.0)


def test_players_table_one_row_per_player_with_versioned_curated() -> None:
    players = _versioned(_players(), seasons=["2025-26", _SEASON], captures=20)
    teams = _versioned(_teams(), seasons=["2025-26", _SEASON], captures=20)
    status = _versioned(_status(), seasons=["2025-26", _SEASON], captures=20)
    base = players_table(
        _predictions(),
        players=_players(),
        teams=_teams(),
        status=_status(),
        as_of_gw=13,
        horizon=3,
    )
    table = players_table(
        _predictions(),
        players=players,
        teams=teams,
        status=status,
        as_of_gw=13,
        horizon=3,
        season=_SEASON,
    )
    assert table["player_key"].is_unique().all()
    assert table.height == base.height
    assert table["total_xpts"].to_list() == pytest.approx(base["total_xpts"].to_list())


def test_latest_rows_takes_newest_capture_and_respects_as_of() -> None:
    later = _TS + timedelta(hours=2)
    players = pl.concat(
        [
            _players(),
            _players().with_columns(
                pl.lit(7, dtype=pl.Int64).alias("team_key"),
                pl.lit(later).alias("available_at"),
            ),
            _players().with_columns(pl.lit("2025-26").alias("season")),
        ]
    )
    latest = latest_rows(players, "player_key", season=_SEASON)
    assert latest.height == 1
    assert latest["team_key"][0] == 7
    before = latest_rows(
        players, "player_key", season=_SEASON, as_of=_TS + timedelta(hours=1)
    )
    assert before["team_key"][0] == 3
    # Without a season the newest season's newest row wins.
    assert latest_rows(players, "player_key")["team_key"][0] == 7
    # Empty / missing-key frames pass through unchanged.
    empty = pl.DataFrame(schema={"player_key": pl.Int64()})
    assert latest_rows(empty, "player_key").is_empty()
    assert latest_rows(pl.DataFrame({"x": [1]}), "player_key").height == 1


def test_drilldown_uses_latest_team_for_opponent() -> None:
    """A mid-season transfer: drill-down must use the player's latest club."""
    later = _TS + timedelta(hours=2)
    players = pl.concat(
        [
            _players(),
            _players().with_columns(
                pl.lit(7, dtype=pl.Int64).alias("team_key"),
                pl.lit(later).alias("available_at"),
            ),
        ]
    )
    drill = player_drilldown(
        100,
        _predictions(),
        players=players,
        fixtures=_fixtures(),
        teams=_versioned(_teams(), seasons=[_SEASON], captures=5),
        season=_SEASON,
    )
    first = drill.fixtures.filter(pl.col("fixture_id") == 1)
    # Fixture 1 is team 3 (home) v 7: as a team-7 player he's away to ARS.
    assert first["opponent"][0] == "ARS"
    assert first["is_home"][0] is False
