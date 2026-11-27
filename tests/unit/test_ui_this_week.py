"""AppTest smoke: This-week page renders with and without component tables."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.manifests import (
    AppliedRecord,
    AsOfDeadline,
    BuildManifest,
    BuildTableEntry,
    OfficialPointer,
)
from fplt.settings import get_settings
from fplt.store.builds import STAGING_VERSION, write_build_manifest, write_table
from fplt.store.official import write_applied_record, write_official_pointer
from fplt.store.runs import RunWriter

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_TS = datetime(2026, 11, 27, 11, 30, tzinfo=UTC)
_SEASON = "2026-27"
_APP = Path(__file__).resolve().parents[2] / "src" / "fplt" / "ui" / "app.py"
_PRED = "r_20261127T120000Z_predtw"
_PLAN = "r_20261127T120000Z_plantw"
_BUILD = "cafe" * 8


def _status_row(player_key: int, price: int) -> dict[str, object]:
    return {
        "season": _SEASON,
        "player_key": player_key,
        "status": "a",
        "news": "",
        "news_added": None,
        "price": price,
        "ep_next": 5.0,
        "chance_of_playing": 100.0,
        "chance_of_playing_this_round": 100.0,
        "penalties_order": None,
        "corners_and_indirect_freekicks_order": None,
        "direct_freekicks_order": None,
        "available_at": _TS,
        "source": "test",
    }


def _seed(
    data_dir: Path,
    *,
    with_components: bool,
) -> None:
    players = pl.DataFrame(
        {
            "season": [_SEASON, _SEASON],
            "player_key": [100, 200],
            "element_id": [10, 20],
            "team_key": [3, 7],
            "position": ["MID", "FWD"],
            "web_name": ["Saka", "Richarlison"],
            "available_at": [_TS, _TS],
            "source": ["test", "test"],
        }
    )
    teams = pl.DataFrame(
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
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [1],
            "gw": [13],
            "team_h_key": [3],
            "team_a_key": [7],
            "kickoff_time": [_TS],
            "team_h_score": [None],
            "team_a_score": [None],
            "finished": [False],
            "available_at": [_TS],
            "source": ["test"],
        }
    )
    gameweeks = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [13],
            "deadline_time": [_TS],
            "finished": [False],
            "data_checked": [False],
            "available_at": [_TS],
            "source": ["test"],
        }
    )
    status = pl.DataFrame([_status_row(100, 105), _status_row(200, 70)])
    entries = {
        "players": write_table(data_dir, "players", players),
        "teams": write_table(data_dir, "teams", teams),
        "fixtures": write_table(data_dir, "fixtures", fixtures),
        "gameweeks": write_table(data_dir, "gameweeks", gameweeks),
        "player_status": write_table(data_dir, "player_status", status),
    }
    write_build_manifest(
        data_dir,
        BuildManifest(
            build_id=_BUILD,
            created_at=_TS,
            staging_version=STAGING_VERSION,
            code_sha="abc1234",
            raw_manifest_hash="d" * 64,
            raw_cutoff=_TS,
            tables={
                name: BuildTableEntry(
                    sha=entry.sha, rows=entry.rows, schema_hash=entry.schema_hash
                )
                for name, entry in entries.items()
            },
        ),
    )
    predictions = pl.DataFrame(
        {
            "model_id": ["deadbeef", "deadbeef"],
            "as_of_season": [_SEASON, _SEASON],
            "as_of_gw": [13, 13],
            "target_season": [_SEASON, _SEASON],
            "target_gw": [13, 13],
            "player_key": [100, 200],
            "fixture_id": [1, 1],
            "xpts": [6.0, 4.0],
            "xpts_sd": [1.0, 1.0],
            "p_played": [0.9, 0.8],
            "p_60": [0.7, 0.6],
            "p_goal": [0.3, 0.25],
            "p_assist": [0.2, 0.15],
            "p_cs": [0.4, 0.1],
            "p_haul": [0.1, 0.08],
        }
    )
    deadline = AsOfDeadline(season=_SEASON, gw=13, deadline_utc=_TS)
    with RunWriter(
        data_dir,
        kind="official",
        mode="live",
        run_type="predict",
        spec_hash="a" * 64,
        spec_label="dc_layered_v4_native",
        model_id="deadbeef",
        as_of=[deadline],
        horizon=3,
        code_sha="abc1234",
        lockfile_hash="b" * 64,
        host="test",
        started_at=_TS,
        run_id=_PRED,
        checkpoint="T-90m",
        build_id=_BUILD,
        auto_build_id=False,
    ) as writer:
        writer.write_parquet("predictions/player_fixture.parquet", predictions)
        if with_components:
            writer.write_parquet(
                "predictions/components/minutes.parquet",
                pl.DataFrame(
                    {
                        "run_id": [_PRED, _PRED],
                        "as_of_season": [_SEASON, _SEASON],
                        "as_of_gw": [13, 13],
                        "deadline_utc": [_TS, _TS],
                        "target_season": [_SEASON, _SEASON],
                        "target_gw": [13, 13],
                        "horizon": [0, 0],
                        "fixture_id": [1, 1],
                        "player_key": [100, 200],
                        "team_key": [3, 7],
                        "schema_version": [SCHEMA_VERSION, SCHEMA_VERSION],
                        "xmins": [80.0, 60.0],
                        "p_played": [0.9, 0.8],
                        "p_60": [0.7, 0.6],
                    }
                ),
            )
            writer.write_parquet(
                "predictions/components/scoreline.parquet",
                pl.DataFrame(
                    {
                        "run_id": [_PRED, _PRED],
                        "as_of_season": [_SEASON, _SEASON],
                        "as_of_gw": [13, 13],
                        "deadline_utc": [_TS, _TS],
                        "target_season": [_SEASON, _SEASON],
                        "target_gw": [13, 13],
                        "horizon": [0, 0],
                        "fixture_id": [1, 1],
                        "team_key": [3, 7],
                        "schema_version": [SCHEMA_VERSION, SCHEMA_VERSION],
                        "opponent_team_key": [7, 3],
                        "is_home": [True, False],
                        "team_eg": [1.8, 0.9],
                        "opp_eg": [0.9, 1.8],
                        "p_cs": [0.45, 0.2],
                    }
                ),
            )
            writer.write_parquet(
                "predictions/components/involvement.parquet",
                pl.DataFrame(
                    {
                        "run_id": [_PRED, _PRED],
                        "as_of_season": [_SEASON, _SEASON],
                        "as_of_gw": [13, 13],
                        "deadline_utc": [_TS, _TS],
                        "target_season": [_SEASON, _SEASON],
                        "target_gw": [13, 13],
                        "horizon": [0, 0],
                        "fixture_id": [1, 1],
                        "player_key": [100, 200],
                        "team_key": [3, 7],
                        "schema_version": [SCHEMA_VERSION, SCHEMA_VERSION],
                        "goal_weight": [0.3, 0.4],
                        "assist_weight": [0.2, 0.15],
                        "exp_goals": [0.54, 0.36],
                        "exp_assists": [0.36, 0.14],
                        "exp_goals_method": ["team_eg_x_weight", "team_eg_x_weight"],
                    }
                ),
            )
            writer.write_parquet(
                "predictions/points_pmf_gw.parquet",
                pl.DataFrame(
                    {
                        "run_id": [_PRED, _PRED],
                        "as_of_season": [_SEASON, _SEASON],
                        "as_of_gw": [13, 13],
                        "deadline_utc": [_TS, _TS],
                        "target_season": [_SEASON, _SEASON],
                        "target_gw": [13, 13],
                        "horizon": [0, 0],
                        "player_key": [100, 100],
                        "team_key": [3, 3],
                        "schema_version": [SCHEMA_VERSION, SCHEMA_VERSION],
                        "points_bin": [2, 6],
                        "probability": [0.4, 0.6],
                    }
                ),
            )
    plan = pl.DataFrame(
        {
            "model_id": ["deadbeef", "deadbeef"],
            "as_of_season": [_SEASON, _SEASON],
            "as_of_gw": [13, 13],
            "target_gw": [13, 13],
            "step": [0, 0],
            "player_key": [100, 200],
            "position": ["MID", "FWD"],
            "team_key": [3, 7],
            "in_xi": [True, True],
            "captain": [True, False],
            "vice": [False, True],
            "bought": [False, False],
            "sold": [False, False],
            "xpts": [6.0, 4.0],
            "hit": [0.0, 0.0],
        }
    )
    with RunWriter(
        data_dir,
        kind="official",
        mode="live",
        run_type="plan",
        spec_hash="a" * 64,
        spec_label="dc_layered_v4_native",
        model_id="deadbeef",
        as_of=[deadline],
        horizon=3,
        plan_horizon=3,
        code_sha="abc1234",
        lockfile_hash="b" * 64,
        host="test",
        started_at=_TS,
        run_id=_PLAN,
        checkpoint="T-90m",
        line="v4_h3",
        squad_mode="real",
        auto_build_id=False,
    ) as writer:
        writer.write_parquet("plans/plan.parquet", plan)
    write_official_pointer(
        data_dir,
        season=_SEASON,
        gw=13,
        line="v4_h3",
        checkpoint="T-90m",
        pointer=OfficialPointer(
            predict_run_id=_PRED,
            plan_run_id=_PLAN,
            role="primary",
            squad_mode="real",
            written_at=_TS,
        ),
    )
    write_applied_record(
        data_dir,
        season=_SEASON,
        gw=13,
        checkpoint="T-90m",
        record=AppliedRecord(line="v4_h3", plan_run_id=_PLAN, submitted_at=_TS),
    )


@pytest.mark.parametrize("with_components", [True, False])
def test_this_week_app_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_components: bool
) -> None:
    _seed(tmp_path, with_components=with_components)
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        app = AppTest.from_file(str(_APP), default_timeout=15)
        app.run()
        assert not app.exception, app.exception
        titles = [str(item.value).lower() for item in app.title]
        assert any("this week" in value for value in titles)
        assert len(app.dataframe) >= 1
    finally:
        get_settings.cache_clear()
