"""Season-rollover preflight (ops-07) on a synthetic new-season fixture."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from fplt.cli import main
from fplt.data.store import DataStore
from fplt.ops.preflight import (
    PreflightConfig,
    check_scoring_rules,
    check_team_maps,
    format_preflight_report,
    run_preflight,
)

_SEASON = "2099-00"
_TS = datetime(2099, 8, 1, tzinfo=UTC)
_GW1 = datetime(2099, 8, 15, 17, 30, tzinfo=UTC)


def _cfg(**overrides: object) -> PreflightConfig:
    base: dict[str, object] = {
        "expected_team_count": 2,
        "max_unmapped_share": 0.1,
        "top_unmapped": 5,
    }
    base.update(overrides)
    return PreflightConfig.model_validate(base)


def _write_maps(
    tmp_path: Path,
    *,
    understat_teams: dict[int, int],
    api_teams: dict[int, int],
    fd_teams: dict[str, int],
) -> tuple[Path, Path, Path]:
    understat = tmp_path / "understat_maps.yaml"
    api = tmp_path / "api_maps.yaml"
    fd = tmp_path / "fd_maps.yaml"
    understat.write_text(
        yaml.safe_dump({"teams": understat_teams, "players": {}}),
        encoding="utf-8",
    )
    api.write_text(
        yaml.safe_dump({"teams": api_teams, "players": {}}),
        encoding="utf-8",
    )
    fd.write_text(yaml.safe_dump({"teams": fd_teams}), encoding="utf-8")
    return understat, api, fd


def _season_frames(
    *,
    promoted_key: int = 91,
    include_unmapped: bool = True,
) -> dict[str, pl.DataFrame]:
    teams = pl.DataFrame(
        {
            "season": [_SEASON, _SEASON],
            "team_key": [3, promoted_key],
            "team_id": [1, 2],
            "name": ["Arsenal", "Promoted FC"],
            "short_name": ["ARS", "PRO"],
            "available_at": [_TS, _TS],
            "source": ["fpl_api", "fpl_api"],
        }
    )
    players = pl.DataFrame(
        {
            "season": [_SEASON, _SEASON],
            "player_key": [100, 200],
            "element_id": [1, 2],
            "team_key": [3, promoted_key],
            "position": ["MID", "FWD"],
            "web_name": ["A", "B"],
            "available_at": [_TS, _TS],
            "source": ["fpl_api", "fpl_api"],
        }
    )
    gameweeks = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [1],
            "deadline_time": [_GW1],
            "finished": [False],
            "data_checked": [False],
            "available_at": [_TS],
            "source": ["fpl_api"],
        }
    )
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [1],
            "gw": [1],
            "team_h_key": [3],
            "team_a_key": [promoted_key],
            "kickoff_time": [_GW1],
            "team_h_score": [None],
            "team_a_score": [None],
            "finished": [False],
            "available_at": [_TS],
            "source": ["fpl_api"],
        }
    ).with_columns(
        pl.col("team_h_score").cast(pl.Int64),
        pl.col("team_a_score").cast(pl.Int64),
    )
    # 1 mapped + 1 unmapped understat row → 50% unmapped when include_unmapped
    understat_rows = [
        {
            "season": _SEASON,
            "match_id": 1,
            "understat_player_id": 501,
            "player_key": 100,
            "team_key": 3,
            "xg": 0.1,
            "xa": 0.0,
            "shots": 1,
            "key_passes": 0,
            "xg_chain": 0.1,
            "xg_buildup": 0.0,
            "minutes": 90,
            "available_at": _TS,
            "source": "understat",
        }
    ]
    if include_unmapped:
        understat_rows.append(
            {
                "season": _SEASON,
                "match_id": 1,
                "understat_player_id": 999,
                "player_key": None,
                "team_key": promoted_key,
                "xg": 0.2,
                "xa": 0.1,
                "shots": 2,
                "key_passes": 1,
                "xg_chain": 0.3,
                "xg_buildup": 0.1,
                "minutes": 80,
                "available_at": _TS,
                "source": "understat",
            }
        )
    understat = pl.DataFrame(understat_rows).with_columns(
        pl.col("player_key").cast(pl.Int64)
    )
    injuries = pl.DataFrame(
        {
            "season": [_SEASON, _SEASON],
            "player_key": [100, None],
            "api_player_id": [10, 11],
            "team_key": [3, promoted_key],
            "api_team_id": [42, 35],
            "api_fixture_id": [1, 2],
            "fixture_id": [None, None],
            "injury_type": ["Injury", "Injury"],
            "reason": ["Knee", "Ankle"],
            "category": ["out", "out"],
            "available_at": [_TS, _TS],
            "source": ["api_football", "api_football"],
        }
    ).with_columns(
        pl.col("player_key").cast(pl.Int64),
        pl.col("fixture_id").cast(pl.Int64),
        pl.col("team_key").cast(pl.Int64),
    )
    return {
        "teams": teams,
        "players": players,
        "gameweeks": gameweeks,
        "fixtures": fixtures,
        "understat_player_matches": understat,
        "player_injuries": injuries,
    }


def _chip_yaml() -> str:
    return yaml.safe_dump(
        {
            "one_chip_per_gameweek": True,
            "free_hit_keeps_free_transfers": True,
            "windows": [
                {
                    "id": "all",
                    "start_event": 1,
                    "stop_event": 38,
                    "carries_over": False,
                }
            ],
            "grants": [
                {"name": "wildcard", "window": "all"},
                {"name": "freehit", "window": "all"},
                {"name": "bboost", "window": "all"},
                {"name": "3xc", "window": "all"},
            ],
        }
    )


def _scoring_yaml() -> str:
    sample = Path("configs/scoring/2026-27.yaml").read_text(encoding="utf-8")
    return sample


def test_missing_scoring_file_fails(tmp_path: Path) -> None:
    scoring = tmp_path / "scoring"
    scoring.mkdir()
    result = check_scoring_rules(_SEASON, scoring_dir=scoring)
    assert result.ok is False
    assert "no scoring rules" in result.detail


def test_promoted_team_missing_from_maps_fails(tmp_path: Path) -> None:
    frames = _season_frames(promoted_key=91)
    understat, api, fd = _write_maps(
        tmp_path,
        understat_teams={83: 3},  # Arsenal only — promoted missing
        api_teams={42: 3},
        fd_teams={"Arsenal": 3},
    )
    result = check_team_maps(
        _SEASON,
        frames["teams"],
        config=_cfg(),
        understat_maps=understat,
        api_football_maps=api,
        football_data_maps=fd,
    )
    assert result.ok is False
    assert "Promoted FC" in result.detail


def test_preflight_reports_unmapped_and_passes_when_mapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = _season_frames(include_unmapped=True)
    scoring = tmp_path / "scoring"
    chips = tmp_path / "chips"
    scoring.mkdir()
    chips.mkdir()
    (scoring / f"{_SEASON}.yaml").write_text(_scoring_yaml(), encoding="utf-8")
    (chips / f"{_SEASON}.yaml").write_text(_chip_yaml(), encoding="utf-8")

    understat, api, fd = _write_maps(
        tmp_path,
        understat_teams={83: 3, 35: 91},
        api_teams={42: 3, 35: 91},
        fd_teams={"Arsenal": 3, "Promoted FC": 91},
    )

    # Unmapped share 0.5 > 0.1 → fail
    report = run_preflight(
        _SEASON,
        config=_cfg(max_unmapped_share=0.1),
        scoring_dir=scoring,
        chips_dir=chips,
        understat_maps=understat,
        api_football_maps=api,
        football_data_maps=fd,
        teams=frames["teams"],
        gameweeks=frames["gameweeks"],
        players=frames["players"],
        fixtures=frames["fixtures"],
        understat_player_matches=frames["understat_player_matches"],
        player_injuries=frames["player_injuries"],
        skip_plan=True,
    )
    assert report.ok is False
    unmapped = next(c for c in report.checks if c.name == "unmapped_players")
    assert unmapped.ok is False
    assert "understat" in unmapped.detail
    assert "999" in unmapped.detail

    # Raise threshold and drop unmapped → pass (still skip plan / models via stub)
    frames_ok = _season_frames(include_unmapped=False)
    # injuries still 50% unmapped — raise threshold
    report_ok = run_preflight(
        _SEASON,
        config=_cfg(max_unmapped_share=0.6),
        scoring_dir=scoring,
        chips_dir=chips,
        understat_maps=understat,
        api_football_maps=api,
        football_data_maps=fd,
        teams=frames_ok["teams"],
        gameweeks=frames_ok["gameweeks"],
        players=frames_ok["players"],
        fixtures=frames_ok["fixtures"],
        understat_player_matches=frames_ok["understat_player_matches"],
        player_injuries=frames_ok["player_injuries"],
        plan_probe=lambda: None,
    )
    assert report_ok.ok is True
    text = format_preflight_report(report_ok)
    assert "PASS" in text
    assert "GW1 deadline" in text


def test_dry_run_plan_probe_failure_surfaces() -> None:
    from fplt.ops.preflight import check_dry_run_plan

    frames = _season_frames(include_unmapped=False)

    def _boom() -> None:
        raise RuntimeError("solver missing")

    result = check_dry_run_plan(
        _SEASON,
        gameweeks=frames["gameweeks"],
        players=frames["players"],
        fixtures=frames["fixtures"],
        plan_probe=_boom,
    )
    assert result.ok is False
    assert "solver missing" in result.detail


def test_structural_dry_run_plan_ok() -> None:
    from fplt.ops.preflight import check_dry_run_plan

    frames = _season_frames(include_unmapped=False)
    result = check_dry_run_plan(
        _SEASON,
        gameweeks=frames["gameweeks"],
        players=frames["players"],
        fixtures=frames["fixtures"],
        plan_probe=lambda: None,
    )
    assert result.ok is True


def test_cli_preflight_exits_nonzero_on_missing_season(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    store = DataStore(tmp_path)
    # Empty store — preflight fails.
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "preflight",
                "--season",
                "2099-00",
                "--data-dir",
                str(tmp_path),
                "--skip-plan",
            ]
        )
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "scoring_rules" in out
    del store


def test_preflight_report_failures_property() -> None:
    from fplt.ops.preflight import CheckResult, PreflightReport

    report = PreflightReport(
        season=_SEASON,
        checks=(
            CheckResult("a", True, "ok"),
            CheckResult("b", False, "warn only", severity="warn"),
            CheckResult("c", False, "hard fail", severity="fail"),
        ),
    )
    assert report.ok is False
    assert [c.name for c in report.failures] == ["c"]


def test_load_preflight_config_errors(tmp_path: Path) -> None:
    from fplt.ops.preflight import load_preflight_config

    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="preflight config"):
        load_preflight_config(missing)
    bad = tmp_path / "bad.yaml"
    bad.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_preflight_config(bad)


def test_run_preflight_rejects_bad_season() -> None:
    with pytest.raises(ValueError, match="file name"):
        run_preflight("../evil", skip_plan=True)


def test_missing_chip_rules_fails(tmp_path: Path) -> None:
    from fplt.ops.preflight import check_chip_rules

    chips = tmp_path / "chips"
    chips.mkdir()
    result = check_chip_rules(_SEASON, chips_dir=chips)
    assert result.ok is False
    assert "no chip rules" in result.detail


def test_team_maps_empty_season_and_count_mismatch(tmp_path: Path) -> None:
    understat, api, fd = _write_maps(
        tmp_path,
        understat_teams={83: 3},
        api_teams={42: 3},
        fd_teams={"Arsenal": 3},
    )
    empty_season = pl.DataFrame(
        {
            "season": ["2098-99"],
            "team_key": [3],
            "team_id": [1],
            "name": ["Arsenal"],
            "short_name": ["ARS"],
            "available_at": [_TS],
            "source": ["fpl_api"],
        }
    )
    result = check_team_maps(
        _SEASON,
        empty_season,
        config=_cfg(),
        understat_maps=understat,
        api_football_maps=api,
        football_data_maps=fd,
    )
    assert result.ok is False
    assert "no staged teams" in result.detail

    frames = _season_frames()
    result_count = check_team_maps(
        _SEASON,
        frames["teams"],
        config=_cfg(expected_team_count=20),
        understat_maps=understat,
        api_football_maps=api,
        football_data_maps=fd,
    )
    assert result_count.ok is False
    assert "expected 20 teams" in result_count.detail


def test_unmapped_players_no_rows_for_season() -> None:
    from fplt.ops.preflight import check_unmapped_players

    other = pl.DataFrame(
        {
            "season": ["2098-99"],
            "match_id": [1],
            "understat_player_id": [1],
            "player_key": [None],
            "team_key": [3],
            "xg": [0.1],
            "xa": [0.0],
            "shots": [1],
            "key_passes": [0],
            "xg_chain": [0.1],
            "xg_buildup": [0.0],
            "minutes": [90],
            "available_at": [_TS],
            "source": ["understat"],
        }
    ).with_columns(pl.col("player_key").cast(pl.Int64))
    result = check_unmapped_players(
        _SEASON,
        understat_player_matches=other,
        player_injuries=None,
        player_status=None,
        config=_cfg(),
    )
    assert "no rows for season" in result.detail


def test_gw1_deadline_error_paths() -> None:
    from datetime import datetime

    from fplt.ops.preflight import check_gw1_deadline

    no_gw1 = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [2],
            "deadline_time": [_GW1],
            "finished": [False],
            "data_checked": [False],
            "available_at": [_TS],
            "source": ["fpl_api"],
        }
    )
    assert check_gw1_deadline(_SEASON, no_gw1).ok is False

    not_dt = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [1],
            "deadline_time": ["2099-08-15"],
            "finished": [False],
            "data_checked": [False],
            "available_at": [_TS],
            "source": ["fpl_api"],
        }
    )
    assert "not a datetime" in check_gw1_deadline(_SEASON, not_dt).detail

    naive = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [1],
            "deadline_time": [datetime(2099, 8, 15, 17, 30)],
            "finished": [False],
            "data_checked": [False],
            "available_at": [_TS],
            "source": ["fpl_api"],
        }
    )
    assert "timezone" in check_gw1_deadline(_SEASON, naive).detail


def test_model_specs_and_structural_dry_run_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.ops.preflight import check_dry_run_plan, check_model_specs

    monkeypatch.setattr(
        "fplt.ops.preflight.load_production_ops",
        lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("no ops")),
    )
    assert check_model_specs(_SEASON).ok is False

    monkeypatch.setattr(
        "fplt.ops.preflight.load_production_ops",
        lambda *_a, **_k: type("Ops", (), {"models": ["missing_model"]})(),
    )
    monkeypatch.setattr("fplt.ops.preflight.load_model_specs", lambda *_a, **_k: {})
    missing = check_model_specs(_SEASON)
    assert missing.ok is False
    assert "missing specs" in missing.detail

    frames = _season_frames(include_unmapped=False)
    # Structural path: specs fail → dry_run fails.
    bad_specs = check_dry_run_plan(
        _SEASON,
        gameweeks=frames["gameweeks"],
        players=frames["players"],
        fixtures=frames["fixtures"],
        plan_probe=None,
    )
    assert bad_specs.ok is False

    monkeypatch.setattr(
        "fplt.ops.preflight.check_model_specs",
        lambda *_a, **_k: type(
            "C", (), {"ok": True, "detail": "ok", "name": "model_specs"}
        )(),
    )
    no_players = check_dry_run_plan(
        _SEASON,
        gameweeks=frames["gameweeks"],
        players=None,
        fixtures=frames["fixtures"],
        plan_probe=None,
    )
    assert "no staged players" in no_players.detail

    no_fixtures = check_dry_run_plan(
        _SEASON,
        gameweeks=frames["gameweeks"],
        players=frames["players"],
        fixtures=None,
        plan_probe=None,
    )
    assert "no staged fixtures" in no_fixtures.detail

    ok = check_dry_run_plan(
        _SEASON,
        gameweeks=frames["gameweeks"],
        players=frames["players"],
        fixtures=frames["fixtures"],
        plan_probe=None,
    )
    assert ok.ok is True
    assert "structural dry-run ok" in ok.detail


def test_table_helpers_and_build_fpl_targets(tmp_path: Path) -> None:
    from fplt.ops.preflight import (
        _optional_status,
        _table,
        build_fpl_targets,
    )

    assert _table(None, "teams", None) is None
    override = pl.DataFrame({"x": [1]})
    assert _table(None, "teams", override) is override
    assert _optional_status(None) is None

    store = DataStore(tmp_path)
    assert _table(store, "teams", None) is None
    assert _optional_status(store) is None

    frames = _season_frames(include_unmapped=False)
    targets = build_fpl_targets(
        frames["players"],
        season=_SEASON,
        bootstrap_elements=[
            {
                "code": 100,
                "first_name": "Ada",
                "second_name": "Lovelace",
                "web_name": "A",
                "birth_date": "1990-01-02",
            },
            {
                "code": 200,
                "web_name": "B",
                "dob": "",
            },
        ],
    )
    assert len(targets) == 2
    by_key = {t.player_key: t for t in targets}
    assert by_key[100].full_name == "Ada Lovelace"
    assert by_key[100].dob is not None
    assert by_key[200].full_name == "B"
