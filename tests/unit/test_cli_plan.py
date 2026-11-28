"""CLI plan dispatch: --from-api, FPLT_ENTRY_ID default, --entry-json wins."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fplt.cli import main
from fplt.optimise.decisions import DEFAULT_LIVE_CHIP_ARM
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON
from fplt.optimise.public_squad import ResolvedSquad
from fplt.optimise.state import SquadState
from fplt.settings import get_settings


def _resolved(squad: SquadState, source: str = "my_team") -> ResolvedSquad:
    return ResolvedSquad(state=squad, source=source)  # type: ignore[arg-type]


def test_plan_from_api_resolves_squad_then_runs_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FPLT_ENTRY_ID", "42")
    get_settings.cache_clear()
    view = object()
    squad = SquadState.model_validate(
        {
            "bank": 0,
            "free_transfers": 1,
            "ft_bank_cap": 2,
            "hit_cost": 4,
            "decay": 1.0,
            "bench_weight": 0.0,
            "owned": [],
        }
    )
    resolve = MagicMock(return_value=_resolved(squad))
    run = MagicMock(
        return_value=(object(), tmp_path / "plan.parquet", "# ok\n", "r_test")
    )
    monkeypatch.setattr(
        "fplt.cli.PITData",
        lambda store: MagicMock(view=MagicMock(return_value=view)),
    )
    monkeypatch.setattr("fplt.cli.DataStore", lambda root: object())
    monkeypatch.setattr("fplt.cli.resolve_squad_state", resolve)
    monkeypatch.setattr("fplt.cli.run_plan", run)
    try:
        main(
            [
                "plan",
                "--model",
                "rolling_mean_v1",
                "--season",
                "2024-25",
                "--gw",
                "3",
                "--horizon",
                "1",
                "--from-api",
            ]
        )
    finally:
        get_settings.cache_clear()
    resolve.assert_called_once()
    kwargs = resolve.call_args.kwargs
    assert kwargs["entry_json"] is None
    assert kwargs["event"] == 3
    run.assert_called_once()
    assert run.call_args.kwargs["squad"] is squad
    assert run.call_args.kwargs["squad_source"] == "my_team"
    assert run.call_args.args[3] == 3


def test_plan_defaults_to_resolve_when_entry_id_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FPLT_ENTRY_ID", "99")
    get_settings.cache_clear()
    squad = SquadState.model_validate(
        {
            "bank": 10,
            "free_transfers": 1,
            "ft_bank_cap": 2,
            "hit_cost": 4,
            "decay": 1.0,
            "bench_weight": 0.0,
            "owned": [],
        }
    )
    resolve = MagicMock(return_value=_resolved(squad))
    run = MagicMock(return_value=(object(), tmp_path / "p.parquet", "r\n", "r_test"))
    monkeypatch.setattr(
        "fplt.cli.PITData",
        lambda store: MagicMock(view=MagicMock(return_value=object())),
    )
    monkeypatch.setattr("fplt.cli.DataStore", lambda root: object())
    monkeypatch.setattr("fplt.cli.resolve_squad_state", resolve)
    monkeypatch.setattr("fplt.cli.run_plan", run)
    try:
        main(
            [
                "plan",
                "--model",
                "rolling_mean_v1",
                "--season",
                "2024-25",
                "--gw",
                "1",
            ]
        )
    finally:
        get_settings.cache_clear()
    resolve.assert_called_once()
    assert run.call_args.kwargs["squad"] is squad
    assert run.call_args.kwargs["squad_source"] == "my_team"
    assert run.call_args.args[4] == DEFAULT_PLAN_HORIZON
    assert DEFAULT_PLAN_HORIZON == 3


def test_plan_entry_json_wins_over_from_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = tmp_path / "picks.json"
    entry.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FPLT_ENTRY_ID", "42")
    get_settings.cache_clear()
    squad = SquadState.model_validate(
        {
            "bank": 5,
            "free_transfers": 1,
            "ft_bank_cap": 2,
            "hit_cost": 4,
            "decay": 1.0,
            "bench_weight": 0.0,
            "owned": [],
        }
    )
    resolve = MagicMock(return_value=_resolved(squad, "entry_json"))
    run = MagicMock(return_value=(object(), tmp_path / "p.parquet", "r\n", "r_test"))
    monkeypatch.setattr(
        "fplt.cli.PITData",
        lambda store: MagicMock(view=MagicMock(return_value=object())),
    )
    monkeypatch.setattr("fplt.cli.DataStore", lambda root: object())
    monkeypatch.setattr("fplt.cli.resolve_squad_state", resolve)
    monkeypatch.setattr("fplt.cli.run_plan", run)
    try:
        main(
            [
                "plan",
                "--model",
                "rolling_mean_v1",
                "--season",
                "2024-25",
                "--gw",
                "2",
                "--from-api",
                "--entry-json",
                str(entry),
            ]
        )
    finally:
        get_settings.cache_clear()
    assert resolve.call_args.kwargs["entry_json"] == entry
    assert run.call_args.kwargs["squad"] is squad
    assert run.call_args.kwargs["squad_source"] == "entry_json"


def test_plan_from_api_without_entry_id_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("FPLT_ENTRY_ID", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit) as raised:
            main(
                [
                    "plan",
                    "--model",
                    "rolling_mean_v1",
                    "--season",
                    "2024-25",
                    "--gw",
                    "1",
                    "--from-api",
                ]
            )
    finally:
        get_settings.cache_clear()
    assert raised.value.code == 2
    assert "FPLT_ENTRY_ID" in capsys.readouterr().err


def test_plan_without_entry_skips_resolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("FPLT_ENTRY_ID", raising=False)
    get_settings.cache_clear()
    resolve = MagicMock()
    run = MagicMock(return_value=(object(), tmp_path / "p.parquet", "r\n", "r_test"))
    monkeypatch.setattr("fplt.cli.resolve_squad_state", resolve)
    monkeypatch.setattr("fplt.cli.run_plan", run)
    try:
        main(
            [
                "plan",
                "--model",
                "rolling_mean_v1",
                "--season",
                "2024-25",
                "--gw",
                "1",
            ]
        )
    finally:
        get_settings.cache_clear()
    resolve.assert_not_called()
    squad = run.call_args.kwargs["squad"]
    assert squad.owned == []
    assert squad.bank == 1000
    assert run.call_args.kwargs["chip_arm"] == DEFAULT_LIVE_CHIP_ARM
    assert DEFAULT_LIVE_CHIP_ARM == "optimised"


def test_plan_cli_passes_chip_arm_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("FPLT_ENTRY_ID", raising=False)
    get_settings.cache_clear()
    run = MagicMock(return_value=(object(), tmp_path / "p.parquet", "r\n", "r_test"))
    monkeypatch.setattr("fplt.cli.run_plan", run)
    try:
        main(
            [
                "plan",
                "--model",
                "rolling_mean_v1",
                "--season",
                "2024-25",
                "--gw",
                "1",
                "--chip-arm",
                "off",
            ]
        )
    finally:
        get_settings.cache_clear()
    assert run.call_args.kwargs["chip_arm"] == "off"
    assert run.call_args.kwargs["force_play"] is True


def test_plan_cli_passes_no_force_play(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("FPLT_ENTRY_ID", raising=False)
    get_settings.cache_clear()
    run = MagicMock(return_value=(object(), tmp_path / "p.parquet", "r\n", "r_test"))
    monkeypatch.setattr("fplt.cli.run_plan", run)
    try:
        main(
            [
                "plan",
                "--model",
                "rolling_mean_v1",
                "--season",
                "2024-25",
                "--gw",
                "1",
                "--no-force-play",
            ]
        )
    finally:
        get_settings.cache_clear()
    assert run.call_args.kwargs["force_play"] is False
    assert run.call_args.kwargs["chip_arm"] == DEFAULT_LIVE_CHIP_ARM


def test_plan_rebuild_clears_owned_and_folds_selling_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FPLT_ENTRY_ID", "42")
    get_settings.cache_clear()
    owned = SquadState.model_validate(
        {
            "bank": 21,
            "free_transfers": 2,
            "ft_bank_cap": 5,
            "hit_cost": 4,
            "decay": 0.9,
            "bench_weight": 0.0001,
            "owned": [
                {"player_key": 1, "selling_price": 100},
                {"player_key": 2, "selling_price": 55},
            ],
        }
    )
    resolve = MagicMock(return_value=_resolved(owned))
    run = MagicMock(return_value=(object(), tmp_path / "p.parquet", "r\n", "r_test"))
    monkeypatch.setattr(
        "fplt.cli.PITData",
        lambda store: MagicMock(view=MagicMock(return_value=object())),
    )
    monkeypatch.setattr("fplt.cli.DataStore", lambda root: object())
    monkeypatch.setattr("fplt.cli.resolve_squad_state", resolve)
    monkeypatch.setattr("fplt.cli.run_plan", run)
    try:
        main(
            [
                "plan",
                "--model",
                "rolling_mean_v1",
                "--season",
                "2024-25",
                "--gw",
                "3",
                "--rebuild",
            ]
        )
    finally:
        get_settings.cache_clear()
    squad = run.call_args.kwargs["squad"]
    assert squad.owned == []
    assert squad.bank == 176
    assert squad.free_transfers == 2
