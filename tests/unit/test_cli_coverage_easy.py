"""Easy CLI coverage: argparse errors, thin subcommands, helper paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fplt.cli import (
    _catalog,
    _deps_equivalence,
    _governance_check,
    _legacy,
    _not_implemented,
    _read_auth_token,
    _release,
    _sql,
    _step_weights_for_decay,
    main,
)
from fplt.settings import get_settings


def test_main_no_command_returns() -> None:
    assert main([]) is None


def test_main_quiet_and_verbose_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr(
        "fplt.store.catalog.rebuild_runs_catalog",
        lambda _root: MagicMock(height=0),
    )
    monkeypatch.setattr(
        "fplt.store.catalog.catalog_runs_path",
        lambda root: root / "catalog" / "runs.parquet",
    )
    try:
        main(["-q", "catalog", "rebuild", "--data-dir", str(tmp_path)])
        main(["-v", "catalog", "rebuild", "--data-dir", str(tmp_path)])
    finally:
        get_settings.cache_clear()


def test_main_argparse_missing_required() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["backtest"])
    assert excinfo.value.code == 2


def test_not_implemented_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert _not_implemented("widget") == 2
    assert "not implemented" in capsys.readouterr().err


def test_catalog_without_subcommand_exits() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _catalog(SimpleNamespace(catalog_command=None))
    assert excinfo.value.code == 2


def test_catalog_rebuild_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr(
        "fplt.store.catalog.rebuild_runs_catalog",
        lambda _root: MagicMock(height=3),
    )
    monkeypatch.setattr(
        "fplt.store.catalog.catalog_runs_path",
        lambda root: root / "catalog" / "runs.parquet",
    )
    try:
        main(["catalog", "rebuild", "--data-dir", str(tmp_path)])
    finally:
        get_settings.cache_clear()
    assert "rows=3" in capsys.readouterr().out


def test_sql_one_shot_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import polars as pl

    monkeypatch.setattr(
        "fplt.store.sql.run_sql",
        lambda query, data_dir, rebuild_catalog=False: pl.DataFrame({"n": [1]}),
    )
    _sql(
        SimpleNamespace(
            data_dir=tmp_path,
            rebuild_catalog=False,
            sql_command="SELECT 1",
        )
    )
    assert "1" in capsys.readouterr().out


def test_sql_opens_repl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    con = MagicMock()
    monkeypatch.setattr("fplt.store.sql.open_sql", lambda *_a, **_k: con)
    called: list[object] = []
    monkeypatch.setattr(
        "fplt.store.sql.sql_repl",
        lambda c: called.append(c),
    )
    _sql(
        SimpleNamespace(
            data_dir=tmp_path,
            rebuild_catalog=True,
            sql_command=None,
        )
    )
    assert called == [con]
    con.close.assert_called_once()


def test_legacy_without_subcommand_exits() -> None:
    with pytest.raises(SystemExit) as excinfo:
        _legacy(SimpleNamespace(legacy_command=None))
    assert excinfo.value.code == 2


def test_legacy_import_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = SimpleNamespace(
        n_imported=1,
        n_skipped=1,
        imported=["legacy-a"],
        skipped=["legacy-b"],
    )
    monkeypatch.setattr("fplt.store.legacy.import_legacy", lambda *_a, **_k: result)
    main(
        [
            "legacy",
            "import",
            "--dry-run",
            "--data-dir",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert "would import=1" in out
    assert "legacy-a" in out
    assert "legacy-b" in out


def test_deps_equivalence_missing_refs_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _deps_equivalence(
            SimpleNamespace(
                check_freeze_only=False,
                control_ref=None,
                treatment_ref=None,
                output_dir=None,
                live=False,
                repo_root=None,
            )
        )
    assert excinfo.value.code == 2
    assert "requires" in capsys.readouterr().err


def test_deps_equivalence_check_freeze_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    freeze = SimpleNamespace(state="open", reason="ok")
    monkeypatch.setattr(
        "fplt.automation.deps_equivalence.check_prod_freeze",
        lambda: freeze,
    )
    _deps_equivalence(
        SimpleNamespace(
            check_freeze_only=True,
            control_ref=None,
            treatment_ref=None,
            output_dir=None,
            live=False,
            repo_root=None,
        )
    )
    assert "open" in capsys.readouterr().out

    freeze_blocked = SimpleNamespace(state="frozen", reason="prod freeze")
    monkeypatch.setattr(
        "fplt.automation.deps_equivalence.check_prod_freeze",
        lambda: freeze_blocked,
    )
    with pytest.raises(SystemExit) as excinfo:
        _deps_equivalence(
            SimpleNamespace(
                check_freeze_only=True,
                control_ref=None,
                treatment_ref=None,
                output_dir=None,
                live=False,
                repo_root=None,
            )
        )
    assert excinfo.value.code == 1


def test_governance_check_pass_and_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    body = tmp_path / "body.md"
    body.write_text("promote-eval-run: x\n", encoding="utf-8")
    monkeypatch.setattr(
        "fplt.automation.governance.check_git_production_governance",
        lambda **_k: SimpleNamespace(passed=True, reasons=()),
    )
    monkeypatch.setattr(
        "fplt.settings.get_settings",
        lambda: SimpleNamespace(data_dir=tmp_path),
    )
    _governance_check(
        SimpleNamespace(
            pr_body="",
            pr_body_file=body,
            base="main",
            head="HEAD",
        )
    )
    assert "ok" in capsys.readouterr().out

    monkeypatch.setattr(
        "fplt.automation.governance.check_git_production_governance",
        lambda **_k: SimpleNamespace(passed=False, reasons=("bad",)),
    )
    with pytest.raises(SystemExit) as excinfo:
        _governance_check(
            SimpleNamespace(
                pr_body="x",
                pr_body_file=None,
                base="main",
                head="HEAD",
            )
        )
    assert excinfo.value.code == 1
    assert "bad" in capsys.readouterr().err


def test_release_usage_without_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _release(SimpleNamespace(release_command=None))
    assert excinfo.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_release_smoke_ok_and_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from fplt.automation.release import ReleaseGateError

    monkeypatch.setattr(
        "fplt.automation.release.run_smoke_plan",
        lambda: SimpleNamespace(detail="smoke ok"),
    )
    _release(SimpleNamespace(release_command="smoke"))
    assert "smoke ok" in capsys.readouterr().out

    def _boom() -> None:
        raise ReleaseGateError("nope")

    monkeypatch.setattr("fplt.automation.release.run_smoke_plan", _boom)
    with pytest.raises(SystemExit) as excinfo:
        _release(SimpleNamespace(release_command="smoke"))
    assert excinfo.value.code == 1


def test_step_weights_for_decay() -> None:
    assert _step_weights_for_decay("geometric", 3) is None
    weights = _step_weights_for_decay("skill", 3)
    assert weights is not None
    assert len(weights) == 3
    with pytest.raises(ValueError, match="unknown"):
        _step_weights_for_decay("nope", 2)


def test_read_auth_token_from_arg_and_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _read_auth_token("abc") == "abc"
    monkeypatch.setattr(
        "fplt.cli.sys.stdin",
        SimpleNamespace(read=lambda: " from-stdin ", isatty=lambda: False),
    )
    assert _read_auth_token("-").strip() == "from-stdin"
    monkeypatch.setattr(
        "fplt.cli.sys.stdin",
        SimpleNamespace(read=lambda: "", isatty=lambda: False),
    )
    with pytest.raises(ValueError, match="empty token"):
        _read_auth_token(None)


def test_data_without_subcommand_exits() -> None:
    from fplt.cli import _data

    with pytest.raises(SystemExit) as excinfo:
        _data(SimpleNamespace(data_command=None))
    assert excinfo.value.code == 2


def test_idmap_without_subcommand_exits() -> None:
    from fplt.cli import _idmap

    with pytest.raises(SystemExit) as excinfo:
        _idmap(SimpleNamespace(idmap_command=None))
    assert excinfo.value.code == 2


def test_drift_check_no_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr(
        "fplt.automation.drift.run_live_drift_monitor",
        lambda *_a, **_k: [],
    )
    try:
        with pytest.raises(SystemExit) as excinfo:
            main(["drift-check", "--no-notify"])
        assert excinfo.value.code == 0
    finally:
        get_settings.cache_clear()
    assert "no production models" in capsys.readouterr().out
