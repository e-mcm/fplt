"""Post-downtime ops-catchup wraps feed refresh → catch_up."""

from pathlib import Path

import pytest

from fplt.automation.ops_catchup import (
    OpsCatchupResult,
    format_ops_catchup_status,
    run_ops_catchup,
)
from fplt.automation.ops_feeds import FeedRefreshResult
from fplt.cli import main
from fplt.data.snapshots import SnapshotFetchError
from fplt.settings import get_settings
from fplt.types import GWKey

_ROOT = Path(__file__).resolve().parents[2]


def test_format_ops_catchup_status_lists_planned_and_pending() -> None:
    text = format_ops_catchup_status(
        OpsCatchupResult(
            snapshots_fetched=2,
            staged_snapshots=1,
            understat_matches=3,
            api_football_captures=4,
            planned=[GWKey("2024-25", 1)],
            still_pending=[GWKey("2024-25", 2)],
            models=["dc_layered_v4"],
            horizon=3,
        )
    )
    assert "snapshots fetched: 2" in text
    assert "staged snapshots:  1" in text
    assert "understat matches: 3" in text
    assert "api_football:      4" in text
    assert "dc_layered_v4 / H=3" in text
    assert "planned this run:  2024-25 gw1" in text
    assert "still pending:     2024-25 gw2" in text
    assert text.endswith("\n")


def test_format_ops_catchup_status_uses_none_when_empty() -> None:
    text = format_ops_catchup_status(
        OpsCatchupResult(
            snapshots_fetched=0,
            staged_snapshots=0,
            understat_matches=0,
            api_football_captures=0,
            planned=[],
            still_pending=[],
            models=["rolling_mean_v1"],
            horizon=1,
        )
    )
    assert "planned this run:  none" in text
    assert "still pending:     none" in text


def test_run_ops_catchup_calls_existing_library_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    calls: list[str] = []
    planned = [GWKey("2024-25", 1)]

    def refresh(
        data_dir: Path | None = None,
        *,
        source: str = "auto",
        remote: str = "origin",
        branch: str = "snapshots",
        settings: object = None,
    ) -> FeedRefreshResult:
        calls.append(f"refresh:{source}:{remote}:{branch}:{data_dir}")
        return FeedRefreshResult(
            snapshots_fetched=1,
            staged_snapshots=5,
            understat_matches=7,
            api_football_captures=9,
        )

    def catch_up(
        data_dir: Path,
        models: list[str] | None = None,
        horizon: int = 3,
    ) -> list[GWKey]:
        calls.append(f"catch_up:{models}:{horizon}")
        return planned

    def pending(
        store: object,
        root: Path,
        models: list[str],
        horizon: int = 3,
    ) -> list[GWKey]:
        calls.append("pending")
        return []

    notify_calls: list[tuple[str, str]] = []

    def notify(settings: object, title: str, message: str, **_: object) -> bool:
        notify_calls.append((title, message))
        return True

    monkeypatch.setattr("fplt.automation.ops_catchup.refresh_ops_feeds", refresh)
    monkeypatch.setattr("fplt.automation.ops_catchup.catch_up", catch_up)
    monkeypatch.setattr("fplt.automation.ops_catchup.pending_checked", pending)
    monkeypatch.setattr("fplt.automation.ops_catchup.reports", lambda _d: "# GW 1\nok")
    monkeypatch.setattr("fplt.automation.ops_catchup.notify_recommendation", notify)

    result = run_ops_catchup(
        models=["rolling_mean_v1"],
        horizon=1,
        source="branch",
        remote="/tmp/remote",
        branch="snapshots",
    )
    assert result.snapshots_fetched == 1
    assert result.staged_snapshots == 5
    assert result.understat_matches == 7
    assert result.api_football_captures == 9
    assert result.planned == planned
    assert result.still_pending == []
    assert result.models == ["rolling_mean_v1"]
    assert result.horizon == 1
    assert calls == [
        f"refresh:branch:/tmp/remote:snapshots:{tmp_path}",
        "catch_up:['rolling_mean_v1']:1",
        "pending",
    ]
    assert notify_calls == [("fplt recommendation", "# GW 1\nok")]
    get_settings.cache_clear()


def test_run_ops_catchup_skips_notify_when_nothing_planned_or_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    notify_calls: list[str] = []

    monkeypatch.setattr(
        "fplt.automation.ops_catchup.refresh_ops_feeds",
        lambda *a, **k: FeedRefreshResult(0, 0, 0, 0),
    )
    monkeypatch.setattr("fplt.automation.ops_catchup.catch_up", lambda *a, **k: [])
    monkeypatch.setattr(
        "fplt.automation.ops_catchup.pending_checked", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "fplt.automation.ops_catchup.notify_recommendation",
        lambda *a, **k: notify_calls.append("sent") or True,
    )

    empty = run_ops_catchup(models=["rolling_mean_v1"], horizon=1)
    assert empty.planned == []
    assert notify_calls == []

    monkeypatch.setattr(
        "fplt.automation.ops_catchup.catch_up",
        lambda *a, **k: [GWKey("2024-25", 1)],
    )
    monkeypatch.setattr("fplt.automation.ops_catchup.reports", lambda _d: "report")
    skipped = run_ops_catchup(models=["rolling_mean_v1"], horizon=1, notify=False)
    assert skipped.planned == [GWKey("2024-25", 1)]
    assert notify_calls == []
    get_settings.cache_clear()


def test_run_ops_catchup_propagates_snapshot_fetch_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    def boom(*_a: object, **_k: object) -> FeedRefreshResult:
        raise SnapshotFetchError("missing branch")

    monkeypatch.setattr("fplt.automation.ops_catchup.refresh_ops_feeds", boom)
    with pytest.raises(SnapshotFetchError, match="missing branch"):
        run_ops_catchup()
    get_settings.cache_clear()


def test_cli_ops_catchup_prints_status_and_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FPLT_ENV_FILE", str(tmp_path / "env"))
    get_settings.cache_clear()
    result = OpsCatchupResult(
        snapshots_fetched=0,
        staged_snapshots=0,
        understat_matches=0,
        api_football_captures=0,
        planned=[],
        still_pending=[],
        models=["dc_layered_v4"],
        horizon=3,
    )
    seen: dict[str, object] = {}

    def fake_run(**kwargs: object) -> OpsCatchupResult:
        seen.update(kwargs)
        return result

    monkeypatch.setattr(
        "fplt.automation.ops_catchup.run_ops_catchup",
        fake_run,
    )
    main(["ops-catchup", "--no-notify", "--horizon", "3"])
    out = capsys.readouterr().out
    assert "ops-catchup status" in out
    assert "still pending:     none" in out
    assert "entry_id" in out or "FPLT_ENTRY_ID" in out or "token" in out.lower()
    assert seen.get("source") == "auto"
    get_settings.cache_clear()


def test_cli_ops_catchup_passes_source_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    seen: dict[str, object] = {}

    def fake_run(**kwargs: object) -> OpsCatchupResult:
        seen.update(kwargs)
        return OpsCatchupResult(
            snapshots_fetched=0,
            staged_snapshots=0,
            understat_matches=0,
            api_football_captures=0,
            planned=[],
            still_pending=[],
            models=["rolling_mean_v1"],
            horizon=3,
        )

    monkeypatch.setattr("fplt.automation.ops_catchup.run_ops_catchup", fake_run)
    main(["ops-catchup", "--source", "branch", "--no-notify"])
    assert seen.get("source") == "branch"
    get_settings.cache_clear()


def test_cli_ops_catchup_exits_on_fetch_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    def boom(**_k: object) -> OpsCatchupResult:
        raise SnapshotFetchError("boom")

    monkeypatch.setattr("fplt.automation.ops_catchup.run_ops_catchup", boom)
    with pytest.raises(SystemExit) as exc:
        main(["ops-catchup"])
    assert exc.value.code == 2
    get_settings.cache_clear()


def test_cli_ops_catchup_exits_on_remote_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.data.remote import RemoteError

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    def boom(**_k: object) -> OpsCatchupResult:
        raise RemoteError("rclone failed")

    monkeypatch.setattr("fplt.automation.ops_catchup.run_ops_catchup", boom)
    with pytest.raises(SystemExit) as exc:
        main(["ops-catchup", "--source", "bucket"])
    assert exc.value.code == 2
    get_settings.cache_clear()


def test_makefile_and_deploy_readme_document_ops_catchup() -> None:
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    readme = (_ROOT / "deploy" / "README.md").read_text(encoding="utf-8")
    design = (_ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    assert "ops-catchup:" in makefile
    assert "fplt ops-catchup" in makefile
    assert "make ops-catchup" in readme
    assert "Post-travel" in readme or "downtime catch-up" in readme
    assert "make ops-catchup" in design
