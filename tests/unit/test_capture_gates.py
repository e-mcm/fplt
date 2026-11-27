"""ops-06 / A10 capture-workflow gate wiring and CLI."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request

import polars as pl
import pytest

from fplt.cli import main
from fplt.data.capture_gates import (
    check_new_captures,
    evaluate_capture,
    format_gates_status,
    parse_since,
    previous_good_snapshot,
)
from fplt.data.gates import GATE_NULL_SHARE, GATE_PLAYER_COUNT, DataGatesConfig
from fplt.data.quarantine import (
    append_quarantine,
    capture_relpath,
    load_quarantine,
    quarantined_capture_paths,
)
from fplt.data.raw import iter_snapshots
from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.settings import Settings, get_settings


def _tight_config() -> DataGatesConfig:
    return DataGatesConfig.model_validate(
        {
            "player_count_max_delta": 1,
            "max_null_share": {"now_cost": 0.0, "status": 0.0, "team": 0.0},
            "expected_team_count": 2,
        }
    )


def _pass_config() -> DataGatesConfig:
    return DataGatesConfig.model_validate(
        {
            "player_count_max_delta": 50,
            "max_null_share": {"now_cost": 0.0, "status": 0.0, "team": 0.0},
            "expected_team_count": 2,
        }
    )


_GOOD_AT = datetime(2024, 8, 10, 0, 0, 0, tzinfo=UTC)
_BAD_AT = datetime(2024, 8, 18, 0, 0, 0, tzinfo=UTC)
_TEAMS = [
    {"id": 1, "code": 3, "name": "A", "short_name": "AAA"},
    {"id": 2, "code": 7, "name": "B", "short_name": "BBB"},
]
_FIXTURES = [
    {
        "id": 1,
        "event": 1,
        "team_h": 1,
        "team_a": 2,
        "kickoff_time": "2024-08-17T14:00:00Z",
        "finished": True,
    },
    {
        "id": 2,
        "event": 2,
        "team_h": 1,
        "team_a": 2,
        "kickoff_time": "2024-08-24T14:00:00Z",
        "finished": False,
    },
]


def _bootstrap(*, n_players: int = 2, null_cost: bool = False) -> dict[str, Any]:
    elements: list[dict[str, Any]] = []
    for i in range(n_players):
        row: dict[str, Any] = {
            "id": i + 1,
            "code": 100 + i,
            "team": (i % 2) + 1,
            "element_type": 3,
            "web_name": f"P{i}",
            "status": "a",
            "now_cost": 50,
        }
        if null_cost and i == 0:
            row["now_cost"] = None
        elements.append(row)
    return {
        "teams": _TEAMS,
        "elements": elements,
        "events": [
            {
                "id": 1,
                "deadline_time": "2024-08-16T17:30:00Z",
                "finished": True,
                "data_checked": True,
                "is_next": False,
            },
            {
                "id": 2,
                "deadline_time": "2024-08-23T17:30:00Z",
                "finished": False,
                "data_checked": False,
                "is_next": True,
            },
        ],
    }


def _write_capture(raw_root: Path, stamp: datetime, bootstrap: dict[str, Any]) -> Path:
    label = stamp.strftime("%Y-%m-%dT%H-%M-%SZ")
    captured = raw_root / "fpl_api" / "season=2024-25" / f"captured_at={label}"
    captured.mkdir(parents=True)
    for name, payload in (("bootstrap_static", bootstrap), ("fixtures", _FIXTURES)):
        with gzip.open(captured / f"{name}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    return captured


def test_parse_since_iso_and_relative() -> None:
    now = datetime(2024, 8, 20, 12, 0, 0, tzinfo=UTC)
    assert parse_since("2024-08-01", now=now) == datetime(
        2024, 8, 1, 0, 0, 0, tzinfo=UTC
    )
    assert parse_since("7d", now=now) == now - timedelta(days=7)
    assert parse_since("24h", now=now) == now - timedelta(hours=24)
    with pytest.raises(ValueError, match="invalid --since"):
        parse_since("nope", now=now)


def test_evaluate_capture_quarantines_and_alerts_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr("fplt.data.capture_gates.load_data_gates_config", _tight_config)
    raw = tmp_path / "raw"
    _write_capture(raw, _GOOD_AT, _bootstrap(n_players=2))
    bad_dir = _write_capture(raw, _BAD_AT, _bootstrap(n_players=2, null_cost=True))
    snaps = iter_snapshots(raw)
    bad = snaps[-1]
    posts: list[Request] = []

    def fake_post(request: Request) -> object:
        posts.append(request)
        return object()

    first = evaluate_capture(
        tmp_path,
        raw,
        bad,
        available_at=_BAD_AT,
        notify_url="https://ntfy.test/fplt",
        post=fake_post,
    )
    assert not first.report.passed
    assert GATE_NULL_SHARE in first.new_failure_types
    assert len(posts) == 1
    assert capture_relpath(raw, bad_dir) in quarantined_capture_paths(tmp_path)

    # Same failure type again: quarantine grows, no second alert.
    later = _BAD_AT + timedelta(hours=1)
    worse = _write_capture(raw, later, _bootstrap(n_players=2, null_cost=True))
    second = evaluate_capture(
        tmp_path,
        raw,
        iter_snapshots(raw)[-1],
        available_at=later,
        notify_url="https://ntfy.test/fplt",
        post=fake_post,
    )
    assert GATE_NULL_SHARE not in second.new_failure_types
    assert len(posts) == 1
    assert capture_relpath(raw, worse) in quarantined_capture_paths(tmp_path)
    get_settings.cache_clear()


def test_previous_good_skips_quarantined(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    good = _write_capture(raw, _GOOD_AT, _bootstrap())
    bad = _write_capture(raw, _BAD_AT, _bootstrap(null_cost=True))
    append_quarantine(
        tmp_path,
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "source": ["fpl_api"],
                "capture_path": [capture_relpath(raw, bad)],
                "captured_at": [_BAD_AT],
                "gate": [GATE_NULL_SHARE],
                "detail": ["x"],
                "previous_capture_path": [None],
                "available_at": [_BAD_AT],
                "schema_version": [SCHEMA_VERSION],
            }
        ),
    )
    later = _BAD_AT + timedelta(days=1)
    _write_capture(raw, later, _bootstrap())
    prev = previous_good_snapshot(
        raw, season="2024-25", before=later, data_dir=tmp_path
    )
    assert prev is not None
    assert prev.capture_dir == good


def test_format_gates_status_and_cli_since(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    path = "fpl_api/season=2024-25/captured_at=2024-08-18T00-00-00Z"
    append_quarantine(
        tmp_path,
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "source": ["fpl_api"],
                "capture_path": [path],
                "captured_at": [_BAD_AT],
                "gate": [GATE_PLAYER_COUNT],
                "detail": ["player count blew up"],
                "previous_capture_path": [None],
                "available_at": [_BAD_AT],
                "schema_version": [SCHEMA_VERSION],
            }
        ),
    )
    text = format_gates_status(tmp_path, since=_GOOD_AT)
    assert "1 row(s)" in text
    assert GATE_PLAYER_COUNT in text
    main(["data", "gates", "--since", "2024-08-01", "--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "player_count" in out
    get_settings.cache_clear()


def test_check_new_captures_merges_and_skips_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    monkeypatch.setattr("fplt.data.capture_gates.load_data_gates_config", _pass_config)
    data_dir = tmp_path / "data"
    raw_history = data_dir / "raw"
    _write_capture(raw_history, _GOOD_AT, _bootstrap())
    new_raw = tmp_path / "new_raw"
    _write_capture(new_raw, _BAD_AT, _bootstrap(null_cost=True))
    settings = Settings(data_dir=data_dir, data_bucket=None, ntfy_url=None)
    result = check_new_captures(
        data_dir,
        new_raw=new_raw,
        settings=settings,
        notify_alerts=False,
        sync_remote=False,
        available_at=_BAD_AT,
    )
    assert len(result.actions) == 1
    assert len(result.failed) == 1
    assert GATE_NULL_SHARE in result.alerted
    assert not load_quarantine(data_dir).is_empty()
    get_settings.cache_clear()


def test_cli_gates_requires_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", "/tmp/fplt-gates-cli")
    get_settings.cache_clear()
    with pytest.raises(SystemExit) as exc:
        main(["data", "gates"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--since" in err
    get_settings.cache_clear()


def test_format_gates_status_empty_and_outside_window(tmp_path: Path) -> None:
    empty = format_gates_status(tmp_path, since=_GOOD_AT)
    assert "0 row(s); none on disk" in empty

    append_quarantine(
        tmp_path,
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "source": ["fpl_api"],
                "capture_path": ["fpl_api/season=2024-25/captured_at=old"],
                "captured_at": [_GOOD_AT],
                "gate": [GATE_NULL_SHARE],
                "detail": ["stale"],
                "previous_capture_path": [None],
                "available_at": [_GOOD_AT],
                "schema_version": [SCHEMA_VERSION],
            }
        ),
    )
    later = format_gates_status(tmp_path, since=_BAD_AT)
    assert "0 row(s)" in later
    assert "(no rows in window)" in later


def test_evaluate_capture_defaults_available_at_and_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("fplt.data.capture_gates.load_data_gates_config", _pass_config)
    raw = tmp_path / "raw"
    _write_capture(raw, _GOOD_AT, _bootstrap())
    snap = iter_snapshots(raw)[0]
    action = evaluate_capture(tmp_path, raw, snap)
    assert action.report.passed
    assert action.new_failure_types == ()
    assert load_quarantine(tmp_path).is_empty()


def test_check_new_captures_latest_per_season_and_sync_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("fplt.data.capture_gates.load_data_gates_config", _pass_config)
    data_dir = tmp_path / "data"
    raw = data_dir / "raw"
    _write_capture(raw, _GOOD_AT, _bootstrap())
    _write_capture(raw, _BAD_AT, _bootstrap())
    calls: list[str] = []

    monkeypatch.setattr(
        "fplt.data.capture_gates.pull_quarantine",
        lambda *a, **k: calls.append("pull") or False,
    )
    monkeypatch.setattr(
        "fplt.data.capture_gates.push_quarantine",
        lambda *a, **k: calls.append("push") or False,
    )
    monkeypatch.setattr(
        "fplt.data.capture_gates._pull_fpl_api_history",
        lambda *a, **k: calls.append("history"),
    )
    settings = Settings(data_dir=data_dir, data_bucket=None, ntfy_url=None)
    result = check_new_captures(
        data_dir,
        settings=settings,
        notify_alerts=False,
        sync_remote=True,
        available_at=_BAD_AT,
    )
    assert "pull" in calls and "push" in calls and "history" in calls
    # Latest capture per season only (two stamps → one action).
    assert len(result.actions) == 1
    assert result.actions[0].report.passed


def test_check_new_captures_skips_already_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("fplt.data.capture_gates.load_data_gates_config", _pass_config)
    data_dir = tmp_path / "data"
    raw = data_dir / "raw"
    first = _write_capture(raw, _GOOD_AT, _bootstrap())
    _write_capture(raw, _BAD_AT, _bootstrap())
    append_quarantine(
        data_dir,
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "source": ["fpl_api"],
                "capture_path": [capture_relpath(raw, first)],
                "captured_at": [_GOOD_AT],
                "gate": [GATE_NULL_SHARE],
                "detail": ["x"],
                "previous_capture_path": [None],
                "available_at": [_GOOD_AT],
                "schema_version": [SCHEMA_VERSION],
            }
        ),
    )
    # Quarantined older snap skipped; still gates the newer one.
    result = check_new_captures(
        data_dir,
        settings=Settings(data_dir=data_dir, data_bucket=None),
        sync_remote=False,
        available_at=_BAD_AT,
    )
    assert len(result.actions) == 1
    assert _BAD_AT.strftime("%Y-%m-%dT%H-%M-%SZ") in result.actions[0].capture_path


def test_pull_fpl_api_history_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fplt.data import capture_gates as mod
    from fplt.data.remote import RemoteError

    raw = tmp_path / "raw"
    settings = Settings(data_dir=tmp_path, data_bucket=None)

    monkeypatch.setattr(mod.Remote, "from_settings", classmethod(lambda cls, s: None))
    mod._pull_fpl_api_history(raw, settings)
    assert not (raw / "fpl_api").exists()

    remote = object()
    monkeypatch.setattr(
        mod.Remote, "from_settings", classmethod(lambda cls, s: remote)
    )
    pulls: list[tuple[object, str, Path]] = []

    def fake_pull(rem: object, prefix: str, dest: Path) -> None:
        pulls.append((rem, prefix, dest))

    monkeypatch.setattr(mod, "pull", fake_pull)
    mod._pull_fpl_api_history(raw, settings)
    assert pulls == [(remote, "raw/fpl_api", raw / "fpl_api")]
    assert (raw / "fpl_api").is_dir()

    def missing_pull(rem: object, prefix: str, dest: Path) -> None:
        raise RemoteError("Directory not found")

    monkeypatch.setattr(mod, "pull", missing_pull)
    mod._pull_fpl_api_history(raw, settings)  # soft-skip

    def other_pull(rem: object, prefix: str, dest: Path) -> None:
        raise RemoteError("auth failed")

    monkeypatch.setattr(mod, "pull", other_pull)
    with pytest.raises(RemoteError, match="auth failed"):
        mod._pull_fpl_api_history(raw, settings)


def test_parse_since_minutes_and_aware() -> None:
    now = datetime(2024, 8, 20, 12, 0, 0, tzinfo=UTC)
    assert parse_since("30m", now=now) == now - timedelta(minutes=30)
    aware = datetime(2024, 8, 1, 12, 0, 0, tzinfo=UTC)
    assert parse_since(aware.isoformat(), now=now) == aware
