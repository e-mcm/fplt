"""Extra status.py edge coverage."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from fplt.automation.status import (
    _checkpoint_state,
    _last_error,
    _last_plans,
    _latest_capture,
    _next_deadline,
    _plans_for_model,
    _prod_sha,
    _stamp_from_path,
    _token_expiry,
    build_status,
    format_status_human,
    last_error_path,
    publish_ops_status,
    push_status,
    write_status,
)
from fplt.data.remote import Remote, RemoteError
from fplt.data.store import DataStore
from fplt.schemas.status import (
    LastErrorStatus,
    NextDeadlineStatus,
    PlanDeadlineStatus,
    TokenExpiryStatus,
)

_NOW = datetime(2024, 8, 20, 12, 0, tzinfo=UTC)


def test_write_status_skips_public(tmp_path: Path) -> None:
    doc = build_status(tmp_path, now=_NOW, models=["rolling_mean_v1"], horizon=1)
    path = write_status(tmp_path, doc, write_public=False)
    assert path.is_file()
    assert not (tmp_path / "status" / "status.public.json").exists()


def test_push_status_missing_and_wrong_kind(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        push_status(tmp_path)
    write_status(
        tmp_path,
        build_status(tmp_path, now=_NOW, models=["rolling_mean_v1"], horizon=1),
    )
    with pytest.raises(ValueError, match="data bucket"):
        push_status(
            tmp_path,
            remote=Remote(target="fplt:fplt-lab", kind="lab", env={"K": "1"}),
        )


def test_push_status_skipped_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_status(
        tmp_path,
        build_status(tmp_path, now=_NOW, models=["rolling_mean_v1"], horizon=1),
    )
    monkeypatch.setattr(
        "fplt.automation.status.Remote.from_settings",
        lambda _s: None,
    )
    assert push_status(tmp_path) is None


def test_publish_ops_status_tolerates_push_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: object, **_k: object) -> None:
        raise RemoteError("nope")

    monkeypatch.setattr("fplt.automation.status.push_status", _boom)
    doc = publish_ops_status(tmp_path, now=_NOW, models=["rolling_mean_v1"], horizon=1)
    assert doc.schema_version == 1


def test_format_status_human_rich_fields(tmp_path: Path) -> None:
    doc = build_status(tmp_path, now=_NOW, models=["rolling_mean_v1"], horizon=1)
    rich = doc.model_copy(
        update={
            "last_plans": [
                PlanDeadlineStatus(
                    model="rolling_mean_v1",
                    model_id="rm",
                    season="2024-25",
                    gw=1,
                    horizon=1,
                    written_at=_NOW,
                    path="plans/rm/as_of=2024-25_gw01_h01/part.parquet",
                ),
                PlanDeadlineStatus(model="missing", reason="no plan parquet"),
            ],
            "next_deadline": NextDeadlineStatus(
                season="2024-25",
                gw=2,
                deadline_utc=_NOW + timedelta(days=1),
                checkpoints_done=["fetch"],
                checkpoints_due=["plan"],
                reason="partial",
            ),
            "token_expiry": TokenExpiryStatus(
                expires_at=_NOW + timedelta(hours=2), expired=False
            ),
            "last_error": LastErrorStatus(at=_NOW, message="boom", source="ops-tick"),
            "last_error_reason": None,
        }
    )
    text = format_status_human(rich)
    assert "plan[rolling_mean_v1]" in text
    assert "plan[missing]" in text
    assert "next_deadline: 2024-25 gw2" in text
    assert "checkpoints:" in text
    assert "token_expiry:" in text
    assert "last_error: boom" in text


def test_unknown_model_and_bad_plan_dirs(tmp_path: Path) -> None:
    plans = _last_plans(tmp_path, ["definitely_not_a_model"], 1)
    assert plans[0].reason
    assert "not registered" in plans[0].reason

    base = tmp_path / "plans" / "mid"
    (base / "as_of=not-a-stamp").mkdir(parents=True)
    (base / "as_of=not-a-stamp" / "part.parquet").write_text("x", encoding="utf-8")
    good = base / "as_of=2024-25_gw01_h02"
    good.mkdir(parents=True)
    (good / "part.parquet").write_text("x", encoding="utf-8")
    assert _plans_for_model(tmp_path, "mid", horizon=1) == []
    found = _plans_for_model(tmp_path, "mid", horizon=2)
    assert len(found) == 1


def test_next_deadline_edges(tmp_path: Path) -> None:
    assert "missing" in (_next_deadline(tmp_path, _NOW).reason or "")
    (tmp_path / "staged").mkdir()
    reason = _next_deadline(tmp_path, _NOW).reason or ""
    assert "unreadable" in reason or "no gameweeks" in reason

    store = DataStore(tmp_path)
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "gw": [1],
                "deadline_time": [_NOW - timedelta(days=1)],
                "finished": [True],
                "data_checked": [False],
                "available_at": [_NOW],
                "source": ["test"],
            }
        ),
    )
    assert "no upcoming" in (_next_deadline(tmp_path, _NOW).reason or "")


def test_checkpoint_probe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fplt.automation.checkpoints as cp

    monkeypatch.setattr(
        cp,
        "resolve_done_sets",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("probe fail")),
    )
    done, due, reason = _checkpoint_state(
        tmp_path, _NOW, _NOW + timedelta(days=1), season="2024-25", gw=1
    )
    assert done is None
    assert due is None
    assert reason
    assert "probe failed" in reason


def test_token_expiry_missing_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "fplt.automation.status.auth_status",
        lambda **_k: SimpleNamespace(
            token=SimpleNamespace(expires_at=None, expired=False)
        ),
    )
    status = _token_expiry(_NOW)
    assert status.expires_at is None
    assert status.reason
    assert "expiry claim" in status.reason


def test_prod_sha_empty_and_last_error_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fplt.automation.status.subprocess.run",
        lambda *_a, **_k: SimpleNamespace(stdout="   \n"),
    )
    sha, reason = _prod_sha(tmp_path)
    assert sha is None
    assert reason
    assert "empty" in reason

    path = last_error_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")
    err, reason = _last_error(tmp_path)
    assert err is None
    assert reason
    assert "unreadable" in reason


def test_latest_capture_edges(tmp_path: Path) -> None:
    assert _latest_capture(tmp_path / "missing", "fpl_api")[2]
    raw = tmp_path / "raw"
    raw.mkdir()
    assert "no understat" in (_latest_capture(raw, "understat")[2] or "")
    assert "no api_football" in (_latest_capture(raw, "api_football")[2] or "")
    assert "no football_data" in (_latest_capture(raw, "football_data")[2] or "")
    assert "unknown source" in (_latest_capture(raw, "weird")[2] or "")

    e0 = raw / "football_data" / "season=2024-25" / "E0.csv"
    e0.parent.mkdir(parents=True)
    e0.write_text("Div\nE0\n", encoding="utf-8")
    stamp, path, reason = _latest_capture(raw, "football_data")
    assert stamp is not None
    assert path is not None
    assert reason is None

    dated = (
        raw
        / "understat"
        / "season=2024-25"
        / "dates"
        / "captured_at=2024-08-01"
        / "dates.json.gz"
    )
    dated.parent.mkdir(parents=True)
    dated.write_bytes(b"")
    assert _stamp_from_path(dated) == datetime(2024, 8, 1, tzinfo=UTC)


def test_last_error_valid_sidecar(tmp_path: Path) -> None:
    path = last_error_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "at": _NOW.isoformat(),
                "message": "failed",
                "source": "ops-tick",
            }
        ),
        encoding="utf-8",
    )
    err, reason = _last_error(tmp_path)
    assert reason is None
    assert err is not None
    assert err.message == "failed"
