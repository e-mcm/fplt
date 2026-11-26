"""ops-10: status.json schema, builder, public split, and CLI."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from fplt.auth import set_credentials
from fplt.automation.status import (
    build_status,
    format_status_human,
    public_status,
    publish_ops_status,
    push_status,
    record_last_error,
    write_status,
)
from fplt.cli import main
from fplt.data.quarantine import append_quarantine
from fplt.data.remote import Remote
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.schemas.status import (
    PRIVATE_STATUS_FIELDS,
    STATUS_SOURCES,
    CaptureSourceStatus,
    StatusDocument,
    TokenExpiryStatus,
)
from fplt.settings import get_settings
from tests.builders import build_league

_NOW = datetime(2024, 8, 20, 12, 0, tzinfo=UTC)


def _jwt(*, exp: datetime) -> str:
    import base64

    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload = (
        base64.urlsafe_b64encode(
            json.dumps({"exp": int(exp.timestamp()), "sub": "u"}).encode()
        )
        .rstrip(b"=")
        .decode()
    )
    return f"{header}.{payload}.sig"


def _write_gz(path: Path, payload: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload or {}, handle)


def _seed_captures(raw: Path) -> None:
    _write_gz(
        raw
        / "fpl_api"
        / "season=2024-25"
        / "captured_at=2024-08-19T10-00-00Z"
        / "bootstrap_static.json.gz"
    )
    _write_gz(
        raw
        / "understat"
        / "season=2024-25"
        / "teams"
        / "captured_at=2024-08-18T09-00-00Z"
        / "teams.json.gz"
    )
    _write_gz(
        raw
        / "api_football"
        / "season=2024-25"
        / "injuries"
        / "captured_at=2024-08-17T08-00-00Z"
        / "injuries.json.gz"
    )
    e0 = raw / "football_data" / "season=2024-25" / "captured_at=2024-08-01" / "E0.csv"
    e0.parent.mkdir(parents=True, exist_ok=True)
    e0.write_text("Div,Date\nE0,01/08/2024\n", encoding="utf-8")


def _seed_plan(
    data_dir: Path, model_name: str, season: str, gw: int, horizon: int
) -> Path:
    specs = load_model_specs()
    model_id = specs[model_name].model_id
    stamp = f"as_of={season}_gw{gw:02d}_h{horizon:02d}"
    path = data_dir / "plans" / model_id / stamp / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"x": [1]}).write_parquet(path)
    return path


def test_capture_status_requires_reason_when_null() -> None:
    with pytest.raises(ValidationError, match="reason"):
        CaptureSourceStatus(source="fpl_api", captured_at=None, reason=None)


def test_build_status_missing_sources_report_null_and_reason(tmp_path: Path) -> None:
    doc = build_status(tmp_path, now=_NOW, models=["rolling_mean_v1"], horizon=1)
    assert doc.schema_version == 1
    assert {item.source for item in doc.last_captures} == set(STATUS_SOURCES)
    for item in doc.last_captures:
        assert item.captured_at is None
        assert item.reason
    assert len(doc.last_plans) == 1
    assert doc.last_plans[0].written_at is None
    assert "no plan" in (doc.last_plans[0].reason or "")
    assert doc.next_deadline.deadline_utc is None
    assert doc.next_deadline.reason
    assert doc.token_expiry.expires_at is None
    assert doc.token_expiry.reason
    assert doc.prod_sha is None
    assert doc.prod_sha_reason
    assert doc.last_error is None
    assert doc.last_error_reason == "no recorded error"
    assert doc.quarantines == []
    names = {item.name for item in doc.heartbeats}
    assert "ops_tick" in names
    assert "snapshot" in names


def test_build_status_reads_captures_plans_quarantine_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    raw = tmp_path / "raw"
    _seed_captures(raw)
    store = DataStore(tmp_path)
    build_league(store)
    _seed_plan(tmp_path, "rolling_mean_v1", "2024-25", 1, 1)

    exp = _NOW + timedelta(hours=4)
    env_path = tmp_path / "env"
    set_credentials(token=_jwt(exp=exp), env_path=env_path)
    monkeypatch.setenv("FPLT_ENV_FILE", str(env_path))

    available = _NOW - timedelta(minutes=5)
    append_quarantine(
        tmp_path,
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "source": ["fpl_api"],
                "capture_path": [
                    "fpl_api/season=2024-25/captured_at=2024-08-19T10-00-00Z"
                ],
                "captured_at": [datetime(2024, 8, 19, 10, tzinfo=UTC)],
                "gate": ["player_count"],
                "detail": ["too few players"],
                "previous_capture_path": [None],
                "available_at": [available],
                "schema_version": [1],
            }
        ),
    )

    doc = build_status(
        tmp_path,
        now=_NOW,
        models=["rolling_mean_v1"],
        horizon=1,
        repo_root=tmp_path,
    )
    by_source = {item.source: item for item in doc.last_captures}
    assert by_source["fpl_api"].captured_at == datetime(2024, 8, 19, 10, tzinfo=UTC)
    assert by_source["understat"].captured_at == datetime(2024, 8, 18, 9, tzinfo=UTC)
    assert by_source["api_football"].captured_at == datetime(2024, 8, 17, 8, tzinfo=UTC)
    assert by_source["football_data"].captured_at == datetime(2024, 8, 1, tzinfo=UTC)
    assert doc.last_plans[0].season == "2024-25"
    assert doc.last_plans[0].gw == 1
    assert doc.last_plans[0].written_at is not None
    assert doc.next_deadline.season == "2024-25"
    assert doc.next_deadline.gw == 2
    assert doc.next_deadline.deadline_utc is not None
    # ops-02 is on main: checkpoint lists populate (may be empty before T−24h).
    assert doc.next_deadline.reason is None
    assert doc.next_deadline.checkpoints_done is not None
    assert doc.next_deadline.checkpoints_due is not None
    assert doc.token_expiry.expires_at == exp
    assert doc.token_expiry.expired is False
    assert len(doc.quarantines) == 1
    assert doc.quarantines[0].gate == "player_count"
    assert doc.prod_sha is None
    assert "prod tag" in (doc.prod_sha_reason or "")
    hb = {item.name: item for item in doc.heartbeats}
    assert hb["snapshot"].last_at == by_source["fpl_api"].captured_at
    assert hb["ops_tick"].last_at == _NOW
    get_settings.cache_clear()


def test_public_status_strips_private_fields(tmp_path: Path) -> None:
    doc = build_status(tmp_path, now=_NOW, models=["rolling_mean_v1"], horizon=1)
    public = public_status(doc)
    for key in PRIVATE_STATUS_FIELDS:
        assert key not in public
    assert "last_error_reason" not in public
    assert "last_captures" in public
    assert "token_expiry" not in public
    # Round-trip private document still validates.
    StatusDocument.model_validate(doc.model_dump())


def test_token_expiry_schema_requires_reason() -> None:
    with pytest.raises(ValidationError, match="reason"):
        TokenExpiryStatus(expires_at=None, reason=None)


def test_write_and_push_status_mutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.data import remote as remote_mod

    doc = build_status(tmp_path, now=_NOW, models=["rolling_mean_v1"], horizon=1)
    path = write_status(tmp_path, doc)
    assert path.is_file()
    assert (tmp_path / "status" / "status.public.json").is_file()
    private = json.loads(path.read_text(encoding="utf-8"))
    public = json.loads(
        (tmp_path / "status" / "status.public.json").read_text(encoding="utf-8")
    )
    assert "token_expiry" in private
    assert "token_expiry" not in public

    seen: dict[str, object] = {}

    def fake_rclone(args: list[str], *, env: dict[str, str]) -> None:
        seen["args"] = args
        seen["env"] = env

    monkeypatch.setattr(remote_mod, "_rclone", fake_rclone)
    remote = Remote(target="fplt:fplt-data", kind="data", env={"K": "1"})
    pushed = push_status(tmp_path, remote=remote)
    assert pushed is remote
    assert seen["args"] == [
        "copyto",
        str(path),
        "fplt:fplt-data/status/status.json",
        "--fast-list",
    ]
    assert "--immutable" not in seen["args"]  # type: ignore[operator]


def test_publish_ops_status_records_and_clears_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fplt.automation.status.push_status",
        lambda *a, **k: None,
    )
    record_last_error(tmp_path, RuntimeError("boom"), source="ops-tick", now=_NOW)
    doc = publish_ops_status(
        tmp_path,
        now=_NOW,
        models=["rolling_mean_v1"],
        horizon=1,
        clear_error=False,
    )
    assert doc.last_error is not None
    assert "boom" in doc.last_error.message
    cleared = publish_ops_status(
        tmp_path,
        now=_NOW + timedelta(minutes=1),
        models=["rolling_mean_v1"],
        horizon=1,
        clear_error=True,
    )
    assert cleared.last_error is None
    assert cleared.last_error_reason == "no recorded error"


def test_format_status_human_mentions_fields(tmp_path: Path) -> None:
    doc = build_status(tmp_path, now=_NOW, models=["rolling_mean_v1"], horizon=1)
    text = format_status_human(doc)
    assert "fplt status" in text
    assert "capture[fpl_api]" in text
    assert "token_expiry" in text
    assert text.endswith("\n")


def test_cli_status_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    main(["status", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["schema_version"] == 1
    assert "token_expiry" in payload
    assert (tmp_path / "status" / "status.json").is_file()

    main(["status", "--json", "--public"])
    public = json.loads(capsys.readouterr().out)
    assert "token_expiry" not in public
    assert "last_captures" in public
    get_settings.cache_clear()
