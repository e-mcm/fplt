"""Stage B4 PR2: push finished runs to data (official) or lab (else)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fplt.data.remote import Remote
from fplt.models.predictions import empty_predictions
from fplt.schemas.manifests import AsOfDeadline
from fplt.settings import Settings
from fplt.store.push import (
    IncompleteRunError,
    bucket_kind_for_run,
    maybe_push_run,
    push_run,
)
from fplt.store.runs import RunWriter, run_dir

_TS = datetime(2026, 11, 24, 12, 56, 1, tzinfo=UTC)


def _deadline() -> AsOfDeadline:
    return AsOfDeadline(season="2026-27", gw=12, deadline_utc=_TS)


def _finish_run(
    tmp_path: Path,
    *,
    run_id: str,
    kind: str = "adhoc",
) -> Path:
    with RunWriter(
        tmp_path,
        kind=kind,  # type: ignore[arg-type]
        mode="live",
        run_type="predict",
        spec_hash="a" * 64,
        spec_label="demo",
        model_id="deadbeef",
        as_of=[_deadline()],
        horizon=3,
        code_sha="abc1234",
        code_dirty=False,
        lockfile_hash="b" * 64,
        host="test",
        raw_manifest_hash="c" * 64,
        staged_digests={},
        started_at=_TS,
        run_id=run_id,
    ) as writer:
        writer.write_parquet(
            "predictions/player_fixture.parquet",
            empty_predictions(),
        )
    return writer.path


def test_bucket_kind_routes_official_to_data() -> None:
    assert bucket_kind_for_run("official") == "data"
    assert bucket_kind_for_run("adhoc") == "lab"
    assert bucket_kind_for_run("experiment") == "lab"
    assert bucket_kind_for_run("benchmark") == "lab"


def test_push_run_refuses_incomplete(tmp_path: Path) -> None:
    path = run_dir(tmp_path, "r_20261124T125601Z_incmpl")
    path.mkdir(parents=True)
    (path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(IncompleteRunError, match="_SUCCESS"):
        push_run(
            tmp_path,
            "r_20261124T125601Z_incmpl",
            remote=Remote(target=str(tmp_path / "bucket"), kind="lab", env={}),
        )


def test_push_run_adhoc_to_lab_local_dir(tmp_path: Path) -> None:
    run_id = "r_20261124T125601Z_adhoc1"
    _finish_run(tmp_path, run_id=run_id, kind="adhoc")
    bucket = tmp_path / "lab-bucket"
    remote = Remote(target=str(bucket), kind="lab", env={})
    pushed = push_run(tmp_path, run_id, remote=remote)
    assert pushed.kind == "lab"
    dest = bucket / f"runs/run={run_id}"
    assert (dest / "_SUCCESS").is_file()
    assert (dest / "manifest.json").is_file()
    assert (dest / "predictions" / "player_fixture.parquet").is_file()
    # #371: catalog inbox rides with the run so catch-up can skip cold caches.
    assert (bucket / "catalog" / "inbox" / f"run={run_id}.parquet").is_file()


def test_push_run_official_to_data_local_dir(tmp_path: Path) -> None:
    run_id = "r_20261124T125601Z_offic1"
    _finish_run(tmp_path, run_id=run_id, kind="official")
    bucket = tmp_path / "data-bucket"
    remote = Remote(target=str(bucket), kind="data", env={})
    push_run(tmp_path, run_id, remote=remote)
    assert (bucket / f"runs/run={run_id}" / "_SUCCESS").is_file()
    assert (bucket / "catalog" / "inbox" / f"run={run_id}.parquet").is_file()


def test_push_run_skips_missing_catalog_inbox(tmp_path: Path) -> None:
    """#371: push still succeeds when the inbox row was already compacted away."""
    from fplt.store.catalog import catalog_inbox_dir

    run_id = "r_20261124T125601Z_noibox"
    _finish_run(tmp_path, run_id=run_id, kind="adhoc")
    inbox = catalog_inbox_dir(tmp_path) / f"run={run_id}.parquet"
    assert inbox.is_file()
    inbox.unlink()
    bucket = tmp_path / "lab-bucket"
    remote = Remote(target=str(bucket), kind="lab", env={})
    push_run(tmp_path, run_id, remote=remote)
    assert (bucket / f"runs/run={run_id}" / "_SUCCESS").is_file()
    assert not (bucket / "catalog" / "inbox" / f"run={run_id}.parquet").exists()


def test_push_run_rejects_wrong_remote_kind(tmp_path: Path) -> None:
    run_id = "r_20261124T125601Z_wrong1"
    _finish_run(tmp_path, run_id=run_id, kind="official")
    remote = Remote(target=str(tmp_path / "lab"), kind="lab", env={})
    with pytest.raises(ValueError, match="data bucket"):
        push_run(tmp_path, run_id, remote=remote)


def test_push_run_requires_configured_bucket(tmp_path: Path) -> None:
    run_id = "r_20261124T125601Z_nobkt1"
    _finish_run(tmp_path, run_id=run_id, kind="adhoc")
    with pytest.raises(ValueError, match="lab bucket is not configured"):
        push_run(tmp_path, run_id, settings=Settings())


def test_maybe_push_run_skips_when_bucket_unset(tmp_path: Path) -> None:
    run_id = "r_20261124T125601Z_skip01"
    _finish_run(tmp_path, run_id=run_id, kind="adhoc")
    assert maybe_push_run(tmp_path, run_id, settings=Settings()) is None


def test_maybe_push_run_pushes_when_lab_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_id = "r_20261124T125601Z_auto01"
    _finish_run(tmp_path, run_id=run_id, kind="adhoc")
    bucket = tmp_path / "lab-bucket"
    settings = Settings(lab_bucket="fplt-lab")

    def fake_from_settings(_s: Settings, bucket: str | None = None) -> Remote | None:
        assert bucket == "fplt-lab"
        return Remote(target=str(tmp_path / "lab-bucket"), kind="lab", env={})

    monkeypatch.setattr("fplt.store.push.Remote.from_settings", fake_from_settings)
    remote = maybe_push_run(tmp_path, run_id, settings=settings)
    assert remote is not None
    assert remote.kind == "lab"
    assert (bucket / f"runs/run={run_id}" / "_SUCCESS").is_file()
