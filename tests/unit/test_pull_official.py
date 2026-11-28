"""Local-Remote tests for ``fplt data pull --official`` (#280)."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from pydantic import SecretStr

from fplt.cli import main
from fplt.data.remote import Remote, RemoteError
from fplt.models.predictions import empty_predictions
from fplt.schemas.manifests import (
    AsOfDeadline,
    BuildManifest,
    BuildTableEntry,
    OfficialPointer,
)
from fplt.settings import Settings, get_settings
from fplt.store.api import Store
from fplt.store.builds import STAGING_VERSION, write_build_manifest, write_table
from fplt.store.official import write_official_pointer
from fplt.store.pull_official import (
    OfficialPullError,
    infer_official_season,
    mirror_official_index,
    pull_official,
    require_data_remote,
    resolve_checkpoint_target,
    resolve_latest_gw,
)
from fplt.store.runs import RunWriter, run_dir

_TS = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)
_SEASON = "2026-27"
_HAS_RCLONE = shutil.which("rclone") is not None


pytestmark = pytest.mark.skipif(not _HAS_RCLONE, reason="rclone binary not installed")


def _succeed(
    data_dir: Path,
    run_id: str,
    *,
    run_type: str = "predict",
    build_id: str | None = None,
    plan_horizon: int | None = None,
    line: str | None = None,
) -> str:
    kwargs: dict[str, object] = {
        "kind": "official",
        "mode": "live",
        "run_type": run_type,
        "spec_hash": "a" * 64,
        "spec_label": "dc_layered_v4_native",
        "model_id": "deadbeef",
        "as_of": [AsOfDeadline(season=_SEASON, gw=13, deadline_utc=_TS)],
        "horizon": plan_horizon or 3,
        "code_sha": "abc1234",
        "code_dirty": False,
        "lockfile_hash": "b" * 64,
        "host": "test",
        "raw_manifest_hash": "c" * 64,
        "staged_digests": {},
        "started_at": _TS,
        "run_id": run_id,
        "checkpoint": "T-90m",
        "build_id": build_id,
    }
    if plan_horizon is not None:
        kwargs["plan_horizon"] = plan_horizon
    if line is not None:
        kwargs["line"] = line
        kwargs["squad_mode"] = "real" if line == "v4_h3" else "virtual"
    with RunWriter(data_dir, **kwargs) as writer:  # type: ignore[arg-type]
        if run_type == "predict":
            writer.write_parquet(
                "predictions/player_fixture.parquet",
                empty_predictions(),
            )
        else:
            writer.write_parquet(
                "plans/plan.parquet",
                pl.DataFrame({"player_key": [1], "action": ["start"]}),
            )
    return writer.run_id


def _pointer(
    data_dir: Path,
    *,
    gw: int,
    line: str,
    predict: str,
    plan: str,
    checkpoint: str = "T-90m",
    role: str = "primary",
) -> None:
    write_official_pointer(
        data_dir,
        season=_SEASON,
        gw=gw,
        line=line,
        checkpoint=checkpoint,
        pointer=OfficialPointer(
            predict_run_id=predict,
            plan_run_id=plan,
            role=role,  # type: ignore[arg-type]
            squad_mode="real" if role == "primary" else "virtual",
            written_at=_TS,
        ),
    )


def _seed_bucket(bucket: Path) -> tuple[str, str, str, str]:
    """Seed a local bucket stand-in; return predict/plan ids and build id."""
    players = pl.DataFrame(
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
    entry = write_table(bucket, "players", players)
    build_id = "cafe" * 8
    write_build_manifest(
        bucket,
        BuildManifest(
            build_id=build_id,
            created_at=_TS,
            staging_version=STAGING_VERSION,
            code_sha="abc",
            raw_manifest_hash="d" * 64,
            raw_cutoff=_TS,
            tables={
                "players": BuildTableEntry(
                    sha=entry.sha, rows=entry.rows, schema_hash=entry.schema_hash
                )
            },
        ),
    )
    predict = _succeed(bucket, "r_20261127T120000Z_pred01", build_id=build_id)
    plan13 = _succeed(
        bucket,
        "r_20261127T120000Z_plan13",
        run_type="plan",
        plan_horizon=3,
        line="v4_h3",
    )
    plan12 = _succeed(
        bucket,
        "r_20261127T120000Z_plan12",
        run_type="plan",
        plan_horizon=3,
        line="v4_h3",
    )
    plan_h5 = _succeed(
        bucket,
        "r_20261127T120000Z_planh5",
        run_type="plan",
        plan_horizon=5,
        line="v4_h5",
    )
    _pointer(bucket, gw=12, line="v4_h3", predict=predict, plan=plan12)
    _pointer(bucket, gw=13, line="v4_h3", predict=predict, plan=plan13)
    _pointer(
        bucket,
        gw=13,
        line="v4_h5",
        predict=predict,
        plan=plan_h5,
        role="candidate",
    )
    _pointer(
        bucket,
        gw=13,
        line="v4_h3",
        predict=predict,
        plan=plan13,
        checkpoint="T-24h",
    )
    return predict, plan13, plan_h5, build_id


@pytest.fixture
def bucket_local(tmp_path: Path) -> tuple[Path, Path, Remote]:
    bucket = tmp_path / "bucket"
    local = tmp_path / "data"
    local.mkdir()
    _seed_bucket(bucket)
    remote = Remote(target=str(bucket), kind="data", env={})
    return bucket, local, remote


def test_require_data_remote_names_fplt_r2_settings() -> None:
    with pytest.raises(RemoteError, match="FPLT_R2_ACCOUNT_ID"):
        require_data_remote(Settings())
    with pytest.raises(RemoteError, match="FPLT_R2_"):
        require_data_remote(Settings(data_bucket="fplt-data"))


def test_require_data_remote_builds_when_complete() -> None:
    remote = require_data_remote(
        Settings(
            data_bucket="fplt-data",
            r2_account_id="acct",
            r2_access_key_id="key",
            r2_secret_access_key=SecretStr("secret"),
        )
    )
    assert remote.target == "fplt:fplt-data"


def test_latest_resolves_highest_primary_gw(
    bucket_local: tuple[Path, Path, Remote],
) -> None:
    _bucket, local, remote = bucket_local
    root = mirror_official_index(local, remote)
    assert infer_official_season(root, None) == _SEASON
    assert resolve_latest_gw(root, season=_SEASON) == 13
    assert resolve_latest_gw(root, season=_SEASON, checkpoint="T-24h") == 13


def test_checkpoint_manual_and_latest_aliases(
    tmp_path: Path,
) -> None:
    """``--checkpoint manual|latest`` pick newest manual-* / any pointer (#364)."""
    bucket = tmp_path / "bucket"
    local = tmp_path / "data"
    local.mkdir()
    predict = _succeed(bucket, "r_20261127T120000Z_pred_m")
    plan_t90 = _succeed(
        bucket,
        "r_20261127T120000Z_plan_t90",
        run_type="plan",
        plan_horizon=3,
        line="v4_h3",
    )
    plan_manual_old = _succeed(
        bucket,
        "r_20261127T120000Z_plan_m1",
        run_type="plan",
        plan_horizon=3,
        line="v4_h3",
    )
    plan_manual_new = _succeed(
        bucket,
        "r_20261127T120000Z_plan_m2",
        run_type="plan",
        plan_horizon=3,
        line="v4_h3",
    )
    _pointer(
        bucket,
        gw=13,
        line="v4_h3",
        predict=predict,
        plan=plan_t90,
        checkpoint="T-90m",
    )
    # Older manual on gw13, newer manual stamp on gw13; T-90m also present.
    write_official_pointer(
        bucket,
        season=_SEASON,
        gw=13,
        line="v4_h3",
        checkpoint="manual-20261127T100000Z",
        pointer=OfficialPointer(
            predict_run_id=predict,
            plan_run_id=plan_manual_old,
            role="primary",
            squad_mode="real",
            written_at=datetime(2026, 11, 27, 10, 0, tzinfo=UTC),
        ),
    )
    write_official_pointer(
        bucket,
        season=_SEASON,
        gw=13,
        line="v4_h3",
        checkpoint="manual-20261128T073015Z",
        pointer=OfficialPointer(
            predict_run_id=predict,
            plan_run_id=plan_manual_new,
            role="primary",
            squad_mode="real",
            written_at=datetime(2026, 11, 28, 7, 30, 15, tzinfo=UTC),
        ),
    )
    # Older GW manual must not win over gw13.
    write_official_pointer(
        bucket,
        season=_SEASON,
        gw=12,
        line="v4_h3",
        checkpoint="manual-20261120T120000Z",
        pointer=OfficialPointer(
            predict_run_id=predict,
            plan_run_id=plan_manual_old,
            role="primary",
            squad_mode="real",
            written_at=datetime(2026, 11, 20, 12, 0, tzinfo=UTC),
        ),
    )
    remote = Remote(target=str(bucket), kind="data", env={})
    root = mirror_official_index(local, remote)

    gw_m, cp_m = resolve_checkpoint_target(root, season=_SEASON, checkpoint="manual")
    assert gw_m == 13
    assert cp_m == "manual-20261128T073015Z"
    assert resolve_latest_gw(root, season=_SEASON, checkpoint="manual") == 13

    gw_l, cp_l = resolve_checkpoint_target(root, season=_SEASON, checkpoint="latest")
    assert gw_l == 13
    assert cp_l == "manual-20261128T073015Z"

    # When T-* is newer than manuals, latest prefers it.
    write_official_pointer(
        bucket,
        season=_SEASON,
        gw=13,
        line="v4_h3",
        checkpoint="T-30m",
        pointer=OfficialPointer(
            predict_run_id=predict,
            plan_run_id=plan_t90,
            role="primary",
            squad_mode="real",
            written_at=datetime(2026, 11, 28, 8, 0, tzinfo=UTC),
        ),
    )
    root2 = mirror_official_index(local, remote)
    gw2, cp2 = resolve_checkpoint_target(root2, season=_SEASON, checkpoint="latest")
    assert (gw2, cp2) == (13, "T-30m")
    # manual alias still ignores T-*.
    assert resolve_checkpoint_target(root2, season=_SEASON, checkpoint="manual") == (
        13,
        "manual-20261128T073015Z",
    )


def test_latest_lists_availability_when_missing(
    tmp_path: Path,
) -> None:
    bucket = tmp_path / "bucket"
    local = tmp_path / "data"
    local.mkdir()
    predict = _succeed(bucket, "r_20261127T120000Z_pred02")
    plan = _succeed(
        bucket,
        "r_20261127T120000Z_plan02",
        run_type="plan",
        plan_horizon=3,
        line="v4_h3",
    )
    _pointer(
        bucket,
        gw=10,
        line="v4_h3",
        predict=predict,
        plan=plan,
        checkpoint="T-24h",
    )
    remote = Remote(target=str(bucket), kind="data", env={})
    root = mirror_official_index(local, remote)
    with pytest.raises(OfficialPullError, match="available:.*gw=10 checkpoint=T-24h"):
        resolve_latest_gw(root, season=_SEASON, checkpoint="T-90m")


def test_line_and_checkpoint_selection(
    bucket_local: tuple[Path, Path, Remote],
) -> None:
    _bucket, local, remote = bucket_local
    result = pull_official(local, remote, season=_SEASON, gw=13, line="v4_h5")
    assert result.plan_run_ids == ("r_20261127T120000Z_planh5",)
    again = pull_official(
        local, remote, season=_SEASON, gw=13, checkpoint="T-24h", line="v4_h3"
    )
    assert again.plan_run_ids == ("r_20261127T120000Z_plan13",)


def test_missing_run_errors_clearly(tmp_path: Path) -> None:
    bucket = tmp_path / "bucket"
    local = tmp_path / "data"
    local.mkdir()
    plan = _succeed(
        bucket,
        "r_20261127T120000Z_plan03",
        run_type="plan",
        plan_horizon=3,
        line="v4_h3",
    )
    # Pointer references a predict run that was never written to the bucket.
    base = bucket / "official" / f"season={_SEASON}" / "gw=14" / "checkpoint=T-90m"
    base.mkdir(parents=True)
    (base / "line=v4_h3.json").write_text(
        OfficialPointer(
            predict_run_id="r_20261127T120000Z_missing",
            plan_run_id=plan,
            role="primary",
            squad_mode="real",
            written_at=_TS,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    # Plan exists so require_run_succeeded isn't needed (wrote pointer by hand).
    remote = Remote(target=str(bucket), kind="data", env={})
    with pytest.raises(OfficialPullError, match="r_20261127T120000Z_missing"):
        pull_official(local, remote, season=_SEASON, gw=14)


def test_idempotent_second_pull_and_store_load(
    bucket_local: tuple[Path, Path, Remote],
) -> None:
    _bucket, local, remote = bucket_local
    first = pull_official(local, remote, season=_SEASON, gw=13, line="v4_h3")
    assert first.n_written > 0
    second = pull_official(local, remote, season=_SEASON, gw=13, line="v4_h3")
    assert second.n_written == 0
    store = Store(local)
    plan = store.official(season=_SEASON, gw=13, line="v4_h3")
    assert plan is not None
    assert plan.run_id == first.plan_run_ids[0]
    frame = store.load("player_fixture", run=first.predict_run_ids[0]).collect()
    assert isinstance(frame, pl.DataFrame)
    assert (local / "curated" / "builds" / f"build={first.build_ids[0]}.json").is_file()


def test_official_pull_includes_run_and_ops_logs(
    bucket_local: tuple[Path, Path, Remote],
) -> None:
    """``fplt data pull --official`` brings down run.log.jsonl and ops_log.jsonl."""
    from fplt.utils.logging import RUN_LOG_RELATIVE, ops_log_path, write_ops_log_once

    bucket, local, remote = bucket_local
    predict = "r_20261127T120000Z_pred01"
    assert (run_dir(bucket, predict) / RUN_LOG_RELATIVE).is_file()
    cp_dir = bucket / "official" / f"season={_SEASON}" / "gw=13" / "checkpoint=T-90m"
    write_ops_log_once(
        cp_dir,
        [
            '{"ts":"2026-11-28T03:04:09.123Z","level":"INFO",'
            '"logger":"fplt.test","message":"ops","checkpoint":"T-90m"}'
        ],
    )
    pull_official(local, remote, season=_SEASON, gw=13, line="v4_h3")
    assert (run_dir(local, predict) / RUN_LOG_RELATIVE).is_file()
    assert ops_log_path(
        local / "official" / f"season={_SEASON}" / "gw=13" / "checkpoint=T-90m"
    ).is_file()


def test_immutable_local_conflict_errors(
    bucket_local: tuple[Path, Path, Remote],
) -> None:
    _bucket, local, remote = bucket_local
    pull_official(local, remote, season=_SEASON, gw=13, line="v4_h3")
    predict = "r_20261127T120000Z_pred01"
    target = run_dir(local, predict) / "manifest.json"
    target.write_text("tampered", encoding="utf-8")
    with pytest.raises(RemoteError, match="immutable"):
        pull_official(local, remote, season=_SEASON, gw=13, line="v4_h3")


def test_cli_official_latest(
    bucket_local: tuple[Path, Path, Remote],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _bucket, local, remote = bucket_local
    monkeypatch.setenv("FPLT_DATA_DIR", str(local))
    get_settings.cache_clear()
    monkeypatch.setattr(
        "fplt.store.pull_official.require_data_remote", lambda settings=None: remote
    )
    try:
        main(
            [
                "data",
                "pull",
                "--official",
                "latest",
                "--season",
                _SEASON,
                "--data-dir",
                str(local),
            ]
        )
    finally:
        get_settings.cache_clear()
    out = capsys.readouterr().out
    assert "gw=13" in out
    assert "r_20261127T120000Z_plan13" in out
