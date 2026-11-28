"""End-to-end weekly ops pipeline on a tiny fixture world (#307).

Runs the real ``run_ops_tick`` path at T−90m with only the network / R2
replaced by a local bucket directory. Asserts history guard, official
predict/plan tables (including ``points_pmf``), pointers, timing, pull, and
this-week analysis helpers. Marked slow; uses Python MC (``dc_layered_v4_e2e``).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest
from pydantic import SecretStr

from fplt.analysis.this_week import (
    default_official_selection,
    plan_panel,
    players_table,
)
from fplt.automation.checkpoints import (
    write_notified_marker,
    write_plan_checkpoint_marker,
)
from fplt.automation.history_guard import HistoryGuardError
from fplt.automation.official_slate import run_official_checkpoint
from fplt.automation.ops_tick import run_ops_tick
from fplt.data.remote import Remote
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.plan import plan_parquet
from fplt.settings import Settings, get_settings
from fplt.store.api import Store
from fplt.store.official import official_checkpoint_dir
from fplt.store.pull_official import (
    mirror_official_index,
    pull_official,
    resolve_latest_gw,
)
from fplt.types import GWKey
from fplt.ui.data import DashboardStore
from fplt.utils.timing import ops_timing_path
from tests.builders_e2e import (
    CURRENT_SEASON,
    E2E_LINE,
    E2E_MODEL,
    NOW_T90M,
    e2e_production_ops,
    seed_e2e_bucket,
)

_HAS_RCLONE = shutil.which("rclone") is not None

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _HAS_RCLONE, reason="rclone binary not installed"),
]


def _patch_local_remote(monkeypatch: pytest.MonkeyPatch, remote: Remote) -> None:
    """Point every ``Remote.from_settings`` call at the local test bucket."""

    def _from_settings(_settings: Settings, bucket: str | None = None) -> Remote:
        del bucket
        return remote

    monkeypatch.setattr(
        "fplt.data.remote.Remote.from_settings",
        classmethod(lambda cls, s, bucket=None: _from_settings(s, bucket)),
    )


def _seed_catchup_plan_markers(data_dir: Path) -> None:
    """Mark every checked GW as planned so catch_up finds nothing pending.

    Prior vaastav seasons are history-guard fodder (2 clubs) and cannot satisfy
    the 15-man MILP; the T−90m official path under test covers fit → predict →
    plan for the current GW2. Catch_up still runs for real with an empty list.
    """
    from tests.builders_e2e import PRIOR_SEASONS

    spec = load_model_specs()[E2E_MODEL]
    keys = [(CURRENT_SEASON, 1), *((season, 1) for season in PRIOR_SEASONS)]
    for season, gw in keys:
        path = plan_parquet(data_dir, spec.model_id, season, gw, 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"player_key": [1], "step": [0]}).write_parquet(path)


def _configure_tick(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_vaastav: bool = True,
    min_prior: int = 3,
) -> tuple[Path, Path, Remote, Settings]:
    bucket = tmp_path / "bucket"
    data = tmp_path / "data"
    data.mkdir()
    seed_e2e_bucket(bucket)
    if not with_vaastav:
        shutil.rmtree(bucket / "raw" / "vaastav")
    remote = Remote(target=str(bucket), kind="data", env={})
    _patch_local_remote(monkeypatch, remote)
    monkeypatch.setenv("FPLT_DATA_DIR", str(data))
    monkeypatch.setenv("FPLT_DATA_BUCKET", "fplt-data")
    monkeypatch.setenv("FPLT_R2_ACCOUNT_ID", "test")
    monkeypatch.setenv("FPLT_R2_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("FPLT_R2_SECRET_ACCESS_KEY", "test")
    monkeypatch.delenv("FPLT_ENTRY_ID", raising=False)
    monkeypatch.setenv("FPLT_MC_BACKEND", "python")
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()
    ops = e2e_production_ops(min_prior_seasons=min_prior)
    monkeypatch.setattr("fplt.automation.ops_tick.load_production_ops", lambda: ops)
    # Force chip_arm=off: run_ops_tick does not forward the default override.
    original = run_official_checkpoint

    def _official(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("chip_arm", "off")
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("fplt.automation.ops_tick.run_official_checkpoint", _official)
    # Benchmark archive pulls OpenFPL (network); e2e replaces network only.
    monkeypatch.setattr(
        "fplt.automation.cycle.archive_if_missing",
        lambda *a, **k: None,
    )
    settings = Settings(
        data_dir=data,
        data_bucket="fplt-data",
        r2_account_id="test",
        r2_access_key_id=SecretStr("test"),
        r2_secret_access_key=SecretStr("test"),
    )
    key = GWKey(CURRENT_SEASON, 2)
    write_plan_checkpoint_marker(data, key, "T-24h", primary_line=E2E_LINE)
    write_plan_checkpoint_marker(data, key, "T-6h", primary_line=E2E_LINE)
    write_notified_marker(data, key)
    _seed_catchup_plan_markers(data)
    return bucket, data, remote, settings


def test_ops_pipeline_e2e_t90m(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Real feeds → history guard → official slate → pull → this-week."""
    _bucket, data, remote, settings = _configure_tick(tmp_path, monkeypatch)

    result = run_ops_tick(
        data,
        now=NOW_T90M,
        source="bucket",
        settings=settings,
        notify=False,
    )
    assert "T-90m" in result.checkpoints_run

    store = Store(data)
    plan_manifest = store.official(
        season=CURRENT_SEASON, gw=2, line=E2E_LINE, checkpoint="T-90m"
    )
    assert plan_manifest is not None, "official plan pointer missing"
    assert plan_manifest.kind == "official"
    assert plan_manifest.run_type == "plan"

    plan = store.load("plans", run=plan_manifest).collect()
    assert not plan.is_empty(), "plan.parquet empty"
    assert "player_key" in plan.columns
    assert plan.filter(pl.col("in_xi"))["player_key"].n_unique() == 11

    # Predict run: parent of the plan, with components + points PMFs.
    predict_id = plan_manifest.parent_run_id
    assert predict_id, "plan missing parent_run_id (predict)"
    player_fixture = store.load("player_fixture", run=predict_id).collect()
    assert not player_fixture.is_empty(), "player_fixture empty"
    # Official predict writes minutes / scoreline / involvement (§3.1); not other.
    for component in ("minutes", "scoreline", "involvement"):
        frame = store.load(component, run=predict_id).collect()
        assert not frame.is_empty(), f"component table {component!r} empty"
    points_pmf = store.load("points_pmf", run=predict_id).collect()
    assert not points_pmf.is_empty(), (
        "points_pmf empty — official predict must write distributions "
        "(breaking this wiring should fail here)"
    )
    points_pmf_gw = store.load("points_pmf_gw", run=predict_id).collect()
    assert not points_pmf_gw.is_empty(), "points_pmf_gw empty"

    cp_dir = official_checkpoint_dir(
        data, season=CURRENT_SEASON, gw=2, checkpoint="T-90m"
    )
    timing = ops_timing_path(cp_dir)
    assert timing.is_file(), f"ops_timing.json missing at {timing}"

    # Pull into a second dir and exercise this-week helpers.
    pull_dir = tmp_path / "pulled"
    pull_dir.mkdir()
    mirror_official_index(pull_dir, remote)
    assert (
        resolve_latest_gw(
            pull_dir / "official",
            season=CURRENT_SEASON,
            primary_line=E2E_LINE,
        )
        == 2
    )
    pulled = pull_official(
        pull_dir,
        remote,
        season=CURRENT_SEASON,
        gw=2,
        line=E2E_LINE,
    )
    assert pulled.predict_run_ids
    assert pulled.plan_run_ids

    dash = DashboardStore(pull_dir)
    pointers = dash.official_lines()
    assert not pointers.is_empty()
    selection = default_official_selection(pointers)
    assert selection is not None
    assert selection.gw == 2
    assert selection.checkpoint == "T-90m"

    pulled_store = Store(pull_dir)
    pulled_plan = pulled_store.load("plans", run=pulled.plan_run_ids[0]).collect()
    staged = DataStore(data)
    players = staged.read("players")
    teams = staged.read("teams")
    status = staged.read("player_status")
    panel = plan_panel(pulled_plan, players=players, teams=teams, status=status)
    assert not panel.xi.is_empty(), "this-week plan panel XI empty"

    pulled_pred = pulled_store.load(
        "player_fixture", run=pulled.predict_run_ids[0]
    ).collect()
    table = players_table(
        pulled_pred,
        players=players,
        teams=teams,
        status=status,
        as_of_gw=2,
        horizon=1,
    )
    assert not table.is_empty(), "this-week players_table empty"


def test_ops_pipeline_e2e_history_guard_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing vaastav priors makes the history guard fail loud."""
    _bucket, data, _remote, settings = _configure_tick(
        tmp_path, monkeypatch, with_vaastav=False
    )
    with pytest.raises(HistoryGuardError, match="prior seasons"):
        run_ops_tick(
            data,
            now=NOW_T90M,
            source="bucket",
            settings=settings,
            notify=False,
        )


def test_ops_pipeline_e2e_deterministic_xpts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two ticks with the same seeds agree on player_fixture xpts."""
    results: list[pl.DataFrame] = []
    for index in range(2):
        run_root = tmp_path / f"run{index}"
        run_root.mkdir()
        _bucket, data, _remote, settings = _configure_tick(run_root, monkeypatch)
        run_ops_tick(
            data,
            now=NOW_T90M,
            source="bucket",
            settings=settings,
            notify=False,
        )
        store = Store(data)
        plan_manifest = store.official(season=CURRENT_SEASON, gw=2, line=E2E_LINE)
        assert plan_manifest is not None
        assert plan_manifest.parent_run_id
        frame = (
            store.load("player_fixture", run=plan_manifest.parent_run_id)
            .collect()
            .select("player_key", "fixture_id", "xpts")
            .sort(["player_key", "fixture_id"])
        )
        results.append(frame)
    assert results[0].equals(results[1])
