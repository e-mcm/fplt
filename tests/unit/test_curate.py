"""Incremental curate + weekly full-rebuild parity (#345)."""

from __future__ import annotations

import gzip
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fplt.automation.curate import (
    CurateParityError,
    capture_dirs_after,
    curated_latest_path,
    load_curated_latest,
    run_curate,
    write_curated_latest,
)
from fplt.data.quarantine import append_quarantine
from fplt.data.raw import ingest_raw, iter_event_live, iter_snapshots
from fplt.data.remote import Remote, RemoteObject
from fplt.data.store import DataStore
from fplt.schemas.manifests import CuratedLatestPointer
from fplt.store.builds import (
    build_from_staged,
    clear_staged,
    hydrate_staged_from_build,
    load_build_manifest,
)

_BOOTSTRAP = {
    "teams": [
        {"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS"},
        {"id": 2, "code": 7, "name": "Spurs", "short_name": "TOT"},
    ],
    "elements": [
        {
            "id": 10,
            "code": 100,
            "team": 1,
            "element_type": 3,
            "web_name": "Saka",
            "status": "a",
            "news": "",
            "now_cost": 100,
            "ep_next": "6.0",
            "chance_of_playing_next_round": 100,
        }
    ],
    "events": [
        {
            "id": 1,
            "deadline_time": "2024-08-16T17:30:00Z",
            "finished": False,
            "data_checked": False,
        }
    ],
}
_FIXTURES = [
    {
        "id": 50,
        "event": 1,
        "team_h": 1,
        "team_a": 2,
        "kickoff_time": "2024-08-17T14:00:00Z",
        "finished": False,
        "team_h_score": None,
        "team_a_score": None,
    }
]
_EVENT_LIVE = {
    "elements": [
        {
            "id": 10,
            "stats": {
                "minutes": 90,
                "total_points": 6,
                "goals_scored": 0,
                "assists": 1,
                "clean_sheets": 1,
                "goals_conceded": 0,
                "own_goals": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "saves": 0,
                "bonus": 0,
                "bps": 20,
                "influence": "0.0",
                "creativity": "0.0",
                "threat": "0.0",
                "ict_index": "0.0",
                "expected_goals": "0.10",
                "expected_assists": "0.20",
                "expected_goal_involvements": "0.30",
                "expected_goals_conceded": "0.40",
                "defensive_contribution": 0,
            },
            "explain": [{"fixture": 50}],
        }
    ]
}


def _stamp(i: int) -> str:
    return f"2024-08-{10 + i:02d}T12-00-00Z"


def _write_bootstrap(raw: Path, *, index: int) -> None:
    stamp = _stamp(index)
    dest = raw / "fpl_api" / "season=2024-25" / f"captured_at={stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(_BOOTSTRAP))
    payload["elements"][0]["now_cost"] = 100 + index
    with gzip.open(dest / "bootstrap_static.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    with gzip.open(dest / "fixtures.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(_FIXTURES, handle)


def _write_event_live(raw: Path, *, index: int) -> None:
    stamp = _stamp(index)
    dest = (
        raw
        / "fpl_api"
        / "season=2024-25"
        / "event_live"
        / "gw=1"
        / f"captured_at={stamp}"
    )
    dest.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest / "event_live.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(_EVENT_LIVE, handle)


def _table_shas(data_dir: Path, build_id: str) -> dict[str, str]:
    manifest = load_build_manifest(data_dir, build_id)
    return {name: entry.sha for name, entry in sorted(manifest.tables.items())}


def test_curated_latest_pointer_round_trip(tmp_path: Path) -> None:
    pointer = CuratedLatestPointer(
        build_id="abcd" * 16,
        created_at=datetime(2026, 11, 28, 12, 0, tzinfo=UTC),
        raw_cutoff=datetime(2026, 11, 28, 11, 0, tzinfo=UTC),
        code_sha="deadbeef",
    )
    write_curated_latest(tmp_path, pointer)
    loaded = load_curated_latest(tmp_path)
    assert loaded == pointer
    assert curated_latest_path(tmp_path).is_file()


def test_capture_dirs_after_filters_cutoff_and_dot_paths() -> None:
    after = datetime(2024, 8, 11, 12, 0, tzinfo=UTC)
    objects = [
        RemoteObject(
            path="fpl_api/season=2024-25/captured_at=2024-08-10T12-00-00Z/b.json.gz",
            size=1,
        ),
        RemoteObject(
            path="fpl_api/season=2024-25/captured_at=2024-08-12T12-00-00Z/b.json.gz",
            size=1,
        ),
        RemoteObject(
            path=".cache/captured_at=2024-08-13T12-00-00Z/x.json.gz",
            size=1,
        ),
        RemoteObject(
            path=(
                "fpl_api/season=2024-25/event_live/gw=1/"
                "captured_at=2024-08-14T12-00-00Z/event_live.json.gz"
            ),
            size=1,
        ),
    ]
    dirs = capture_dirs_after(objects, after)
    assert dirs == [
        "fpl_api/season=2024-25/captured_at=2024-08-12T12-00-00Z",
        "fpl_api/season=2024-25/event_live/gw=1/captured_at=2024-08-14T12-00-00Z",
    ]


def test_event_live_staged_once_not_per_bootstrap(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for i in range(3):
        _write_bootstrap(raw, index=i)
    _write_event_live(raw, index=2)
    store = DataStore(tmp_path)
    count = ingest_raw(store, raw)
    # 3 bootstraps + 1 event_live (not 3× event_live).
    assert count == 4
    assert len(iter_snapshots(raw)) == 3
    assert len(iter_event_live(raw)) == 1
    stats = store.read("player_fixture_stats")
    assert stats.height == 1


def _build_prefix(
    root: Path, *, n_bootstraps: int, event_live_indices: list[int]
) -> None:
    raw = root / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    for i in range(n_bootstraps):
        _write_bootstrap(raw, index=i)
    for i in event_live_indices:
        if i < n_bootstraps:
            _write_event_live(raw, index=i)


def test_incremental_vs_full_same_table_shas(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for i in range(4):
        _write_bootstrap(raw, index=i)
    _write_event_live(raw, index=1)
    _write_event_live(raw, index=3)

    full_root = tmp_path / "full"
    shutil.copytree(raw, full_root / "raw")
    ingest_raw(DataStore(full_root), full_root / "raw")
    full = build_from_staged(full_root, code_sha="full")

    # Incremental: first two bootstraps (+ event_live index 1), then remainder.
    inc_root = tmp_path / "inc"
    _build_prefix(inc_root, n_bootstraps=2, event_live_indices=[1])
    ingest_raw(DataStore(inc_root), inc_root / "raw")
    first = build_from_staged(inc_root, code_sha="inc1")

    if (inc_root / "raw").exists():
        shutil.rmtree(inc_root / "raw")
    shutil.copytree(raw, inc_root / "raw")
    clear_staged(inc_root)
    hydrate_staged_from_build(inc_root, first.build_id)
    cutoff = datetime(2024, 8, 11, 12, 0, tzinfo=UTC)
    snaps = [s for s in iter_snapshots(inc_root / "raw") if s.captured_at > cutoff]
    ingest_raw(
        DataStore(inc_root),
        inc_root / "raw",
        snapshots=snaps,
        event_live_after=cutoff,
    )
    second = build_from_staged(inc_root, code_sha="inc2")

    assert _table_shas(inc_root, second.build_id) == _table_shas(
        full_root, full.build_id
    )
    assert second.build_id == full.build_id


@given(st.integers(min_value=1, max_value=4))
@settings(max_examples=8, deadline=None)
def test_hypothesis_split_point_parity(split: int) -> None:
    """Incremental vs full at a random split of the capture sequence."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "raw"
        for i in range(5):
            _write_bootstrap(raw, index=i)
        _write_event_live(raw, index=2)

        full_root = root / "full"
        shutil.copytree(raw, full_root / "raw")
        ingest_raw(DataStore(full_root), full_root / "raw")
        full = build_from_staged(full_root, code_sha="full")

        cutoff = datetime(2024, 8, 10 + split - 1, 12, 0, tzinfo=UTC)
        inc_root = root / "inc"
        live = [2] if split > 2 else []
        _build_prefix(inc_root, n_bootstraps=split, event_live_indices=live)
        ingest_raw(DataStore(inc_root), inc_root / "raw")
        first = build_from_staged(inc_root, code_sha="p1")

        if (inc_root / "raw").exists():
            shutil.rmtree(inc_root / "raw")
        shutil.copytree(raw, inc_root / "raw")
        clear_staged(inc_root)
        hydrate_staged_from_build(inc_root, first.build_id)
        snaps = [s for s in iter_snapshots(inc_root / "raw") if s.captured_at > cutoff]
        ingest_raw(
            DataStore(inc_root),
            inc_root / "raw",
            snapshots=snaps,
            event_live_after=cutoff,
        )
        second = build_from_staged(inc_root, code_sha="p2")
        assert _table_shas(inc_root, second.build_id) == _table_shas(
            full_root, full.build_id
        )


def test_no_new_raw_leaves_pointer_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pointer = CuratedLatestPointer(
        build_id="cafe" * 16,
        created_at=datetime(2026, 11, 28, tzinfo=UTC),
        raw_cutoff=datetime(2026, 11, 28, 12, 0, tzinfo=UTC),
        code_sha="abc",
    )
    write_curated_latest(tmp_path, pointer)
    # Ensure build manifest exists for hydrate path (not reached when no new).
    remote = Remote(target=str(tmp_path / "bucket"), kind="data", env={})
    (tmp_path / "bucket" / "raw").mkdir(parents=True)

    def fake_lsjson(_remote: Remote, _prefix: str) -> list[RemoteObject]:
        return [
            RemoteObject(
                path="fpl_api/season=2024-25/captured_at=2026-11-27T00-00-00Z/b.json.gz",
                size=1,
            )
        ]

    monkeypatch.setattr("fplt.automation.curate.lsjson", fake_lsjson)
    monkeypatch.setattr("fplt.automation.curate.pull_build", lambda *a, **k: 0)
    monkeypatch.setattr("fplt.automation.curate.pull_quarantine", lambda *a, **k: False)

    result = run_curate(
        tmp_path,
        data_remote=remote,
        push_remote=remote,
        notify_parity=False,
    )
    assert result.skipped_no_new is True
    assert result.pointer_updated is False
    assert load_curated_latest(tmp_path) == pointer


def test_pull_capture_dirs_only_requests_newer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.automation import curate as curate_mod

    requested: list[str] = []

    def fake_pull(_remote: Remote, prefix: str, dest: Path) -> list[Path]:
        requested.append(prefix)
        dest.mkdir(parents=True, exist_ok=True)
        return []

    monkeypatch.setattr(curate_mod, "pull", fake_pull)
    remote = Remote(target=str(tmp_path / "bucket"), kind="data", env={})
    dirs = [
        "fpl_api/season=2024-25/captured_at=2024-08-12T12-00-00Z",
        "fpl_api/season=2024-25/event_live/gw=1/captured_at=2024-08-14T12-00-00Z",
    ]
    curate_mod.pull_capture_dirs(tmp_path, remote, dirs)
    assert requested == [f"raw/{d}" for d in dirs]


def test_quarantined_skipped_in_ingest(tmp_path: Path) -> None:
    import polars as pl

    raw = tmp_path / "raw"
    _write_bootstrap(raw, index=0)
    _write_bootstrap(raw, index=1)
    bad = "fpl_api/season=2024-25/captured_at=2024-08-11T12-00-00Z"
    append_quarantine(
        tmp_path,
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "source": ["fpl_api"],
                "capture_path": [bad],
                "captured_at": [datetime(2024, 8, 11, 12, 0, tzinfo=UTC)],
                "gate": ["player_count"],
                "detail": ["test"],
                "previous_capture_path": [None],
                "available_at": [datetime(2024, 8, 11, 12, 0, tzinfo=UTC)],
                "schema_version": [1],
            }
        ),
    )
    store = DataStore(tmp_path)
    count = ingest_raw(store, raw)
    assert count == 1
    status = store.read("player_status")
    assert status["price"].to_list() == [100]


def test_snapshot_workflow_has_curate_job() -> None:
    import yaml

    doc = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / ".github/workflows/snapshot.yml"
        ).read_text(encoding="utf-8")
    )
    curate = doc["jobs"]["curate"]
    assert curate["needs"] == "capture"
    assert curate["environment"] == "r2"
    assert curate["concurrency"]["group"] == "curate"
    assert curate["concurrency"]["cancel-in-progress"] is False
    run = next(s for s in curate["steps"] if s.get("name") == "Curate")
    assert "fplt data curate" in run["run"]


def test_curate_full_workflow_schedule_and_parity_cli() -> None:
    import yaml

    doc = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2] / ".github/workflows/curate-full.yml"
        ).read_text(encoding="utf-8")
    )
    assert doc["concurrency"]["group"] == "curate"
    job = doc["jobs"]["curate_full"]
    assert job["environment"] == "r2"
    run = next(s for s in job["steps"] if s.get("name") == "Full curate + parity")
    assert "fplt data curate --full" in run["run"]


def test_full_parity_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Prior pointer names a different build (different table content).
    prior = tmp_path / "prior"
    _write_bootstrap(prior / "raw", index=0)
    ingest_raw(DataStore(prior), prior / "raw")
    prior_build = build_from_staged(prior, code_sha="prior")

    root = tmp_path / "full"
    _write_bootstrap(root / "raw", index=1)
    shutil.copytree(prior / "curated", root / "curated")
    write_curated_latest(
        root,
        CuratedLatestPointer(
            build_id=prior_build.build_id,
            created_at=prior_build.created_at,
            raw_cutoff=datetime(2024, 8, 10, 12, 0, tzinfo=UTC),
            code_sha="prior",
        ),
    )

    monkeypatch.setattr("fplt.automation.curate.fetch_raw", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.curate.pull_quarantine", lambda *a, **k: False)

    with pytest.raises(CurateParityError, match="parity failed"):
        run_curate(
            root,
            full=True,
            source="branch",
            notify_parity=False,
            data_remote=None,
        )
