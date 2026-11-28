"""Cover exception branches touched by the B/BLE/TRY400 lint ratchet (#311)."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import polars as pl
import pytest

from fplt.backtest.compare import CompareRun
from fplt.backtest.promote_eval import (
    EvalSpec,
    PreflightResult,
    PromoteEvalPlan,
    _actuals_for_equivalence,
    _check_leak,
    _folds_from_compare,
    _member_prediction_keys,
    _write_audit,
)
from fplt.data.fpl_client import FPLAPIError, FPLClient
from fplt.data.store import DataStore
from fplt.models.spec import ModelSpec, _package_version
from fplt.schemas.manifests import AsOfDeadline
from fplt.store.experiments import arm_reuse_key, find_reusable_arm
from fplt.store.runs import fplt_mc_version
from tests.unit.test_promote_eval import _base_payload, _fake_compare

_TS = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
_API = "https://fantasy.premierleague.com/api"


def _plan(**overrides: object) -> PromoteEvalPlan:
    values: dict[str, object] = {
        "eval_name": "demo",
        "kind": "config",
        "lever": "hyperparam",
        "control": "c",
        "treatment": "t",
        "seasons": ["2024-25"],
        "from_gw": 1,
        "to_gw": 1,
        "seasons_back": 1,
        "modes": ["walk_forward"],
        "suites": EvalSpec.model_validate(_base_payload()).suites,
        "commands": [],
        "diff": [],
        "notes": "",
        "window_note": "",
        "warnings": [],
        "dry_run": True,
    }
    values.update(overrides)
    return PromoteEvalPlan(**values)  # type: ignore[arg-type]


def test_package_version_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_name: str) -> str:
        raise PackageNotFoundError("fplt")

    monkeypatch.setattr("fplt.models.spec.version", boom)
    assert _package_version() == "0.0.0"


def test_fplt_mc_version_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(_name: str) -> str:
        raise PackageNotFoundError("fplt-mc")

    monkeypatch.setattr("fplt.store.runs.version", boom)
    assert fplt_mc_version() is None


def test_find_reusable_arm_skips_unreadable_manifest(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "run=r_bad"
    run.mkdir(parents=True)
    (run / "_SUCCESS").write_text("", encoding="utf-8")
    (run / "manifest.json").write_text("{not-json", encoding="utf-8")
    key = arm_reuse_key(
        spec_hash="a" * 64,
        code_sha="abc1234",
        lockfile_hash="b" * 64,
        build_id=None,
        deadlines=[AsOfDeadline(season="2024-25", gw=1, deadline_utc=_TS)],
        horizon=1,
        seeds={"base": 1},
    )
    assert find_reusable_arm(tmp_path, key) is None


def test_fpl_client_logs_transport_and_non_json_errors() -> None:
    def raise_connect(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    http = httpx.Client(transport=httpx.MockTransport(raise_connect), base_url=_API)
    api = FPLClient(http, base_url=_API)
    try:
        with pytest.raises(FPLAPIError, match="request failed"):
            api.entry(1)
    finally:
        api.close()
        http.close()

    def not_json(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    http = httpx.Client(transport=httpx.MockTransport(not_json), base_url=_API)
    api = FPLClient(http, base_url=_API)
    try:
        with pytest.raises(FPLAPIError, match="non-JSON"):
            api.entry(1)
    finally:
        api.close()
        http.close()


def test_fpl_client_post_logs_transport_error() -> None:
    def raise_connect(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    http = httpx.Client(transport=httpx.MockTransport(raise_connect), base_url=_API)
    api = FPLClient(http, base_url=_API)
    try:
        with pytest.raises(FPLAPIError, match="request failed"):
            api._post_json("/x/", {}, empty_ok=True)
    finally:
        api.close()
        http.close()


def test_actuals_for_equivalence_when_store_read_fails(tmp_path: Path) -> None:
    store = MagicMock(spec=DataStore)
    store.read.side_effect = FileNotFoundError("missing")
    with patch("fplt.backtest.promote_eval.DataStore", return_value=store):
        assert _actuals_for_equivalence(tmp_path, _plan()) is None


def _empty_scored() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": pl.Series([], dtype=pl.Utf8),
            "gw": pl.Series([], dtype=pl.Int64),
            "player_key": pl.Series([], dtype=pl.Int64),
            "fixture_id": pl.Series([], dtype=pl.Int64),
        }
    )


def _predict_run_parquet(data_dir: Path, member: ModelSpec) -> Path:
    """Write a succeeded predict run and return its player_fixture path."""
    from fplt.store.catalog import scan_succeeded_runs
    from fplt.store.runs import run_dir, write_predict_run
    from fplt.types import GWKey

    write_predict_run(
        data_dir,
        spec=member,
        as_of=GWKey("2025-26", 1),
        deadline_utc=_TS,
        horizon=1,
        frame=pl.DataFrame(
            {
                "model_id": [member.model_id],
                "run_kind": ["live"],
                "as_of_season": ["2025-26"],
                "as_of_gw": [1],
                "target_season": ["2025-26"],
                "target_gw": [1],
                "horizon": [0],
                "player_key": [1],
                "position": ["MID"],
                "fixture_id": [10],
                "xpts": [2.5],
                "xpts_sd": [0.5],
                "p_played": [None],
                "p_60": [None],
                "p_goal": [None],
                "p_assist": [None],
                "p_cs": [None],
                "p_haul": [None],
                "actual_pts": [None],
                "actual_minutes": [None],
                "actual_goals": [None],
                "actual_assists": [None],
                "actual_cs": [None],
            }
        ),
    )
    run_id = scan_succeeded_runs(data_dir)["run_id"][0]
    return run_dir(data_dir, str(run_id)) / "predictions" / "player_fixture.parquet"


def test_member_prediction_keys_skips_corrupt_parquet(tmp_path: Path) -> None:
    member = ModelSpec(name="m", kind="rolling_mean", params={"window": 5})
    path = _predict_run_parquet(tmp_path, member)
    path.write_bytes(b"not-parquet")
    keys, source = _member_prediction_keys(tmp_path, member, _empty_scored())
    assert keys.is_empty()
    assert source == "universe"


def test_member_prediction_keys_skips_missing_and_noxpts(tmp_path: Path) -> None:
    """Missing parquet and parquet without xpts are skipped (runs/ layout)."""
    member = ModelSpec(name="m", kind="rolling_mean", params={"window": 5})
    path = _predict_run_parquet(tmp_path, member)
    path.unlink()
    keys, source = _member_prediction_keys(tmp_path, member, _empty_scored())
    assert keys.is_empty()
    assert source == "universe"

    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"player_key": [1], "fixture_id": [10]}).write_parquet(path)
    keys, source = _member_prediction_keys(tmp_path, member, _empty_scored())
    assert keys.is_empty()
    assert source == "universe"


def test_member_prediction_keys_reads_season_gw_columns(tmp_path: Path) -> None:
    """Predict-run frames with season/gw columns contribute coverage keys."""
    member = ModelSpec(name="m", kind="rolling_mean", params={"window": 5})
    path = _predict_run_parquet(tmp_path, member)
    pl.DataFrame(
        {
            "season": ["2025-26"],
            "gw": [1],
            "player_key": [7],
            "fixture_id": [11],
            "xpts": [3.0],
        }
    ).write_parquet(path)
    keys, source = _member_prediction_keys(tmp_path, member, _empty_scored())
    assert source == "predictions"
    assert keys.to_dicts() == [
        {"season": "2025-26", "gw": 1, "player_key": 7, "fixture_id": 11}
    ]


def test_check_leak_skips_corrupt_feature_parquet(tmp_path: Path) -> None:
    payload = _base_payload(
        kind="feature",
        change={
            "lever": "feature_version",
            "orientation": "opt_in",
            "features": ["demo_v1"],
            "diff": ["features: +demo_v1"],
        },
    )
    spec = EvalSpec.model_validate(payload)
    feature = tmp_path / "features" / "demo_v1"
    feature.mkdir(parents=True)
    (feature / "part.parquet").write_bytes(b"not-parquet")
    store = MagicMock(spec=DataStore)
    fixtures = pl.DataFrame(
        {
            "season": ["2024-25"],
            "gw": [1],
            "fixture_id": [1],
        }
    )
    store.table.return_value = pl.LazyFrame(
        {
            "season": ["2024-25"],
            "gw": [1],
            "deadline_time": [datetime(2024, 8, 1, tzinfo=UTC)],
            "available_at": [datetime(2024, 7, 1, tzinfo=UTC)],
        }
    )
    with patch(
        "fplt.backtest.promote_eval._target_fixtures",
        return_value=fixtures,
    ):
        assert _check_leak(spec, _plan(kind="feature"), tmp_path, store) is None


def test_folds_from_compare_when_ledger_missing(tmp_path: Path) -> None:
    spec = ModelSpec(name="m", kind="rolling_mean", params={"window": 5})
    compare = _fake_compare(
        summary=pl.DataFrame(),
        runs=(CompareRun(model_name="m", mode="walk_forward", spec=spec),),
    )
    with patch(
        "fplt.backtest.promote_eval.read_predictions",
        side_effect=OSError("gone"),
    ):
        assert _folds_from_compare(compare, tmp_path) == []


def test_write_audit_package_version_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_name: str) -> str:
        raise PackageNotFoundError("fplt")

    monkeypatch.setattr("fplt.backtest.promote_eval.package_version", boom)
    spec = EvalSpec.model_validate(_base_payload())
    control = ModelSpec(name="c", kind="rolling_mean", params={"window": 5})
    treatment = ModelSpec(name="t", kind="rolling_mean", params={"window": 3})
    preflight = PreflightResult(
        ok=True,
        git_sha="abc",
        control_spec=control,
        treatment_spec=treatment,
    )
    out = tmp_path / "out"
    out.mkdir()
    with patch("fplt.backtest.promote_eval._hash_staged", return_value={}):
        _write_audit(
            out,
            spec,
            _plan(),
            preflight,
            data_dir=tmp_path,
            argv=["promote-eval"],
            wall_clock={},
            peak_rss={},
            folds=[],
            verdict="reject",
        )
    payload = (out / "audit.json").read_text(encoding="utf-8")
    assert '"fplt_version": "0.0.0"' in payload
