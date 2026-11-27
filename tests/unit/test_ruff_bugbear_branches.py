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


def test_member_prediction_keys_skips_corrupt_parquet(tmp_path: Path) -> None:
    member = ModelSpec(name="m", kind="rolling_mean", params={"window": 5})
    root = tmp_path / "predictions" / member.model_id / "as_of=x"
    root.mkdir(parents=True)
    (root / "part.parquet").write_bytes(b"not-parquet")
    scored = pl.DataFrame(
        {
            "season": pl.Series([], dtype=pl.Utf8),
            "gw": pl.Series([], dtype=pl.Int64),
            "player_key": pl.Series([], dtype=pl.Int64),
            "fixture_id": pl.Series([], dtype=pl.Int64),
        }
    )
    keys, source = _member_prediction_keys(tmp_path, member, scored)
    assert keys.is_empty()
    assert isinstance(source, str)


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
