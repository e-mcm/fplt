"""Content-addressed produce_ledger cache (P2.5)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from fplt.backtest.compare import compare_models
from fplt.backtest.ledger import read_predictions
from fplt.backtest.produce_ledger import (
    LedgerCacheKey,
    StaleLedgerCacheError,
    build_ledger_cache_key,
    hash_staged_data,
    produce_ledger,
    read_ledger_cache,
    resolved_spec_sha256,
    write_ledger_cache,
)
from fplt.backtest.promote_eval import execute_promote_eval, load_eval_spec
from fplt.data.store import DataStore
from fplt.models.spec import ModelSpec, TrainWindow

_PRIOR = "2023-24"
_TEST = "2024-25"
_START = datetime(2024, 7, 1, tzinfo=UTC)
_GW1 = datetime(2024, 8, 16, 17, tzinfo=UTC)
_GW1_RESULT = datetime(2024, 8, 17, 18, tzinfo=UTC)
_GW2 = datetime(2024, 8, 23, 17, tzinfo=UTC)
_PRIOR_RESULT = datetime(2023, 8, 12, 17, tzinfo=UTC)
_ZERO = (
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
)


def test_hash_staged_data_hashes_each_table(tmp_path: Path) -> None:
    store = _store(tmp_path)
    hashes = hash_staged_data(tmp_path)
    assert "fixtures" in hashes
    assert "gameweeks" in hashes
    assert len(hashes["fixtures"]) == 64
    # Changing a staged file changes its hash.
    path = store.path("fixtures")
    before = hashes["fixtures"]
    path.write_bytes(path.read_bytes() + b"\n")
    assert hash_staged_data(tmp_path)["fixtures"] != before


def test_cache_key_includes_required_fields(tmp_path: Path) -> None:
    _store(tmp_path)
    spec = ModelSpec(
        name="rolling_mean_v1",
        kind="rolling_mean",
        params={"n_sims": 100},
        seed=7,
        train_window=TrainWindow(seasons_back=2),
        refit="every_gw",
    )
    key = build_ledger_cache_key(
        spec,
        data_dir=tmp_path,
        season=_TEST,
        test_from_gw=1,
        test_to_gw=2,
        fit_gw=1,
        mode="walk_forward",
        steps=1,
        seasons_back=2,
        lock_root=tmp_path,
    )
    payload = key.to_dict()
    assert payload["resolved_spec_sha256"] == resolved_spec_sha256(spec)
    assert payload["uv_lock_sha256"] == "missing"
    assert payload["data_hashes"]
    assert payload["season"] == _TEST
    assert payload["test_from_gw"] == 1
    assert payload["test_to_gw"] == 2
    assert payload["seasons_back"] == 2
    assert payload["n_sims"] == 100
    assert payload["seed"] == 7
    assert payload["eval_seed"] is None
    assert len(key.digest()) == 64

    paired = build_ledger_cache_key(
        spec,
        data_dir=tmp_path,
        season=_TEST,
        test_from_gw=1,
        test_to_gw=2,
        fit_gw=1,
        mode="walk_forward",
        steps=1,
        seasons_back=2,
        lock_root=tmp_path,
        eval_seed=0,
    )
    assert paired.to_dict()["eval_seed"] == 0
    assert paired.digest() != key.digest()


def test_produce_ledger_writes_and_hits_cache(tmp_path: Path) -> None:
    _store(tmp_path)
    spec = ModelSpec(
        name="rolling_mean_v1",
        kind="rolling_mean",
        train_window=TrainWindow(seasons_back=2),
        refit="every_gw",
    )
    first = produce_ledger(
        tmp_path,
        spec,
        _TEST,
        mode="walk_forward",
        test_from_gw=1,
        test_to_gw=2,
        steps=1,
        seasons_back=2,
        lock_root=tmp_path,
    )
    assert first.cache_hit is False
    assert (first.cache_dir / "ledger.parquet").is_file()
    assert (first.cache_dir / "key.json").is_file()
    stored = json.loads((first.cache_dir / "key.json").read_text(encoding="utf-8"))
    assert stored == first.key.to_dict()
    assert first.frame.height > 0

    second = produce_ledger(
        tmp_path,
        spec,
        _TEST,
        mode="walk_forward",
        test_from_gw=1,
        test_to_gw=2,
        steps=1,
        seasons_back=2,
        lock_root=tmp_path,
    )
    assert second.cache_hit is True
    assert second.cache_dir == first.cache_dir
    assert second.frame.select(sorted(second.frame.columns)).equals(
        first.frame.select(sorted(first.frame.columns))
    )

    paired = produce_ledger(
        tmp_path,
        spec,
        _TEST,
        mode="walk_forward",
        test_from_gw=1,
        test_to_gw=2,
        steps=1,
        seasons_back=2,
        lock_root=tmp_path,
        eval_seed=0,
    )
    assert paired.cache_hit is False
    assert paired.key.eval_seed == 0
    assert paired.cache_dir != first.cache_dir


def test_stale_key_on_read_raises(tmp_path: Path) -> None:
    _store(tmp_path)
    spec = ModelSpec(
        name="rolling_mean_v1",
        kind="rolling_mean",
        train_window=TrainWindow(seasons_back=2),
        refit="every_gw",
    )
    produced = produce_ledger(
        tmp_path,
        spec,
        _TEST,
        mode="fixed",
        test_from_gw=1,
        test_to_gw=1,
        steps=1,
        seasons_back=2,
        lock_root=tmp_path,
    )
    # Tamper with key.json while leaving the digest directory in place.
    bad = produced.key.to_dict()
    bad["seed"] = 999
    (produced.cache_dir / "key.json").write_text(
        json.dumps(bad, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(StaleLedgerCacheError, match="stale ledger cache"):
        read_ledger_cache(produced.cache_dir, produced.key)


def test_compare_models_reuses_cached_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _store(tmp_path)
    calls = {"n": 0}
    real_walk = __import__(
        "fplt.backtest.produce_ledger", fromlist=["walk_forward"]
    ).walk_forward

    def counting_walk(*args: Any, **kwargs: Any) -> pl.DataFrame:
        calls["n"] += 1
        return real_walk(*args, **kwargs)

    monkeypatch.setattr("fplt.backtest.produce_ledger.walk_forward", counting_walk)
    first = compare_models(
        tmp_path,
        _TEST,
        models=["rolling_mean_v1"],
        test_from_gw=1,
        test_to_gw=2,
        seasons_back=2,
        modes=["walk_forward"],
        steps=1,
        use_cache=True,
    )
    assert calls["n"] == 1
    assert first.runs[0].cache_hit is False
    # Clear the shared prediction ledger so a second compare must re-append.
    ledger_path = tmp_path / "ledger" / "predictions" / "part.parquet"
    if ledger_path.exists():
        ledger_path.unlink()
    second = compare_models(
        tmp_path,
        _TEST,
        models=["rolling_mean_v1"],
        test_from_gw=1,
        test_to_gw=2,
        seasons_back=2,
        modes=["walk_forward"],
        steps=1,
        use_cache=True,
    )
    assert calls["n"] == 1  # cache hit; walk_forward not called again
    assert second.runs[0].cache_hit is True
    assert read_predictions(tmp_path).height > 0


def test_promote_eval_stale_cache_exits_2(tmp_path: Path) -> None:
    """Execute maps StaleLedgerCacheError to exit 2 (preflight fail)."""
    from unittest.mock import patch

    from fplt.backtest.promote_eval import PreflightResult
    from fplt.models.spec import load_model_specs

    spec = load_eval_spec("pens_order_involvement_ab")
    specs = load_model_specs()
    out = tmp_path / "out"

    def boom(*_a: Any, **_k: Any) -> Any:
        raise StaleLedgerCacheError("stored key ≠ computed key")

    def ok_preflight(*_a: Any, **_k: Any) -> PreflightResult:
        return PreflightResult(
            ok=True,
            control_spec=specs[spec.control],
            treatment_spec=specs[spec.treatment],
        )

    with (
        patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False),
        patch(
            "fplt.backtest.promote_eval.preflight_promote_eval",
            side_effect=ok_preflight,
        ),
        patch(
            "fplt.backtest.promote_eval._run_components",
            return_value=(None, None, None),
        ),
        patch("fplt.backtest.promote_eval.compare_models", side_effect=boom),
    ):
        result = execute_promote_eval(
            spec,
            data_dir=tmp_path,
            dry_run=False,
            to_gw=1,
            output_dir=out,
            argv=["promote-eval", "--execute"],
        )
    assert result.exit_code == 2
    assert result.verdict == "drop"
    assert "stale ledger cache" in result.recommendation
    assert any("stale ledger cache" in f for f in result.preflight.failures)


def test_write_read_round_trip(tmp_path: Path) -> None:
    key = LedgerCacheKey(
        resolved_spec_sha256="a" * 64,
        uv_lock_sha256="b" * 64,
        data_hashes={"fixtures": "c" * 64},
        season=_TEST,
        test_from_gw=1,
        test_to_gw=1,
        fit_gw=1,
        mode="walk_forward",
        steps=1,
        seasons_back=2,
        n_sims=None,
        seed=0,
    )
    frame = pl.DataFrame(
        {
            "model_id": ["m1"],
            "run_kind": ["walk_forward"],
            "as_of_season": [_TEST],
            "as_of_gw": [1],
            "target_season": [_TEST],
            "target_gw": [1],
            "horizon": [1],
            "player_key": [1],
            "position": ["MID"],
            "fixture_id": [1],
            "xpts": [1.0],
            "xpts_sd": [0.0],
            "p_haul": [None],
            "p_60": [None],
            "actual_pts": [None],
            "actual_minutes": [None],
        }
    )
    cache_dir = tmp_path / "cache"
    write_ledger_cache(cache_dir, key, frame)
    loaded = read_ledger_cache(cache_dir, key)
    assert loaded.height == 1


def _store(root: Path, source: str = "fpl_api") -> DataStore:
    store = DataStore(root)
    store.write(
        "players",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST],
                    "player_key": [100],
                    "element_id": [10],
                    "team_key": [1],
                    "web_name": ["A"],
                    "position": ["MID"],
                    "available_at": [_START],
                }
            )
        ),
    )
    store.write(
        "teams",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST, _TEST],
                    "team_key": [1, 2],
                    "team_id": [1, 2],
                    "name": ["Home", "Away"],
                    "short_name": ["HOM", "AWY"],
                    "available_at": [_START, _START],
                }
            )
        ),
    )
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_PRIOR, _TEST, _TEST],
                    "gw": [1, 1, 2],
                    "deadline_time": [
                        _PRIOR_RESULT.replace(hour=11),
                        _GW1,
                        _GW2,
                    ],
                    "finished": [True, True, True],
                    "data_checked": [True, True, True],
                    "available_at": [
                        _PRIOR_RESULT.replace(hour=11),
                        _GW1,
                        _GW2,
                    ],
                }
            )
        ),
    )
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_PRIOR, _TEST, _TEST],
                    "fixture_id": [1, 10, 20],
                    "event": [1, 1, 2],
                    "gw": [1, 1, 2],
                    "kickoff_time": [_PRIOR_RESULT, _GW1_RESULT, _GW2],
                    "team_h_key": [1, 1, 1],
                    "team_a_key": [2, 2, 2],
                    "team_h_score": [1, 2, 0],
                    "team_a_score": [0, 1, 0],
                    "finished": [True, True, True],
                    "available_at": [_PRIOR_RESULT, _GW1_RESULT, _GW2],
                }
            )
        ),
    )
    stats_rows = []
    for season, gw, fixture_id, pts, minutes, at in (
        (_PRIOR, 1, 1, 2, 90, _PRIOR_RESULT),
        (_TEST, 1, 10, 6, 90, _GW1_RESULT),
        (_TEST, 2, 20, 4, 90, _GW2),
    ):
        row = {
            "season": season,
            "gw": gw,
            "player_key": 100,
            "fixture_id": fixture_id,
            "minutes": minutes,
            "total_points": pts,
            "was_home": True,
            "source": source,
            "available_at": at,
            **{col: 0 for col in _ZERO},
        }
        stats_rows.append(row)
    store.write("player_fixture_stats", _utc(pl.DataFrame(stats_rows)))
    store.write(
        "player_status",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST, _TEST],
                    "gw": [1, 2],
                    "player_key": [100, 100],
                    "status": ["a", "a"],
                    "chance_of_playing": [100, 100],
                    "news": ["", ""],
                    "ep_this": [0.0, 0.0],
                    "ep_next": [0.0, 0.0],
                    "now_cost": [50, 50],
                    "available_at": [_GW1, _GW2],
                    "source": [source, source],
                }
            )
        ),
    )
    return store


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for name, dtype in frame.schema.items():
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
            exprs.append(pl.col(name).dt.replace_time_zone("UTC"))
    return frame.with_columns(exprs) if exprs else frame
