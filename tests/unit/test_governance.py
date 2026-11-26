"""ops-09 production model governance CI check."""

from __future__ import annotations

from pathlib import Path

import pytest

from fplt.automation.governance import (
    PRODUCTION_RELPATH,
    check_production_governance,
    extract_promote_eval_runs,
    model_spec_relpath,
    parse_production_models,
    resolve_promote_eval_verdict,
)

_PROD = """\
models:
  - dc_prod_v1
plan_horizon: 3
pipeline:
  data:
    sources:
      odds_capture: false
      fplcache: false
      api_football_injuries: true
    capture_checkpoint: T-90m
  model: dc_prod_v1
  planner:
    horizon: 3
    decay: 0.9
    pool:
      target: 130
    mip_gap: 0.001
    time_limit_s: 600
    objective: mean
  policy:
    max_hits: 1
    hit_min_gain: 4.0
    chips:
      scheduler: season_v1
      arm: optimised
      force_play: true
    captain: max_xpts
    lineup_reopt: true
    price_forecast: mean_gw_delta
    bench_weight: 0.0
  env:
    mc_backend: native
"""

_SPEC_V1 = "name: dc_prod_v1\nkind: rolling_mean\nparams: {window: 3}\n"
_SPEC_V1_EDITED = "name: dc_prod_v1\nkind: rolling_mean\nparams: {window: 5}\n"
_SPEC_V2 = "name: dc_prod_v2\nkind: rolling_mean\nparams: {window: 5}\n"


def _prod_retarget(name: str) -> str:
    return _PROD.replace("dc_prod_v1", name)


def test_parse_production_models() -> None:
    assert parse_production_models(_PROD) == ["dc_prod_v1"]


def test_extract_promote_eval_runs() -> None:
    body = (
        "See board.\n\npromote-eval-run: pe-abc123\nAlso promote-eval-run: other_id\n"
    )
    assert extract_promote_eval_runs(body) == ["pe-abc123", "other_id"]


def test_unchanged_config_passes() -> None:
    result = check_production_governance(
        changed_paths=["src/fplt/cli.py", "README.md"],
        base_production_yaml=_PROD,
        head_production_yaml=_PROD,
        base_model_specs={model_spec_relpath("dc_prod_v1"): _SPEC_V1},
        head_model_specs={model_spec_relpath("dc_prod_v1"): _SPEC_V1},
        pr_body="",
    )
    assert result.passed
    assert result.reasons == ()


def test_changed_production_without_eval_link_fails() -> None:
    head = _prod_retarget("dc_prod_v2")
    result = check_production_governance(
        changed_paths=[PRODUCTION_RELPATH, model_spec_relpath("dc_prod_v2")],
        base_production_yaml=_PROD,
        head_production_yaml=head,
        base_model_specs={model_spec_relpath("dc_prod_v1"): _SPEC_V1},
        head_model_specs={
            model_spec_relpath("dc_prod_v1"): _SPEC_V1,
            model_spec_relpath("dc_prod_v2"): _SPEC_V2,
        },
        pr_body="Flip production to v2.",
        verdict_for_run=lambda _run_id: "keep",
    )
    assert not result.passed
    assert any("without a promote-eval link" in reason for reason in result.reasons)


def test_changed_production_with_passing_eval_link_passes() -> None:
    head = _prod_retarget("dc_prod_v2")
    result = check_production_governance(
        changed_paths=[PRODUCTION_RELPATH, model_spec_relpath("dc_prod_v2")],
        base_production_yaml=_PROD,
        head_production_yaml=head,
        base_model_specs={model_spec_relpath("dc_prod_v1"): _SPEC_V1},
        head_model_specs={
            model_spec_relpath("dc_prod_v1"): _SPEC_V1,
            model_spec_relpath("dc_prod_v2"): _SPEC_V2,
        },
        pr_body="promote-eval-run: pe-keep-1\n",
        verdict_for_run=lambda run_id: {"pe-keep-1": "keep"}.get(run_id),
    )
    assert result.passed
    assert result.promote_eval_runs == ("pe-keep-1",)


def test_changed_production_with_failing_verdict_fails() -> None:
    head = _prod_retarget("dc_prod_v2")
    result = check_production_governance(
        changed_paths=[PRODUCTION_RELPATH],
        base_production_yaml=_PROD,
        head_production_yaml=head,
        base_model_specs={model_spec_relpath("dc_prod_v1"): _SPEC_V1},
        head_model_specs={model_spec_relpath("dc_prod_v1"): _SPEC_V1},
        pr_body="promote-eval-run: pe-drop-1\n",
        verdict_for_run=lambda run_id: {"pe-drop-1": "drop"}.get(run_id),
    )
    assert not result.passed
    assert any("no passing (keep) verdict" in reason for reason in result.reasons)


def test_in_place_spec_edit_fails_even_with_link() -> None:
    rel = model_spec_relpath("dc_prod_v1")
    result = check_production_governance(
        changed_paths=[rel],
        base_production_yaml=_PROD,
        head_production_yaml=_PROD,
        base_model_specs={rel: _SPEC_V1},
        head_model_specs={rel: _SPEC_V1_EDITED},
        pr_body="promote-eval-run: pe-keep-1\n",
        verdict_for_run=lambda _run_id: "keep",
    )
    assert not result.passed
    assert result.in_place_edits == (rel,)
    assert any("in-place edit" in reason for reason in result.reasons)


def test_resolve_verdict_from_summary_json(tmp_path: Path) -> None:
    run_id = "pe-local-1"
    summary = tmp_path / "runs" / f"run={run_id}" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"verdict": "keep"}\n', encoding="utf-8")
    assert resolve_promote_eval_verdict(run_id, data_dir=tmp_path) == "keep"


def test_resolve_verdict_missing(tmp_path: Path) -> None:
    assert resolve_promote_eval_verdict("missing-run", data_dir=tmp_path) is None


def test_parse_production_models_rejects_empty() -> None:
    with pytest.raises(ValueError, match="models"):
        parse_production_models("plan_horizon: 3\n")
