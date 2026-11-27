"""ops-09 production model governance CI check."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fplt.automation.governance import (
    PRODUCTION_RELPATH,
    GovernanceResult,
    check_git_production_governance,
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


def test_additive_ops_key_without_slate_change_passes() -> None:
    """ops-09: min_prior_seasons alone does not require a promote-eval (#285)."""
    head = _PROD + "min_prior_seasons: 3\n"
    result = check_production_governance(
        changed_paths=[PRODUCTION_RELPATH],
        base_production_yaml=_PROD,
        head_production_yaml=head,
        base_model_specs={model_spec_relpath("dc_prod_v1"): _SPEC_V1},
        head_model_specs={model_spec_relpath("dc_prod_v1"): _SPEC_V1},
        pr_body="Add history-guard threshold.",
    )
    assert result.passed
    assert result.gated_paths == ()
    assert result.reasons == ()


def test_live_slate_changed_edge_cases() -> None:
    from fplt.automation.governance import _live_slate_changed

    assert _live_slate_changed(None, None) is False
    assert _live_slate_changed(None, _PROD) is True
    assert _live_slate_changed(_PROD, None) is True
    assert _live_slate_changed("[]\n", _PROD) is True
    assert _live_slate_changed(_PROD, _PROD) is False


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


def test_parse_production_models_rejects_non_mapping() -> None:
    with pytest.raises(ValueError, match="mapping"):
        parse_production_models("- just a list\n")


def test_parse_production_models_rejects_blank_names() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_production_models("models:\n  - ''\n")


def test_governance_result_failures_property() -> None:
    ok = GovernanceResult(passed=True, reasons=("ignored",))
    assert ok.failures == ()
    bad = GovernanceResult(passed=False, reasons=("boom",))
    assert bad.failures == ("boom",)


def test_default_verdict_lookup_used_when_gated(
    tmp_path: Path,
) -> None:
    run_id = "pe-from-data"
    summary = tmp_path / "runs" / f"run={run_id}" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"verdict": "KEEP"}\n', encoding="utf-8")
    head = _prod_retarget("dc_prod_v2")
    result = check_production_governance(
        changed_paths=[PRODUCTION_RELPATH],
        base_production_yaml=_PROD,
        head_production_yaml=head,
        base_model_specs={model_spec_relpath("dc_prod_v1"): _SPEC_V1},
        head_model_specs={model_spec_relpath("dc_prod_v1"): _SPEC_V1},
        pr_body=f"promote-eval-run: {run_id}\n",
        data_dir=tmp_path,
    )
    assert result.passed


def test_missing_verdict_details_fail() -> None:
    head = _prod_retarget("dc_prod_v2")
    result = check_production_governance(
        changed_paths=[PRODUCTION_RELPATH],
        base_production_yaml=_PROD,
        head_production_yaml=head,
        base_model_specs={model_spec_relpath("dc_prod_v1"): _SPEC_V1},
        head_model_specs={model_spec_relpath("dc_prod_v1"): _SPEC_V1},
        pr_body="promote-eval-run: pe-gone\n",
        verdict_for_run=lambda _run_id: None,
    )
    assert not result.passed
    assert any("pe-gone=missing" in reason for reason in result.reasons)


def test_in_place_skips_when_base_spec_missing() -> None:
    rel = model_spec_relpath("dc_prod_v1")
    result = check_production_governance(
        changed_paths=[rel],
        base_production_yaml=_PROD,
        head_production_yaml=_PROD,
        base_model_specs={},
        head_model_specs={rel: _SPEC_V1_EDITED},
        pr_body="promote-eval-run: pe-keep-1\n",
        verdict_for_run=lambda _run_id: "keep",
    )
    # Gated (listed at head/base) but not in-place without a base text.
    assert result.gated_paths == (rel,)
    assert result.in_place_edits == ()
    assert result.passed


def test_deleted_listed_spec_counts_as_in_place() -> None:
    rel = model_spec_relpath("dc_prod_v1")
    result = check_production_governance(
        changed_paths=[rel],
        base_production_yaml=_PROD,
        head_production_yaml=_PROD,
        base_model_specs={rel: _SPEC_V1},
        head_model_specs={},
        pr_body="promote-eval-run: pe-keep-1\n",
        verdict_for_run=lambda _run_id: "keep",
    )
    assert not result.passed
    assert result.in_place_edits == (rel,)


def test_resolve_verdict_from_registry_yaml(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "\n".join(
            [
                "version: 1",
                "entries:",
                "  - id: pe-reg-1",
                "    date: '2026-01-01'",
                "    path: modelspec",
                "    verdict: keep",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert resolve_promote_eval_verdict("pe-reg-1", registry_path=registry) == "keep"
    assert resolve_promote_eval_verdict("other", registry_path=registry) is None


def test_resolve_verdict_bad_registry_returns_none(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text("not: valid: registry\n", encoding="utf-8")
    assert resolve_promote_eval_verdict("x", registry_path=registry) is None


def test_resolve_verdict_from_promote_eval_tree(tmp_path: Path) -> None:
    run_id = "pe-nested"
    summary = tmp_path / "promote_eval" / "board" / run_id / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"verdict": "keep"}\n', encoding="utf-8")
    assert resolve_promote_eval_verdict(run_id, data_dir=tmp_path) == "keep"


def test_resolve_verdict_ignores_invalid_summary(tmp_path: Path) -> None:
    run_id = "pe-bad-summary"
    summary = tmp_path / "runs" / f"run={run_id}" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("[1, 2, 3]\n", encoding="utf-8")
    assert resolve_promote_eval_verdict(run_id, data_dir=tmp_path) is None
    summary.write_text('{"verdict": ""}\n', encoding="utf-8")
    assert resolve_promote_eval_verdict(run_id, data_dir=tmp_path) is None
    summary.write_text("{not-json", encoding="utf-8")
    assert resolve_promote_eval_verdict(run_id, data_dir=tmp_path) is None


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_check_git_production_governance_on_temp_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    prod = repo / PRODUCTION_RELPATH
    prod.parent.mkdir(parents=True)
    prod.write_text(_PROD, encoding="utf-8")
    spec = repo / model_spec_relpath("dc_prod_v1")
    spec.parent.mkdir(parents=True)
    spec.write_text(_SPEC_V1, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "base"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Unrelated change → pass.
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "docs"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    ok = check_git_production_governance(
        repo=repo,
        base_ref=base,
        head_ref="HEAD",
        pr_body="",
    )
    assert ok.passed

    # Production retarget without promote-eval link → fail.
    prod.write_text(_prod_retarget("dc_prod_v2"), encoding="utf-8")
    (repo / model_spec_relpath("dc_prod_v2")).write_text(_SPEC_V2, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "prod"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    bad = check_git_production_governance(
        repo=repo,
        base_ref=base,
        head_ref="HEAD",
        pr_body="no link",
        verdict_for_run=lambda _rid: "keep",
    )
    assert not bad.passed
    assert any("without a promote-eval link" in r for r in bad.reasons)
