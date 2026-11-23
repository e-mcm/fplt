"""Committed promote-results registry loader and audit links."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fplt.backtest.results_registry import (
    PromoteResult,
    ResultsRegistry,
    default_results_registry_path,
    entry_by_id,
    load_results_registry,
    provisional_entries,
)


def test_committed_registry_loads_and_links_audits() -> None:
    """Seeded registry resolves and matches committed audit digests."""
    registry = load_results_registry()
    assert registry.version == 1
    assert default_results_registry_path().is_file()
    assert len(registry.entries) >= 11

    defcon = entry_by_id(registry, "defcon_rate")
    assert defcon.verdict == "keep"
    assert defcon.provisional is True
    assert defcon.provisional_reason is not None
    assert "FDB-255" in defcon.provisional_reason
    assert defcon.eval_spec == "defcon_rate_ab"
    assert defcon.treatment == "dc_layered_v3"
    assert defcon.windows[0].audit == "results/defcon_rate/primary/audit.json"
    assert defcon.windows[0].git_sha is not None
    roles = {w.role for w in defcon.windows}
    assert {"primary", "confirm", "development", "protocol_confirm"} <= roles
    protocol = next(w for w in defcon.windows if w.role == "protocol_confirm")
    assert protocol.audit == "results/defcon_rate/protocol_confirm/audit.json"
    assert protocol.from_gw == 20 and protocol.to_gw == 38

    price = entry_by_id(registry, "price_within_horizon")
    assert price.verdict == "keep"
    assert price.provisional is True
    assert price.eval_spec is None
    assert price.path == "decision_policy"

    provisional_ids = {e.id for e in provisional_entries(registry)}
    assert provisional_ids == {"defcon_rate", "price_within_horizon"}

    prematch = entry_by_id(registry, "prematch_1x2_prior")
    assert prematch.verdict == "drop"
    assert prematch.provisional is False
    assert "fixtures" in prematch.windows[0].data_sha256

    cs = entry_by_id(registry, "cs_only_player_rmse")
    assert cs.verdict == "keep"
    assert cs.provisional is False
    assert cs.eval_spec == "cs_only_player_rmse_ab"
    # Keep-audit arms (pre-flip); production flip is dc_layered_v4 (FDB-267).
    assert cs.control == "dc_layered_v3"
    assert cs.treatment == "dc_layered_cs_v1"
    assert "dc_layered_v4" in (cs.notes or "")
    roles = {w.role for w in cs.windows}
    assert roles == {"development", "confirm"}


def test_registry_eval_specs_exist_when_named() -> None:
    """Every named EvalSpec still has a YAML under configs/evals/."""
    registry = load_results_registry()
    evals = Path("configs/evals")
    for entry in registry.entries:
        if entry.eval_spec is None:
            continue
        path = evals / f"{entry.eval_spec}.yaml"
        assert path.is_file(), f"missing EvalSpec for {entry.id}: {path}"


def test_missing_audit_raises(tmp_path: Path) -> None:
    """Referenced audit paths must exist when require_audits is on."""
    payload = {
        "version": 1,
        "entries": [
            {
                "id": "demo",
                "date": "2026-11-23",
                "path": "modelspec",
                "verdict": "drop",
                "eval_spec": "demo_ab",
                "windows": [
                    {
                        "role": "primary",
                        "seasons": ["2024-25"],
                        "from_gw": 1,
                        "to_gw": 10,
                        "audit": "results/missing/audit.json",
                    }
                ],
            }
        ],
    }
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="audit not found"):
        load_results_registry(path, evals_root=tmp_path)


def test_data_digest_mismatch_raises(tmp_path: Path) -> None:
    """Registry data_sha256 must match the committed audit.json digests."""
    audit_dir = tmp_path / "results" / "demo" / "primary"
    audit_dir.mkdir(parents=True)
    (audit_dir / "audit.json").write_text(
        '{"data": {"fixtures": {"sha256": "' + ("a" * 64) + '"}}}\n',
        encoding="utf-8",
    )
    payload = {
        "version": 1,
        "entries": [
            {
                "id": "demo",
                "date": "2026-11-23",
                "path": "modelspec",
                "verdict": "keep",
                "windows": [
                    {
                        "role": "primary",
                        "seasons": ["2024-25"],
                        "from_gw": 1,
                        "to_gw": 2,
                        "audit": "results/demo/primary/audit.json",
                        "data_sha256": {"fixtures": "b" * 64},
                    }
                ],
            }
        ],
    }
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        load_results_registry(path, evals_root=tmp_path)


def test_provisional_requires_reason() -> None:
    """Provisional keeps must document why they are provisional."""
    with pytest.raises(ValidationError, match="provisional_reason"):
        PromoteResult.model_validate(
            {
                "id": "demo",
                "date": "2026-11-23",
                "path": "modelspec",
                "verdict": "keep",
                "provisional": True,
            }
        )


def test_duplicate_ids_rejected() -> None:
    """Registry entry ids must be unique."""
    entry = {
        "id": "demo",
        "date": "2026-11-23",
        "path": "decision_policy",
        "verdict": "aborted",
    }
    with pytest.raises(ValidationError, match="duplicate"):
        ResultsRegistry.model_validate({"version": 1, "entries": [entry, entry]})


def test_entry_by_id_missing() -> None:
    registry = ResultsRegistry.model_validate(
        {
            "version": 1,
            "entries": [
                {
                    "id": "demo",
                    "date": "2026-11-23",
                    "path": "modelspec",
                    "verdict": "drop",
                }
            ],
        }
    )
    with pytest.raises(KeyError):
        entry_by_id(registry, "nope")
