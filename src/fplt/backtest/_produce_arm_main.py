"""Subprocess entrypoint: produce one prediction ledger arm from a JSON request.

Invoked as ``uv run --directory <worktree> --frozen python <this> <request.json>``.
Uses the worktree's installed dependencies while importing produce helpers from
the same source tree when present, else from the installed ``fplt`` package.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal, cast


def main(argv: list[str] | None = None) -> int:
    """Read a request JSON, produce one arm, write parquet + metadata."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: _produce_arm_main.py <request.json>", file=sys.stderr)
        return 2
    request_path = Path(args[0])
    payload: dict[str, Any] = json.loads(request_path.read_text(encoding="utf-8"))

    # Prefer the worktree / installed package; fall back to host helper import.
    from fplt.backtest.produce_ledger import produce_ledger
    from fplt.models.spec import load_model_specs

    models_dir = Path(payload["models_dir"]) if payload.get("models_dir") else None
    specs = load_model_specs(models_dir)
    model_name = str(payload["model_name"])
    if model_name not in specs:
        print(f"ModelSpec {model_name!r} missing", file=sys.stderr)
        return 2
    spec = specs[model_name]
    mode = cast(Literal["walk_forward", "fixed"], str(payload["mode"]))
    fit_gw = payload.get("fit_gw")
    seasons_back = payload.get("seasons_back")
    lock_root = Path(payload["lock_root"]) if payload.get("lock_root") else None
    out_parquet = Path(payload["out_parquet"])
    produced = produce_ledger(
        Path(payload["data_dir"]),
        spec,
        str(payload["season"]),
        mode=mode,
        test_from_gw=int(payload["test_from_gw"]),
        test_to_gw=int(payload["test_to_gw"]),
        fit_gw=None if fit_gw is None else int(fit_gw),
        steps=int(payload.get("steps") or 1),
        seasons_back=None if seasons_back is None else int(seasons_back),
        use_cache=bool(payload.get("use_cache", True)),
        lock_root=lock_root,
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    produced.frame.write_parquet(out_parquet)
    meta = {
        "rows": produced.frame.height,
        "cache_hit": produced.cache_hit,
        "digest": produced.key.digest(),
        "uv_lock_sha256": produced.key.uv_lock_sha256,
        "resolved_spec_sha256": produced.key.resolved_spec_sha256,
    }
    out_parquet.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
