#!/usr/bin/env python3
"""Assemble and persist the nightly performance bench (#312).

Runs (or accepts) fit/predict/plan timings on the e2e fixture world, mc-bench,
and pytest top-25 durations; writes write-once
``$LAB_ROOT/bench/date=<UTC>/results.json``; evaluates the regression rule and
optionally notifies via ntfy. Never exits non-zero solely for a timing
regression.

Examples::

    uv run python scripts/nightly_bench.py --lab-dir /tmp/lab --skip-fixture \\
        --phases-json phases.json --mc-json mc.json --junit junit.xml
    uv run python scripts/nightly_bench.py --lab-dir /tmp/lab --run-fixture
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from fplt.data.remote import Remote
    from fplt.settings import Settings

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_fixture(data_dir: Path, bucket_dir: Path) -> dict[str, float]:
    """Seed the e2e world and return consolidated fit/predict/plan phases."""
    import polars as pl
    import pytest
    from pydantic import SecretStr

    from fplt.automation.bench import consolidate_fixture_phases
    from fplt.automation.checkpoints import (
        write_notified_marker,
        write_plan_checkpoint_marker,
    )
    from fplt.automation.official_slate import run_official_checkpoint
    from fplt.automation.ops_tick import run_ops_tick
    from fplt.data.remote import Remote
    from fplt.models.spec import load_model_specs
    from fplt.optimise.plan import plan_parquet
    from fplt.settings import Settings, get_settings
    from fplt.types import GWKey
    from fplt.utils.timing import load_timing_records
    from tests.builders_e2e import (
        CURRENT_SEASON,
        E2E_LINE,
        E2E_MODEL,
        NOW_T90M,
        PRIOR_SEASONS,
        e2e_production_ops,
        seed_e2e_bucket,
    )

    # Use pytest's MonkeyPatch so Remote / production ops patches restore cleanly.
    monkeypatch = pytest.MonkeyPatch()
    try:
        bucket_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        seed_e2e_bucket(bucket_dir)
        remote = Remote(target=str(bucket_dir), kind="data", env={})

        def _from_settings(
            _cls: type[Remote], _settings: Settings, bucket: str | None = None
        ) -> Remote:
            del bucket
            return remote

        monkeypatch.setattr(
            "fplt.data.remote.Remote.from_settings",
            classmethod(_from_settings),
        )
        monkeypatch.setenv("FPLT_DATA_DIR", str(data_dir))
        monkeypatch.setenv("FPLT_MC_BACKEND", "python")
        monkeypatch.setenv("JOB_STARTED_AT", str(int(NOW_T90M.timestamp()) - 30))
        monkeypatch.delenv("FPLT_ENTRY_ID", raising=False)
        get_settings.cache_clear()
        ops = e2e_production_ops(min_prior_seasons=3)
        monkeypatch.setattr("fplt.automation.ops_tick.load_production_ops", lambda: ops)

        def _official(*args: object, **kwargs: object) -> object:
            kwargs.setdefault("chip_arm", "off")
            return run_official_checkpoint(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            "fplt.automation.ops_tick.run_official_checkpoint", _official
        )

        def _skip_archive(*_a: object, **_k: object) -> None:
            return None

        monkeypatch.setattr(
            "fplt.automation.cycle.archive_if_missing",
            _skip_archive,
        )

        spec = load_model_specs()[E2E_MODEL]
        for season, gw in [
            (CURRENT_SEASON, 1),
            *((s, 1) for s in PRIOR_SEASONS),
        ]:
            path = plan_parquet(data_dir, spec.model_id, season, gw, 1)
            path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"player_key": [1], "step": [0]}).write_parquet(path)

        key = GWKey(CURRENT_SEASON, 2)
        write_plan_checkpoint_marker(data_dir, key, "T-24h", primary_line=E2E_LINE)
        write_plan_checkpoint_marker(data_dir, key, "T-6h", primary_line=E2E_LINE)
        write_notified_marker(data_dir, key)
        settings = Settings(
            data_dir=data_dir,
            data_bucket="fplt-data",
            r2_account_id="bench",
            r2_access_key_id=SecretStr("bench"),
            r2_secret_access_key=SecretStr("bench"),
        )
        run_ops_tick(
            data_dir,
            now=NOW_T90M,
            source="bucket",
            settings=settings,
            notify=False,
        )
        merged: dict[str, float] = {}
        for record in load_timing_records(data_dir, limit=20):
            phases_obj: object = record.get("phases", {})
            if not isinstance(phases_obj, dict):
                continue
            raw_phases = cast(dict[str, object], phases_obj)
            for name, value in raw_phases.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                seconds = float(value)
                if seconds <= 0:
                    continue
                key = str(name)
                merged[key] = max(merged.get(key, 0.0), seconds)
        return consolidate_fixture_phases(merged)
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def _mc_bench_payload(*, n_sims: int, repeats: int) -> dict[str, float]:
    from scripts.mc_bench import run_bench

    rows = run_bench(n_sims=n_sims, repeats=repeats)
    return {name: float(ms) for name, ms in rows}


def _pytest_top25(junit: Path, *, top: int = 25) -> tuple[float, list[dict[str, Any]]]:
    from scripts.pytest_profile import parse_junit_durations

    cases = parse_junit_durations(junit)
    suite_s = float(sum(c.seconds for c in cases))
    top_rows = [{"nodeid": c.nodeid, "seconds": float(c.seconds)} for c in cases[:top]]
    return suite_s, top_rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry: assemble results, write-once, evaluate regressions."""
    from fplt.automation.bench import (
        build_bench_results,
        evaluate_regression,
        format_regression_alerts,
        format_trend_report,
        list_bench_results,
        load_bench_config,
        notify_regression_alerts,
        write_bench_results,
    )
    from fplt.settings import get_settings
    from fplt.utils.logging import configure_logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lab-dir",
        type=Path,
        required=True,
        help="Local directory standing in for fplt-lab (bench/ written here)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Working data dir for the fixture tick (default: lab-dir/_bench_data)",
    )
    parser.add_argument(
        "--bucket-dir",
        type=Path,
        help="Local bucket dir for the fixture (default: lab-dir/_bench_bucket)",
    )
    parser.add_argument(
        "--run-fixture",
        action="store_true",
        help="Run the e2e fixture world tick for fit/predict/plan timings",
    )
    parser.add_argument(
        "--skip-fixture",
        action="store_true",
        help="Do not run the fixture world (use --phases-json)",
    )
    parser.add_argument(
        "--phases-json",
        type=Path,
        help='Precomputed {"phases": {...}} JSON (skips fixture when set)',
    )
    parser.add_argument(
        "--mc-json",
        type=Path,
        help="Precomputed mc-bench {step: ms} JSON (else run mc-bench)",
    )
    parser.add_argument(
        "--skip-mc",
        action="store_true",
        help="Skip mc-bench collection",
    )
    parser.add_argument("--mc-n-sims", type=int, default=10_000)
    parser.add_argument("--mc-repeats", type=int, default=3)
    parser.add_argument(
        "--junit",
        type=Path,
        help="Pytest JUnit XML for suite duration + top-25",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="UTC date YYYY-MM-DD (default: today UTC)",
    )
    parser.add_argument(
        "--git-sha",
        type=str,
        default=os.environ.get("GITHUB_SHA"),
        help="Commit sha recorded in results (default: GITHUB_SHA)",
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip ntfy even when regressions fire",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Idempotent write when today's results.json already exists",
    )
    parser.add_argument(
        "--pull-history",
        action="store_true",
        help="Pull existing bench/ from FPLT_LAB_BUCKET into --lab-dir first",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push today's bench/date=<UTC>/ partition to FPLT_LAB_BUCKET",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging",
    )
    args = parser.parse_args(argv)
    configure_logging(verbose=bool(args.verbose))

    lab_dir: Path = args.lab_dir
    lab_dir.mkdir(parents=True, exist_ok=True)
    settings: Settings = get_settings()

    if args.pull_history:
        _pull_bench_history(lab_dir, settings)

    phases: dict[str, float] = {}

    if args.phases_json is not None:
        payload = _load_json(args.phases_json)
        raw = payload.get("phases", payload)
        phases.update({str(k): float(v) for k, v in raw.items()})
    elif args.run_fixture or not args.skip_fixture:
        data_dir = args.data_dir or (lab_dir / "_bench_data")
        bucket_dir = args.bucket_dir or (lab_dir / "_bench_bucket")
        logger.info("running e2e fixture world under %s", data_dir)
        phases.update(_run_fixture(data_dir, bucket_dir))
    else:
        logger.warning("no fixture phases (--skip-fixture without --phases-json)")

    mc_bench: dict[str, float] = {}
    if args.mc_json is not None:
        mc_bench = {str(k): float(v) for k, v in _load_json(args.mc_json).items()}
    elif not args.skip_mc:
        logger.info("running mc-bench")
        mc_bench = _mc_bench_payload(n_sims=args.mc_n_sims, repeats=args.mc_repeats)
    if mc_bench:
        phases["mc_bench_total"] = sum(mc_bench.values()) / 1000.0  # ms → s

    pytest_top25: list[dict[str, Any]] = []
    if args.junit is not None:
        suite_s, pytest_top25 = _pytest_top25(args.junit)
        phases["pytest_suite"] = suite_s

    day = args.date or date.today().isoformat()
    results = build_bench_results(
        day=day,
        phases=phases,
        recorded_at=datetime.now(UTC),
        git_sha=args.git_sha,
        mc_bench=mc_bench,
        pytest_top25=pytest_top25,
        meta={"source": "nightly_bench"},
    )
    path = write_bench_results(lab_dir, results, exist_ok=bool(args.exist_ok))
    logger.info("wrote %s", path)

    cfg = load_bench_config()
    history = list_bench_results(lab_dir)
    report = evaluate_regression(history, config=cfg)
    print(format_trend_report(history, config=cfg))
    if report.alerted:
        print(format_regression_alerts(report), file=sys.stderr)
        if not args.no_notify:
            notify_regression_alerts(settings, report)
    if args.push:
        _push_bench_day(lab_dir, day, settings)
    # Timing regressions never fail the process.
    return 0


def _lab_remote(settings: Settings) -> Remote | None:
    """Build a lab-bucket Remote, or None when FPLT_LAB_BUCKET is unset."""
    from fplt.data.remote import Remote

    if not settings.lab_bucket:
        logger.warning("soft skip: FPLT_LAB_BUCKET unset")
        return None
    return Remote.from_settings(settings, bucket=settings.lab_bucket)


def _pull_bench_history(lab_dir: Path, settings: Settings) -> None:
    """Pull existing ``bench/`` history from the lab bucket into ``lab_dir``."""
    from fplt.data.remote import RemoteError, pull

    remote = _lab_remote(settings)
    if remote is None:
        return
    dest = lab_dir / "bench"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        pull(remote, "bench", dest)
    except (RemoteError, OSError, ValueError) as exc:
        logger.warning("soft skip: pull bench/ failed: %s", exc)


def _push_bench_day(lab_dir: Path, day: str, settings: Settings) -> None:
    """Push today's ``bench/date=<UTC>/`` partition to the lab bucket."""
    from fplt.automation.bench import bench_date_dir
    from fplt.data.remote import RemoteError, push

    remote = _lab_remote(settings)
    if remote is None:
        return
    src = bench_date_dir(lab_dir, day)
    if not src.is_dir():
        logger.warning("nothing to push under %s", src)
        return
    try:
        push(remote, src, f"bench/date={day}")
    except (RemoteError, OSError, ValueError) as exc:
        logger.exception("push bench/date=%s failed: %s", day, exc)
        raise
    logger.info("pushed bench/date=%s", day)


if __name__ == "__main__":
    raise SystemExit(main())
