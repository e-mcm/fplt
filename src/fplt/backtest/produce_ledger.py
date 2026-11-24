"""Produce one arm's prediction ledger with a content-addressed parquet cache.

P2.5 splits produce from compare. ``produce_ledger`` walks one ModelSpec over a
shared window and stores the result under ``data_dir/ledger_cache/<digest>/``
keyed by resolved-spec hash, ``uv.lock`` hash, staged-data hashes, window,
``seasons_back``, ``n_sims``, and seed. The key is written beside the parquet
and verified on every read. This is plain parquet on disk, not ``joblib``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl

from fplt.backtest.walk_forward import walk_forward
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.contracts import PointsPredictor
from fplt.models.spec import ModelSpec, get_model
from fplt.types import GWKey
from fplt.utils.rng import paired_eval_seed
from fplt.utils.timing import timing_artifacts

logger = logging.getLogger(__name__)

CompareMode = Literal["walk_forward", "fixed"]
_KEY_VERSION = 1
_CACHE_DIRNAME = "ledger_cache"
_LEDGER_FILE = "ledger.parquet"
_KEY_FILE = "key.json"


class StaleLedgerCacheError(ValueError):
    """Cached ``key.json`` does not match the key just computed."""


@dataclass(frozen=True)
class LedgerCacheKey:
    """Content-addressed identity for one produced prediction ledger."""

    resolved_spec_sha256: str
    uv_lock_sha256: str
    data_hashes: dict[str, str]
    season: str
    test_from_gw: int
    test_to_gw: int
    fit_gw: int
    mode: CompareMode
    steps: int
    seasons_back: int | None
    n_sims: int | None
    seed: int
    eval_seed: int | None = None
    version: int = _KEY_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Canonical JSON-serialisable form (sorted data hash keys)."""
        return {
            "version": self.version,
            "resolved_spec_sha256": self.resolved_spec_sha256,
            "uv_lock_sha256": self.uv_lock_sha256,
            "data_hashes": dict(sorted(self.data_hashes.items())),
            "season": self.season,
            "test_from_gw": self.test_from_gw,
            "test_to_gw": self.test_to_gw,
            "fit_gw": self.fit_gw,
            "mode": self.mode,
            "steps": self.steps,
            "seasons_back": self.seasons_back,
            "n_sims": self.n_sims,
            "seed": self.seed,
            "eval_seed": self.eval_seed,
        }

    def digest(self) -> str:
        """Directory name: SHA-256 of the canonical key payload."""
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LedgerCacheKey:
        """Parse a stored ``key.json`` payload."""
        mode = payload["mode"]
        if mode not in ("walk_forward", "fixed"):
            raise StaleLedgerCacheError(f"unknown cache mode {mode!r}")
        data_hashes_obj: object = payload.get("data_hashes") or {}
        if not isinstance(data_hashes_obj, dict):
            raise StaleLedgerCacheError("data_hashes must be an object")
        typed_hashes = cast(dict[object, object], data_hashes_obj)
        data_hashes = {
            str(raw_key): str(raw_val) for raw_key, raw_val in typed_hashes.items()
        }
        return cls(
            resolved_spec_sha256=str(payload["resolved_spec_sha256"]),
            uv_lock_sha256=str(payload["uv_lock_sha256"]),
            data_hashes=data_hashes,
            season=str(payload["season"]),
            test_from_gw=int(payload["test_from_gw"]),
            test_to_gw=int(payload["test_to_gw"]),
            fit_gw=int(payload["fit_gw"]),
            mode=mode,  # type: ignore[arg-type]
            steps=int(payload["steps"]),
            seasons_back=(
                None
                if payload.get("seasons_back") is None
                else int(payload["seasons_back"])
            ),
            n_sims=(None if payload.get("n_sims") is None else int(payload["n_sims"])),
            seed=int(payload["seed"]),
            eval_seed=(
                None if payload.get("eval_seed") is None else int(payload["eval_seed"])
            ),
            version=int(payload.get("version", _KEY_VERSION)),
        )


@dataclass(frozen=True)
class ProduceLedgerResult:
    """One produced (or cache-hit) prediction frame for an arm × mode."""

    frame: pl.DataFrame
    spec: ModelSpec
    key: LedgerCacheKey
    cache_dir: Path
    cache_hit: bool


def repo_root() -> Path:
    """Repository root (``src/fplt/backtest`` → three parents up)."""
    return Path(__file__).resolve().parents[3]


def uv_lock_sha256(*, root: Path | None = None) -> str:
    """SHA-256 of ``uv.lock`` at the repo root."""
    path = (root or repo_root()) / "uv.lock"
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved_spec_sha256(spec: ModelSpec) -> str:
    """Identity hash of a ModelSpec (alias of ``spec_hash`` since B8).

    Fills typed component defaults and excludes ``name``. Prefer importing
    ``fplt.models.spec_identity.spec_hash`` at new call sites.
    """
    from fplt.models.spec_identity import spec_hash

    return spec_hash(spec)


def hash_staged_data(data_dir: Path) -> dict[str, str]:
    """SHA-256 of every ``staged/<table>/part.parquet`` under ``data_dir``."""
    staged = data_dir / "staged"
    if not staged.is_dir():
        return {}
    out: dict[str, str] = {}
    for path in sorted(staged.glob("*/part.parquet")):
        out[path.parent.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def n_sims_of(spec: ModelSpec) -> int | None:
    """Top-level ``params.n_sims`` when set."""
    value = spec.params.get("n_sims")
    return int(value) if value is not None else None


def build_ledger_cache_key(
    spec: ModelSpec,
    *,
    data_dir: Path,
    season: str,
    test_from_gw: int,
    test_to_gw: int,
    fit_gw: int,
    mode: CompareMode,
    steps: int,
    seasons_back: int | None,
    lock_root: Path | None = None,
    eval_seed: int | None = None,
) -> LedgerCacheKey:
    """Build the cache key for one arm × window × mode."""
    return LedgerCacheKey(
        resolved_spec_sha256=resolved_spec_sha256(spec),
        uv_lock_sha256=uv_lock_sha256(root=lock_root),
        data_hashes=hash_staged_data(data_dir),
        season=season,
        test_from_gw=test_from_gw,
        test_to_gw=test_to_gw,
        fit_gw=fit_gw,
        mode=mode,
        steps=steps,
        seasons_back=seasons_back,
        n_sims=n_sims_of(spec),
        seed=int(spec.seed),
        eval_seed=eval_seed,
    )


def ledger_cache_dir(data_dir: Path, key: LedgerCacheKey) -> Path:
    """Content-addressed directory for ``key`` under ``data_dir``."""
    return data_dir / _CACHE_DIRNAME / key.digest()


def read_ledger_cache(cache_dir: Path, expected: LedgerCacheKey) -> pl.DataFrame:
    """Load ``ledger.parquet`` after verifying ``key.json`` matches ``expected``.

    Args:
        cache_dir: Directory that should contain ``key.json`` and the parquet.
        expected: Key just computed for this produce call.

    Returns:
        Cached prediction rows.

    Raises:
        StaleLedgerCacheError: Missing files or stored key ≠ expected.
        FileNotFoundError: Cache directory or parquet is absent.
    """
    key_path = cache_dir / _KEY_FILE
    ledger_path = cache_dir / _LEDGER_FILE
    if not key_path.is_file():
        raise FileNotFoundError(f"ledger cache key missing: {key_path}")
    if not ledger_path.is_file():
        raise FileNotFoundError(f"ledger cache parquet missing: {ledger_path}")
    stored = LedgerCacheKey.from_dict(json.loads(key_path.read_text(encoding="utf-8")))
    if stored.to_dict() != expected.to_dict():
        raise StaleLedgerCacheError(
            f"stale ledger cache at {cache_dir}: stored key ≠ computed key"
        )
    return pl.read_parquet(ledger_path)


def write_ledger_cache(
    cache_dir: Path, key: LedgerCacheKey, frame: pl.DataFrame
) -> None:
    """Write ``key.json`` and ``ledger.parquet`` under ``cache_dir``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / _KEY_FILE).write_text(
        json.dumps(key.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frame.write_parquet(cache_dir / _LEDGER_FILE)


def produce_ledger(
    data_dir: Path,
    spec: ModelSpec,
    season: str,
    *,
    mode: CompareMode,
    test_from_gw: int,
    test_to_gw: int,
    fit_gw: int | None = None,
    steps: int = 1,
    seasons_back: int | None = None,
    use_cache: bool = True,
    lock_root: Path | None = None,
    model: PointsPredictor | None = None,
    eval_seed: int | None = None,
) -> ProduceLedgerResult:
    """Fit/predict one arm for ``mode`` and optionally read/write the cache.

    Args:
        data_dir: Root with ``staged/``; cache lives under ``ledger_cache/``.
        spec: Resolved ModelSpec (refit / seasons_back already applied).
        season: Evaluation season such as ``"2024-25"``.
        mode: ``walk_forward`` or ``fixed``.
        test_from_gw: First scored as-of gameweek.
        test_to_gw: Last scored as-of gameweek, inclusive.
        fit_gw: Fixed-mode fit deadline; defaults to ``test_from_gw``.
        steps: Horizon length passed to ``walk_forward``.
        seasons_back: Window override recorded in the cache key (may also be
            baked into ``spec.train_window``).
        use_cache: When true, hit/write the content-addressed parquet cache.
        lock_root: Optional override for locating ``uv.lock``.
        model: Optional pre-built predictor (tests); otherwise ``get_model``.
        eval_seed: When set, activate promote-eval CRN streams
            (``paired_eval_seed``) so arms that share this seed share fixture
            noise. ``None`` keeps production ``seed_for(model_id, ...)``.

    Returns:
        Prediction rows plus cache metadata.

    Raises:
        StaleLedgerCacheError: Cache present but ``key.json`` does not match.
        ValueError: Invalid window or mode.
    """
    if mode not in ("walk_forward", "fixed"):
        raise ValueError(f"unknown compare mode {mode!r}")
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    fit_at = test_from_gw if fit_gw is None else fit_gw
    if not 1 <= test_from_gw <= test_to_gw <= 38:
        raise ValueError(
            f"test gameweeks must be in 1..38, got {test_from_gw}..{test_to_gw}"
        )
    if not 1 <= fit_at <= 38:
        raise ValueError(f"fit_gw must be in 1..38, got {fit_at}")
    if fit_at > test_from_gw:
        raise ValueError(
            f"fit_gw ({fit_at}) must be at or before test_from_gw ({test_from_gw})"
        )

    key_seasons_back = (
        seasons_back if seasons_back is not None else spec.train_window.seasons_back
    )
    key = build_ledger_cache_key(
        spec,
        data_dir=data_dir,
        season=season,
        test_from_gw=test_from_gw,
        test_to_gw=test_to_gw,
        fit_gw=fit_at,
        mode=mode,
        steps=steps,
        seasons_back=key_seasons_back,
        lock_root=lock_root,
        eval_seed=eval_seed,
    )
    cache_dir = ledger_cache_dir(data_dir, key)

    if use_cache and (cache_dir / _LEDGER_FILE).is_file():
        frame = read_ledger_cache(cache_dir, key)
        logger.info(
            "produce_ledger cache hit model=%s mode=%s digest=%s rows=%s",
            spec.name,
            mode,
            key.digest()[:12],
            frame.height,
        )
        return ProduceLedgerResult(
            frame=frame,
            spec=spec,
            key=key,
            cache_dir=cache_dir,
            cache_hit=True,
        )

    store = DataStore(data_dir)
    pit = PITData(store)
    predictor = model if model is not None else get_model(spec.kind)(spec)
    logger.info(
        "produce_ledger model=%s mode=%s refit=%s model_id=%s eval_seed=%s",
        spec.name,
        mode,
        spec.refit,
        spec.model_id,
        eval_seed,
    )
    started = time.perf_counter()
    with timing_artifacts(data_dir):
        if eval_seed is None:
            frame = _predict_window(
                predictor,
                pit,
                season,
                mode=mode,
                fit_gw=fit_at,
                test_from_gw=test_from_gw,
                test_to_gw=test_to_gw,
                steps=steps,
            )
        else:
            with paired_eval_seed(int(eval_seed)):
                frame = _predict_window(
                    predictor,
                    pit,
                    season,
                    mode=mode,
                    fit_gw=fit_at,
                    test_from_gw=test_from_gw,
                    test_to_gw=test_to_gw,
                    steps=steps,
                )
    logger.info(
        "produce_ledger wrote %s rows in %.1fs (cache_miss digest=%s)",
        frame.height,
        time.perf_counter() - started,
        key.digest()[:12],
    )
    if use_cache:
        write_ledger_cache(cache_dir, key, frame)
    return ProduceLedgerResult(
        frame=frame,
        spec=spec,
        key=key,
        cache_dir=cache_dir,
        cache_hit=False,
    )


def _predict_window(
    model: PointsPredictor,
    pit: PITData,
    season: str,
    *,
    mode: CompareMode,
    fit_gw: int,
    test_from_gw: int,
    test_to_gw: int,
    steps: int,
) -> pl.DataFrame:
    """Fit/predict one mode; keep only as-of rows inside the test window."""
    if mode == "walk_forward":
        as_ofs = [GWKey(season, gw) for gw in range(test_from_gw, test_to_gw + 1)]
        return walk_forward(model, pit, as_ofs, steps=steps)
    as_ofs = [GWKey(season, gw) for gw in range(fit_gw, test_to_gw + 1)]
    frame = walk_forward(model, pit, as_ofs, steps=steps)
    if frame.is_empty() or fit_gw == test_from_gw:
        return frame
    return frame.filter(
        (pl.col("as_of_gw") >= test_from_gw) & (pl.col("as_of_gw") <= test_to_gw)
    )


__all__ = [
    "LedgerCacheKey",
    "ProduceLedgerResult",
    "StaleLedgerCacheError",
    "build_ledger_cache_key",
    "hash_staged_data",
    "ledger_cache_dir",
    "n_sims_of",
    "produce_ledger",
    "read_ledger_cache",
    "repo_root",
    "resolved_spec_sha256",
    "uv_lock_sha256",
    "write_ledger_cache",
]
