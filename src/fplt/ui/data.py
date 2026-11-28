"""Read-only dashboard adapter over ``Store`` plus local fallbacks.

Full catalog helpers (``Store.official``, ``Store.runs``, ``status.json``)
land with the Stage C catalog PR. Until then this adapter scans
``$FPLT_DATA_DIR`` and keeps a ``Store`` handle ready for those methods.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, cast

import polars as pl

from fplt.analysis.last_week import post_gw_catalog
from fplt.analysis.overview import (
    official_pointers_frame,
    plan_summary_frame,
    quarantine_summary_frame,
    runs_catalog_frame,
)
from fplt.analysis.run_timing import discover_ops_timing, load_ops_timing_file
from fplt.automation.checkpoints import alarm_marker_path
from fplt.data.benchmark_projections import read_benchmark_archive
from fplt.data.quarantine import load_quarantine
from fplt.schemas.manifests import AppliedRecord, OfficialPointer, RunManifest
from fplt.schemas.timing import TimingRecord
from fplt.settings import get_settings
from fplt.store import Store
from fplt.store.builds import CuratedBuildSource
from fplt.store.legacy import parse_as_of_dir
from fplt.store.official import official_checkpoint_dir
from fplt.store.push import load_run_manifest
from fplt.store.runs import run_dir
from fplt.utils.timing import load_timing_records, ops_timing_path

_KNOWN_ALARMS: tuple[str, ...] = ("retry", "urgent", "missing_official_slate")

logger = logging.getLogger(__name__)

_SUCCESS = "_SUCCESS"
_MANIFEST = "manifest.json"
_OFFICIAL_RE = re.compile(
    r"^season=(?P<season>[^/]+)/gw=(?P<gw>\d+)/"
    r"checkpoint=(?P<checkpoint>[^/]+)/line=(?P<line>[^/]+)\.json$"
)


class DashboardStore:
    """Thin read adapter for the Streamlit overview (and later pages).

    Prefer ``Store`` catalog methods when they exist; otherwise scan the
    local data dir. Never writes.
    """

    def __init__(self, data_dir: Path, store: Store | None = None) -> None:
        """Bind to a local data root.

        Args:
            data_dir: ``$FPLT_DATA_DIR`` mirror.
            store: Optional pre-built ``Store`` (tests).
        """
        self.data_dir = data_dir
        self._store = store if store is not None else Store(data_dir)

    @classmethod
    def from_settings(cls) -> DashboardStore:
        """Construct from process settings."""
        return cls(get_settings().data_dir)

    def runs_catalog(self, *, limit: int = 50) -> pl.DataFrame:
        """Return recent run manifests as an overview catalog table.

        Args:
            limit: Max rows (newest ``run_id`` first).

        Returns:
            Frame from ``runs_catalog_frame``.
        """
        # TODO(catalog): call Store.list_runs / Store.runs when step 8 lands.
        _ = self._store  # keep the Store handle for the catalog swap-in
        manifests = _scan_run_manifests(self.data_dir)
        return runs_catalog_frame(manifests[:limit])

    def official_lines(self) -> pl.DataFrame:
        """Return official line pointers, or empty when none exist.

        Returns:
            Frame from ``official_pointers_frame``.
        """
        # TODO(catalog): call Store.official(...) when B12 + catalog land.
        return official_pointers_frame(_scan_official_pointers(self.data_dir))

    def latest_plan_summary(self) -> pl.DataFrame:
        """Summarise the newest local plan (run or legacy ``plans/`` tree).

        Returns:
            One-row ``plan_summary_frame``, or empty when no plan is found.
        """
        plan = self._latest_plan_frame()
        return plan_summary_frame(plan)

    def quarantine_summary(self) -> pl.DataFrame:
        """Return quarantine rollup (ops-06), empty when none.

        Returns:
            Frame from ``quarantine_summary_frame``.
        """
        return quarantine_summary_frame(load_quarantine(self.data_dir))

    def status_payload(self) -> dict[str, Any] | None:
        """Load ``status/status.json`` when present (ops-10); else ``None``.

        Returns:
            Parsed JSON object, or ``None`` when the file is missing.
        """
        # TODO(ops-10): prefer Store.status() once fplt status writes this.
        path = self.data_dir / "status" / "status.json"
        if not path.is_file():
            return None
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            logger.warning("status.json is not an object: %s", path)
            return None
        return cast(dict[str, Any], raw)

    def applied_line(self, *, season: str, gw: int, checkpoint: str) -> str | None:
        """Return the applied line name for a checkpoint, if present."""
        record = self.applied_record(season=season, gw=gw, checkpoint=checkpoint)
        return None if record is None else record.line

    def applied_record(
        self, *, season: str, gw: int, checkpoint: str
    ) -> AppliedRecord | None:
        """Return the validated ``applied.json`` for a checkpoint, if present."""
        path = (
            official_checkpoint_dir(
                self.data_dir, season=season, gw=gw, checkpoint=checkpoint
            )
            / "applied.json"
        )
        if not path.is_file():
            return None
        try:
            return AppliedRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            logger.debug("skip applied %s: %s", path, exc)
            return None

    def run_manifest(self, run_id: str) -> RunManifest | None:
        """Load a succeeded run manifest via ``Store``."""
        return self._store.manifest(run_id)

    def load_table(self, table: str, *, run_id: str) -> pl.DataFrame:
        """Collect one run table; empty frame when the file is missing."""
        try:
            return self._store.load(table, run=run_id).collect()
        except FileNotFoundError:
            return pl.DataFrame()

    def curated_tables(self, build_id: str | None) -> dict[str, pl.DataFrame]:
        """Load UI label tables from a curated build (empty when missing)."""
        names = ("players", "teams", "fixtures", "gameweeks", "player_status")
        if not build_id:
            return {name: pl.DataFrame() for name in names}
        try:
            source = CuratedBuildSource.load(self.data_dir, build_id)
        except FileNotFoundError:
            return {name: pl.DataFrame() for name in names}
        return {name: source.table(name).collect() for name in names}

    def ops_timing(
        self, *, season: str, gw: int, checkpoint: str
    ) -> TimingRecord | None:
        """Load official ``ops_timing.json`` for a checkpoint, if present."""
        path = ops_timing_path(
            official_checkpoint_dir(
                self.data_dir, season=season, gw=gw, checkpoint=checkpoint
            )
        )
        return load_ops_timing_file(path)

    def remote_timing_records(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent ``kind=remote`` timing artifacts for transfer rows."""
        return load_timing_records(self.data_dir, kind="remote", limit=limit)

    def ops_timing_history(
        self, *, season: str | None = None, checkpoint: str | None = None
    ) -> list[tuple[str, int, str, TimingRecord]]:
        """Scan pulled ``ops_timing.json`` files for trend charts."""
        return discover_ops_timing(self.data_dir, season=season, checkpoint=checkpoint)

    def post_gw_runs(self) -> pl.DataFrame:
        """Return succeeded ``purpose=post_gw`` runs for the Last-week selector.

        Returns:
            Frame from ``post_gw_catalog`` (newest GW first).
        """
        return post_gw_catalog(_scan_run_manifests(self.data_dir))

    def post_gw_scorecards(self, *, season: str, through_gw: int) -> pl.DataFrame:
        """Load and concat scorecards from post_gw runs through ``through_gw``.

        Args:
            season: Season key.
            through_gw: Inclusive gameweek ceiling.

        Returns:
            Combined scorecard rows (possibly empty).
        """
        catalog = self.post_gw_runs().filter(
            (pl.col("season") == season) & (pl.col("gw") <= through_gw)
        )
        frames: list[pl.DataFrame] = []
        for run_id in catalog["run_id"].to_list():
            frame = self.load_table("scorecard", run_id=str(run_id))
            if not frame.is_empty():
                frames.append(frame)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed")

    def benchmark_projections(self, *, season: str, gw: int) -> pl.DataFrame:
        """Load deadline ``benchmark_projections`` for ``season`` / ``gw``."""
        frame = read_benchmark_archive(self.data_dir)
        if frame.is_empty():
            return frame
        return frame.filter(
            (pl.col("as_of_season") == season) & (pl.col("as_of_gw") == gw)
        )

    def player_outcomes(self, *, season: str, gw: int) -> pl.DataFrame:
        """Load player–fixture outcomes for one GW (empty when absent)."""
        from fplt.data.store import DataStore

        try:
            frame = DataStore(self.data_dir).read("player_fixture_outcomes")
        except (OSError, KeyError, ValueError):
            return pl.DataFrame()
        if frame.is_empty():
            return frame
        part = frame.filter((pl.col("season") == season) & (pl.col("gw") == gw))
        if part.is_empty() or "pts" not in part.columns:
            return pl.DataFrame()
        return part.select("player_key", "fixture_id", "pts")

    def official_predict_table(
        self, *, season: str, gw: int, line: str | None = None
    ) -> pl.DataFrame:
        """Load primary (or named) T−90m ``player_fixture`` predictions."""
        pointers = self.official_lines().filter(
            (pl.col("season") == season)
            & (pl.col("gw") == gw)
            & (pl.col("checkpoint") == "T-90m")
        )
        if pointers.is_empty():
            pointers = self.official_lines().filter(
                (pl.col("season") == season) & (pl.col("gw") == gw)
            )
        if pointers.is_empty():
            return pl.DataFrame()
        if line is not None:
            hit = pointers.filter(pl.col("line") == line)
            if not hit.is_empty():
                pointers = hit
        elif "role" in pointers.columns:
            primary = pointers.filter(pl.col("role") == "primary")
            if not primary.is_empty():
                pointers = primary
        run_id = str(pointers["predict_run_id"][0])
        return self.load_table("player_fixture", run_id=run_id)

    def gw_alarms(self, *, season: str, gw: int) -> list[dict[str, Any]]:
        """Return alarm marker summaries for a gameweek (empty when none)."""
        from fplt.types import GWKey

        rows: list[dict[str, Any]] = []
        key = GWKey(season, gw)
        for alarm_id in _KNOWN_ALARMS:
            path = alarm_marker_path(self.data_dir, key, alarm_id)
            if not path.is_file():
                continue
            try:
                payload: object = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                rows.append({"key": alarm_id, "value": "present", "detail": None})
                continue
            if not isinstance(payload, dict):
                rows.append({"key": alarm_id, "value": "present", "detail": None})
                continue
            data = cast(dict[str, Any], payload)
            rows.append(
                {
                    "key": str(data.get("alarm", alarm_id)),
                    "value": str(data.get("kind", "alarm")),
                    "detail": None
                    if data.get("written_at") is None
                    else str(data.get("written_at")),
                }
            )
        return rows

    def status_freshness_rows(self) -> list[dict[str, Any]]:
        """Shape ``status.json`` capture / heartbeat freshness for ops review."""
        payload = self.status_payload()
        if payload is None:
            return []
        rows: list[dict[str, Any]] = []
        captures = payload.get("last_captures")
        if isinstance(captures, list):
            for item_raw in cast(list[object], captures):
                if not isinstance(item_raw, dict):
                    continue
                item = cast(dict[str, Any], item_raw)
                source = item.get("source")
                name_alt = item.get("name")
                name = (
                    str(source)
                    if source is not None
                    else str(name_alt)
                    if name_alt is not None
                    else "capture"
                )
                at = item.get("captured_at")
                if at is None:
                    at = item.get("last_at")
                reason = item.get("reason")
                rows.append(
                    {
                        "key": name,
                        "value": None if at is None else str(at),
                        "detail": None if reason is None else str(reason),
                    }
                )
        heartbeats = payload.get("heartbeats")
        if isinstance(heartbeats, list):
            for item_raw in cast(list[object], heartbeats):
                if not isinstance(item_raw, dict):
                    continue
                item = cast(dict[str, Any], item_raw)
                hb_name = item.get("name")
                last_at = item.get("last_at")
                reason = item.get("reason")
                rows.append(
                    {
                        "key": f"hb:{'job' if hb_name is None else hb_name}",
                        "value": None if last_at is None else str(last_at),
                        "detail": None if reason is None else str(reason),
                    }
                )
        return rows

    def status_last_error(self) -> str | None:
        """Return the last-error message from ``status.json``, if any."""
        payload = self.status_payload()
        if payload is None:
            return None
        err_raw = payload.get("last_error")
        if not isinstance(err_raw, dict):
            return None
        err = cast(dict[str, Any], err_raw)
        message = err.get("message")
        return None if message is None else str(message)

    def _latest_plan_frame(self) -> pl.DataFrame:
        for run_id in reversed(_succeeded_run_ids(self.data_dir)):
            path = run_dir(self.data_dir, run_id) / "plans" / "plan.parquet"
            if path.is_file():
                return pl.read_parquet(path)
        return _latest_legacy_plan(self.data_dir)


def _scan_run_manifests(data_dir: Path) -> list[RunManifest]:
    manifests: list[RunManifest] = []
    for run_id in reversed(_succeeded_run_ids(data_dir)):
        try:
            manifests.append(load_run_manifest(data_dir, run_id))
        except (FileNotFoundError, ValueError) as exc:
            logger.debug("skip run %s: %s", run_id, exc)
    # Include non-succeeded manifests that still have a file (running/failed).
    root = data_dir / "runs"
    if not root.is_dir():
        return manifests
    seen = {item.run_id for item in manifests}
    for path in sorted(root.glob("run=*"), reverse=True):
        if not path.is_dir():
            continue
        run_id = path.name.removeprefix("run=")
        if run_id in seen:
            continue
        manifest_path = path / _MANIFEST
        if not manifest_path.is_file():
            continue
        try:
            manifests.append(
                RunManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
            )
        except ValueError as exc:
            logger.debug("skip invalid manifest %s: %s", run_id, exc)
    manifests.sort(key=lambda item: item.run_id, reverse=True)
    return manifests


def _succeeded_run_ids(data_dir: Path) -> list[str]:
    root = data_dir / "runs"
    if not root.is_dir():
        return []
    ids: list[str] = []
    for path in sorted(root.glob("run=*")):
        if path.is_dir() and (path / _SUCCESS).is_file():
            ids.append(path.name.removeprefix("run="))
    return ids


def _scan_official_pointers(
    data_dir: Path,
) -> list[tuple[str, int, str, str, OfficialPointer]]:
    root = data_dir / "official"
    if not root.is_dir():
        return []
    rows: list[tuple[str, int, str, str, OfficialPointer]] = []
    for path in sorted(root.rglob("line=*.json")):
        rel = path.relative_to(root).as_posix()
        match = _OFFICIAL_RE.match(rel)
        if match is None:
            continue
        try:
            pointer = OfficialPointer.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            logger.debug("skip official pointer %s: %s", rel, exc)
            continue
        rows.append(
            (
                match.group("season"),
                int(match.group("gw")),
                match.group("checkpoint"),
                match.group("line"),
                pointer,
            )
        )
    return rows


def _latest_legacy_plan(data_dir: Path) -> pl.DataFrame:
    root = data_dir / "plans"
    if not root.is_dir():
        return pl.DataFrame()
    candidates: list[tuple[str, Path]] = []
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        for as_dir in model_dir.glob("as_of=*"):
            parsed = parse_as_of_dir(as_dir.name)
            if parsed is None:
                continue
            part = as_dir / "part.parquet"
            if part.is_file():
                candidates.append((as_dir.name, part))
    if not candidates:
        return pl.DataFrame()
    candidates.sort(key=lambda item: item[0], reverse=True)
    return pl.read_parquet(candidates[0][1])
