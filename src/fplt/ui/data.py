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

from fplt.analysis.overview import (
    official_pointers_frame,
    plan_summary_frame,
    quarantine_summary_frame,
    runs_catalog_frame,
)
from fplt.data.quarantine import load_quarantine
from fplt.schemas.manifests import OfficialPointer, RunManifest
from fplt.settings import get_settings
from fplt.store import Store
from fplt.store.legacy import parse_as_of_dir
from fplt.store.push import load_run_manifest
from fplt.store.runs import run_dir

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
