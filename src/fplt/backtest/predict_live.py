"""Fit or load a predictor and write live predictions for the next gameweeks."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

# Importing the package registers the predictors with get_model.
import fplt.models.baselines  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.composite  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.direct  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.ensemble  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.external.openfpl  # noqa: F401  # pyright: ignore[reportUnusedImport]
from fplt.backtest.ledger import append_predictions
from fplt.backtest.walk_forward import allowed_seasons, horizon_for, ledger_rows
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.contracts import PointsPredictor
from fplt.models.predictions import require_scored_predictions, write_predictions
from fplt.models.spec import ModelSpec, artifact_dir, get_model, load_model_specs
from fplt.types import GWKey

logger = logging.getLogger(__name__)


def resolve_as_of(
    store: DataStore,
    *,
    season: str | None = None,
    gw: int | None = None,
    now: datetime | None = None,
) -> GWKey:
    """Pick the deadline to predict from.

    When ``season`` and ``gw`` are set, that deadline is used. Otherwise the
    earliest staged gameweek whose deadline is still after ``now`` wins. If
    every deadline has passed, the earliest unchecked gameweek is used.

    Args:
        store: Staged tables, including ``gameweeks``.
        season: Optional season label such as ``"2024-25"``.
        gw: Optional gameweek number. Must be set with ``season``.
        now: Clock reading. Defaults to the current UTC time.

    Returns:
        The as-of gameweek key.

    Raises:
        ValueError: ``season``/``gw`` are mismatched, out of range, or no
            staged gameweek can be predicted.
        KeyError: An explicit ``season``/``gw`` is not in the store.
    """
    if (season is None) != (gw is None):
        raise ValueError("season and gw must be given together")
    if season is not None and gw is not None:
        if not 1 <= gw <= 38:
            raise ValueError(f"gameweek must be in 1..38, got {gw}")
        key = GWKey(season, gw)
        store.deadline(key)
        return key

    frame = store.read("gameweeks")
    if frame.is_empty():
        raise ValueError("no staged gameweeks to predict")
    latest = frame.sort("available_at").unique(subset=["season", "gw"], keep="last")
    moment = _aware(now if now is not None else datetime.now(UTC))
    upcoming: list[tuple[datetime, GWKey]] = []
    unchecked: list[GWKey] = []
    for row in latest.iter_rows(named=True):
        key = GWKey(str(row["season"]), int(row["gw"]))
        deadline = _aware(row["deadline_time"])
        if deadline > moment:
            upcoming.append((deadline, key))
        if not bool(row["data_checked"]):
            unchecked.append(key)
    if upcoming:
        upcoming.sort(key=lambda item: (item[0], item[1]))
        return upcoming[0][1]
    if unchecked:
        return sorted(unchecked)[0]
    raise ValueError("no upcoming gameweek to predict")


def run_predict(
    data_dir: Path,
    model_name: str,
    horizon: int,
    *,
    season: str | None = None,
    gw: int | None = None,
    now: datetime | None = None,
    append_ledger: bool = True,
) -> Path:
    """Fit or load ``model_name`` and write predictions for the next gameweeks.

    Loads an existing artifact for the resolved as-of deadline when present;
    otherwise fits on the PIT view and saves the artifact. Predictions are
    written under ``predictions/<model_id>/as_of=.../part.parquet``. When
    ``append_ledger`` is true, the same rows are appended to the ledger with
    ``run_kind="live"``.

    Args:
        data_dir: Root that contains ``staged/`` and receives ``predictions/``.
        model_name: Spec name from ``configs/models``.
        horizon: Number of gameweeks to predict, starting at the as-of.
        season: Optional season override for the as-of deadline.
        gw: Optional gameweek override. Must be set with ``season``.
        now: Clock used when resolving the as-of automatically.
        append_ledger: Also append ``LedgerSchema`` rows for this run.

    Returns:
        The parquet path written under ``predictions/``.

    Raises:
        ValueError: The horizon, model name, or as-of resolution is invalid,
            or every predicted ``xpts`` is 0.
        KeyError: The model kind is unregistered, or an explicit gameweek is
            missing from staged data.
        FileExistsError: Predictions for this model and deadline already exist.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    specs = load_model_specs()
    if model_name not in specs:
        raise ValueError(f"unknown model spec {model_name!r}")
    spec = specs[model_name]
    store = DataStore(data_dir)
    as_of = resolve_as_of(store, season=season, gw=gw, now=now)
    logger.info(
        "predict model=%s as_of=%s horizon=%s",
        model_name,
        as_of,
        horizon,
    )
    pit = PITData(store)
    seasons = allowed_seasons(as_of, spec.train_window.seasons_back)
    view = pit.view(as_of).within_seasons(seasons)
    model = _fit_or_load(data_dir, spec, as_of, view)
    predicted = model.predict(view, horizon_for(as_of, horizon))
    require_scored_predictions(predicted, model_name)
    path = write_predictions(data_dir, spec, as_of, predicted)
    logger.info("wrote %s prediction rows to %s", predicted.height, path)
    if append_ledger:
        append_predictions(data_dir, ledger_rows(predicted, view, as_of, "live"))
    return path


def _fit_or_load(
    data_dir: Path, spec: ModelSpec, as_of: GWKey, view: PITView
) -> PointsPredictor:
    dest = artifact_dir(data_dir, spec, as_of)
    cls = get_model(spec.kind)
    if (dest / "spec.json").exists():
        logger.info("loading artifact from %s", dest)
        loaded: PointsPredictor = cls.load(dest, spec)
        return loaded
    logger.info("fitting %s and saving to %s", spec.name, dest)
    model: PointsPredictor = cls(spec)
    model.fit(view)
    model.save(dest)
    return model


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
