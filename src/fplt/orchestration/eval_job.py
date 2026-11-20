"""Multi-model walk-forward eval into the prediction ledger.

Runs the core leaderboard set (baselines, layered, LightGBM) for one season
and optionally OpenFPL when configured. Afterward ``make leaderboard`` reads
the same ledger. This is not a CI quality gate.
"""

# Dagster does not ship type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUnknownParameterType=false
# pyright: reportMissingTypeArgument=false
#
# Do not enable ``from __future__ import annotations`` here: Dagster resolves
# the ``config: EvalBoardConfig`` annotation at decoration time.

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import dagster as dg
from dagster import Config

# Side-effect imports so kinds resolve even before backtest/run.py registers
# them (see PR #27 for the shared import on the CLI path).
import fplt.models.direct  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.external.openfpl  # noqa: F401  # pyright: ignore[reportUnusedImport]
from fplt.backtest.ledger import append_predictions, fill_actuals
from fplt.backtest.run import run_backtest
from fplt.backtest.walk_forward import walk_forward
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import ModelSpec, get_model, load_model_specs
from fplt.settings import get_settings
from fplt.types import GWKey

logger = logging.getLogger(__name__)

CORE_MODELS: tuple[str, ...] = (
    "rolling_mean_v1",
    "ep_next_v1",
    "dc_layered_v1",
    "lgbm_direct_v1",
)
OPENFPL_MODEL = "openfpl_v1"
OPENFPL_COMPARE_FROM = "2024-25"

Refit = Literal["every_gw", "every_n_gw", "never"]


@dataclass(frozen=True)
class EvalResult:
    """Specs that wrote predictions during one eval run."""

    specs: tuple[ModelSpec, ...]
    skipped_openfpl: str | None = None


class EvalBoardConfig(Config):
    """Dagster run config for ``eval_board_job``."""

    season: str
    from_gw: int = 1
    to_gw: int = 38
    with_openfpl: bool = False
    openfpl_model_dir: str | None = None
    steps: int = 1


def resolve_eval_models(
    season: str,
    *,
    models: list[str] | None = None,
    with_openfpl: bool = False,
    openfpl_model_dir: str | None = None,
) -> list[str]:
    """Choose which YAML specs to backtest for ``season``.

    Defaults to ``CORE_MODELS``. OpenFPL is added when ``with_openfpl`` is
    true, ``openfpl_model_dir`` is set, or ``configs/models/openfpl_v1.yaml``
    already has a non-empty ``params.model_dir``. Seasons before
    ``OPENFPL_COMPARE_FROM`` cannot include OpenFPL.

    Args:
        season: Evaluation season such as ``"2024-25"``.
        models: Explicit spec names; when set, OpenFPL flags are ignored.
        with_openfpl: Force-include ``openfpl_v1``.
        openfpl_model_dir: Non-empty path implies OpenFPL should run.

    Returns:
        Spec names in run order.

    Raises:
        ValueError: OpenFPL was requested for a training-data season, or an
            explicit list is empty.
    """
    if models is not None:
        if not models:
            raise ValueError("models must be non-empty")
        _reject_early_openfpl(season, models)
        return list(models)
    chosen = list(CORE_MODELS)
    include_openfpl = (
        with_openfpl or bool(openfpl_model_dir) or _yaml_openfpl_model_dir_set()
    )
    if include_openfpl:
        _reject_early_openfpl(season, [OPENFPL_MODEL])
        chosen.append(OPENFPL_MODEL)
    return chosen


def run_eval_board(
    data_dir: Path,
    season: str,
    from_gw: int = 1,
    to_gw: int = 38,
    *,
    models: list[str] | None = None,
    with_openfpl: bool = False,
    openfpl_model_dir: str | None = None,
    refit: Refit | None = None,
    seasons_back: int | None = None,
    steps: int = 1,
) -> EvalResult:
    """Walk-forward backtest the eval set into ``data_dir``'s ledger.

    OpenFPL uses ``openfpl_model_dir`` when provided (overrides the YAML
    ``model_dir``). An empty ``model_dir`` makes the OpenFPL runner emit
    zeros — those rows are not real OpenFPL scores. Set ``model_dir`` (YAML
    or ``--openfpl-model-dir``) after ``uv sync`` in ``external/openfpl``.

    Args:
        data_dir: Root with ``staged/``; receives ``ledger/``.
        season: Evaluation season.
        from_gw: First as-of gameweek.
        to_gw: Last as-of gameweek, inclusive.
        models: Explicit specs; default is the core set (± OpenFPL).
        with_openfpl: Include OpenFPL even when YAML ``model_dir`` is empty.
        openfpl_model_dir: Optional override for OpenFPL's fitted ensemble.
        refit: Override each spec's refit policy.
        seasons_back: Override each spec's training window.
        steps: Horizon length passed to ``walk_forward``.

    Returns:
        Specs that ran, plus a skip note when OpenFPL was declined.

    Raises:
        ValueError: Invalid range, unknown spec, or OpenFPL season floor.
        KeyError: A spec kind is not registered.
    """
    chosen = resolve_eval_models(
        season,
        models=models,
        with_openfpl=with_openfpl,
        openfpl_model_dir=openfpl_model_dir,
    )
    skipped = (
        None
        if models is not None
        else _openfpl_skip_note(
            season,
            chosen,
            with_openfpl=with_openfpl,
            openfpl_model_dir=openfpl_model_dir,
        )
    )
    others = [name for name in chosen if name != OPENFPL_MODEL]
    include_openfpl = OPENFPL_MODEL in chosen
    ran: list[ModelSpec] = []
    if others:
        ran.extend(
            run_backtest(
                data_dir,
                season,
                from_gw,
                to_gw,
                others,
                refit=refit,
                seasons_back=seasons_back,
                steps=steps,
            )
        )
    if include_openfpl:
        ran.append(
            _run_openfpl(
                data_dir,
                season,
                from_gw,
                to_gw,
                model_dir=openfpl_model_dir,
                refit=refit,
                seasons_back=seasons_back,
                steps=steps,
            )
        )
        fill_actuals(data_dir, DataStore(data_dir))
    if skipped:
        logger.info(skipped)
    return EvalResult(specs=tuple(ran), skipped_openfpl=skipped)


def _run_openfpl(
    data_dir: Path,
    season: str,
    from_gw: int,
    to_gw: int,
    *,
    model_dir: str | None,
    refit: Refit | None,
    seasons_back: int | None,
    steps: int,
) -> ModelSpec:
    """Backtest OpenFPL, optionally overriding ``params.model_dir``."""
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if seasons_back is not None and seasons_back < 1:
        raise ValueError(f"seasons_back must be positive, got {seasons_back}")
    if not 1 <= from_gw <= to_gw <= 38:
        raise ValueError(f"gameweeks must be in 1..38, got {from_gw}..{to_gw}")
    specs = load_model_specs()
    if OPENFPL_MODEL not in specs:
        raise ValueError(f"unknown model spec {OPENFPL_MODEL!r}")
    spec = _configured(specs[OPENFPL_MODEL], refit, seasons_back)
    if model_dir is not None:
        params = dict(spec.params)
        params["model_dir"] = model_dir
        spec = spec.model_copy(update={"params": params})
    if not str(spec.params.get("model_dir") or "").strip():
        logger.warning(
            "openfpl_v1 model_dir is empty; the runner returns zeros "
            "(not OpenFPL scores). Set params.model_dir or --openfpl-model-dir."
        )
    store = DataStore(data_dir)
    pit = PITData(store)
    as_ofs = [GWKey(season, gw) for gw in range(from_gw, to_gw + 1)]
    model = get_model(spec.kind)(spec)
    frame = walk_forward(model, pit, as_ofs, steps=steps)
    append_predictions(data_dir, frame)
    return spec


def _configured(
    spec: ModelSpec, refit: Refit | None, seasons_back: int | None
) -> ModelSpec:
    updates: dict[str, object] = {}
    if refit is not None:
        updates["refit"] = refit
    if seasons_back is not None:
        updates["train_window"] = spec.train_window.model_copy(
            update={"seasons_back": seasons_back}
        )
    if not updates:
        return spec
    return spec.model_copy(update=updates)


def _yaml_openfpl_model_dir_set() -> bool:
    specs = load_model_specs()
    openfpl = specs.get(OPENFPL_MODEL)
    if openfpl is None:
        return False
    value = openfpl.params.get("model_dir")
    return isinstance(value, str) and bool(value.strip())


def _reject_early_openfpl(season: str, models: list[str]) -> None:
    if OPENFPL_MODEL not in models:
        return
    if _season_start(season) < _season_start(OPENFPL_COMPARE_FROM):
        raise ValueError(
            f"OpenFPL is only comparable from {OPENFPL_COMPARE_FROM} onwards, "
            f"got {season}"
        )


def _season_start(season: str) -> int:
    head, _, tail = season.partition("-")
    if len(head) != 4 or not head.isdigit() or len(tail) != 2:
        raise ValueError(f"season must look like 2024-25, got {season!r}")
    return int(head)


def _openfpl_skip_note(
    season: str,
    chosen: list[str],
    *,
    with_openfpl: bool,
    openfpl_model_dir: str | None,
) -> str | None:
    if OPENFPL_MODEL in chosen:
        return None
    if with_openfpl or openfpl_model_dir:
        return None
    if _season_start(season) < _season_start(OPENFPL_COMPARE_FROM):
        return (
            f"skipped {OPENFPL_MODEL}: season {season} is before {OPENFPL_COMPARE_FROM}"
        )
    return (
        f"skipped {OPENFPL_MODEL}: set params.model_dir in its YAML, "
        "pass --openfpl-model-dir, or --with-openfpl"
    )


@dg.op
def eval_board_op(context: dg.OpExecutionContext, config: EvalBoardConfig) -> list[str]:
    """Run the multi-model eval for the configured season."""
    result = run_eval_board(
        get_settings().data_dir,
        config.season,
        config.from_gw,
        config.to_gw,
        with_openfpl=config.with_openfpl,
        openfpl_model_dir=config.openfpl_model_dir,
        steps=config.steps,
    )
    names = [spec.name for spec in result.specs]
    context.log.info("eval_board ran %s", ", ".join(names) or "nothing")
    if result.skipped_openfpl:
        context.log.info(result.skipped_openfpl)
    return names


@dg.job(name="eval_board")
def eval_board_job() -> None:
    """Walk-forward the core models (and optional OpenFPL) into the ledger."""
    eval_board_op()
