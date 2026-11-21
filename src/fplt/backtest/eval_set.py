"""Shared model set for multi-model eval and fair compare runs."""

from __future__ import annotations

from fplt.models.external.openfpl import DEFAULT_MODEL_DIR
from fplt.models.spec import load_model_specs

CORE_MODELS: tuple[str, ...] = (
    "rolling_mean_v1",
    "ep_next_v1",
    "dc_layered_v1",
    "lgbm_direct_v1",
)
OPENFPL_MODEL = "openfpl_v1"
OPENFPL_COMPARE_FROM = "2024-25"


def resolve_eval_models(
    season: str,
    *,
    models: list[str] | None = None,
    with_openfpl: bool = False,
    openfpl_model_dir: str | None = None,
) -> list[str]:
    """Choose which YAML specs to backtest for ``season``.

    Defaults to ``CORE_MODELS``. OpenFPL is added when ``with_openfpl`` is
    true, ``openfpl_model_dir`` is set, or the YAML ``params.model_dir`` is a
    path other than the fetched default checkout. Seasons before
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


def openfpl_skip_note(
    season: str,
    chosen: list[str],
    *,
    with_openfpl: bool,
    openfpl_model_dir: str | None,
) -> str | None:
    """Explain why OpenFPL was left out of the default set, if it was."""
    if OPENFPL_MODEL in chosen:
        return None
    if with_openfpl or openfpl_model_dir:
        return None
    if _season_start(season) < _season_start(OPENFPL_COMPARE_FROM):
        return (
            f"skipped {OPENFPL_MODEL}: season {season} is before {OPENFPL_COMPARE_FROM}"
        )
    return (
        f"skipped {OPENFPL_MODEL}: pass --with-openfpl or --openfpl-model-dir "
        "to include it"
    )


def _yaml_openfpl_model_dir_set() -> bool:
    specs = load_model_specs()
    openfpl = specs.get(OPENFPL_MODEL)
    if openfpl is None:
        return False
    value = openfpl.params.get("model_dir")
    if not isinstance(value, str):
        return False
    text = value.strip()
    return bool(text) and text != DEFAULT_MODEL_DIR


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
