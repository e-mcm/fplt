"""Thin multi-arm prediction bakeoff on top of promote-eval / compare.

Championship ranking of several ``ModelSpec`` names on one shared window.
Reuses ``compare_models`` (content-addressed ledger cache), the intervals
SPA/MCS helpers, and optionally pairwise ``execute_promote_eval`` runs.
This is not a second framework and does not invent metric math.

A bakeoff on one window is descriptive. It does not emit ``keep``; confirmation
is a fresh window (feature-eval-pipeline §9).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fplt.backtest.compare import CompareResult, compare_models, format_summary_table
from fplt.backtest.decision_pool import HEADLINE_METRIC, filter_decision_pool
from fplt.backtest.intervals import (
    multi_arm_selection,
    per_gw_losses,
)
from fplt.backtest.ledger import read_predictions
from fplt.backtest.promote_eval import (
    ReportSpec,
    SuitesSpec,
    WindowSpec,
    default_evals_dir,
    execute_promote_eval,
    load_eval_spec,
)
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs

logger = logging.getLogger(__name__)

CompareModeName = Literal["walk_forward", "fixed"]


class BakeoffSpec(BaseModel):
    """Championship bakeoff: N ModelSpecs, one control, one shared window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    question: str
    description: str = ""
    control: str
    models: list[str] = Field(min_length=3)
    window: WindowSpec = Field(default_factory=WindowSpec)
    suites: SuitesSpec = Field(default_factory=SuitesSpec)
    pairwise_evals: list[str] = Field(default_factory=list)
    report: ReportSpec = Field(default_factory=ReportSpec)

    @field_validator("question")
    @classmethod
    def _question_nonblank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must be non-blank")
        return cleaned

    @model_validator(mode="after")
    def _control_in_models(self) -> BakeoffSpec:
        if self.control not in self.models:
            raise ValueError(f"control {self.control!r} must be listed in models")
        if len(set(self.models)) != len(self.models):
            raise ValueError("models must be unique")
        if self.suites.decisions:
            raise ValueError(
                "bakeoff.suites.decisions must be false; use fplt decisions "
                "for the decision scoreboard"
            )
        return self


class BakeoffPlan(BaseModel):
    """Resolved dry-run plan for one BakeoffSpec."""

    model_config = ConfigDict(frozen=True)

    bakeoff_name: str
    control: str
    models: list[str]
    seasons: list[str]
    from_gw: int
    to_gw: int
    seasons_back: int
    modes: list[CompareModeName]
    suites: SuitesSpec
    pairwise_evals: list[str]
    commands: list[str]
    warnings: list[str] = Field(default_factory=list)
    dry_run: bool = True


@dataclass(frozen=True)
class BakeoffResult:
    """Artifacts from an executed (or dry-run) bakeoff."""

    bakeoff_name: str
    exit_code: int
    output_dir: Path
    plan: BakeoffPlan
    ranking: str
    compare: CompareResult | None = None
    summary: pl.DataFrame | None = None
    spa: dict[str, Any] | None = None
    mcs: dict[str, Any] | None = None
    pairwise: tuple[str, ...] = ()


def load_bakeoff_spec(name: str, directory: Path | None = None) -> BakeoffSpec:
    """Load one BakeoffSpec YAML by ``name``.

    Args:
        name: Stem of ``configs/evals/<name>.yaml`` (or a path stem).
        directory: Override evals folder (tests).

    Returns:
        Validated ``BakeoffSpec``.

    Raises:
        FileNotFoundError: No matching YAML.
        ValueError: YAML is empty, not a mapping, or fails validation.
    """
    folder = directory if directory is not None else default_evals_dir()
    path = folder / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    if "kind" in raw:
        raise ValueError(
            f"{path.name} looks like an EvalSpec (has kind=); use "
            "fplt promote-eval for pairwise A/B, or a bakeoff YAML without kind"
        )
    spec = BakeoffSpec.model_validate(raw)
    if spec.name != name:
        raise ValueError(
            f"BakeoffSpec name {spec.name!r} does not match file stem {name!r}"
        )
    return spec


def bakeoff_from_models(
    *,
    name: str,
    control: str,
    models: list[str],
    seasons: list[str],
    from_gw: int,
    to_gw: int,
    seasons_back: int = 4,
    modes: list[CompareModeName] | None = None,
    steps: int = 1,
    question: str | None = None,
    pairwise_evals: list[str] | None = None,
) -> BakeoffSpec:
    """Build a BakeoffSpec from CLI flags (no YAML required).

    Args:
        name: Artifact folder name.
        control: SPA benchmark ModelSpec.
        models: All arms including ``control`` (at least three).
        seasons: Eval seasons.
        from_gw: First scored gameweek.
        to_gw: Last scored gameweek.
        seasons_back: Training window length override for compare.
        modes: Compare modes (default walk_forward only).
        steps: Prediction horizon steps.
        question: Ranking question; a default is used when omitted.
        pairwise_evals: Optional EvalSpec names to run after the multi-arm compare.

    Returns:
        Validated ``BakeoffSpec``.
    """
    chosen_modes = list(modes) if modes else ["walk_forward"]
    return BakeoffSpec(
        name=name,
        question=question
        or (
            f"Which of {', '.join(models)} ranks best vs control "
            f"{control} on the shared window?"
        ),
        control=control,
        models=list(models),
        window=WindowSpec(
            seasons=list(seasons),
            from_gw=from_gw,
            to_gw=to_gw,
            seasons_back=seasons_back,
            modes=chosen_modes,  # type: ignore[arg-type]
        ),
        suites=SuitesSpec(
            prediction=True,
            components=False,
            decisions=False,
            steps=steps,
        ),
        pairwise_evals=list(pairwise_evals or []),
        report=ReportSpec(title=f"Bakeoff: {name}"),
    )


def plan_bakeoff(
    spec: BakeoffSpec,
    *,
    dry_run: bool = True,
    from_gw: int | None = None,
    to_gw: int | None = None,
    seasons: list[str] | None = None,
    models_dir: Path | None = None,
) -> BakeoffPlan:
    """Build the command plan for a BakeoffSpec (dry-run safe).

    Args:
        spec: Loaded bakeoff.
        dry_run: When true, plan only.
        from_gw: Override window start.
        to_gw: Override window end.
        seasons: Override season list.
        models_dir: Optional ``configs/models`` override for existence checks.

    Returns:
        Resolved ``BakeoffPlan`` with printable commands and warnings.
    """
    window = spec.window
    start = from_gw if from_gw is not None else window.from_gw
    end = to_gw if to_gw is not None else window.to_gw
    season_list = list(seasons) if seasons is not None else list(window.seasons)
    if start > end:
        raise ValueError(f"from_gw {start} > to_gw {end}")

    warnings: list[str] = []
    model_names = set(load_model_specs(models_dir))
    for arm in spec.models:
        if arm not in model_names:
            warnings.append(f"ModelSpec {arm!r} not found under configs/models")
    evals_folder = default_evals_dir()
    if models_dir is not None:
        # tests pass a models dir; pair with sibling ../evals when present
        sibling = models_dir.parent / "evals"
        if sibling.is_dir():
            evals_folder = sibling
    for eval_name in spec.pairwise_evals:
        if not (evals_folder / f"{eval_name}.yaml").is_file():
            warnings.append(f"pairwise EvalSpec {eval_name!r} not found")

    commands: list[str] = []
    arms = " ".join(spec.models)
    mode_flag = " ".join(window.modes)
    for season in season_list:
        if spec.suites.prediction:
            commands.append(
                "uv run fplt compare "
                f"--season {season} --models {arms} "
                f"--test-from-gw {start} --test-to-gw {end} "
                f"--seasons-back {window.seasons_back} --modes {mode_flag} "
                f"--steps {spec.suites.steps}"
            )
        if spec.suites.components:
            commands.append(
                f"# components suite: run eval-components per arm on {season} "
                "(bakeoff does not reimplement slot flags)"
            )
    for eval_name in spec.pairwise_evals:
        commands.append(f"uv run fplt promote-eval --eval {eval_name} --execute")
    commands.append("# rank models → ranking.md (SPA / MCS; descriptive, not keep)")
    if not commands:
        commands.append("# no suites enabled")

    return BakeoffPlan(
        bakeoff_name=spec.name,
        control=spec.control,
        models=list(spec.models),
        seasons=season_list,
        from_gw=start,
        to_gw=end,
        seasons_back=window.seasons_back,
        modes=list(window.modes),
        suites=spec.suites,
        pairwise_evals=list(spec.pairwise_evals),
        commands=commands,
        warnings=warnings,
        dry_run=dry_run,
    )


def format_bakeoff_plan(plan: BakeoffPlan) -> str:
    """Render a bakeoff plan for stdout."""
    lines = [
        f"bakeoff: {plan.bakeoff_name}",
        f"control: {plan.control}",
        f"models: {' '.join(plan.models)}",
        f"window: seasons={plan.seasons} gw={plan.from_gw}-{plan.to_gw} "
        f"seasons_back={plan.seasons_back} modes={list(plan.modes)}",
        f"pairwise_evals: {plan.pairwise_evals or '(none)'}",
        "commands:",
    ]
    for cmd in plan.commands:
        lines.append(f"  {cmd}")
    if plan.warnings:
        lines.append("warnings:")
        for warning in plan.warnings:
            lines.append(f"  - {warning}")
    lines.append(
        "note: ranking on this window is descriptive; keep needs a fresh window"
    )
    return "\n".join(lines) + "\n"


def plan_to_json(plan: BakeoffPlan) -> str:
    """Serialize a bakeoff plan as JSON."""
    return plan.model_dump_json(indent=2) + "\n"


def execute_bakeoff(
    spec: BakeoffSpec,
    *,
    data_dir: Path,
    dry_run: bool = False,
    from_gw: int | None = None,
    to_gw: int | None = None,
    seasons: list[str] | None = None,
    models_dir: Path | None = None,
    output_dir: Path | None = None,
    with_openfpl: bool = False,
    openfpl_model_dir: str | None = None,
    use_cache: bool = True,
    evals_dir: Path | None = None,
) -> BakeoffResult:
    """Plan, optionally compare all arms, and write the ranking report.

    Args:
        spec: Loaded BakeoffSpec.
        data_dir: Staged data root (also the ledger / cache root).
        dry_run: When true, write plan only.
        from_gw: Override window start.
        to_gw: Override window end.
        seasons: Override seasons.
        models_dir: Optional models override.
        output_dir: Artifact directory override.
        with_openfpl: Forwarded to compare when OpenFPL is in the set.
        openfpl_model_dir: Optional OpenFPL ``model_dir``.
        use_cache: Reuse content-addressed ledgers (default true).
        evals_dir: Override for pairwise EvalSpec loading.

    Returns:
        ``BakeoffResult`` including exit code for the CLI.
    """
    plan = plan_bakeoff(
        spec,
        dry_run=dry_run,
        from_gw=from_gw,
        to_gw=to_gw,
        seasons=seasons,
        models_dir=models_dir,
    )
    out = output_dir or _default_output_dir(data_dir, spec.name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(plan_to_json(plan), encoding="utf-8")

    if dry_run:
        text = format_bakeoff_plan(plan)
        (out / "ranking.md").write_text(text, encoding="utf-8")
        return BakeoffResult(
            bakeoff_name=spec.name,
            exit_code=0,
            output_dir=out,
            plan=plan,
            ranking=text,
        )

    if plan.warnings and any("not found" in w for w in plan.warnings):
        msg = "; ".join(plan.warnings)
        text = f"bakeoff preflight failed: {msg}\n"
        (out / "ranking.md").write_text(text, encoding="utf-8")
        return BakeoffResult(
            bakeoff_name=spec.name,
            exit_code=2,
            output_dir=out,
            plan=plan,
            ranking=text,
        )

    compare_result: CompareResult | None = None
    summaries: list[pl.DataFrame] = []
    for season in plan.seasons:
        if len(plan.seasons) > 1:
            season_out = out / "compare" / season
        else:
            season_out = out / "compare"
        part = compare_models(
            data_dir,
            season,
            models=list(plan.models),
            test_from_gw=plan.from_gw,
            test_to_gw=plan.to_gw,
            seasons_back=plan.seasons_back,
            modes=list(plan.modes),  # type: ignore[arg-type]
            steps=plan.suites.steps,
            with_openfpl=with_openfpl or ("openfpl_v1" in plan.models),
            openfpl_model_dir=openfpl_model_dir,
            output_dir=season_out,
            use_cache=use_cache,
        )
        summaries.append(part.summary)
        compare_result = part

    assert compare_result is not None
    summary = (
        pl.concat(summaries, how="diagonal_relaxed")
        if len(summaries) > 1
        else compare_result.summary
    )
    summary.write_parquet(out / "summary.parquet")

    spa_payload, mcs_payload = selection_payloads(
        data_dir=data_dir,
        compare=compare_result,
        control=plan.control,
        models=list(plan.models),
    )

    pairwise_names: list[str] = []
    for eval_name in plan.pairwise_evals:
        pair_spec = load_eval_spec(eval_name, directory=evals_dir)
        pair_out = out / "pairwise" / eval_name
        pair_result = execute_promote_eval(
            pair_spec,
            data_dir=data_dir,
            dry_run=False,
            from_gw=plan.from_gw,
            to_gw=plan.to_gw,
            seasons=list(plan.seasons),
            output_dir=pair_out,
            models_dir=models_dir,
        )
        pairwise_names.append(eval_name)
        logger.info(
            "pairwise promote-eval %s → %s (exit %s)",
            eval_name,
            pair_result.verdict,
            pair_result.exit_code,
        )

    ranking = render_ranking(
        spec,
        plan,
        summary=summary,
        spa=spa_payload,
        mcs=mcs_payload,
        pairwise=tuple(pairwise_names),
    )
    (out / "ranking.md").write_text(ranking, encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "bakeoff_name": spec.name,
                "question": spec.question,
                "control": plan.control,
                "models": plan.models,
                "summary": summary.to_dicts(),
                "spa": spa_payload,
                "mcs": mcs_payload,
                "pairwise_evals": pairwise_names,
                "note": (
                    "descriptive ranking on this window; keep needs a fresh window"
                ),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return BakeoffResult(
        bakeoff_name=spec.name,
        exit_code=0,
        output_dir=out,
        plan=plan,
        ranking=ranking,
        compare=compare_result,
        summary=summary,
        spa=spa_payload,
        mcs=mcs_payload,
        pairwise=tuple(pairwise_names),
    )


def selection_payloads(
    *,
    data_dir: Path,
    compare: CompareResult,
    control: str,
    models: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Build SPA/MCS JSON payloads when at least three arms share the window.

    Arms with no per-GW RMSE losses (empty ledgers) are skipped and listed under
    ``skipped_empty`` so a dead CORE_MODELS baseline cannot abort the board.

    Args:
        data_dir: Ledger root.
        compare: Completed multi-arm compare.
        control: Benchmark ModelSpec name.
        models: All arm names (including control).

    Returns:
        ``(spa_dict, mcs_dict)`` or ``(None, None)`` when SPA/MCS does not apply.
    """
    if len(models) < 3 or control not in models:
        return None, None
    id_by_name = {run.model_name: run.spec.model_id for run in compare.runs}
    try:
        ledger = read_predictions(data_dir)
    except Exception:
        return None, None
    model_ids = {id_by_name[name] for name in models if name in id_by_name}
    scored = ledger.filter(pl.col("model_id").is_in(sorted(model_ids)))
    if scored.is_empty():
        return None, None
    try:
        scored = filter_decision_pool(scored, DataStore(data_dir))
    except Exception:
        logger.warning(
            "bakeoff decision pool filter failed; SPA/MCS uses full ledger",
            exc_info=True,
        )
    if scored.is_empty():
        return None, None

    keys: pl.DataFrame | None = None
    per_arm: dict[str, pl.DataFrame] = {}
    skipped_empty: list[str] = []
    for name in models:
        mid = id_by_name.get(name)
        if mid is None:
            return None, None
        losses = per_gw_losses(scored, model_id=mid, metric=HEADLINE_METRIC)
        if losses.is_empty():
            # Arms with no scored ledger rows (e.g. ep_next on vaastav) must
            # not abort SPA/MCS for the rest of the championship set.
            skipped_empty.append(name)
            logger.warning(
                "bakeoff SPA/MCS skipping %s: no per-GW %s losses",
                name,
                HEADLINE_METRIC.upper(),
            )
            continue
        per_arm[name] = losses
        gw_keys = losses.select(["target_season", "target_gw"])
        keys = (
            gw_keys
            if keys is None
            else keys.join(gw_keys, on=["target_season", "target_gw"], how="inner")
        )
    if control not in per_arm:
        return None, None
    if len(per_arm) < 3 or keys is None or keys.is_empty():
        return None, None

    loss_by_arm: dict[str, Any] = {}
    for name, losses in per_arm.items():
        aligned = keys.join(
            losses, on=["target_season", "target_gw"], how="inner"
        ).sort(["target_season", "target_gw"])
        loss_by_arm[name] = aligned["loss"].to_numpy()

    selection = multi_arm_selection(loss_by_arm, control=control)
    if selection is None:
        return None, None
    spa, mcs = selection
    return (
        {
            "pvalue_consistent": spa.pvalue_consistent,
            "pvalue_lower": spa.pvalue_lower,
            "pvalue_upper": spa.pvalue_upper,
            "n_gw": spa.n_gw,
            "n_models": spa.n_models,
            "skipped_empty": skipped_empty,
        },
        {
            "included": list(mcs.included),
            "excluded": list(mcs.excluded),
            "size": mcs.size,
            "n_gw": mcs.n_gw,
            "skipped_empty": skipped_empty,
        },
    )


def render_ranking(
    spec: BakeoffSpec,
    plan: BakeoffPlan,
    *,
    summary: pl.DataFrame,
    spa: dict[str, Any] | None,
    mcs: dict[str, Any] | None,
    pairwise: tuple[str, ...] = (),
) -> str:
    """Render ``ranking.md`` for a multi-arm bakeoff.

    Args:
        spec: BakeoffSpec.
        plan: Resolved plan.
        summary: Overall compare summary.
        spa: Optional SPA payload.
        mcs: Optional MCS payload.
        pairwise: EvalSpec names that also ran.

    Returns:
        Markdown text.
    """
    title = spec.report.title or f"Bakeoff: {spec.name}"
    lines = [
        f"# {title}",
        "",
        f"**Question:** {spec.question}",
        "",
        (
            "Descriptive ranking on this window — not a promote-eval `keep`. "
            "Confirmation needs a fresh window (feature-eval-pipeline §9)."
        ),
        "",
        f"- Control (SPA benchmark): `{plan.control}`",
        f"- Arms: {', '.join(f'`{m}`' for m in plan.models)}",
        (
            f"- Window: seasons={plan.seasons}, GW {plan.from_gw}–{plan.to_gw}, "
            f"seasons_back={plan.seasons_back}, modes={list(plan.modes)}"
        ),
        "",
        "## Metric table",
        "",
        "```",
        format_summary_table(summary).rstrip(),
        "```",
        "",
        (
            f"## Rank by {HEADLINE_METRIC.upper()} "
            "(walk-forward, horizon=1 when present; MAE secondary in table)"
        ),
        "",
    ]
    ranked = _rank_by_headline(summary)
    if not ranked:
        lines.append("_No scored rows._")
    else:
        for i, (name, value) in enumerate(ranked, start=1):
            marker = " ← control" if name == plan.control else ""
            lines.append(f"{i}. `{name}` {HEADLINE_METRIC.upper()}={value:.4f}{marker}")

    lines.extend(["", f"## SPA / MCS ({HEADLINE_METRIC.upper()} per-GW losses)", ""])
    if spa is None or mcs is None:
        lines.append(
            "_SPA/MCS unavailable (need ≥3 arms with aligned gameweek losses)._"
        )
    else:
        lines.append(
            f"- SPA p_consistent={spa.get('pvalue_consistent')} "
            f"(lower={spa.get('pvalue_lower')}, upper={spa.get('pvalue_upper')}); "
            f"n_gw={spa.get('n_gw')}, n_challengers={spa.get('n_models')}"
        )
        included = _str_list(mcs.get("included"))
        lines.append(f"- MCS included: {', '.join(f'`{n}`' for n in included)}")
        excluded = _str_list(mcs.get("excluded"))
        if excluded:
            lines.append(f"- MCS excluded: {', '.join(f'`{n}`' for n in excluded)}")
        else:
            lines.append("- MCS excluded: (none)")
        skipped = _str_list(spa.get("skipped_empty"))
        if skipped:
            lines.append(
                "- Skipped empty arms (no scored ledger rows): "
                + ", ".join(f"`{n}`" for n in skipped)
            )

    lines.extend(["", "## Pairwise promote-eval", ""])
    if pairwise:
        for name in pairwise:
            lines.append(f"- `{name}` → see `pairwise/{name}/recommendation.md`")
    else:
        lines.append(
            "_Not run._ Compose with `pairwise_evals:` or `fplt promote-eval`."
        )

    lines.extend(
        [
            "",
            "## Call",
            "",
            (
                "**inconclusive / descriptive.** Do not promote the winner of this "
                "window. Use a confirmation EvalSpec on a held-out window for `keep`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in cast(list[object], value):
        out.append(str(item))
    return out


def _rank_by_headline(summary: pl.DataFrame) -> list[tuple[str, float]]:
    """Rank arms by the mean-seeking headline metric (RMSE)."""
    metric = HEADLINE_METRIC
    if summary.is_empty() or metric not in summary.columns:
        return []
    frame = summary
    if "mode" in frame.columns:
        wf = frame.filter(pl.col("mode") == "walk_forward")
        if not wf.is_empty():
            frame = wf
    if "horizon" in frame.columns:
        h1 = frame.filter(pl.col("horizon") == 1)
        if not h1.is_empty():
            frame = h1
    rows = (
        frame.select(["model_name", metric])
        .drop_nulls()
        .unique(subset=["model_name"], keep="first")
        .sort(metric)
        .to_dicts()
    )
    out: list[tuple[str, float]] = []
    for row in rows:
        name = row.get("model_name")
        value = row.get(metric)
        if isinstance(name, str) and isinstance(value, int | float):
            out.append((name, float(value)))
    return out


def _default_output_dir(data_dir: Path, bakeoff_name: str) -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return data_dir / "bakeoff" / bakeoff_name / stamp


__all__ = [
    "BakeoffPlan",
    "BakeoffResult",
    "BakeoffSpec",
    "bakeoff_from_models",
    "execute_bakeoff",
    "format_bakeoff_plan",
    "load_bakeoff_spec",
    "plan_bakeoff",
    "plan_to_json",
    "render_ranking",
    "selection_payloads",
]
