"""Season-rollover preflight checks (ops-07).

Validates scoring/chips, team id maps, unmapped-player share, GW1 deadline,
production ModelSpecs, and a dry-run plan probe for a target season.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field

from fplt.automation.production import load_production_ops
from fplt.data.api_football import (
    default_id_maps_path as api_football_id_maps_path,
)
from fplt.data.api_football import (
    load_id_maps as load_api_football_id_maps,
)
from fplt.data.football_data import (
    default_team_map_path,
    load_team_names,
)
from fplt.data.store import DataStore
from fplt.data.understat import (
    default_id_maps_path as understat_id_maps_path,
)
from fplt.data.understat import (
    load_id_maps as load_understat_id_maps,
)
from fplt.models.spec import load_model_specs
from fplt.scoring.chips import ChipRules
from fplt.scoring.rules import Rules

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "ops" / "preflight.yaml"
)
SOURCE_UNDERSTAT = "understat"
SOURCE_API_FOOTBALL = "api_football"
SOURCE_FOOTBALL_DATA = "football_data"


class PreflightConfig(BaseModel):
    """Thresholds for season preflight."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_team_count: int = Field(ge=1, default=20)
    max_unmapped_share: float = Field(ge=0.0, le=1.0, default=0.05)
    top_unmapped: int = Field(ge=1, default=10)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one named preflight check."""

    name: str
    ok: bool
    detail: str
    severity: Literal["fail", "warn"] = "fail"


@dataclass(frozen=True)
class PreflightReport:
    """All preflight checks for one season."""

    season: str
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        """True when every failing-severity check passed."""
        return all(check.ok or check.severity == "warn" for check in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        """Checks that failed at ``fail`` severity."""
        return tuple(
            check for check in self.checks if not check.ok and check.severity == "fail"
        )


def default_preflight_config_path() -> Path:
    """Return the repo-root path to ``configs/ops/preflight.yaml``."""
    return _DEFAULT_CONFIG


def load_preflight_config(path: Path | None = None) -> PreflightConfig:
    """Load preflight thresholds from YAML.

    Args:
        path: Config file. Defaults to ``configs/ops/preflight.yaml``.

    Returns:
        Validated thresholds.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError: The file is not a mapping.
    """
    source = default_preflight_config_path() if path is None else path
    if not source.is_file():
        raise FileNotFoundError(f"preflight config not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{source.name} must contain a mapping")
    return PreflightConfig.model_validate(cast(dict[str, Any], loaded))


def format_preflight_report(report: PreflightReport) -> str:
    """Render a preflight report for the CLI.

    Args:
        report: Output of :func:`run_preflight`.

    Returns:
        Multi-line status text.
    """
    status = "PASS" if report.ok else "FAIL"
    lines = [f"preflight {report.season}: {status}"]
    for check in report.checks:
        mark = "ok" if check.ok else check.severity.upper()
        lines.append(f"  [{mark}] {check.name}: {check.detail}")
    return "\n".join(lines)


def run_preflight(
    season: str,
    *,
    store: DataStore | None = None,
    config: PreflightConfig | None = None,
    scoring_dir: Path | None = None,
    chips_dir: Path | None = None,
    models_dir: Path | None = None,
    understat_maps: Path | None = None,
    api_football_maps: Path | None = None,
    football_data_maps: Path | None = None,
    production_path: Path | None = None,
    teams: pl.DataFrame | None = None,
    gameweeks: pl.DataFrame | None = None,
    players: pl.DataFrame | None = None,
    fixtures: pl.DataFrame | None = None,
    understat_player_matches: pl.DataFrame | None = None,
    player_injuries: pl.DataFrame | None = None,
    plan_probe: Callable[[], None] | None = None,
    skip_plan: bool = False,
) -> PreflightReport:
    """Run season-rollover preflight checks.

    Args:
        season: Season label such as ``2027-28``.
        store: Staged ``DataStore`` used when explicit frames are omitted.
        config: Thresholds; defaults to ``configs/ops/preflight.yaml``.
        scoring_dir: Override ``configs/scoring``.
        chips_dir: Override ``configs/chips``.
        models_dir: Override ``configs/models``.
        understat_maps: Override Understat id maps path.
        api_football_maps: Override API-Football id maps path.
        football_data_maps: Override football-data team names path.
        production_path: Override ``configs/ops/production.yaml``.
        teams: Optional staged teams frame (else ``store.read``).
        gameweeks: Optional staged gameweeks frame.
        players: Optional staged players frame.
        fixtures: Optional staged fixtures frame.
        understat_player_matches: Optional staged Understat player matches.
        player_injuries: Optional staged API-Football injuries.
        plan_probe: Optional callable that raises on dry-run plan failure.
            When ``None`` and ``skip_plan`` is false, a structural plan
            dry-run runs (ModelSpecs + GW1 inputs).
        skip_plan: Skip the dry-run plan check entirely.

    Returns:
        Aggregated :class:`PreflightReport`.
    """
    if not season or Path(season).name != season:
        raise ValueError(f"season must be a file name, got {season!r}")
    cfg = load_preflight_config() if config is None else config
    checks: list[CheckResult] = [
        check_scoring_rules(season, scoring_dir=scoring_dir),
        check_chip_rules(season, chips_dir=chips_dir),
    ]

    teams_frame = _table(store, "teams", teams)
    checks.append(
        check_team_maps(
            season,
            teams_frame,
            config=cfg,
            understat_maps=understat_maps,
            api_football_maps=api_football_maps,
            football_data_maps=football_data_maps,
        )
    )
    checks.append(
        check_unmapped_players(
            season,
            understat_player_matches=_table(
                store, "understat_player_matches", understat_player_matches
            ),
            player_injuries=_table(store, "player_injuries", player_injuries),
            player_status=_optional_status(store),
            config=cfg,
        )
    )
    gw_frame = _table(store, "gameweeks", gameweeks)
    checks.append(check_gw1_deadline(season, gw_frame))
    checks.append(
        check_model_specs(
            season,
            models_dir=models_dir,
            production_path=production_path,
        )
    )
    if not skip_plan:
        checks.append(
            check_dry_run_plan(
                season,
                gameweeks=gw_frame,
                players=_table(store, "players", players),
                fixtures=_table(store, "fixtures", fixtures),
                models_dir=models_dir,
                production_path=production_path,
                plan_probe=plan_probe,
            )
        )
    return PreflightReport(season=season, checks=tuple(checks))


def check_scoring_rules(season: str, *, scoring_dir: Path | None = None) -> CheckResult:
    """Check that ``configs/scoring/<season>.yaml`` loads."""
    try:
        if scoring_dir is None:
            Rules.for_season(season)
        else:
            path = scoring_dir / f"{season}.yaml"
            if not path.is_file():
                raise FileNotFoundError(f"no scoring rules for season {season!r}")
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            Rules.model_validate(loaded)
    except (FileNotFoundError, ValueError) as exc:
        return CheckResult("scoring_rules", False, str(exc))
    return CheckResult("scoring_rules", True, f"configs/scoring/{season}.yaml ok")


def check_chip_rules(season: str, *, chips_dir: Path | None = None) -> CheckResult:
    """Check that ``configs/chips/<season>.yaml`` loads."""
    try:
        if chips_dir is None:
            ChipRules.for_season(season)
        else:
            path = chips_dir / f"{season}.yaml"
            if not path.is_file():
                raise FileNotFoundError(f"no chip rules for season {season!r}")
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            ChipRules.model_validate(loaded)
    except (FileNotFoundError, ValueError) as exc:
        return CheckResult("chip_rules", False, str(exc))
    return CheckResult("chip_rules", True, f"configs/chips/{season}.yaml ok")


def check_team_maps(
    season: str,
    teams: pl.DataFrame | None,
    *,
    config: PreflightConfig,
    understat_maps: Path | None = None,
    api_football_maps: Path | None = None,
    football_data_maps: Path | None = None,
) -> CheckResult:
    """Check every FPL ``team_key`` for ``season`` is mapped in each source."""
    if teams is None or teams.is_empty():
        return CheckResult(
            "team_maps",
            False,
            f"no staged teams for season {season!r}",
        )
    season_teams = teams.filter(pl.col("season") == season)
    if season_teams.is_empty():
        return CheckResult(
            "team_maps",
            False,
            f"no staged teams for season {season!r}",
        )
    keys = sorted({int(v) for v in season_teams["team_key"].to_list()})
    names = {
        int(row["team_key"]): str(row["name"])
        for row in season_teams.select(["team_key", "name"]).unique().to_dicts()
    }
    if len(keys) != config.expected_team_count:
        return CheckResult(
            "team_maps",
            False,
            f"expected {config.expected_team_count} teams, found {len(keys)}",
        )

    _, understat_teams = load_understat_id_maps(
        understat_maps if understat_maps is not None else understat_id_maps_path()
    )
    _, api_teams = load_api_football_id_maps(
        api_football_maps
        if api_football_maps is not None
        else api_football_id_maps_path()
    )
    fd_names = load_team_names(
        football_data_maps
        if football_data_maps is not None
        else default_team_map_path()
    )
    understat_values = set(understat_teams.values())
    api_values = set(api_teams.values())
    fd_values = set(fd_names.values())

    missing: list[str] = []
    for key in keys:
        label = names.get(key, str(key))
        gaps: list[str] = []
        if key not in understat_values:
            gaps.append(SOURCE_UNDERSTAT)
        if key not in api_values:
            gaps.append(SOURCE_API_FOOTBALL)
        if key not in fd_values:
            gaps.append(SOURCE_FOOTBALL_DATA)
        if gaps:
            missing.append(f"{label} ({key}): {', '.join(gaps)}")
    if missing:
        sample = "; ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        return CheckResult(
            "team_maps",
            False,
            f"{len(missing)} team(s) missing from maps: {sample}{more}",
        )
    return CheckResult(
        "team_maps",
        True,
        f"all {len(keys)} teams mapped in understat/api_football/football_data",
    )


def check_unmapped_players(
    season: str,
    *,
    understat_player_matches: pl.DataFrame | None,
    player_injuries: pl.DataFrame | None,
    player_status: pl.DataFrame | None,
    config: PreflightConfig,
) -> CheckResult:
    """Report unmapped-player share by source and top offenders."""
    parts: list[str] = []
    failed = False
    for source, frame, id_col, weight_col in (
        (
            SOURCE_UNDERSTAT,
            understat_player_matches,
            "understat_player_id",
            "minutes",
        ),
        (SOURCE_API_FOOTBALL, player_injuries, "api_player_id", None),
    ):
        if frame is None or frame.is_empty():
            parts.append(f"{source}: no staged rows")
            continue
        season_frame = frame.filter(pl.col("season") == season)
        if season_frame.is_empty():
            parts.append(f"{source}: no rows for season")
            continue
        total = season_frame.height
        unmapped = season_frame.filter(pl.col("player_key").is_null())
        share = unmapped.height / total if total else 0.0
        top = _top_unmapped(
            unmapped,
            id_col=id_col,
            weight_col=weight_col,
            player_status=player_status,
            limit=config.top_unmapped,
        )
        top_txt = ", ".join(top) if top else "none"
        status = "ok" if share <= config.max_unmapped_share else "FAIL"
        if share > config.max_unmapped_share:
            failed = True
        parts.append(
            f"{source}: unmapped_share={share:.3f} ({unmapped.height}/{total}) "
            f"[{status}]; top={top_txt}"
        )
    detail = "; ".join(parts) if parts else "no external player tables"
    return CheckResult("unmapped_players", not failed, detail)


def check_gw1_deadline(season: str, gameweeks: pl.DataFrame | None) -> CheckResult:
    """Parse and report the GW1 deadline for ``season``."""
    if gameweeks is None or gameweeks.is_empty():
        return CheckResult("gw1_deadline", False, "no staged gameweeks")
    gw1 = gameweeks.filter((pl.col("season") == season) & (pl.col("gw") == 1))
    if gw1.is_empty():
        return CheckResult(
            "gw1_deadline",
            False,
            f"no GW1 row for season {season!r}",
        )
    raw = gw1["deadline_time"][0]
    if not isinstance(raw, datetime):
        return CheckResult(
            "gw1_deadline",
            False,
            f"GW1 deadline is not a datetime: {raw!r}",
        )
    if raw.tzinfo is None:
        return CheckResult(
            "gw1_deadline",
            False,
            "GW1 deadline missing timezone (expected UTC)",
        )
    return CheckResult(
        "gw1_deadline",
        True,
        f"GW1 deadline {raw.isoformat()}",
    )


def check_model_specs(
    season: str,
    *,
    models_dir: Path | None = None,
    production_path: Path | None = None,
) -> CheckResult:
    """Load production ModelSpecs (season is recorded for the report only)."""
    del season  # specs are not season-keyed; kept for a uniform check signature
    try:
        ops = load_production_ops(production_path)
        specs = load_model_specs(models_dir)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        return CheckResult("model_specs", False, str(exc))
    missing = [name for name in ops.models if name not in specs]
    if missing:
        return CheckResult(
            "model_specs",
            False,
            f"production models missing specs: {missing}",
        )
    return CheckResult(
        "model_specs",
        True,
        f"loaded production models: {', '.join(ops.models)}",
    )


def check_dry_run_plan(
    season: str,
    *,
    gameweeks: pl.DataFrame | None,
    players: pl.DataFrame | None,
    fixtures: pl.DataFrame | None,
    models_dir: Path | None = None,
    production_path: Path | None = None,
    plan_probe: Callable[[], None] | None = None,
) -> CheckResult:
    """Structural (or injected) dry-run plan check for the new season."""
    if plan_probe is not None:
        try:
            plan_probe()
        except Exception as exc:  # noqa: BLE001 — surface any probe failure
            return CheckResult("dry_run_plan", False, f"plan probe failed: {exc}")
        return CheckResult("dry_run_plan", True, "plan probe ok")

    specs_check = check_model_specs(
        season, models_dir=models_dir, production_path=production_path
    )
    if not specs_check.ok:
        return CheckResult("dry_run_plan", False, specs_check.detail)

    gw_check = check_gw1_deadline(season, gameweeks)
    if not gw_check.ok:
        return CheckResult("dry_run_plan", False, gw_check.detail)

    if players is None or players.filter(pl.col("season") == season).is_empty():
        return CheckResult(
            "dry_run_plan",
            False,
            f"no staged players for season {season!r}",
        )
    if fixtures is None or fixtures.filter(pl.col("season") == season).is_empty():
        return CheckResult(
            "dry_run_plan",
            False,
            f"no staged fixtures for season {season!r}",
        )
    n_players = players.filter(pl.col("season") == season).height
    n_fixtures = fixtures.filter(pl.col("season") == season).height
    return CheckResult(
        "dry_run_plan",
        True,
        f"structural dry-run ok ({n_players} players, {n_fixtures} fixtures)",
    )


def _table(
    store: DataStore | None,
    name: str,
    override: pl.DataFrame | None,
) -> pl.DataFrame | None:
    if override is not None:
        return override
    if store is None:
        return None
    path = store.path(name)
    if not path.is_file():
        return None
    return store.read(name)


def _optional_status(store: DataStore | None) -> pl.DataFrame | None:
    if store is None:
        return None
    path = store.path("player_status")
    if not path.is_file():
        return None
    return store.read("player_status")


def _top_unmapped(
    unmapped: pl.DataFrame,
    *,
    id_col: str,
    weight_col: str | None,
    player_status: pl.DataFrame | None,
    limit: int,
) -> list[str]:
    if unmapped.is_empty():
        return []
    if weight_col is not None and weight_col in unmapped.columns:
        ranked = (
            unmapped.group_by(id_col)
            .agg(pl.col(weight_col).sum().alias("weight"))
            .sort("weight", descending=True)
            .head(limit)
        )
        return [
            f"{int(row[id_col])}(min={int(row['weight'])})" for row in ranked.to_dicts()
        ]
    # Injuries: rank by row count; attach price when the id somehow mapped later.
    ranked = (
        unmapped.group_by(id_col)
        .len()
        .rename({"len": "weight"})
        .sort("weight", descending=True)
        .head(limit)
    )
    # player_status reserved for a future price join once unmapped ids resolve.
    _ = player_status
    return [f"{int(row[id_col])}(n={int(row['weight'])})" for row in ranked.to_dicts()]


def build_fpl_targets(
    players: pl.DataFrame,
    *,
    season: str,
    bootstrap_elements: Sequence[Mapping[str, Any]] | None = None,
) -> list[Any]:
    """Build :class:`~fplt.ops.idmap.TargetPlayer` rows for idmap suggest.

    Args:
        players: Staged players table.
        season: Season filter.
        bootstrap_elements: Optional raw bootstrap ``elements`` for full name
            / DOB when present.

    Returns:
        Target players for :func:`~fplt.ops.idmap.suggest_matches`.
    """
    from datetime import date as date_cls

    from fplt.ops.idmap import TargetPlayer

    extras: dict[int, tuple[str, date_cls | None]] = {}
    if bootstrap_elements is not None:
        for element in bootstrap_elements:
            code = int(element["code"])
            first = str(element.get("first_name") or "")
            second = str(element.get("second_name") or "")
            full = f"{first} {second}".strip() or str(element.get("web_name") or "")
            dob_raw = element.get("birth_date") or element.get("dob")
            dob: date_cls | None = None
            if isinstance(dob_raw, str) and dob_raw:
                dob = date_cls.fromisoformat(dob_raw[:10])
            extras[code] = (full, dob)

    out: list[TargetPlayer] = []
    frame = players.filter(pl.col("season") == season)
    for row in frame.to_dicts():
        key = int(row["player_key"])
        full, dob = extras.get(key, (str(row["web_name"]), None))
        out.append(
            TargetPlayer(
                player_key=key,
                web_name=str(row["web_name"]),
                full_name=full,
                team_key=int(row["team_key"]),
                position=str(row["position"]),
                dob=dob,
            )
        )
    return out
