"""Shared prediction scenarios for sample-average XI / bench / captain.

Wave D1 (M2/M3): the transfer MILP still sees expected points, but lineup
selection can maximise *mean realised points* across joint scenarios with
autosub and captain/vice fallback applied per draw. Predictors that emit
samples feed ``PredictionSampleSchema`` / ``ScenarioOutcomeSchema``; tests
may build independent draws from marginals via :func:`independent_scenarios`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import ModelSpec, get_model
from fplt.optimise.realised import score_gameweek
from fplt.schemas.scenarios import PredictionSampleSchema, ScenarioOutcomeSchema
from fplt.types import GWKey

_POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
# Default SAA draw count when the caller does not override (full predict
# ``n_sims`` is often 5k–10k; lineup enumeration needs far fewer).
DEFAULT_SAA_SIMS = 500


@dataclass(frozen=True)
class ScenarioBundle:
    """Dense player × sim matrices aligned on ``player_keys``.

    ``points[i, s]`` / ``minutes[i, s]`` are the outcome for
    ``player_keys[i]`` in shared scenario ``s``.
    """

    player_keys: np.ndarray
    points: np.ndarray
    minutes: np.ndarray

    @property
    def n_players(self) -> int:
        """Number of players in the bundle."""
        return int(self.player_keys.shape[0])

    @property
    def n_sims(self) -> int:
        """Number of shared scenarios."""
        return int(self.points.shape[1])

    def index_of(self, player_key: int) -> int:
        """Row index for ``player_key``.

        Raises:
            KeyError: The player is not in the bundle.
        """
        matches = np.flatnonzero(self.player_keys == int(player_key))
        if matches.size == 0:
            raise KeyError(player_key)
        return int(matches[0])


def scenario_bundle_from_frame(frame: pl.DataFrame) -> ScenarioBundle:
    """Pivot a long scenario frame into dense ``(P, S)`` matrices.

    Args:
        frame: Rows matching ``ScenarioOutcomeSchema``. Every player must
            share the same set of ``sim`` indices.

    Returns:
        A :class:`ScenarioBundle` with players sorted by ``player_key`` and
        sims sorted ascending.

    Raises:
        ValueError: Validation fails or sim coverage is incomplete.
    """
    checked = ScenarioOutcomeSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    if checked.is_empty():
        return ScenarioBundle(
            player_keys=np.zeros(0, dtype=np.int64),
            points=np.zeros((0, 0), dtype=np.float64),
            minutes=np.zeros((0, 0), dtype=np.int64),
        )
    sims = sorted(int(value) for value in checked["sim"].unique().to_list())
    players = sorted(int(value) for value in checked["player_key"].unique().to_list())
    expected = len(players) * len(sims)
    if checked.height != expected:
        raise ValueError(
            f"scenario frame must be a full player×sim grid "
            f"({len(players)}×{len(sims)}={expected}), got {checked.height}"
        )
    keyed = checked.sort("player_key", "sim")
    points = np.asarray(keyed["points"].to_list(), dtype=np.float64).reshape(
        len(players), len(sims)
    )
    minutes = np.asarray(keyed["minutes"].to_list(), dtype=np.int64).reshape(
        len(players), len(sims)
    )
    return ScenarioBundle(
        player_keys=np.asarray(players, dtype=np.int64),
        points=points,
        minutes=minutes,
    )


def scenario_outcomes_from_samples(
    samples: pl.DataFrame,
    *,
    season: str,
    gw: int,
) -> pl.DataFrame:
    """Aggregate fixture-level joint samples to player–gameweek outcomes.

    Double gameweeks sum ``points`` and ``minutes`` across fixtures that
    share the same ``sim`` index (CRN key). Blank weeks yield an empty frame.

    Args:
        samples: Rows matching ``PredictionSampleSchema``.
        season: Season label to keep.
        gw: Gameweek to keep.

    Returns:
        A frame matching ``ScenarioOutcomeSchema``.

    Raises:
        ValueError: Validation fails.
    """
    checked = PredictionSampleSchema.validate(samples)
    assert isinstance(checked, pl.DataFrame)
    chosen = checked.filter(
        (pl.col("target_season") == season) & (pl.col("target_gw") == gw)
    )
    if chosen.is_empty():
        return ScenarioOutcomeSchema.validate(
            pl.DataFrame(
                schema={
                    "player_key": pl.Int64(),
                    "sim": pl.Int64(),
                    "points": pl.Float64(),
                    "minutes": pl.Int64(),
                }
            )
        )
    aggregated = chosen.group_by("player_key", "sim").agg(
        pl.col("points").sum(),
        pl.col("minutes").sum(),
    )
    return ScenarioOutcomeSchema.validate(aggregated)


def scenario_bundle_from_samples(
    samples: pl.DataFrame,
    *,
    season: str,
    gw: int,
) -> ScenarioBundle:
    """Build a :class:`ScenarioBundle` from joint predictor samples.

    Args:
        samples: Rows matching ``PredictionSampleSchema``.
        season: Season label to keep.
        gw: Gameweek to keep.

    Returns:
        Dense player × sim matrices for lineup SAA.

    Raises:
        ValueError: Validation fails or the player×sim grid is incomplete.
    """
    return scenario_bundle_from_frame(
        scenario_outcomes_from_samples(samples, season=season, gw=gw)
    )


def make_joint_scenario_provider(
    spec: ModelSpec,
    store: DataStore,
    season: str,
    *,
    n_sims: int = DEFAULT_SAA_SIMS,
) -> Callable[[int], ScenarioBundle]:
    """Fit the layered model at each deadline and emit step-0 joint scenarios.

    Used by ``simulate_decisions`` / ``run_plan`` when ``joint_scenarios`` is
    on. Re-fits at every ``gw`` so samples honour PIT. ``n_sims`` defaults to
    :data:`DEFAULT_SAA_SIMS` (not the ModelSpec predict ``n_sims``).

    Args:
        spec: Layered ``dc_layered`` ModelSpec (must implement ``sample``).
        store: Staged tables for PIT views.
        season: Fixture season.
        n_sims: Draws per fixture.

    Returns:
        ``provider(gw) -> ScenarioBundle`` for that deadline's gameweek.

    Raises:
        ValueError: ``n_sims`` is invalid or the model kind cannot sample.
        TypeError: The constructed model has no ``sample`` method.
    """
    if n_sims < 1:
        raise ValueError(f"n_sims must be a positive int, got {n_sims!r}")
    import fplt.models.composite as _composite  # noqa: F401

    _ = _composite  # register dc_layered for get_model
    cls = get_model(spec.kind)
    probe = cls(spec)
    if not callable(getattr(probe, "sample", None)):
        raise TypeError(
            f"model kind {spec.kind!r} does not implement sample(); "
            "joint scenarios require a SampledPointsPredictor"
        )
    pit = PITData(store)

    def provider(gw: int) -> ScenarioBundle:
        view = pit.view(GWKey(season, int(gw)))
        model = cls(spec)
        model.fit(view)
        samples = model.sample(view, [GWKey(season, int(gw))], int(n_sims))
        return scenario_bundle_from_samples(samples, season=season, gw=int(gw))

    return provider


def joint_scenarios_for_deadline(
    spec: ModelSpec,
    data_dir: Path,
    season: str,
    gameweek: int,
    *,
    n_sims: int = DEFAULT_SAA_SIMS,
) -> ScenarioBundle:
    """Fit once at ``gameweek`` and return joint scenarios for that GW.

    Args:
        spec: Layered model spec.
        data_dir: Root with ``staged/``.
        season: Season label.
        gameweek: Deadline / target gameweek.
        n_sims: Draws per fixture.

    Returns:
        A :class:`ScenarioBundle` covering the sampled player pool.
    """
    provider = make_joint_scenario_provider(
        spec, DataStore(data_dir), season, n_sims=n_sims
    )
    return provider(gameweek)


def independent_scenarios(
    players: pl.DataFrame,
    *,
    n_sims: int,
    rng: np.random.Generator,
) -> ScenarioBundle:
    """Build independent per-player draws from marginal xPts / play probs.

    Used when a predictor has no joint ``sample`` method. Minutes are a
    Bernoulli play draw (90 or 0); points are ``xpts / max(p_played, ε)``
    when playing and 0 otherwise so the marginal mean stays near ``xpts``.

    Args:
        players: ``player_key``, ``xpts``, and optionally ``p_played``
            (default 1.0) / ``p_haul`` (ignored here; captaincy uses it
            elsewhere).
        n_sims: Number of independent draws.
        rng: Seeded generator.

    Returns:
        A :class:`ScenarioBundle` over the given players.

    Raises:
        ValueError: ``n_sims`` is not positive or required columns missing.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    required = {"player_key", "xpts"}
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"players missing columns: {sorted(missing)}")
    ordered = players.sort("player_key")
    keys = np.asarray(ordered["player_key"].to_list(), dtype=np.int64)
    xpts = np.asarray(ordered["xpts"].to_list(), dtype=np.float64)
    if "p_played" in ordered.columns:
        p_played = np.asarray(ordered["p_played"].to_list(), dtype=np.float64)
        p_played = np.clip(np.nan_to_num(p_played, nan=1.0), 0.0, 1.0)
    else:
        p_played = np.ones(keys.shape[0], dtype=np.float64)
    play = rng.random((keys.shape[0], n_sims)) < p_played[:, None]
    conditional = np.divide(
        xpts,
        np.maximum(p_played, 1e-6),
        out=np.zeros_like(xpts),
        where=p_played > 0,
    )
    points = np.where(play, conditional[:, None], 0.0)
    minutes = np.where(play, 90, 0).astype(np.int64)
    return ScenarioBundle(player_keys=keys, points=points, minutes=minutes)


def mean_realised_lineup(
    picks: pl.DataFrame,
    bundle: ScenarioBundle,
    *,
    chip: str | None = None,
) -> float:
    """Mean realised net points of a lineup across shared scenarios.

    Args:
        picks: Fifteen (or fewer) rows with ``player_key``, ``position``,
            ``in_xi``, ``captain``, ``vice``, and optional ``bench_slot``.
        bundle: Shared scenarios covering every ``player_key`` in ``picks``.
        chip: Optional chip for the gameweek (``bboost`` / ``3xc``).

    Returns:
        ``(1/S) Σ_s score_gameweek(picks | scenario s)``.

    Raises:
        ValueError: Empty sims, or a pick is missing from the bundle.
        KeyError: A ``player_key`` is not in ``bundle``.
    """
    scores = realised_lineup_scores(picks, bundle, chip=chip)
    return float(scores.mean()) if scores.size else 0.0


def realised_lineup_scores(
    picks: pl.DataFrame,
    bundle: ScenarioBundle,
    *,
    chip: str | None = None,
) -> np.ndarray:
    """Per-scenario realised nets for a fixed lineup.

    Args:
        picks: Lineup rows (see :func:`mean_realised_lineup`).
        bundle: Shared scenarios.
        chip: Optional chip for the gameweek.

    Returns:
        Length-``S`` array of nets from :func:`score_gameweek`.

    Raises:
        ValueError: The bundle has no sims.
        KeyError: A pick is missing from the bundle.
    """
    if bundle.n_sims == 0:
        raise ValueError("scenario bundle has no sims")
    required = {"player_key", "position", "in_xi", "captain", "vice"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"picks missing columns: {sorted(missing)}")
    kept = picks.filter(~pl.col("sold")) if "sold" in picks.columns else picks
    rows = kept.select(
        [col for col in kept.columns if col in required | {"bench_slot"}]
    ).to_dicts()
    indices = [bundle.index_of(int(row["player_key"])) for row in rows]
    scores = np.empty(bundle.n_sims, dtype=np.float64)
    for sim in range(bundle.n_sims):
        frame_rows: list[dict[str, object]] = []
        for row, index in zip(rows, indices, strict=True):
            entry: dict[str, object] = {
                "player_key": int(row["player_key"]),
                "position": str(row["position"]),
                "in_xi": bool(row["in_xi"]),
                "captain": bool(row["captain"]),
                "vice": bool(row["vice"]),
                "actual_pts": float(bundle.points[index, sim]),
                "actual_minutes": int(bundle.minutes[index, sim]),
            }
            if "bench_slot" in row and row["bench_slot"] is not None:
                entry["bench_slot"] = int(row["bench_slot"])
            frame_rows.append(entry)
        scores[sim] = float(
            score_gameweek(pl.DataFrame(frame_rows), hit=0.0, chip=chip)
        )
    return scores


def default_bench_slots(picks: pl.DataFrame) -> pl.DataFrame:
    """Assign ``bench_slot`` 1–4 by position then ``player_key`` for non-XI.

    XI rows get ``bench_slot`` null. Used when a lineup has no explicit order.

    Args:
        picks: Rows with ``player_key``, ``position``, ``in_xi``.

    Returns:
        A copy with a ``bench_slot`` column.
    """
    if "bench_slot" in picks.columns:
        return picks
    non_xi = picks.filter(~pl.col("in_xi"))
    bench = (
        non_xi.with_columns(
            pl.col("position").replace_strict(_POS_ORDER, default=99).alias("_pos_ord")
        )
        .sort("_pos_ord", "player_key")
        .with_columns(pl.Series("bench_slot", list(range(1, non_xi.height + 1))))
        .select("player_key", "bench_slot")
    )
    return picks.join(bench, on="player_key", how="left").with_columns(
        pl.col("bench_slot").cast(pl.Int64)
    )
