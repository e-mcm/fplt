"""Candidate-pool pruning policy for long-horizon MILPs.

Binary count scales with priced players × horizon. Before heuristic top-k
cuts, drop players who are exactly dominated on price and per-step xPts
(§2.5). Then keep the union of, per position: top decayed Σ xPts, top
points-above-replacement per £ (vs the cheapest playable option), top raw
xPts at each step, top raw Σ xPts over rolling 2–3-step windows, and the
cheapest *playing* fillers — plus any currently owned players. Defaults
are tuned so a full FPL pool (~700) shrinks to about 130 candidates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import highspy
import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fplt.schemas.plan import CandidateSchema

# Minimum pool depth so a legal 15 can still be formed after pruning.
# Also the squad quota *q* used by exact dominance.
MIN_POS_POOL: Mapping[str, int] = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

# FPL ≤3-per-club rule; dominance must leave a club-feasible set of size *q*.
MAX_PER_CLUB = 3

# Legacy top-K defaults from the first prune PR (#132).
LEGACY_TOP_K: Mapping[str, int] = {
    "GK": 20,
    "DEF": 50,
    "MID": 60,
    "FWD": 30,
}

# Nominal target pool after heuristic union (FDB-159). Scaled from the
# prior ~200 quotas via the pruning-loss harness (synthetic 640-player /
# 24-scenario H=5 sweep). Dominance often leaves ~100–110 regardless;
# these quotas still bind when the pre-filter is less aggressive.
DEFAULT_TARGET_POOL = 130
DEFAULT_TOP_XPTS: Mapping[str, int] = {
    "GK": 16,
    "DEF": 36,
    "MID": 46,
    "FWD": 23,
}
DEFAULT_TOP_VALUE: Mapping[str, int] = {
    "GK": 10,
    "DEF": 23,
    "MID": 29,
    "FWD": 13,
}
DEFAULT_KEEP_CHEAPEST = 4
DEFAULT_MIN_P_60 = 0.5

# Per-step / short-window tops catch one-week punts, DGW spikes, and Free
# Hit targets that a horizon sum misses.
DEFAULT_TOP_PER_STEP: Mapping[str, int] = {
    "GK": 2,
    "DEF": 5,
    "MID": 6,
    "FWD": 3,
}
DEFAULT_TOP_WINDOW: Mapping[str, int] = {
    "GK": 2,
    "DEF": 5,
    "MID": 6,
    "FWD": 3,
}
DEFAULT_WINDOW_LENGTHS: tuple[int, ...] = (2, 3)

# Auto-enable when the planner horizon reaches this unless the caller
# forces prune on/off. Live default is H=3; full-pool H=3 continuous
# seasons can hang in HiGHS (forecast-combine GW16), so shortlist at
# the live horizon rather than only at H≥5.
PRUNE_AUTO_MIN_HORIZON = 3


def _validate_window_lengths(value: Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    out = tuple(int(length) for length in value)
    for length in out:
        if length < 2:
            raise ValueError(f"window_lengths entries must be >= 2, got {length}")
    return out


def _validate_pos_counts(
    value: Mapping[str, int] | None,
    *,
    field_name: str,
) -> dict[str, int] | None:
    if value is None:
        return None
    out = {str(position): int(count) for position, count in value.items()}
    for position, minimum in MIN_POS_POOL.items():
        if position not in out:
            raise ValueError(f"{field_name} missing position {position!r}")
        if out[position] < 0:
            raise ValueError(f"{field_name}[{position!r}] must be >= 0")
        if out[position] > 0 and out[position] < minimum:
            raise ValueError(
                f"{field_name}[{position!r}]={out[position]} is below the "
                f"squad minimum {minimum}"
            )
    return out


class CandidatePrunePolicy(BaseModel):
    """Per-position keep rules for shrinking the optimiser candidate pool.

    Filter order in :meth:`apply`:

    1. Exact dominance pre-filter (never drops ``owned_keys``).
    2. Always keep ``owned_keys``.
    3. Union top ``top_xpts`` by decayed Σ xPts per position.
    4. Union top ``top_value`` by points-above-replacement per £:
       ``(xPts_i − xPts_r) / (price_i − price_r)`` with decayed xPts,
       where *r* is the cheapest playable option at the position
       (``p_60 >= min_p_60``; fall back to plain cheapest when none
       pass). Players priced at or below *r* are skipped.
    5. Union top ``top_per_step`` by raw xPts at each horizon step.
    6. Union top ``top_window`` by raw Σ xPts over each rolling window
       whose length is in ``window_lengths`` (default 2 and 3).
    7. Union the ``keep_cheapest`` lowest-priced *playing* players per
       position (``p_60 >= min_p_60``; fall back to plain cheapest when
       too few pass).
    8. Top up any position that fell below squad minima.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    top_xpts: dict[str, int] | None = Field(
        default_factory=lambda: dict(DEFAULT_TOP_XPTS),
        description=(
            "Per-position keep counts by decayed Σ horizon xPts. None / "
            "empty disables. Defaults target ~130 with ``top_value``."
        ),
    )
    top_value: dict[str, int] | None = Field(
        default_factory=lambda: dict(DEFAULT_TOP_VALUE),
        description=(
            "Per-position keep counts by points-above-replacement per £ "
            "(decayed xPts vs cheapest playable). None disables."
        ),
    )
    top_per_step: dict[str, int] | None = Field(
        default_factory=lambda: dict(DEFAULT_TOP_PER_STEP),
        description=(
            "Per-position keep counts by raw xPts at each step. Catches "
            "one-week punts and Free Hit targets a horizon sum misses."
        ),
    )
    top_window: dict[str, int] | None = Field(
        default_factory=lambda: dict(DEFAULT_TOP_WINDOW),
        description=(
            "Per-position keep counts by raw Σ xPts over each rolling "
            "window in ``window_lengths``. None disables."
        ),
    )
    window_lengths: tuple[int, ...] = Field(
        default=DEFAULT_WINDOW_LENGTHS,
        description=(
            "Rolling window lengths (steps) for ``top_window``. Empty "
            "disables window tops even when ``top_window`` is set."
        ),
    )
    keep_cheapest: int = Field(
        default=DEFAULT_KEEP_CHEAPEST,
        ge=0,
        description=(
            "Cheapest playing fillers to retain per position "
            "(budget / Free Hit fillers)."
        ),
    )
    min_p_60: float = Field(
        default=DEFAULT_MIN_P_60,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum mean ``p_60`` for a player to count as a playing "
            "filler. When too few players pass, fall back to plain cheapest."
        ),
    )
    auto_min_horizon: int = Field(
        default=PRUNE_AUTO_MIN_HORIZON,
        ge=1,
        description="Horizon at which ``plan_squad`` auto-enables this policy.",
    )

    @field_validator("top_xpts")
    @classmethod
    def _top_xpts_ok(cls, value: Mapping[str, int] | None) -> dict[str, int] | None:
        return _validate_pos_counts(value, field_name="top_xpts")

    @field_validator("top_value")
    @classmethod
    def _top_value_ok(cls, value: Mapping[str, int] | None) -> dict[str, int] | None:
        return _validate_pos_counts(value, field_name="top_value")

    @field_validator("top_per_step")
    @classmethod
    def _top_per_step_ok(cls, value: Mapping[str, int] | None) -> dict[str, int] | None:
        return _validate_pos_counts(value, field_name="top_per_step")

    @field_validator("top_window")
    @classmethod
    def _top_window_ok(cls, value: Mapping[str, int] | None) -> dict[str, int] | None:
        return _validate_pos_counts(value, field_name="top_window")

    @field_validator("window_lengths")
    @classmethod
    def _window_lengths_ok(cls, value: Sequence[int] | None) -> tuple[int, ...]:
        return _validate_window_lengths(value)

    @model_validator(mode="after")
    def _some_keep_path(self) -> CandidatePrunePolicy:
        has_xpts = self.top_xpts is not None and any(
            count > 0 for count in self.top_xpts.values()
        )
        has_value = self.top_value is not None and any(
            count > 0 for count in self.top_value.values()
        )
        has_step = self.top_per_step is not None and any(
            count > 0 for count in self.top_per_step.values()
        )
        has_window = (
            self.top_window is not None
            and any(count > 0 for count in self.top_window.values())
            and bool(self.window_lengths)
        )
        if (
            not has_xpts
            and not has_value
            and not has_step
            and not has_window
            and self.keep_cheapest <= 0
        ):
            raise ValueError(
                "CandidatePrunePolicy needs at least one active keep rule "
                "(top_xpts, top_value, top_per_step, top_window, or "
                "keep_cheapest)"
            )
        return self

    @classmethod
    def target_pool(
        cls,
        *,
        top_xpts: Mapping[str, int] | None = None,
        top_value: Mapping[str, int] | None = None,
        top_per_step: Mapping[str, int] | None = None,
        top_window: Mapping[str, int] | None = None,
        window_lengths: Sequence[int] | None = None,
        keep_cheapest: int = DEFAULT_KEEP_CHEAPEST,
        min_p_60: float = DEFAULT_MIN_P_60,
        auto_min_horizon: int = PRUNE_AUTO_MIN_HORIZON,
    ) -> CandidatePrunePolicy:
        """Return the default ~130-player policy (tunable overrides)."""
        return cls(
            top_xpts=dict(DEFAULT_TOP_XPTS if top_xpts is None else top_xpts),
            top_value=dict(DEFAULT_TOP_VALUE if top_value is None else top_value),
            top_per_step=dict(
                DEFAULT_TOP_PER_STEP if top_per_step is None else top_per_step
            ),
            top_window=dict(DEFAULT_TOP_WINDOW if top_window is None else top_window),
            window_lengths=(
                DEFAULT_WINDOW_LENGTHS
                if window_lengths is None
                else tuple(window_lengths)
            ),
            keep_cheapest=keep_cheapest,
            min_p_60=min_p_60,
            auto_min_horizon=auto_min_horizon,
        )

    @classmethod
    def legacy_top_k(
        cls,
        *,
        top_k: Mapping[str, int] | None = None,
        keep_cheapest: int = 8,
        min_p_60: float = DEFAULT_MIN_P_60,
        auto_min_horizon: int = PRUNE_AUTO_MIN_HORIZON,
    ) -> CandidatePrunePolicy:
        """Return the #132 top-K + cheapest policy (no value / step cuts)."""
        return cls(
            top_xpts=dict(LEGACY_TOP_K if top_k is None else top_k),
            top_value=None,
            top_per_step=None,
            top_window=None,
            window_lengths=(),
            keep_cheapest=keep_cheapest,
            min_p_60=min_p_60,
            auto_min_horizon=auto_min_horizon,
        )

    def should_auto_apply(self, horizon: int) -> bool:
        """True when ``horizon`` reaches ``auto_min_horizon``."""
        return horizon >= self.auto_min_horizon

    def apply(
        self,
        players: pl.DataFrame,
        points: pl.DataFrame,
        *,
        owned_keys: Sequence[int] = (),
        decay: float = 1.0,
    ) -> pl.DataFrame:
        """Return a ``CandidateSchema`` subset of ``players``.

        Args:
            players: Candidates matching ``CandidateSchema``.
            points: Horizon rows with at least ``player_key``, ``step``,
                ``xpts``. Optional ``p_60`` enables playing-filler filtering.
            owned_keys: Player keys that must remain selectable.
            decay: Per-step discount matching the planner
                (``SquadState.decay``). Rank scores use
                ``Σ_t decay^t · xPts_t`` with 0-indexed ``t``.

        Returns:
            Pruned candidates sorted by ``player_key``.

        Raises:
            ValueError: Owned keys are missing from ``players``, ``decay``
                is invalid, or the pruned pool cannot form a legal fifteen.
        """
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        checked = CandidateSchema.validate(players.sort("player_key"))
        assert isinstance(checked, pl.DataFrame)
        pool = checked
        if pool.is_empty():
            return pool

        keep: set[int] = {int(key) for key in owned_keys}
        missing_owned = sorted(keep - set(pool["player_key"].to_list()))
        if missing_owned:
            raise ValueError(f"owned players are not in the pool: {missing_owned}")

        # Exact dominance before heuristic top-k (§2.5).
        pool = drop_dominated_players(pool, points, owned_keys=sorted(keep))
        ranked = _rank_players(pool, points, decay=decay, min_p_60=self.min_p_60)

        if self.top_xpts is not None:
            for position, quota in self.top_xpts.items():
                if quota <= 0:
                    continue
                keep.update(_top_keys(ranked, position, quota, "sum_xpts"))

        if self.top_value is not None:
            for position, quota in self.top_value.items():
                if quota <= 0:
                    continue
                keep.update(_top_keys(ranked, position, quota, "par_value"))

        if self.top_per_step is not None:
            keep.update(_top_keys_at_each_step(pool, points, quotas=self.top_per_step))

        if self.top_window is not None and self.window_lengths:
            keep.update(
                _top_keys_at_windows(
                    pool,
                    points,
                    quotas=self.top_window,
                    window_lengths=self.window_lengths,
                )
            )

        if self.keep_cheapest > 0:
            for position in MIN_POS_POOL:
                keep.update(
                    _cheapest_keys(
                        ranked,
                        position,
                        self.keep_cheapest,
                        min_p_60=self.min_p_60,
                    )
                )

        keep = _top_up_minima(ranked, keep)
        pruned = pool.filter(pl.col("player_key").is_in(sorted(keep))).sort(
            "player_key"
        )
        for position, minimum in MIN_POS_POOL.items():
            have = pruned.filter(pl.col("position") == position).height
            if have < minimum:
                raise ValueError(
                    f"pruned pool has {have} {position} players; need >= {minimum}"
                )
        return pruned


# Default: ~130-player union of top xPts, PAR value, step/window, cheapest.
DEFAULT_PRUNE_POLICY = CandidatePrunePolicy.target_pool()


def drop_dominated_players(
    players: pl.DataFrame,
    points: pl.DataFrame,
    *,
    owned_keys: Sequence[int] = (),
) -> pl.DataFrame:
    """Drop players exactly dominated on price and per-step xPts.

    Player *i* is dropped when at least *q* same-position others each have
    ``price ≤ price_i`` and ``xPts ≥ xPts_i`` at every horizon step, with
    a strict improvement on price or at least one step (Pareto), and the
    dominators admit a club-feasible selection of size *q* (≤
    ``MAX_PER_CLUB`` from any club). *q* is the position squad quota in
    ``MIN_POS_POOL``. Owned players are never dropped. Dominated keys that
    are required for a club-feasible fifteen are restored afterwards
    (cross-position club usage is not visible to the per-position test).

    Args:
        players: Candidates matching ``CandidateSchema``.
        points: Horizon rows with ``player_key``, ``step``, ``xpts``.
        owned_keys: Player keys that must survive.

    Returns:
        A ``CandidateSchema`` subset of ``players``, sorted by ``player_key``.
    """
    checked = CandidateSchema.validate(players.sort("player_key"))
    assert isinstance(checked, pl.DataFrame)
    if checked.is_empty():
        return checked
    protected = {int(key) for key in owned_keys}
    dominated = _dominated_keys(checked, points, protected=protected)
    if not dominated:
        return checked
    survivors = set(checked["player_key"].to_list()) - dominated
    # Dominance is position-local; other positions also consume club slots.
    # Restore dominated keys until a legal fifteen still exists under the
    # club cap (keeps the cut lossless on thin / concentrated pools).
    survivors = _restore_for_squad_feasibility(checked, survivors, dominated)
    return checked.filter(pl.col("player_key").is_in(sorted(survivors))).sort(
        "player_key"
    )


def prune_candidates(
    players: pl.DataFrame,
    points: pl.DataFrame,
    *,
    owned_keys: Sequence[int] = (),
    policy: CandidatePrunePolicy | None = None,
    top_k: Mapping[str, int] | None = None,
    keep_cheapest: int | None = None,
    decay: float = 1.0,
) -> pl.DataFrame:
    """Drop low-value candidates via ``CandidatePrunePolicy``.

    Args:
        players: Candidates matching ``CandidateSchema``.
        points: Horizon points. Optional ``p_60`` enables playing fillers.
        owned_keys: Player keys that must remain selectable.
        policy: Policy to apply. Defaults to ``DEFAULT_PRUNE_POLICY``.
            Passing ``top_k`` / ``keep_cheapest`` without ``policy`` builds
            a legacy top-K policy for backward compatibility.
        top_k: Legacy override — forces ``CandidatePrunePolicy.legacy_top_k``.
        keep_cheapest: Legacy override for cheapest-per-position count.
        decay: Planner decay passed through to ranking
            (``Σ decay^step · xPts``).

    Returns:
        A ``CandidateSchema`` subset of ``players``, sorted by ``player_key``.
    """
    if policy is not None and (top_k is not None or keep_cheapest is not None):
        raise ValueError("pass policy= or top_k/keep_cheapest=, not both")
    if policy is None and (top_k is not None or keep_cheapest is not None):
        active = CandidatePrunePolicy.legacy_top_k(
            top_k=top_k,
            keep_cheapest=DEFAULT_KEEP_CHEAPEST
            if keep_cheapest is None
            else keep_cheapest,
        )
    elif policy is None:
        active = DEFAULT_PRUNE_POLICY
    else:
        active = policy
    return active.apply(players, points, owned_keys=owned_keys, decay=decay)


def _rank_players(
    pool: pl.DataFrame,
    points: pl.DataFrame,
    *,
    decay: float = 1.0,
    min_p_60: float = DEFAULT_MIN_P_60,
) -> pl.DataFrame:
    """Attach decayed Σ xPts, PAR value, and mean ``p_60`` to ``pool``.

    Points-above-replacement per £ uses the cheapest *playable* option
    per position as replacement *r* (same ``p_60 >= min_p_60`` notion as
    playing fillers; fall back to plain cheapest when none pass):

    ``par_value = (sum_xpts_i − sum_xpts_r) / (price_i − price_r)``

    Players priced at or below *r* get a null ``par_value`` and are
    skipped by the top-value cut.
    """
    if points.is_empty():
        ranked = pool.with_columns(
            pl.lit(0.0).alias("sum_xpts"),
            pl.lit(None).cast(pl.Float64).alias("p_60"),
        )
        return _attach_par_value(ranked, min_p_60=min_p_60)
    required = {"player_key", "xpts"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"points missing columns: {sorted(missing)}")
    if decay != 1.0 and "step" not in points.columns:
        raise ValueError("points missing column 'step' required for decayed ranking")

    scored = points
    if "step" in points.columns:
        # Match plan_squad: weight = decay**step (step is 0-indexed).
        scored = points.with_columns(
            (
                pl.col("xpts")
                * (pl.lit(float(decay)) ** pl.col("step").cast(pl.Float64()))
            ).alias("_decayed_xpts")
        )
        xpts_expr = pl.col("_decayed_xpts").sum()
    else:
        xpts_expr = pl.col("xpts").sum()

    aggs: list[pl.Expr] = [xpts_expr.alias("sum_xpts")]
    if "p_60" in points.columns:
        aggs.append(pl.col("p_60").mean().alias("p_60"))
    totals = scored.group_by("player_key").agg(aggs)
    ranked = pool.join(totals, on="player_key", how="left").with_columns(
        pl.col("sum_xpts").fill_null(0.0)
    )
    if "p_60" not in ranked.columns:
        ranked = ranked.with_columns(pl.lit(None).cast(pl.Float64).alias("p_60"))
    return _attach_par_value(ranked, min_p_60=min_p_60)


def _attach_par_value(
    ranked: pl.DataFrame,
    *,
    min_p_60: float,
) -> pl.DataFrame:
    """Attach ``par_value`` vs the cheapest playable option per position."""
    if ranked.is_empty():
        return ranked.with_columns(pl.lit(None).cast(pl.Float64).alias("par_value"))
    replacements: list[dict[str, object]] = []
    for position in MIN_POS_POOL:
        part = ranked.filter(pl.col("position") == position)
        if part.is_empty():
            continue
        playing = part.filter(
            pl.col("p_60").is_not_null() & (pl.col("p_60") >= min_p_60)
        )
        # Same playable notion as keep_cheapest fillers: prefer p_60, else
        # fall back to the plain cheapest at the position.
        source = playing if playing.height > 0 else part
        replacement = source.sort(["price", "player_key"]).head(1)
        replacements.append(
            {
                "position": position,
                "price_r": int(replacement["price"][0]),
                "xpts_r": float(replacement["sum_xpts"][0]),
            }
        )
    if not replacements:
        return ranked.with_columns(pl.lit(None).cast(pl.Float64).alias("par_value"))
    rep = pl.DataFrame(replacements)
    joined = ranked.join(rep, on="position", how="left")
    return joined.with_columns(
        pl.when(pl.col("price") > pl.col("price_r"))
        .then(
            (pl.col("sum_xpts") - pl.col("xpts_r"))
            / (
                pl.col("price").cast(pl.Float64())
                - pl.col("price_r").cast(pl.Float64())
            )
        )
        .otherwise(None)
        .alias("par_value")
    ).drop("price_r", "xpts_r")


def _cheapest_keys(
    ranked: pl.DataFrame,
    position: str,
    quota: int,
    *,
    min_p_60: float,
) -> set[int]:
    """Return the ``quota`` cheapest keys, preferring playing fillers.

    Prefer players with mean ``p_60 >= min_p_60``. If fewer than ``quota``
    pass (or ``p_60`` is absent), fall back to the plain cheapest at the
    position.
    """
    part = ranked.filter(pl.col("position") == position)
    if part.is_empty() or quota <= 0:
        return set()
    playing = part.filter(pl.col("p_60").is_not_null() & (pl.col("p_60") >= min_p_60))
    source = playing if playing.height >= quota else part
    keys = source.sort(["price", "player_key"]).head(quota)["player_key"].to_list()
    return {int(key) for key in keys}


def _top_keys(
    ranked: pl.DataFrame,
    position: str,
    quota: int,
    score_col: str,
) -> set[int]:
    part = ranked.filter(pl.col("position") == position).filter(
        pl.col(score_col).is_not_null()
    )
    if part.is_empty() or quota <= 0:
        return set()
    keys = (
        part.sort([score_col, "player_key"], descending=[True, False])
        .head(quota)["player_key"]
        .to_list()
    )
    return {int(key) for key in keys}


def _points_with_position(
    pool: pl.DataFrame,
    points: pl.DataFrame,
) -> pl.DataFrame | None:
    """Join position onto horizon points; None when step tops cannot run."""
    if points.is_empty() or "step" not in points.columns:
        return None
    required = {"player_key", "xpts"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"points missing columns: {sorted(missing)}")
    return points.join(
        pool.select("player_key", "position"),
        on="player_key",
        how="inner",
    )


def _top_keys_at_each_step(
    pool: pl.DataFrame,
    points: pl.DataFrame,
    *,
    quotas: Mapping[str, int],
) -> set[int]:
    """Keep top ``quotas[pos]`` by raw xPts at every horizon step.

    Uses undecayed step xPts so a Free Hit / one-week punt ranks on that
    week's forecast, not on a decayed horizon sum.
    """
    keyed = _points_with_position(pool, points)
    if keyed is None:
        return set()
    keep: set[int] = set()
    steps = sorted({int(step) for step in keyed["step"].to_list()})
    for position, quota in quotas.items():
        if quota <= 0:
            continue
        part = keyed.filter(pl.col("position") == position)
        if part.is_empty():
            continue
        for step in steps:
            keys = (
                part.filter(pl.col("step") == step)
                .sort(["xpts", "player_key"], descending=[True, False])
                .head(quota)["player_key"]
                .to_list()
            )
            keep.update(int(key) for key in keys)
    return keep


def _top_keys_at_windows(
    pool: pl.DataFrame,
    points: pl.DataFrame,
    *,
    quotas: Mapping[str, int],
    window_lengths: Sequence[int],
) -> set[int]:
    """Keep top ``quotas[pos]`` by raw Σ xPts over each rolling window.

    Window scores sum undecayed step xPts so a short Bench Boost / double
    stretch ranks on those weeks' forecasts.
    """
    keyed = _points_with_position(pool, points)
    if keyed is None:
        return set()
    steps = sorted({int(step) for step in keyed["step"].to_list()})
    if not steps:
        return set()
    keep: set[int] = set()
    for length in window_lengths:
        if length < 2 or length > len(steps):
            continue
        for start in range(len(steps) - length + 1):
            window = steps[start : start + length]
            windowed = (
                keyed.filter(pl.col("step").is_in(window))
                .group_by(["player_key", "position"])
                .agg(pl.col("xpts").sum().alias("window_xpts"))
            )
            for position, quota in quotas.items():
                if quota <= 0:
                    continue
                part = windowed.filter(pl.col("position") == position)
                if part.is_empty():
                    continue
                keys = (
                    part.sort(["window_xpts", "player_key"], descending=[True, False])
                    .head(quota)["player_key"]
                    .to_list()
                )
                keep.update(int(key) for key in keys)
    return keep


def _top_up_minima(ranked: pl.DataFrame, keep: set[int]) -> set[int]:
    filled = set(keep)
    for position, minimum in MIN_POS_POOL.items():
        part = ranked.filter(pl.col("position") == position)
        have = part.filter(pl.col("player_key").is_in(sorted(filled))).height
        if have >= minimum:
            continue
        need = minimum - have
        extras = (
            part.filter(~pl.col("player_key").is_in(sorted(filled)))
            .sort(["sum_xpts", "player_key"], descending=[True, False])
            .head(need)["player_key"]
            .to_list()
        )
        filled.update(int(key) for key in extras)
    return filled


def _dominated_keys(
    pool: pl.DataFrame,
    points: pl.DataFrame,
    *,
    protected: set[int],
) -> set[int]:
    """Return keys that are exactly dominated under club-feasible quotas."""
    xpts_by_key = _per_step_xpts(pool, points)
    dropped: set[int] = set()
    for position, quota in MIN_POS_POOL.items():
        part = pool.filter(pl.col("position") == position)
        if part.height <= quota:
            continue
        rows = list(part.iter_rows(named=True))
        for candidate in rows:
            key = int(candidate["player_key"])
            if key in protected:
                continue
            price = int(candidate["price"])
            vector = xpts_by_key[key]
            dom_teams: list[int] = []
            for other in rows:
                other_key = int(other["player_key"])
                if other_key == key:
                    continue
                if not _weakly_dominates(
                    price_j=int(other["price"]),
                    xpts_j=xpts_by_key[other_key],
                    price_i=price,
                    xpts_i=vector,
                ):
                    continue
                dom_teams.append(int(other["team_key"]))
            if _club_feasible_count(dom_teams, quota):
                dropped.add(key)
    return dropped


def _per_step_xpts(
    pool: pl.DataFrame,
    points: pl.DataFrame,
) -> dict[int, tuple[float, ...]]:
    """Map each pool player to a dense per-step xPts vector (missing → 0)."""
    keys = [int(key) for key in pool["player_key"].to_list()]
    if points.is_empty() or "step" not in points.columns:
        return {key: () for key in keys}
    required = {"player_key", "xpts"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"points missing columns: {sorted(missing)}")
    steps = sorted({int(step) for step in points["step"].to_list()})
    if not steps:
        return {key: () for key in keys}
    index = {step: i for i, step in enumerate(steps)}
    width = len(steps)
    vectors: dict[int, list[float]] = {key: [0.0] * width for key in keys}
    key_set = set(keys)
    for row in points.select("player_key", "step", "xpts").iter_rows(named=True):
        key = int(row["player_key"])
        if key not in key_set:
            continue
        vectors[key][index[int(row["step"])]] = float(row["xpts"])
    return {key: tuple(values) for key, values in vectors.items()}


def _weakly_dominates(
    *,
    price_j: int,
    xpts_j: tuple[float, ...],
    price_i: int,
    xpts_i: tuple[float, ...],
) -> bool:
    """True when *j* is ≤ price and ≥ xPts on every step, strict somewhere.

    Equal clones do not dominate each other, so a pack of identicals cannot
    wipe itself out under the quota rule.
    """
    if price_j > price_i:
        return False
    if len(xpts_j) != len(xpts_i):
        raise ValueError("per-step xPts vectors must share the same steps")
    strict = price_j < price_i
    for left, right in zip(xpts_j, xpts_i, strict=True):
        if left < right:
            return False
        if left > right:
            strict = True
    return strict


def _club_feasible_count(team_keys: Sequence[int], quota: int) -> bool:
    """True when ``team_keys`` can supply ``quota`` under ``MAX_PER_CLUB``."""
    if quota <= 0:
        return True
    counts = Counter(int(team) for team in team_keys)
    return sum(min(MAX_PER_CLUB, count) for count in counts.values()) >= quota


def _restore_for_squad_feasibility(
    pool: pl.DataFrame,
    survivors: set[int],
    dominated: set[int],
) -> set[int]:
    """Re-add dominated keys until position + club quotas admit a fifteen."""
    kept = set(survivors)
    if _pool_admits_squad(pool.filter(pl.col("player_key").is_in(sorted(kept)))):
        return kept
    # Prefer restoring underrepresented clubs so the club-cap bite clears.
    team_of = {
        int(row["player_key"]): int(row["team_key"])
        for row in pool.select("player_key", "team_key").iter_rows(named=True)
    }
    pending = sorted(
        dominated,
        key=lambda key: (
            sum(1 for other in kept if team_of[other] == team_of[key]),
            key,
        ),
    )
    for key in pending:
        kept.add(key)
        if _pool_admits_squad(pool.filter(pl.col("player_key").is_in(sorted(kept)))):
            return kept
    return kept


def _pool_admits_squad(pool: pl.DataFrame) -> bool:
    """True when ``pool`` can form a 15 under position quotas and club cap."""
    for position, quota in MIN_POS_POOL.items():
        if pool.filter(pl.col("position") == position).height < quota:
            return False
    count = pool.height
    if count < 15:
        return False
    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("log_to_console", False)
    for _ in range(count):
        solver.addVar(0.0, 1.0)
    kinds = np.full(count, int(highspy.HighsVarType.kInteger), dtype=np.uint8)
    status = solver.changeColsIntegrality(
        count, np.arange(count, dtype=np.int32), kinds
    )
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS rejected integrality: {status}")
    picks = np.arange(count, dtype=np.int32)
    _add_sum(solver, picks, 15, 15)
    positions = pool["position"].to_list()
    clubs = pool["team_key"].to_list()
    for position, quota in MIN_POS_POOL.items():
        members = [
            int(picks[i]) for i, name in enumerate(positions) if name == position
        ]
        _add_sum(solver, members, quota, quota)
    for club in sorted({int(c) for c in clubs}):
        members = [int(picks[i]) for i, key in enumerate(clubs) if int(key) == club]
        _add_sum(solver, members, 0, MAX_PER_CLUB)
    solver.changeObjectiveSense(highspy.ObjSense.kMinimize)
    run = solver.run()
    if run != highspy.HighsStatus.kOk:
        return False
    return solver.getModelStatus() == highspy.HighsModelStatus.kOptimal


def _add_sum(
    solver: highspy.Highs,
    columns: Sequence[int] | np.ndarray,
    lower: float,
    upper: float,
) -> None:
    cols = np.asarray(list(columns), dtype=np.int32)
    weights = np.ones(cols.shape[0], dtype=np.float64)
    status = solver.addRow(lower, upper, int(cols.shape[0]), cols, weights)
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS rejected a constraint: {status}")
