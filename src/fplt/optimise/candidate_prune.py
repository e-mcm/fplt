"""Candidate-pool pruning policy for long-horizon MILPs.

Binary count scales with priced players × horizon. Keep the union of, per
position: top Σ xPts, top Σ xPts / price, and the cheapest fillers — plus
any currently owned players. Defaults are tuned so a full FPL pool (~700)
shrinks to about 200 candidates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fplt.schemas.plan import CandidateSchema

# Minimum pool depth so a legal 15 can still be formed after pruning.
MIN_POS_POOL: Mapping[str, int] = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

# Legacy top-K defaults from the first prune PR (#132).
LEGACY_TOP_K: Mapping[str, int] = {
    "GK": 20,
    "DEF": 50,
    "MID": 60,
    "FWD": 30,
}

# Tuned for ~200 unique players on 2025-26 dc_layered_v2 (GW1/19/30 H=8).
DEFAULT_TOP_XPTS: Mapping[str, int] = {
    "GK": 25,
    "DEF": 55,
    "MID": 70,
    "FWD": 35,
}
DEFAULT_TOP_VALUE: Mapping[str, int] = {
    "GK": 15,
    "DEF": 35,
    "MID": 45,
    "FWD": 20,
}
DEFAULT_KEEP_CHEAPEST = 4

# Auto-enable when the planner horizon reaches this unless the caller
# forces prune on/off.
PRUNE_AUTO_MIN_HORIZON = 5


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

    1. Always keep ``owned_keys``.
    2. Union top ``top_xpts`` by Σ xPts per position.
    3. Union top ``top_value`` by Σ xPts / price per position.
    4. Union the ``keep_cheapest`` lowest-priced players per position.
    5. Top up any position that fell below squad minima.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    top_xpts: dict[str, int] | None = Field(
        default_factory=lambda: dict(DEFAULT_TOP_XPTS),
        description=(
            "Per-position keep counts by Σ horizon xPts. None / empty "
            "disables. Defaults target ~200 with ``top_value``."
        ),
    )
    top_value: dict[str, int] | None = Field(
        default_factory=lambda: dict(DEFAULT_TOP_VALUE),
        description=("Per-position keep counts by Σ xPts / price. None disables."),
    )
    keep_cheapest: int = Field(
        default=DEFAULT_KEEP_CHEAPEST,
        ge=0,
        description=(
            "Cheapest players to retain per position (budget / Free Hit fillers)."
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

    @model_validator(mode="after")
    def _some_keep_path(self) -> CandidatePrunePolicy:
        has_xpts = self.top_xpts is not None and any(
            count > 0 for count in self.top_xpts.values()
        )
        has_value = self.top_value is not None and any(
            count > 0 for count in self.top_value.values()
        )
        if not has_xpts and not has_value and self.keep_cheapest <= 0:
            raise ValueError(
                "CandidatePrunePolicy needs at least one active keep rule "
                "(top_xpts, top_value, or keep_cheapest)"
            )
        return self

    @classmethod
    def target_pool(
        cls,
        *,
        top_xpts: Mapping[str, int] | None = None,
        top_value: Mapping[str, int] | None = None,
        keep_cheapest: int = DEFAULT_KEEP_CHEAPEST,
        auto_min_horizon: int = PRUNE_AUTO_MIN_HORIZON,
    ) -> CandidatePrunePolicy:
        """Return the default ~200-player policy (tunable overrides)."""
        return cls(
            top_xpts=dict(DEFAULT_TOP_XPTS if top_xpts is None else top_xpts),
            top_value=dict(DEFAULT_TOP_VALUE if top_value is None else top_value),
            keep_cheapest=keep_cheapest,
            auto_min_horizon=auto_min_horizon,
        )

    @classmethod
    def legacy_top_k(
        cls,
        *,
        top_k: Mapping[str, int] | None = None,
        keep_cheapest: int = 8,
        auto_min_horizon: int = PRUNE_AUTO_MIN_HORIZON,
    ) -> CandidatePrunePolicy:
        """Return the #132 top-K + cheapest policy (no value cut)."""
        return cls(
            top_xpts=dict(LEGACY_TOP_K if top_k is None else top_k),
            top_value=None,
            keep_cheapest=keep_cheapest,
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
    ) -> pl.DataFrame:
        """Return a ``CandidateSchema`` subset of ``players``.

        Args:
            players: Candidates matching ``CandidateSchema``.
            points: Horizon rows with at least ``player_key``, ``step``,
                ``xpts``.
            owned_keys: Player keys that must remain selectable.

        Returns:
            Pruned candidates sorted by ``player_key``.

        Raises:
            ValueError: Owned keys are missing from ``players``, or the
                pruned pool cannot form a legal fifteen.
        """
        checked = CandidateSchema.validate(players.sort("player_key"))
        assert isinstance(checked, pl.DataFrame)
        pool = checked
        if pool.is_empty():
            return pool

        ranked = _rank_players(pool, points)
        keep: set[int] = {int(key) for key in owned_keys}
        missing_owned = sorted(keep - set(ranked["player_key"].to_list()))
        if missing_owned:
            raise ValueError(f"owned players are not in the pool: {missing_owned}")

        if self.top_xpts is not None:
            for position, quota in self.top_xpts.items():
                if quota <= 0:
                    continue
                keep.update(_top_keys(ranked, position, quota, "sum_xpts"))

        if self.top_value is not None:
            for position, quota in self.top_value.items():
                if quota <= 0:
                    continue
                keep.update(_top_keys(ranked, position, quota, "xpts_per_price"))

        if self.keep_cheapest > 0:
            for position in MIN_POS_POOL:
                part = ranked.filter(pl.col("position") == position)
                if part.is_empty():
                    continue
                cheap = (
                    part.sort(["price", "player_key"])
                    .head(self.keep_cheapest)["player_key"]
                    .to_list()
                )
                keep.update(int(key) for key in cheap)

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


# Default: ~200-player union of top xPts, top value, and cheapest.
DEFAULT_PRUNE_POLICY = CandidatePrunePolicy.target_pool()


def prune_candidates(
    players: pl.DataFrame,
    points: pl.DataFrame,
    *,
    owned_keys: Sequence[int] = (),
    policy: CandidatePrunePolicy | None = None,
    top_k: Mapping[str, int] | None = None,
    keep_cheapest: int | None = None,
) -> pl.DataFrame:
    """Drop low-value candidates via ``CandidatePrunePolicy``.

    Args:
        players: Candidates matching ``CandidateSchema``.
        points: Horizon points.
        owned_keys: Player keys that must remain selectable.
        policy: Policy to apply. Defaults to ``DEFAULT_PRUNE_POLICY``.
            Passing ``top_k`` / ``keep_cheapest`` without ``policy`` builds
            a legacy top-K policy for backward compatibility.
        top_k: Legacy override — forces ``CandidatePrunePolicy.legacy_top_k``.
        keep_cheapest: Legacy override for cheapest-per-position count.

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
    return active.apply(players, points, owned_keys=owned_keys)


def _rank_players(pool: pl.DataFrame, points: pl.DataFrame) -> pl.DataFrame:
    if points.is_empty():
        return pool.with_columns(
            pl.lit(0.0).alias("sum_xpts"),
            pl.lit(0.0).alias("xpts_per_price"),
        )
    required = {"player_key", "xpts"}
    missing = required - set(points.columns)
    if missing:
        raise ValueError(f"points missing columns: {sorted(missing)}")

    totals = points.group_by("player_key").agg(pl.col("xpts").sum().alias("sum_xpts"))
    ranked = pool.join(totals, on="player_key", how="left").with_columns(
        pl.col("sum_xpts").fill_null(0.0)
    )
    # price is in tenths of a million; ratio scale only ranks within a pool.
    return ranked.with_columns(
        (
            pl.col("sum_xpts")
            / pl.col("price").cast(pl.Float64()).clip(lower_bound=1.0)
        ).alias("xpts_per_price")
    )


def _top_keys(
    ranked: pl.DataFrame,
    position: str,
    quota: int,
    score_col: str,
) -> set[int]:
    part = ranked.filter(pl.col("position") == position)
    if part.is_empty() or quota <= 0:
        return set()
    keys = (
        part.sort([score_col, "player_key"], descending=[True, False])
        .head(quota)["player_key"]
        .to_list()
    )
    return {int(key) for key in keys}


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
