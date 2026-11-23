"""xG/xA involvement shares blended with FPL goals and assists.

When FPL-native ``expected_goals`` / ``expected_assists`` are visible on
``player_fixture_stats``, goal and assist rates mix those Opta xG/xA per 90
with shrunk FPL goals/assists (``xg_blend``). Understat match rows fill gaps
for players with no FPL xG yet. Players with neither keep the FPL-only rates.
``own_goal_weight`` and ``p_assist`` are estimated from the same visible
history so the layered simulator can drop the YAML guesses.
"""

from __future__ import annotations

import polars as pl

from fplt.data.pit import PITView
from fplt.models.components.involvement.rates import (
    POSITIONS,
    appearance_history,
    current_teams,
    latest_rows,
    penalty_takers,
    position_goal_assist_rates,
    set_piece_takers,
    shrunk_rate,
    team_share,
    validated_involvement,
)
from fplt.models.spec import ModelSpec, register_component
from fplt.schemas.components import INVOLVEMENT_COLUMNS

_DEFAULT_OWN_GOAL = 0.05
_DEFAULT_P_ASSIST = 0.75
_DEFAULT_P_PENALTY = 0.12


class InvolvementXg:
    """Team-relative goal/assist weights from FPL/Understat xG/xA plus FPL.

    ``window`` and ``shrinkage`` match ``InvolvementShares``. ``xg_blend`` in
    ``[0, 1]`` is the weight on the xG rate when both signals exist.
    Preference for the xG signal: FPL-native ``expected_goals`` /
    ``expected_assists`` on ``player_fixture_stats``, then Understat
    ``understat_player_matches``. ``penalty_bonus`` is added to the blended
    goal rate for the primary penalty taker (``penalties_order == 1``),
    falling back to a visible penalty miss when order is null. Set
    ``use_penalties_order: false`` to force the miss-only heuristic for A/B
    (default ``true``). When ``use_penalty_event`` is true, skip the rate
    bump and expose ``p_penalty`` / ``penalty_taker_keys`` so the simulator
    can allocate a fraction of goals as penalties. Optional
    ``use_set_piece_orders`` adds ``set_piece_bonus`` for primary corner /
    free-kick takers (default off). After ``fit``, ``own_goal_weight`` and
    ``p_assist`` hold calibrated simulator knobs (defaults when history is
    empty).
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        window = spec.params.get("window", 10)
        shrinkage = spec.params.get("shrinkage", 5.0)
        penalty = spec.params.get("penalty_bonus", 0.3)
        blend = spec.params.get("xg_blend", 0.7)
        use_order = spec.params.get("use_penalties_order", True)
        use_event = spec.params.get("use_penalty_event", False)
        p_pen = spec.params.get("p_penalty", _DEFAULT_P_PENALTY)
        use_set = spec.params.get("use_set_piece_orders", False)
        set_bonus = spec.params.get("set_piece_bonus", 0.1)
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive int, got {window!r}")
        shrinkage_value = float(shrinkage)
        penalty_value = float(penalty)
        blend_value = float(blend)
        p_pen_value = float(p_pen)
        set_bonus_value = float(set_bonus)
        if shrinkage_value < 0:
            raise ValueError(f"shrinkage must be non-negative, got {shrinkage_value}")
        if penalty_value < 0:
            raise ValueError(f"penalty_bonus must be non-negative, got {penalty_value}")
        if not 0.0 <= blend_value <= 1.0:
            raise ValueError(f"xg_blend must be in [0, 1], got {blend_value}")
        if not isinstance(use_order, bool):
            raise ValueError(f"use_penalties_order must be a bool, got {use_order!r}")
        if not isinstance(use_event, bool):
            raise ValueError(f"use_penalty_event must be a bool, got {use_event!r}")
        if not 0.0 <= p_pen_value <= 1.0:
            raise ValueError(f"p_penalty must be in [0, 1], got {p_pen_value}")
        if not isinstance(use_set, bool):
            raise ValueError(f"use_set_piece_orders must be a bool, got {use_set!r}")
        if set_bonus_value < 0:
            raise ValueError(
                f"set_piece_bonus must be non-negative, got {set_bonus_value}"
            )
        self._window = window
        self._shrinkage = shrinkage_value
        self._penalty = penalty_value
        self._xg_blend = blend_value
        self._use_penalties_order = use_order
        self._use_penalty_event = use_event
        self._p_penalty_param = p_pen_value
        self._use_set_piece_orders = use_set
        self._set_piece_bonus = set_bonus_value
        self.own_goal_weight = _DEFAULT_OWN_GOAL
        self.p_assist = _DEFAULT_P_ASSIST
        self.p_penalty = 0.0
        self.penalty_taker_keys: set[int] = set()
        self._fitted = False
        self._goal: dict[int, float] = {}
        self._assist: dict[int, float] = {}

    def fit(self, view: PITView) -> None:
        """Estimate blended rates and simulator knobs from ``view``."""
        self._fitted = True
        self._goal = {}
        self._assist = {}
        self.own_goal_weight = _DEFAULT_OWN_GOAL
        self.p_assist = _DEFAULT_P_ASSIST
        self.p_penalty = 0.0
        self.penalty_taker_keys = set()
        history = appearance_history(view)
        fpl_goal, fpl_assist = _fpl_rates(history, self._window, self._shrinkage)
        xg_goal, xg_assist = _xg_rates(view, history, self._window, self._shrinkage)
        takers = penalty_takers(
            view, history, use_penalties_order=self._use_penalties_order
        )
        self.penalty_taker_keys = set(takers)
        if self._use_penalty_event:
            self.p_penalty = self._p_penalty_param
        set_takers: set[int] = (
            set_piece_takers(view) if self._use_set_piece_orders else set()
        )
        players = set(fpl_goal) | set(fpl_assist) | set(xg_goal) | set(xg_assist)
        for player in players:
            self._goal[player] = _blend_rate(
                fpl_goal.get(player), xg_goal.get(player), self._xg_blend
            )
            self._assist[player] = _blend_rate(
                fpl_assist.get(player), xg_assist.get(player), self._xg_blend
            )
            if player in takers and not self._use_penalty_event:
                self._goal[player] = self._goal.get(player, 0.0) + self._penalty
            if player in set_takers:
                self._goal[player] = self._goal.get(player, 0.0) + self._set_piece_bonus
        for player in set_takers - players:
            self._goal[player] = self._set_piece_bonus
        if self._use_penalty_event:
            for player in takers - players:
                self._goal.setdefault(player, 0.0)
        if not history.is_empty():
            self.own_goal_weight, self.p_assist = _sim_params(history)

    def predict(self, view: PITView, player_fixtures: pl.DataFrame) -> pl.DataFrame:
        """Team-relative weights for each player-fixture.

        Args:
            view: Deadline view. The current squad comes from here.
            player_fixtures: Columns ``player_key`` and ``fixture_id``.

        Returns:
            A frame matching ``InvolvementSchema``.

        Raises:
            RuntimeError: ``fit`` has not been called.
            ValueError: ``player_fixtures`` lacks a required column.
        """
        if not self._fitted:
            raise RuntimeError("fit the involvement model before predicting")
        missing = {"player_key", "fixture_id"} - set(player_fixtures.columns)
        if missing:
            raise ValueError(f"player_fixtures is missing {sorted(missing)}")
        asked = player_fixtures.select("player_key", "fixture_id").unique(
            maintain_order=True
        )
        if asked.is_empty():
            return validated_involvement(pl.DataFrame(schema=INVOLVEMENT_COLUMNS))
        teams = current_teams(view)
        frame = asked.join(teams, on="player_key", how="left").with_columns(
            pl.col("team_key").fill_null(-pl.col("player_key"))
        )
        goal = frame["player_key"].replace_strict(self._goal, default=0.0)
        assist = frame["player_key"].replace_strict(self._assist, default=0.0)
        frame = frame.with_columns(goal.alias("goal_rate"), assist.alias("assist_rate"))
        frame = frame.with_columns(
            pl.col("goal_rate").sum().over("fixture_id", "team_key").alias("goal_sum"),
            pl.col("assist_rate")
            .sum()
            .over("fixture_id", "team_key")
            .alias("assist_sum"),
            pl.len().over("fixture_id", "team_key").alias("squad"),
        )
        frame = frame.with_columns(
            team_share("goal_rate", "goal_sum").alias("goal_weight"),
            team_share("assist_rate", "assist_sum").alias("assist_weight"),
        )
        return validated_involvement(frame.select(list(INVOLVEMENT_COLUMNS)))


def _fpl_rates(
    history: pl.DataFrame, window: int, shrinkage: float
) -> tuple[dict[int, float], dict[int, float]]:
    if history.is_empty():
        return {}, {}
    prior = position_goal_assist_rates(history)
    recent = latest_rows(history, window)
    goal: dict[int, float] = {}
    assist: dict[int, float] = {}
    grouped = recent.group_by("player_key").agg(
        pl.col("position").last(),
        pl.col("goals_scored").sum(),
        pl.col("assists").sum(),
        pl.col("minutes").sum(),
    )
    for row in grouped.iter_rows(named=True):
        player = int(row["player_key"])
        position = row["position"] if row["position"] in POSITIONS else ""
        goal_prior, assist_prior = prior.get(position, (0.0, 0.0))
        nineties = float(row["minutes"]) / 90.0
        goal[player] = shrunk_rate(
            float(row["goals_scored"]), nineties, goal_prior, shrinkage
        )
        assist[player] = shrunk_rate(
            float(row["assists"]), nineties, assist_prior, shrinkage
        )
    return goal, assist


def _xg_rates(
    view: PITView, history: pl.DataFrame, window: int, shrinkage: float
) -> tuple[dict[int, float], dict[int, float]]:
    """Prefer FPL-native Opta xG; fall back to Understat for gaps."""
    fpl_goal, fpl_assist = _fpl_native_xg_rates(history, window, shrinkage)
    ust_goal, ust_assist = _understat_xg_rates(view, history, window, shrinkage)
    goal = {**ust_goal, **fpl_goal}
    assist = {**ust_assist, **fpl_assist}
    return goal, assist


def _fpl_native_xg_rates(
    history: pl.DataFrame, window: int, shrinkage: float
) -> tuple[dict[int, float], dict[int, float]]:
    if history.is_empty() or "expected_goals" not in history.columns:
        return {}, {}
    recent = latest_rows(history, window)
    has_xg = pl.col("expected_goals").is_not_null()
    has_xa = pl.col("expected_assists").is_not_null()
    with_xg = recent.filter(has_xg | has_xa)
    if with_xg.is_empty():
        return {}, {}
    keyed = with_xg.with_columns(
        pl.col("expected_goals").fill_null(0.0),
        pl.col("expected_assists").fill_null(0.0),
    )
    prior = _xg_position_rates(keyed, "expected_goals", "expected_assists")
    goal: dict[int, float] = {}
    assist: dict[int, float] = {}
    grouped = keyed.group_by("player_key").agg(
        pl.col("position").last(),
        pl.col("expected_goals").sum().alias("xg"),
        pl.col("expected_assists").sum().alias("xa"),
        pl.col("minutes").sum(),
    )
    for row in grouped.iter_rows(named=True):
        player = int(row["player_key"])
        position = row["position"] if row["position"] in POSITIONS else ""
        goal_prior, assist_prior = prior.get(position, (0.0, 0.0))
        nineties = float(row["minutes"]) / 90.0
        goal[player] = shrunk_rate(float(row["xg"]), nineties, goal_prior, shrinkage)
        assist[player] = shrunk_rate(
            float(row["xa"]), nineties, assist_prior, shrinkage
        )
    return goal, assist


def _understat_xg_rates(
    view: PITView, history: pl.DataFrame, window: int, shrinkage: float
) -> tuple[dict[int, float], dict[int, float]]:
    matches = (
        view.table("understat_player_matches")
        .filter(pl.col("player_key").is_not_null())
        .collect()
    )
    if matches.is_empty():
        return {}, {}
    roster = (
        history.select("player_key", "position")
        if not history.is_empty()
        else pl.DataFrame(schema={"player_key": pl.Int64(), "position": pl.Utf8()})
    )
    if roster.is_empty():
        roster = (
            view.latest("players", ["season", "player_key"])
            .select("player_key", "position")
            .collect()
        )
    positions = (
        roster.sort("player_key")
        .unique(subset=["player_key"], keep="last")
        .select("player_key", "position")
    )
    keyed = matches.with_columns(pl.col("player_key").cast(pl.Int64)).join(
        positions, on="player_key", how="left"
    )
    prior = _xg_position_rates(keyed, "xg", "xa")
    recent = _latest_matches(keyed, window)
    goal: dict[int, float] = {}
    assist: dict[int, float] = {}
    grouped = recent.group_by("player_key").agg(
        pl.col("position").last(),
        pl.col("xg").sum(),
        pl.col("xa").sum(),
        pl.col("minutes").sum(),
    )
    for row in grouped.iter_rows(named=True):
        player = int(row["player_key"])
        position = row["position"] if row["position"] in POSITIONS else ""
        goal_prior, assist_prior = prior.get(position, (0.0, 0.0))
        nineties = float(row["minutes"]) / 90.0
        goal[player] = shrunk_rate(float(row["xg"]), nineties, goal_prior, shrinkage)
        assist[player] = shrunk_rate(
            float(row["xa"]), nineties, assist_prior, shrinkage
        )
    return goal, assist


def _xg_position_rates(
    matches: pl.DataFrame, xg_col: str, xa_col: str
) -> dict[str, tuple[float, float]]:
    played = matches.filter(
        (pl.col("minutes") > 0) & pl.col("position").is_in(list(POSITIONS))
    )
    if played.is_empty():
        return {}
    totals = played.group_by("position").agg(
        pl.col(xg_col).sum().alias("xg"),
        pl.col(xa_col).sum().alias("xa"),
        (pl.col("minutes") / 90.0).sum().alias("nineties"),
    )
    rates: dict[str, tuple[float, float]] = {}
    for row in totals.iter_rows(named=True):
        nineties = float(row["nineties"])
        rates[str(row["position"])] = (
            float(row["xg"]) / nineties,
            float(row["xa"]) / nineties,
        )
    return rates


def _latest_matches(matches: pl.DataFrame, window: int) -> pl.DataFrame:
    ranked = matches.sort(["player_key", "available_at", "match_id"]).with_columns(
        pl.col("player_key").cum_count().over("player_key").cast(pl.Int64).alias("_i"),
        pl.len().over("player_key").cast(pl.Int64).alias("_n"),
    )
    return ranked.filter(pl.col("_i") > pl.col("_n") - window).drop("_i", "_n")


def _blend_rate(fpl: float | None, understat: float | None, blend: float) -> float:
    if fpl is not None and understat is not None:
        return blend * understat + (1.0 - blend) * fpl
    if understat is not None:
        return understat
    if fpl is not None:
        return fpl
    return 0.0


def _sim_params(history: pl.DataFrame) -> tuple[float, float]:
    goals = float(history["goals_scored"].sum())
    assists = float(history["assists"].sum())
    own = float(history["own_goals"].sum())
    if goals > 0:
        p_assist = min(1.0, max(0.0, assists / goals))
    else:
        p_assist = _DEFAULT_P_ASSIST
    total = goals + own
    if total > 0 and own < total:
        rate = own / total
        own_goal_weight = rate / (1.0 - rate)
    elif total > 0:
        own_goal_weight = 1.0
    else:
        own_goal_weight = _DEFAULT_OWN_GOAL
    return own_goal_weight, p_assist


register_component("involvement_xg")(InvolvementXg)
