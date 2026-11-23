"""Minutes-bucket other plus additive match-context offsets.

Wraps ``other_player`` for the unconditional baseline (including
``defcon_mode: rate``). Fit learns position-level additive deltas for team
goals band, clean sheet, and player-scored on total other (bonus + saves +
cards + defcon + residual). Cards / defcon / residual stay in the baseline;
offsets mainly absorb bonus structure confirmed by FDB-263.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import polars as pl

from fplt.data.pit import PITView
from fplt.models.components.other.player import OtherPlayerPriors
from fplt.models.spec import ModelSpec, register_component
from fplt.scoring.rules import MatchStats, Rules

_POSITIONS = ("GK", "DEF", "MID", "FWD")
_POS_INDEX = {name: index for index, name in enumerate(_POSITIONS)}
_GOALS_BANDS = ("0", "1", "2+")
_Residual = tuple[str, str, int, int, float]
_IndexFn = Callable[[str, str, int, int], tuple[int, int]]


class OtherMatchConditioned:
    """Other-points prior with sim-time match deltas.

    ``predict`` returns the minutes-bucket baseline (``OtherPointsSchema``).
    ``match_delta`` supplies additive offsets once the simulator has drawn
    team goals, clean sheets, and player goals.
    """

    match_conditioned: Literal[True] = True

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        match_shrinkage = spec.params.get("match_shrinkage", 20.0)
        shrinkage_value = float(match_shrinkage)
        if shrinkage_value < 0:
            raise ValueError(
                f"match_shrinkage must be non-negative, got {shrinkage_value}"
            )
        use_defensive = spec.params.get("use_defensive_contribution", True)
        if not isinstance(use_defensive, bool):
            raise ValueError(
                f"use_defensive_contribution must be a bool, got {use_defensive!r}"
            )
        self._match_shrinkage = shrinkage_value
        self._use_defensive = use_defensive
        base_params = {
            key: value for key, value in spec.params.items() if key != "match_shrinkage"
        }
        self._base = OtherPlayerPriors(
            ModelSpec(
                name=f"{spec.name}__base",
                kind="other_player",
                params=base_params,
                refit=spec.refit,
                seed=spec.seed,
                train_window=spec.train_window,
            )
        )
        self._fitted = False
        n_pos = len(_POSITIONS)
        self._delta_goals = np.zeros((n_pos, 3), dtype=np.float64)
        self._delta_cs = np.zeros((n_pos, 2), dtype=np.float64)
        self._delta_scored = np.zeros((n_pos, 2), dtype=np.float64)

    def fit(self, view: PITView) -> None:
        """Fit the minutes-bucket baseline and match-context offsets."""
        self._base.fit(view)
        self._fitted = True
        self._delta_goals[:] = 0.0
        self._delta_cs[:] = 0.0
        self._delta_scored[:] = 0.0
        history = _match_history(view, use_defensive=self._use_defensive)
        if history.is_empty():
            return
        residuals: list[_Residual] = []
        for row in history.iter_rows(named=True):
            position = str(row["position"])
            if position not in _POS_INDEX:
                continue
            minutes = int(row["minutes"])
            if minutes <= 0:
                continue
            bucket = _bucket(minutes)
            baseline = self._base.expected_total(
                int(row["player_key"]), position, bucket
            )
            residual = float(row["actual_other"]) - baseline
            residuals.append(
                (
                    position,
                    _goals_band(int(row["team_goals"])),
                    int(row["clean_sheet"]),
                    int(row["player_scored"]),
                    residual,
                )
            )
        if not residuals:
            return
        self._delta_goals = _factor_deltas(
            residuals,
            index_fn=_goals_index,
            shape=(len(_POSITIONS), 3),
            shrinkage=self._match_shrinkage,
        )
        self._delta_cs = _factor_deltas(
            residuals,
            index_fn=_cs_index,
            shape=(len(_POSITIONS), 2),
            shrinkage=self._match_shrinkage,
        )
        self._delta_scored = _factor_deltas(
            residuals,
            index_fn=_scored_index,
            shape=(len(_POSITIONS), 2),
            shrinkage=self._match_shrinkage,
        )

    def predict(self, view: PITView, player_fixtures: pl.DataFrame) -> pl.DataFrame:
        """Minutes-bucket baseline rows (``OtherPointsSchema``).

        Args:
            view: Deadline view.
            player_fixtures: Columns ``player_key`` and ``fixture_id``.

        Returns:
            Unconditional other-points frame from the wrapped ``other_player``.

        Raises:
            RuntimeError: ``fit`` has not been called.
        """
        if not self._fitted:
            raise RuntimeError(
                "fit the match-conditioned other model before predicting"
            )
        return self._base.predict(view, player_fixtures)

    def match_delta(
        self,
        *,
        positions: np.ndarray,
        minutes: np.ndarray,
        team_goals: np.ndarray,
        clean_sheet: np.ndarray,
        player_goals: np.ndarray,
    ) -> np.ndarray:
        """Additive other points on top of the minutes-bucket baseline.

        Args:
            positions: Position codes, shape ``(P,)``.
            minutes: Drawn minutes, shape ``(P, n)``.
            team_goals: Team goals for the player's side, shape ``(P, n)``.
            clean_sheet: Clean-sheet indicators after red-void, shape ``(P, n)``.
            player_goals: Goals allocated to the player, shape ``(P, n)``.

        Returns:
            Offsets with shape ``(P, n)``. Zero wherever ``minutes == 0``.

        Raises:
            RuntimeError: ``fit`` has not been called.
            ValueError: Shapes disagree.
        """
        if not self._fitted:
            raise RuntimeError(
                "fit the match-conditioned other model before match_delta"
            )
        played = np.asarray(minutes, dtype=np.float64)
        goals = np.asarray(team_goals, dtype=np.int64)
        sheets = np.asarray(clean_sheet, dtype=np.bool_)
        scored = np.asarray(player_goals, dtype=np.int64)
        if played.ndim != 2:
            raise ValueError(f"minutes must be 2-D, got shape {played.shape}")
        players, sims = played.shape
        if goals.shape != (players, sims):
            raise ValueError(
                f"team_goals must have shape {(players, sims)}, got {goals.shape}"
            )
        if sheets.shape != (players, sims):
            raise ValueError(
                f"clean_sheet must have shape {(players, sims)}, got {sheets.shape}"
            )
        if scored.shape != (players, sims):
            raise ValueError(
                f"player_goals must have shape {(players, sims)}, got {scored.shape}"
            )
        if len(positions) != players:
            raise ValueError(f"positions length {len(positions)} != players {players}")
        pos_idx = np.array(
            [_POS_INDEX.get(str(position), -1) for position in positions],
            dtype=np.int64,
        )
        delta = np.zeros((players, sims), dtype=np.float64)
        eligible = pos_idx >= 0
        if not np.any(eligible):
            return delta
        band = np.where(goals <= 0, 0, np.where(goals == 1, 1, 2))
        cs_idx = sheets.astype(np.int64)
        scored_idx = (scored >= 1).astype(np.int64)
        for player in np.where(eligible)[0]:
            index = int(pos_idx[player])
            delta[player] = (
                self._delta_goals[index, band[player]]
                + self._delta_cs[index, cs_idx[player]]
                + self._delta_scored[index, scored_idx[player]]
            )
            delta[player] = np.where(played[player] > 0, delta[player], 0.0)
        return delta


def _goals_index(position: str, band: str, _cs: int, _scored: int) -> tuple[int, int]:
    return _POS_INDEX[position], _band_index(band)


def _cs_index(position: str, _band: str, cs: int, _scored: int) -> tuple[int, int]:
    return _POS_INDEX[position], cs


def _scored_index(position: str, _band: str, _cs: int, scored: int) -> tuple[int, int]:
    return _POS_INDEX[position], scored


def _factor_deltas(
    residuals: list[_Residual],
    *,
    index_fn: _IndexFn,
    shape: tuple[int, int],
    shrinkage: float,
) -> np.ndarray:
    """Centered, shrunk mean residual for one match factor."""
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.float64)
    pos_sums = np.zeros(shape[0], dtype=np.float64)
    pos_counts = np.zeros(shape[0], dtype=np.float64)
    for position, band, cs, scored, residual in residuals:
        pos = _POS_INDEX[position]
        row, col = index_fn(position, band, cs, scored)
        sums[row, col] += residual
        counts[row, col] += 1.0
        pos_sums[pos] += residual
        pos_counts[pos] += 1.0
    pos_mean = np.divide(
        pos_sums,
        pos_counts,
        out=np.zeros_like(pos_sums),
        where=pos_counts > 0,
    )
    cell_mean = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    )
    centered = cell_mean - pos_mean[:, None]
    denom = counts + shrinkage
    weight = np.divide(
        counts,
        denom,
        out=np.zeros_like(counts),
        where=denom > 0,
    )
    return weight * centered


def _match_history(view: PITView, *, use_defensive: bool) -> pl.DataFrame:
    """Played appearances with actual other and match context."""
    stats = view.latest(
        "player_fixture_stats", ["season", "player_key", "fixture_id"]
    ).collect()
    if stats.is_empty():
        return stats
    roster = (
        view.latest("players", ["season", "player_key"])
        .select("season", "player_key", "team_key", "position")
        .collect()
    )
    fixtures = (
        view.latest("fixtures", ["season", "fixture_id"])
        .select(
            "season",
            "fixture_id",
            "team_h_key",
            "team_a_key",
            "team_h_score",
            "team_a_score",
            "finished",
        )
        .collect()
    )
    empty = pl.DataFrame(
        schema={
            "player_key": pl.Int64(),
            "position": pl.Utf8(),
            "minutes": pl.Int64(),
            "team_goals": pl.Int64(),
            "clean_sheet": pl.Int64(),
            "player_scored": pl.Int64(),
            "actual_other": pl.Float64(),
        }
    )
    if fixtures.is_empty():
        return empty
    history = (
        stats.join(roster, on=["season", "player_key"], how="inner")
        .join(fixtures, on=["season", "fixture_id"], how="inner")
        .filter(pl.col("finished") & pl.col("team_h_score").is_not_null())
        .sort(["player_key", "season", "gw", "fixture_id"])
    )
    rules_cache: dict[str, Rules] = {}
    rows: list[dict[str, object]] = []
    for row in history.iter_rows(named=True):
        minutes = int(row["minutes"])
        if minutes <= 0:
            continue
        season = str(row["season"])
        if season not in rules_cache:
            rules_cache[season] = Rules.for_season(season)
        position = str(row["position"])
        bonus, saves, cards, defensive, other = _components(
            rules_cache[season],
            position,
            row,
            use_defensive=use_defensive,
        )
        team_key = int(row["team_key"])
        if team_key == int(row["team_h_key"]):
            team_goals = int(row["team_h_score"] or 0)
        else:
            team_goals = int(row["team_a_score"] or 0)
        rows.append(
            {
                "player_key": int(row["player_key"]),
                "position": position,
                "minutes": minutes,
                "team_goals": team_goals,
                "clean_sheet": int(int(row.get("clean_sheets") or 0) > 0),
                "player_scored": int(int(row.get("goals_scored") or 0) > 0),
                "actual_other": bonus + saves + cards + defensive + other,
            }
        )
    if not rows:
        return empty
    return pl.DataFrame(rows)


def _components(
    rules: Rules,
    position: str,
    row: dict[str, Any],
    *,
    use_defensive: bool,
) -> tuple[float, float, float, float, float]:
    def points(**fields: int) -> float:
        return float(rules.score(MatchStats(position=position, **fields)))  # type: ignore[arg-type]

    bonus = points(bonus=int(row["bonus"]))
    saves = points(saves=int(row["saves"]))
    cards = points(
        yellow_cards=int(row["yellow_cards"]), red_cards=int(row["red_cards"])
    )
    defensive = 0.0
    if use_defensive and rules.defensive_contributions:
        if "defensive_contribution" in row:
            count = int(row["defensive_contribution"] or 0)
            for rule in rules.defensive_contributions:
                if position in rule.positions and count >= rule.threshold:
                    defensive = float(rule.points)
                    break
    other = points(
        penalties_saved=int(row["penalties_saved"]),
        penalties_missed=int(row["penalties_missed"]),
        own_goals=int(row["own_goals"]),
    )
    return (bonus, saves, cards, defensive, other)


def _bucket(minutes: int) -> str:
    if minutes <= 0:
        return "0"
    if minutes < 60:
        return "1-59"
    return "60+"


def _goals_band(goals: int) -> str:
    if goals <= 0:
        return "0"
    if goals == 1:
        return "1"
    return "2+"


def _band_index(band: str) -> int:
    return _GOALS_BANDS.index(band)


register_component("other_match_conditioned")(OtherMatchConditioned)
