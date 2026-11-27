"""Player-level other-points means, shrunk toward position × minutes bucket.

Bonus uses realised bonus points blended with a BPS→bonus proxy when the
player has few samples in a bucket. Saves, cards, defensive contribution,
and the residual ``other`` bucket follow the same shrink. Defensive points
use staged ``defensive_contribution`` action counts when the season's
scoring rules award them; older seasons stay at zero. Set
``use_defensive_contribution: false`` to force that channel to zero for
A/B evals (default ``true``).

``defcon_mode: rate`` (default ``mean``) replaces the shrunk 0/2 DC prior
with ``P(award | minutes, position, rate)`` via a Poisson action-rate model
(``other_player_defcon_rate_v1``). Released ``other_player_v1`` keeps ``mean``.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from fplt.data.pit import PITView
from fplt.models.component_params import OtherPlayerParams, parse_component_params
from fplt.models.components.other.defcon_rate import (
    action_count,
    award_rule,
    expected_defcon_points,
    fit_player_rates,
    fit_position_rates,
    parse_bucket_minutes,
)
from fplt.models.spec import ModelSpec, register_component
from fplt.schemas.components import (
    MINUTE_BUCKETS,
    OTHER_POINTS_COLUMNS,
    OtherPointsSchema,
)
from fplt.scoring.rules import MatchStats, Rules
from fplt.utils.frames import require_dataframe

_POSITIONS = frozenset({"GK", "DEF", "MID", "FWD"})
_Points = tuple[float, float, float, float, float]
_ZERO: _Points = (0.0, 0.0, 0.0, 0.0, 0.0)
_Key = tuple[int, str]
_PosKey = tuple[str, str]


class OtherPlayerPriors:
    """Shrunk player × minutes-bucket means for non-goal FPL points.

    ``window`` keeps each player's latest appearances per bucket.
    ``shrinkage`` is a pseudo-count of the position × bucket mean.
    ``bps_bonus_weight`` blends a BPS→bonus lookup into the bonus estimate
    when samples are scarce (weight decays as ``n / (n + shrinkage)`` grows).
    ``use_defensive_contribution`` (default true) scores staged defensive
    action counts when the season rules award them; false zeros that channel.
    ``defcon_mode`` is ``mean`` (shrunk realised awards) or ``rate``
    (Poisson P(award) from actions-per-90).
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        params = parse_component_params(OtherPlayerParams, spec.params)
        self._window = params.window
        self._shrinkage = params.shrinkage
        self._bps_weight = params.bps_bonus_weight
        self._use_defensive = params.use_defensive_contribution
        self._defcon_mode = params.defcon_mode
        self._bucket_minutes = parse_bucket_minutes(params.defcon_bucket_minutes)
        self._fitted = False
        self._prior: dict[_PosKey, _Points] = {}
        self._player: dict[_Key, _Points] = {}
        self._bps_bonus: dict[int, float] = {}
        self._defcon_pos_rate: dict[str, float] = {}
        self._defcon_player_rate: dict[int, float] = {}
        self._defcon_rule_by_pos: dict[str, tuple[int, int]] = {}

    def fit(self, view: PITView) -> None:
        """Estimate player and position priors from appearances on ``view``."""
        self._fitted = True
        self._prior = {}
        self._player = {}
        self._bps_bonus = {}
        self._defcon_pos_rate = {}
        self._defcon_player_rate = {}
        self._defcon_rule_by_pos = {}
        history = _scored_history(view, use_defensive=self._use_defensive)
        if history.is_empty():
            return
        self._bps_bonus = _bps_bonus_lookup(history)
        samples: dict[_PosKey, list[_Points]] = {}
        player_rows: dict[_Key, list[tuple[_Points, float]]] = {}
        player_position: dict[int, str] = {}
        zero_dc_in_mean = self._defcon_mode == "rate"
        for row in history.iter_rows(named=True):
            position = str(row["position"])
            if position not in _POSITIONS:
                continue
            player = int(row["player_key"])
            bucket = str(row["minutes_bucket"])
            defensive = 0.0 if zero_dc_in_mean else float(row["defensive_contribution"])
            points = (
                float(row["bonus"]),
                float(row["saves"]),
                float(row["cards"]),
                defensive,
                float(row["other"]),
            )
            samples.setdefault((position, bucket), []).append(points)
            player_rows.setdefault((player, bucket), []).append(
                (points, float(row["bps"]))
            )
            player_position[player] = position
        for key, values in samples.items():
            self._prior[key] = _mean_points(values)
        for key, rows in player_rows.items():
            recent = rows[-self._window :]
            position = player_position.get(key[0], "")
            prior = self._prior.get((position, key[1]), _ZERO)
            self._player[key] = _shrunk_player(
                recent,
                prior,
                self._shrinkage,
                self._bps_weight,
                self._bps_bonus,
            )
        if self._defcon_mode == "rate" and self._use_defensive:
            self._fit_defcon_rates(view)

    def expected_total(
        self, player_key: int, position: str, minutes_bucket: str
    ) -> float:
        """Sum of other channels for one player × minutes bucket.

        Args:
            player_key: Stable player code.
            position: ``GK`` / ``DEF`` / ``MID`` / ``FWD``.
            minutes_bucket: One of ``0``, ``1-59``, ``60+``.

        Returns:
            Bonus + saves + cards + defensive contribution + residual other.

        Raises:
            RuntimeError: ``fit`` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("fit the other-points model before expected_total")
        points = self._player.get((player_key, minutes_bucket))
        if points is None:
            points = self._prior.get((position, minutes_bucket), _ZERO)
        if self._defcon_mode == "rate":
            points = self._with_rate_defcon(
                points, player_key, position, minutes_bucket
            )
        return float(sum(points))

    def predict(self, view: PITView, player_fixtures: pl.DataFrame) -> pl.DataFrame:
        """Three bucket rows for each player-fixture.

        Args:
            view: Deadline view. Position comes from the latest player row.
            player_fixtures: Columns ``player_key`` and ``fixture_id``.

        Returns:
            A frame matching ``OtherPointsSchema``.

        Raises:
            RuntimeError: ``fit`` has not been called.
            ValueError: ``player_fixtures`` lacks a required column.
        """
        if not self._fitted:
            raise RuntimeError("fit the other-points model before predicting")
        missing = {"player_key", "fixture_id"} - set(player_fixtures.columns)
        if missing:
            raise ValueError(f"player_fixtures is missing {sorted(missing)}")
        asked = player_fixtures.select("player_key", "fixture_id").unique(
            maintain_order=True
        )
        if asked.is_empty():
            return _validated(pl.DataFrame(schema=OTHER_POINTS_COLUMNS))
        positions = _positions(view)
        buckets = pl.DataFrame({"minutes_bucket": list(MINUTE_BUCKETS)})
        frame = asked.join(buckets, how="cross").join(
            positions, on="player_key", how="left"
        )
        rows: list[dict[str, object]] = []
        for player, fixture, bucket, position in frame.iter_rows():
            player_key = int(player)
            bucket_name = str(bucket)
            points = self._player.get((player_key, bucket_name))
            if points is None:
                points = self._prior.get((str(position), bucket_name), _ZERO)
            if self._defcon_mode == "rate":
                points = self._with_rate_defcon(
                    points, player_key, str(position), bucket_name
                )
            rows.append(_row(player_key, int(fixture), bucket_name, points))
        return _validated(pl.DataFrame(rows, schema=OTHER_POINTS_COLUMNS))

    def _fit_defcon_rates(self, view: PITView) -> None:
        """Fit position/player actions-per-90 from DC-awarding seasons."""
        stats = view.latest(
            "player_fixture_stats", ["season", "player_key", "fixture_id"]
        ).collect()
        if stats.is_empty():
            return
        roster = (
            view.latest("players", ["season", "player_key"])
            .select("season", "player_key", "position")
            .collect()
        )
        history = stats.join(roster, on=["season", "player_key"], how="inner").sort(
            ["player_key", "season", "gw", "fixture_id"]
        )
        rules_cache: dict[str, Rules] = {}
        rate_samples: list[tuple[str, float, float]] = []
        player_samples: list[tuple[int, str, float, float]] = []
        for row in history.iter_rows(named=True):
            season = str(row["season"])
            if season not in rules_cache:
                rules_cache[season] = Rules.for_season(season)
            rules = rules_cache[season]
            if not rules.defensive_contributions:
                continue
            position = str(row["position"])
            if position not in _POSITIONS:
                continue
            minutes = int(row["minutes"])
            if minutes <= 0:
                continue
            rule = award_rule(rules, position)
            if rule is None:
                continue
            self._defcon_rule_by_pos[position] = rule
            actions = float(action_count(row, position, rules))
            nineties = minutes / 90.0
            rate_samples.append((position, actions, nineties))
            player_samples.append((int(row["player_key"]), position, actions, nineties))
        self._defcon_pos_rate = fit_position_rates(rate_samples)
        self._defcon_player_rate = fit_player_rates(
            player_samples,
            self._defcon_pos_rate,
            window=self._window,
            shrinkage=self._shrinkage,
        )

    def _with_rate_defcon(
        self,
        points: _Points,
        player_key: int,
        position: str,
        bucket: str,
    ) -> _Points:
        """Replace the DC channel with the Poisson rate expectation."""
        if not self._use_defensive:
            return (points[0], points[1], points[2], 0.0, points[4])
        rule = self._defcon_rule_by_pos.get(position)
        if rule is None:
            return (points[0], points[1], points[2], 0.0, points[4])
        threshold, award_points = rule
        rate = self._defcon_player_rate.get(
            player_key, self._defcon_pos_rate.get(position, 0.0)
        )
        minutes = self._bucket_minutes.get(bucket, 0.0)
        expected = expected_defcon_points(
            rate_per_90=rate,
            minutes=minutes,
            threshold=threshold,
            points=award_points,
        )
        return (points[0], points[1], points[2], expected, points[4])


def _scored_history(view: PITView, *, use_defensive: bool = True) -> pl.DataFrame:
    stats = view.latest(
        "player_fixture_stats", ["season", "player_key", "fixture_id"]
    ).collect()
    if stats.is_empty():
        return stats
    roster = (
        view.latest("players", ["season", "player_key"])
        .select("season", "player_key", "position")
        .collect()
    )
    history = stats.join(roster, on=["season", "player_key"], how="inner").sort(
        ["player_key", "season", "gw", "fixture_id"]
    )
    rules_cache: dict[str, Rules] = {}
    rows: list[dict[str, object]] = []
    for row in history.iter_rows(named=True):
        season = str(row["season"])
        if season not in rules_cache:
            rules_cache[season] = Rules.for_season(season)
        bonus, saves, cards, defensive, other = _components(
            rules_cache[season],
            str(row["position"]),
            row,
            use_defensive=use_defensive,
        )
        rows.append(
            {
                "player_key": int(row["player_key"]),
                "position": row["position"],
                "minutes_bucket": _bucket(int(row["minutes"])),
                "bonus": bonus,
                "saves": saves,
                "cards": cards,
                "defensive_contribution": defensive,
                "other": other,
                "bps": int(row.get("bps") or 0),
            }
        )
    return pl.DataFrame(rows)


def _components(
    rules: Rules,
    position: str,
    row: dict[str, Any],
    *,
    use_defensive: bool = True,
) -> _Points:
    def points(**fields: int) -> float:
        return float(rules.score(MatchStats(position=position, **fields)))  # type: ignore[arg-type]

    bonus = points(bonus=int(row["bonus"]))
    saves = points(saves=int(row["saves"]))
    cards = points(
        yellow_cards=int(row["yellow_cards"]), red_cards=int(row["red_cards"])
    )
    defensive = _defensive_points(rules, position, row, use_defensive=use_defensive)
    # Appearance points are not part of "other"; isolate residual awards.
    other = points(
        penalties_saved=int(row["penalties_saved"]),
        penalties_missed=int(row["penalties_missed"]),
        own_goals=int(row["own_goals"]),
    )
    return (bonus, saves, cards, defensive, other)


def _defensive_points(
    rules: Rules,
    position: str,
    row: dict[str, Any],
    *,
    use_defensive: bool = True,
) -> float:
    """Map staged defensive contribution to points for ``position``.

    When ``use_defensive`` is false or the season table has no defensive rules,
    return 0. Otherwise prefer the live ``defensive_contribution`` action count
    (CBI+T for defenders, CBI+T+R for midfielders/forwards) against that
    position's threshold. Frames that predate the staged column fall back to
    summing the action columns through ``Rules.score``.
    """
    if not use_defensive or not rules.defensive_contributions:
        return 0.0
    if "defensive_contribution" in row:
        count = int(row["defensive_contribution"] or 0)
        for rule in rules.defensive_contributions:
            if position in rule.positions and count >= rule.threshold:
                return float(rule.points)
        return 0.0
    return float(
        rules.score(
            MatchStats(
                position=position,  # type: ignore[arg-type]
                clearances_blocks_interceptions=int(
                    row.get("clearances_blocks_interceptions") or 0
                ),
                tackles=int(row.get("tackles") or 0),
                recoveries=int(row.get("recoveries") or 0),
            )
        )
    )


def _bps_bonus_lookup(history: pl.DataFrame) -> dict[int, float]:
    played = history.filter(pl.col("minutes_bucket") != "0")
    if played.is_empty():
        return {}
    grouped = played.group_by("bps").agg(pl.col("bonus").mean().alias("mean_bonus"))
    return {
        int(row["bps"]): float(row["mean_bonus"])
        for row in grouped.iter_rows(named=True)
    }


def _shrunk_player(
    recent: list[tuple[_Points, float]],
    prior: _Points,
    shrinkage: float,
    bps_weight: float,
    bps_bonus: dict[int, float],
) -> _Points:
    count = float(len(recent))
    means = _mean_points([points for points, _ in recent])
    shrunk = tuple(
        (count * means[index] + shrinkage * prior[index]) / (count + shrinkage)
        if count + shrinkage > 0
        else prior[index]
        for index in range(5)
    )
    mean_bps = sum(bps for _, bps in recent) / count
    proxy = bps_bonus.get(int(round(mean_bps)), prior[0])
    blend = bps_weight * shrinkage / (count + shrinkage) if count + shrinkage else 0.0
    bonus = (1.0 - blend) * shrunk[0] + blend * proxy
    return (bonus, shrunk[1], shrunk[2], shrunk[3], shrunk[4])


def _mean_points(values: list[_Points]) -> _Points:
    count = float(len(values))
    return tuple(  # type: ignore[return-value]
        sum(sample[index] for sample in values) / count for index in range(5)
    )


def _bucket(minutes: int) -> str:
    if minutes <= 0:
        return "0"
    if minutes < 60:
        return "1-59"
    return "60+"


def _positions(view: PITView) -> pl.DataFrame:
    roster = (
        view.latest("players", ["season", "player_key"])
        .select("player_key", "position", "available_at")
        .collect()
    )
    if roster.is_empty():
        return pl.DataFrame(schema={"player_key": pl.Int64(), "position": pl.Utf8()})
    return (
        roster.sort("available_at")
        .unique(subset=["player_key"], keep="last")
        .select("player_key", "position")
    )


def _row(
    player: int,
    fixture: int,
    bucket: str,
    prior: _Points,
) -> dict[str, object]:
    bonus, saves, cards, defensive, other = prior
    return {
        "player_key": player,
        "fixture_id": fixture,
        "minutes_bucket": bucket,
        "bonus": bonus,
        "saves": saves,
        "cards": cards,
        "defensive_contribution": defensive,
        "other": other,
    }


def _validated(frame: pl.DataFrame) -> pl.DataFrame:
    checked = OtherPointsSchema.validate(frame)
    checked = require_dataframe(checked)
    return checked


register_component("other_player")(OtherPlayerPriors)
