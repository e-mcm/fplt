"""Position priors for the points the simulator does not draw.

Goals, assists, appearance, clean sheets, and goals conceded are left to
the scoring rules. What remains is the mean, by position and minutes
bucket, of bonus, save points, card points, and the rest. This v1 prior
keeps defensive contribution at zero; ``other_player`` uses staged
``defensive_contribution`` when the season rules award it.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from fplt.data.pit import PITView
from fplt.models.component_params import OtherPriorsParams, parse_component_params
from fplt.models.spec import ModelSpec, register_component
from fplt.schemas.components import (
    MINUTE_BUCKETS,
    OTHER_POINTS_COLUMNS,
    OtherPointsSchema,
)
from fplt.scoring.rules import MatchStats, Rules

_POSITIONS = frozenset({"GK", "DEF", "MID", "FWD"})
_Points = tuple[float, float, float, float, float]
_ZERO: _Points = (0.0, 0.0, 0.0, 0.0, 0.0)


class OtherPointsPriors:
    """Mean bonus, saves, cards, and other points by position and minutes.

    Buckets are ``0``, ``1-59``, and ``60+``. A position with no appearance
    in a bucket gets zeros there. Every player of a position shares that
    position's prior; nothing here is player-specific.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        parse_component_params(OtherPriorsParams, spec.params)
        self._fitted = False
        self._prior: dict[tuple[str, str], _Points] = {}

    def fit(self, view: PITView) -> None:
        """Average visible appearances into position-and-bucket priors."""
        self._fitted = True
        self._prior = {}
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
        history = stats.join(roster, on=["season", "player_key"], how="inner")
        totals: dict[tuple[str, str], list[_Points]] = {}
        rules: dict[str, Rules] = {}
        for row in history.iter_rows(named=True):
            position = row["position"]
            if position not in _POSITIONS:
                continue
            season = str(row["season"])
            if season not in rules:
                rules[season] = Rules.for_season(season)
            bucket = _bucket(int(row["minutes"]))
            totals.setdefault((position, bucket), []).append(
                _components(rules[season], position, row)
            )
        for key, samples in totals.items():
            count = float(len(samples))
            means = [
                sum(sample[index] for sample in samples) / count for index in range(5)
            ]
            self._prior[key] = (means[0], means[1], means[2], means[3], means[4])

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
            prior = self._prior.get((position, str(bucket)), _ZERO)
            rows.append(_row(int(player), int(fixture), str(bucket), prior))
        return _validated(pl.DataFrame(rows, schema=OTHER_POINTS_COLUMNS))


def _bucket(minutes: int) -> str:
    if minutes <= 0:
        return "0"
    if minutes < 60:
        return "1-59"
    return "60+"


def _components(rules: Rules, position: str, row: dict[str, Any]) -> _Points:
    def points(**fields: int) -> float:
        return float(rules.score(MatchStats(position=position, **fields)))  # type: ignore[arg-type]

    bonus = points(bonus=int(row["bonus"]))
    saves = points(saves=int(row["saves"]))
    cards = points(
        yellow_cards=int(row["yellow_cards"]), red_cards=int(row["red_cards"])
    )
    other = points(
        penalties_saved=int(row["penalties_saved"]),
        penalties_missed=int(row["penalties_missed"]),
        own_goals=int(row["own_goals"]),
    )
    return (bonus, saves, cards, 0.0, other)


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
    prior: tuple[float, float, float, float, float],
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
    assert isinstance(checked, pl.DataFrame)
    return checked


register_component("other_priors")(OtherPointsPriors)
