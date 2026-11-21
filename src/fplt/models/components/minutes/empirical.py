"""Empirical minutes: a recent start rate and a shrunk minutes histogram.

Each player is drawn on their own. Nothing in this component stops two
teammates both playing 90 minutes.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fplt.data.pit import PITView
from fplt.models.components.minutes.common import (
    START_FALLBACK,
    SUB_FALLBACK,
    WIDTH,
    availability,
    chance,
    counts,
    normalise,
    positions,
    shrunk,
)
from fplt.models.distributions import MINUTE_BINS, MinutesDist
from fplt.models.spec import ModelSpec, register_component


class EmpiricalMinutes:
    """P(start) from recent appearances, then a shrunk start or sub histogram.

    ``window`` is how many of the player's latest appearances set the start
    rate and the player's own histogram. ``shrinkage`` is the number of
    pseudo-counts taken from the position prior, which uses every visible
    appearance. ``doubtful_chance`` is the chance used when status is ``d``
    and the feed omits ``chance_of_playing``. A missing status row is treated
    as available, which is what the vaastav seasons record.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        window = spec.params.get("window", 5)
        shrinkage = spec.params.get("shrinkage", 5.0)
        doubtful = spec.params.get("doubtful_chance", 0.5)
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive int, got {window!r}")
        shrinkage_value = float(shrinkage)
        doubtful_value = float(doubtful)
        if shrinkage_value < 0:
            raise ValueError(f"shrinkage must be non-negative, got {shrinkage_value}")
        if not 0.0 <= doubtful_value <= 1.0:
            raise ValueError(f"doubtful_chance must be in [0, 1], got {doubtful_value}")
        self._window = window
        self._shrinkage = shrinkage_value
        self._doubtful = doubtful_value
        self._fitted = False
        self._position: dict[int, str] = {}
        self._player_rate: dict[int, tuple[float, float]] = {}
        self._player_start: dict[int, np.ndarray] = {}
        self._player_sub: dict[int, np.ndarray] = {}
        self._position_rate: dict[str, tuple[float, float]] = {}
        self._position_start: dict[str, np.ndarray] = {}
        self._position_sub: dict[str, np.ndarray] = {}

    def fit(self, view: PITView) -> None:
        """Estimate rates and histograms from appearances visible on ``view``."""
        self._fitted = True
        self._position = positions(view)
        stats = (
            view.latest("player_fixture_stats", ["season", "player_key", "fixture_id"])
            .collect()
            .sort(["season", "gw", "fixture_id"])
        )
        self._player_rate = {}
        self._player_start = {}
        self._player_sub = {}
        self._position_rate = {}
        self._position_start = {}
        self._position_sub = {}
        if stats.is_empty():
            return
        start_counts: dict[str, np.ndarray] = {}
        sub_counts: dict[str, np.ndarray] = {}
        seen: dict[str, int] = {}
        started: dict[str, int] = {}
        subbed: dict[str, int] = {}
        grouped = stats.group_by("player_key", maintain_order=True)
        for key, group in grouped:
            player = int(key[0])
            position = self._position.get(player, "UNK")
            self._remember_player(player, group.tail(self._window))
            _remember_position(
                position,
                group,
                start_counts,
                sub_counts,
                seen,
                started,
                subbed,
            )
        for position, total in seen.items():
            self._position_rate[position] = (
                started.get(position, 0) / total,
                subbed.get(position, 0) / total,
            )
            self._position_start[position] = normalise(
                start_counts.get(position), START_FALLBACK
            )
            self._position_sub[position] = normalise(
                sub_counts.get(position), SUB_FALLBACK
            )

    def predict(self, view: PITView, player_fixtures: pl.DataFrame) -> MinutesDist:
        """Minutes distribution for each player-fixture.

        Args:
            view: Deadline view. Status is read from here, not from history
                published later.
            player_fixtures: Columns ``player_key`` and ``fixture_id``.

        Returns:
            A ``MinutesDist`` aligned with ``player_fixtures``.
        """
        if not self._fitted:
            raise RuntimeError("fit the minutes model before predicting")
        missing = {"player_key", "fixture_id"} - set(player_fixtures.columns)
        if missing:
            raise ValueError(f"player_fixtures is missing {sorted(missing)}")
        keys = player_fixtures.select("player_key", "fixture_id")
        chance_map = availability(view)
        pmf = np.zeros((keys.height, WIDTH), dtype=np.float64)
        for index, player in enumerate(keys["player_key"].to_list()):
            status, playing = chance_map.get(int(player), (None, None))
            pmf[index] = self._mix(int(player), status, playing)
        return MinutesDist(keys, MINUTE_BINS, pmf)

    def _remember_player(self, player: int, recent: pl.DataFrame) -> None:
        minutes = recent["minutes"].to_numpy()
        starts = recent["starts"].to_numpy()
        played = len(recent)
        self._player_rate[player] = (
            float(np.sum(starts > 0) / played),
            float(np.sum((starts == 0) & (minutes > 0)) / played),
        )
        self._player_start[player] = counts(minutes[starts > 0])
        self._player_sub[player] = counts(minutes[(starts == 0) & (minutes > 0)])

    def _mix(
        self, player: int, status: str | None, chance_of_playing: float | None
    ) -> np.ndarray:
        position = self._position.get(player, "UNK")
        if player in self._player_rate:
            p_start, p_sub = self._player_rate[player]
        else:
            p_start, p_sub = self._position_rate.get(position, (0.0, 0.0))
        gate = chance(status, chance_of_playing, self._doubtful)
        p_start *= gate
        p_sub *= gate
        # 4/5 + 1/5 is not exactly 1 in float, which would make the zero bin negative.
        played = p_start + p_sub
        if played > 1.0:
            p_start /= played
            p_sub /= played
            played = 1.0
        start = shrunk(
            self._player_start.get(player),
            self._position_start.get(position),
            START_FALLBACK,
            self._shrinkage,
        )
        sub = shrunk(
            self._player_sub.get(player),
            self._position_sub.get(position),
            SUB_FALLBACK,
            self._shrinkage,
        )
        pmf = np.zeros(WIDTH, dtype=np.float64)
        pmf[0] = 1.0 - played
        pmf += p_start * start
        pmf += p_sub * sub
        return pmf


def _remember_position(
    position: str,
    group: pl.DataFrame,
    start_counts: dict[str, np.ndarray],
    sub_counts: dict[str, np.ndarray],
    seen: dict[str, int],
    started: dict[str, int],
    subbed: dict[str, int],
) -> None:
    minutes = group["minutes"].to_numpy()
    starts = group["starts"].to_numpy()
    start_rows = minutes[starts > 0]
    sub_rows = minutes[(starts == 0) & (minutes > 0)]
    start_counts[position] = start_counts.get(position, np.zeros(WIDTH)) + counts(
        start_rows
    )
    sub_counts[position] = sub_counts.get(position, np.zeros(WIDTH)) + counts(sub_rows)
    seen[position] = seen.get(position, 0) + len(group)
    started[position] = started.get(position, 0) + int(len(start_rows))
    subbed[position] = subbed.get(position, 0) + int(len(sub_rows))


register_component("empirical_minutes")(EmpiricalMinutes)
