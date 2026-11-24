"""Minutes v2: multi-window rates, context logits, soft ~11-starter projection.

Independence across teammates remains the default draw model. When
``soft_squad`` is on, expected starters per team-fixture are nudged toward
``target_starters`` before the PMF is built — a soft stand-in for a hard
eleven-starter constraint. Rolling shares and empirical rates exclude
injury / suspension absences (PIT status at each GW deadline). Logistic
features include ``games_since_return`` / ``absence_length``;
``return_ramp_games`` dampens P(start) for the first games back. FPL
status/chance/news gating defaults on (``use_availability_gate``; eval-only
off-fork ``dc_layered_noavail_v1``). Optional ``use_injury_overlay`` applies
staged ``player_injuries``. Optional ``predicted_xi`` mixes P(start) toward
staged ``predicted_lineups`` near kickoff. Production ``dc_layered_v4`` keeps
injury/XI overlays off — enable via ``dc_layered_injuries_v1`` /
``dc_layered_predicted_xi_v1``. Rest / congestion use all-comp kickoffs
(API-Football cups + Europe) when ``competition_fixtures`` is staged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import polars as pl

from fplt.data.pit import PITView
from fplt.models.component_params import (
    EmpiricalMinutesV2Params,
    parse_component_params,
)
from fplt.models.components.minutes.absence import (
    annotate_absences,
    available_rate,
    available_rolling_shares,
    return_metrics,
    return_ramp_factor,
    scale_return_feature,
)
from fplt.models.components.minutes.calibrate import (
    ChanceCalibration,
    calibrated_long_factor,
    effective_chance_pct,
    fit_chance_calibration,
)
from fplt.models.components.minutes.common import (
    START_FALLBACK,
    SUB_FALLBACK,
    WIDTH,
    availability,
    availability_gate,
    counts,
    fixture_injuries,
    horizon_step,
    injuries,
    normalise,
    positions,
    resolve_injury,
    shrunk,
)
from fplt.models.components.minutes.congestion import (
    competition_team_kickoffs,
    team_schedule_features,
)
from fplt.models.components.minutes.predicted_xi import apply_predicted_xi
from fplt.models.distributions import MINUTE_BINS, MinutesDist
from fplt.models.spec import ModelSpec, register_component
from fplt.types import Position

_LONG = MINUTE_BINS >= 60
_START_DIM = 13
_LONG_DIM = 5


class EmpiricalMinutesV2:
    """Logistic P(start) and P(≥60|start) with histogram residuals.

    Rolling start shares over ``windows``, minutes volatility, home/away,
    all-comp rest days, games in the last 7 days, DGW, price, opponent
    strength, and return-from-absence features feed two small logistics fit
    on appearances visible at the deadline. Injury / suspension zeros are
    excluded from rolling shares and empirical rates (P4). Conditional minute
    shapes still come from shrunk start/sub histograms. ``logistic_blend``
    weights the logistic P(start) against the multi-window empirical rate
    (default 0.5). ``return_ramp_games`` dampens blended P(start) for the
    first games back after an absence. Availability (status /
    chance_of_playing) gates both rates after the logits, with
    ``availability_recovery_games`` recovering the gate toward 1.0 for
    horizon steps after the next-round FPL anchor (P3). When
    ``calibrate_chance_flags`` is on (default), snapshot ``player_status``
    replaces face-value chance with fitted play rates (P1) and scales
    P(≥60|start) for flagged players (P2). Optional ``use_injury_overlay``
    also applies staged ``player_injuries`` (default off — production stays
    FPL-tags-only until eval); player-level injury tags use the same
    recovery curve. When ``predicted_xi`` is true, near-deadline predicted
    lineup roles mix into P(start) before soft-squad projection.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        params = parse_component_params(EmpiricalMinutesV2Params, spec.params)
        parsed = _parse_windows(params.windows)
        self._windows = tuple(sorted(set(parsed)))
        self._shrinkage = params.shrinkage
        self._doubtful = params.doubtful_chance
        self._soft_squad = params.soft_squad
        self._target_starters = params.target_starters
        self._soft_strength = params.soft_squad_strength
        self._logistic_blend = params.logistic_blend
        self._l2 = params.l2
        self._return_ramp_games = params.return_ramp_games
        self._availability_recovery_games = params.availability_recovery_games
        self._use_injury_overlay = params.use_injury_overlay
        self._use_availability_gate = params.use_availability_gate
        self._calibrate_chance_flags = params.calibrate_chance_flags
        self._chance_cal_min_count = params.chance_cal_min_count
        self._predicted_xi = params.predicted_xi
        self._xi_window_hours = params.predicted_xi_window_hours
        self._xi_strength = params.predicted_xi_strength
        self._fitted = False
        self._calibration = ChanceCalibration.identity()
        self._position: dict[int, str] = {}
        self._player_rate: dict[int, tuple[float, float]] = {}
        self._player_features: dict[int, dict[str, float]] = {}
        self._player_start: dict[int, np.ndarray] = {}
        self._player_sub: dict[int, np.ndarray] = {}
        self._position_rate: dict[str, tuple[float, float]] = {}
        self._position_start: dict[str, np.ndarray] = {}
        self._position_sub: dict[str, np.ndarray] = {}
        self._start_beta = np.zeros(_START_DIM, dtype=np.float64)
        self._long_beta = np.zeros(_LONG_DIM, dtype=np.float64)

    def fit(self, view: PITView) -> None:
        """Estimate rates, histograms, and context logits from ``view``."""
        self._fitted = True
        self._position = positions(view)
        stats = (
            view.latest("player_fixture_stats", ["season", "player_key", "fixture_id"])
            .collect()
            .sort(["season", "gw", "fixture_id", "available_at"])
        )
        self._player_rate = {}
        self._player_features = {}
        self._player_start = {}
        self._player_sub = {}
        self._position_rate = {}
        self._position_start = {}
        self._position_sub = {}
        self._start_beta = np.zeros(_START_DIM, dtype=np.float64)
        self._long_beta = np.zeros(_LONG_DIM, dtype=np.float64)
        self._calibration = ChanceCalibration.identity()
        if stats.is_empty():
            return
        if self._calibrate_chance_flags:
            self._calibration = fit_chance_calibration(
                view, stats, min_count=self._chance_cal_min_count
            )
        tagged = annotate_absences(view, stats)
        start_counts: dict[str, np.ndarray] = {}
        sub_counts: dict[str, np.ndarray] = {}
        seen: dict[str, int] = {}
        started: dict[str, int] = {}
        subbed: dict[str, int] = {}
        for key, group in tagged.group_by("player_key", maintain_order=True):
            player = int(key[0])
            position = self._position.get(player, "UNK")
            self._remember_player(player, group)
            _accumulate_position(
                position, group, start_counts, sub_counts, seen, started, subbed
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
        training = _training_rows(view, tagged, self._windows)
        if training.height > 0:
            start_x = np.column_stack(
                [
                    np.ones(training.height),
                    training.select(_START_FEATURE_COLS).to_numpy(),
                ]
            )
            long_x = np.column_stack(
                [
                    np.ones(training.height),
                    training.select(_LONG_FEATURE_COLS).to_numpy(),
                ]
            )
            self._start_beta = _fit_logit(
                start_x,
                np.asarray(training["y_start"].to_numpy(), dtype=np.float64),
                self._l2,
            )
            starters = training.filter(pl.col("y_start") > 0.5)
            if starters.height > 0:
                long_x = np.column_stack(
                    [
                        np.ones(starters.height),
                        starters.select(_LONG_FEATURE_COLS).to_numpy(),
                    ]
                )
                self._long_beta = _fit_logit(
                    long_x,
                    np.asarray(starters["y_long"].to_numpy(), dtype=np.float64),
                    self._l2,
                )

    def predict(self, view: PITView, player_fixtures: pl.DataFrame) -> MinutesDist:
        """Minutes distribution for each player-fixture.

        Args:
            view: Deadline view used for status, fixtures, and team maps.
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
        if keys.is_empty():
            return MinutesDist(keys, MINUTE_BINS, np.zeros((0, WIDTH)))
        context = _predict_context(view, keys)
        chance_map = availability(view)
        injury_map = injuries(view) if self._use_injury_overlay else {}
        injury_by_fixture = fixture_injuries(view) if self._use_injury_overlay else {}
        p_start = np.zeros(keys.height, dtype=np.float64)
        p_sub = np.zeros(keys.height, dtype=np.float64)
        p_long = np.zeros(keys.height, dtype=np.float64)
        start_hist = np.zeros((keys.height, WIDTH), dtype=np.float64)
        sub_hist = np.zeros((keys.height, WIDTH), dtype=np.float64)
        for index, row in enumerate(context.iter_rows(named=True)):
            player = int(row["player_key"])
            fixture = int(row["fixture_id"])
            position = self._position.get(player, "UNK")
            tags = chance_map.get(player)
            injury = (
                resolve_injury(player, fixture, injury_map, injury_by_fixture)
                if self._use_injury_overlay
                else None
            )
            target_gw = row.get("gw")
            step = (
                horizon_step(view.as_of.gw, int(target_gw))
                if target_gw is not None
                else 0
            )
            cal = (
                self._calibration
                if self._calibrate_chance_flags and self._use_availability_gate
                else None
            )
            gate = availability_gate(
                tags,
                self._doubtful,
                injury=injury,
                use_injury_overlay=self._use_injury_overlay,
                use_availability_gate=self._use_availability_gate,
                step=step,
                recovery_games=self._availability_recovery_games,
                calibration=cal,
            )
            features = self._player_features.get(player, _empty_player_features())
            start_x = _start_vector(features, row)
            long_x = _long_vector(features, row)
            raw_start = float(_sigmoid(start_x @ self._start_beta))
            raw_long = float(_sigmoid(long_x @ self._long_beta))
            if player in self._player_rate:
                base_start, base_sub = self._player_rate[player]
            else:
                base_start, base_sub = self._position_rate.get(position, (0.0, 0.0))
            # Blend logistic with the multi-window empirical rate so sparse
            # early-season fits do not ignore role stickiness.
            blend = self._logistic_blend
            blended = blend * raw_start + (1.0 - blend) * base_start
            ramp = return_ramp_factor(
                features.get("games_since_return", 0.0),
                features.get("absence_length", 0.0),
                self._return_ramp_games,
                from_suspension=bool(features.get("from_suspension", 0.0)),
            )
            p_start[index] = blended * ramp * gate
            p_sub[index] = base_sub * gate * (1.0 - blended)
            # P2: scale P(≥60|start) for flagged players using snapshot maps.
            if cal is None or tags is None:
                p_long[index] = raw_long
            else:
                pct = effective_chance_pct(
                    tags.status,
                    tags.chance_next,
                    chance_this=tags.chance_this,
                )
                p_long[index] = raw_long * calibrated_long_factor(pct, cal)
            start_hist[index] = shrunk(
                self._player_start.get(player),
                self._position_start.get(position),
                START_FALLBACK,
                self._shrinkage,
            )
            sub_hist[index] = shrunk(
                self._player_sub.get(player),
                self._position_sub.get(position),
                SUB_FALLBACK,
                self._shrinkage,
            )
        if self._predicted_xi:
            p_start, p_sub = apply_predicted_xi(
                view,
                context,
                p_start,
                p_sub,
                window_hours=self._xi_window_hours,
                strength=self._xi_strength,
            )
        if self._soft_squad:
            p_start = _soft_project_starters(
                context, p_start, self._target_starters, self._soft_strength
            )
        pmf = np.zeros((keys.height, WIDTH), dtype=np.float64)
        for index in range(keys.height):
            played = p_start[index] + p_sub[index]
            if played > 1.0:
                p_start[index] /= played
                p_sub[index] /= played
                played = 1.0
            start = _reweight_long(start_hist[index], p_long[index])
            pmf[index, 0] = 1.0 - played
            pmf[index] += p_start[index] * start
            pmf[index] += p_sub[index] * sub_hist[index]
        return MinutesDist(keys, MINUTE_BINS, pmf)

    def _remember_player(self, player: int, group: pl.DataFrame) -> None:
        minutes = group["minutes"].to_numpy()
        starts = group["starts"].to_numpy()
        features = available_rolling_shares(group, self._windows)
        self._player_rate[player] = available_rate(group)
        self._player_features[player] = features
        self._player_start[player] = counts(minutes[starts > 0])
        self._player_sub[player] = counts(minutes[(starts == 0) & (minutes > 0)])


_START_FEATURE_COLS = [
    "starts_share_3",
    "starts_share_5",
    "starts_share_10",
    "minutes_std_5",
    "was_home",
    "rest_days",
    "games_last_7d",
    "is_dgw",
    "price",
    "opp_strength",
    "games_since_return",
    "absence_length",
]
_LONG_FEATURE_COLS = [
    "minutes_mean_5",
    "starts_share_5",
    "minutes_std_5",
    "was_home",
]


def _empty_player_features() -> dict[str, float]:
    return {
        "starts_share_3": 0.0,
        "starts_share_5": 0.0,
        "starts_share_10": 0.0,
        "minutes_mean_5": 0.0,
        "minutes_std_5": 0.0,
        "games_since_return": 0.0,
        "absence_length": 0.0,
        "from_suspension": 0.0,
    }


def _accumulate_position(
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
    # Rate denominator excludes injury / suspension absences (P4).
    available = (
        group.filter(~pl.col("is_absence")) if "is_absence" in group.columns else group
    )
    avail_minutes = available["minutes"].to_numpy()
    avail_starts = available["starts"].to_numpy()
    avail_start_rows = avail_minutes[avail_starts > 0]
    avail_sub_rows = avail_minutes[(avail_starts == 0) & (avail_minutes > 0)]
    seen[position] = seen.get(position, 0) + available.height
    started[position] = started.get(position, 0) + int(len(avail_start_rows))
    subbed[position] = subbed.get(position, 0) + int(len(avail_sub_rows))


def _training_rows(
    view: PITView, stats: pl.DataFrame, windows: tuple[int, ...]
) -> pl.DataFrame:
    """Shifted rolling features joined to fixture context for logistic fit.

    Rolling start / minutes windows skip absence rows (null them before the
    rolling mean) so injury zeros do not dilute the share. Return features
    are shifted so row *t* sees ``games_since_return`` / ``absence_length``
    from appearances strictly before *t*.
    """
    empty = pl.DataFrame(
        schema={
            "y_start": pl.Float64(),
            "y_long": pl.Float64(),
            **{name: pl.Float64() for name in _START_FEATURE_COLS},
            "minutes_mean_5": pl.Float64(),
        }
    )
    if stats.is_empty():
        return empty
    ordered = stats.sort(["player_key", "available_at"])
    if "is_absence" not in ordered.columns:
        ordered = ordered.with_columns(pl.lit(False).alias("is_absence"))
    # Null starts/minutes on absences so rolling means ignore them.
    masked = ordered.with_columns(
        pl.when(pl.col("is_absence"))
        .then(None)
        .otherwise((pl.col("starts") > 0).cast(pl.Float64))
        .alias("_start_flag"),
        pl.when(pl.col("is_absence"))
        .then(None)
        .otherwise(pl.col("minutes").cast(pl.Float64))
        .alias("_minutes_avail"),
    )
    exprs: list[pl.Expr] = []
    shifted_starts = pl.col("_start_flag").shift(1)
    shifted_minutes = pl.col("_minutes_avail").shift(1)
    for window in (3, 5, 10):
        exprs.append(
            shifted_starts.rolling_mean(window_size=window, min_samples=1)
            .over("player_key")
            .alias(f"starts_share_{window}")
        )
    exprs.append(
        shifted_minutes.rolling_mean(window_size=5, min_samples=1)
        .over("player_key")
        .alias("minutes_mean_5")
    )
    exprs.append(
        shifted_minutes.rolling_std(window_size=5, min_samples=2)
        .over("player_key")
        .alias("minutes_std_5")
    )
    rolled = masked.with_columns(exprs)
    return_rows = _shifted_return_features(ordered)
    rolled = rolled.join(
        return_rows, on=["player_key", "fixture_id"], how="left"
    ).drop_nulls(subset=["starts_share_3", "starts_share_5", "starts_share_10"])
    if rolled.is_empty():
        return empty
    context = _fixture_context(view)
    players = (
        view.latest("players", ["season", "player_key"])
        .select("season", "player_key", "team_key", "position")
        .collect()
    )
    status = (
        view.latest("player_status", ["season", "player_key"])
        .select("season", "player_key", "price")
        .collect()
    )
    joined = (
        rolled.join(players, on=["season", "player_key"], how="left")
        .join(status, on=["season", "player_key"], how="left")
        .join(context, on=["season", "fixture_id"], how="left")
        .with_columns(
            (pl.col("team_key") == pl.col("team_h_key"))
            .cast(pl.Float64)
            .alias("was_home"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("team_a_key"))
            .otherwise(pl.col("team_h_key"))
            .alias("opponent_key"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("rest_home"))
            .otherwise(pl.col("rest_away"))
            .alias("rest_days"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("games_home"))
            .otherwise(pl.col("games_away"))
            .alias("games_last_7d"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("dgw_home"))
            .otherwise(pl.col("dgw_away"))
            .alias("is_dgw"),
            (pl.col("starts") > 0).cast(pl.Float64).alias("y_start"),
            (pl.col("minutes") >= 60).cast(pl.Float64).alias("y_long"),
            (pl.col("price").cast(pl.Float64) / 100.0).fill_null(0.5).alias("price"),
            (pl.col("minutes_std_5").fill_null(0.0) / 90.0).alias("minutes_std_5"),
            (pl.col("minutes_mean_5").fill_null(0.0) / 90.0).alias("minutes_mean_5"),
            (pl.col("games_since_return").fill_null(0.0).clip(0.0, 10.0) / 10.0).alias(
                "games_since_return"
            ),
            (pl.col("absence_length").fill_null(0.0).clip(0.0, 10.0) / 10.0).alias(
                "absence_length"
            ),
        )
    )
    ratings = _opponent_ratings(view)
    _ = windows
    return (
        joined.join(ratings, left_on="opponent_key", right_on="team_key", how="left")
        .with_columns(
            pl.col("opp_strength").fill_null(1.0),
            pl.col("was_home").fill_null(0.0),
            pl.col("rest_days").fill_null(1.0),
            pl.col("games_last_7d").fill_null(0.0),
            pl.col("is_dgw").fill_null(0.0),
        )
        .select(
            "y_start",
            "y_long",
            "starts_share_3",
            "starts_share_5",
            "starts_share_10",
            "minutes_std_5",
            "was_home",
            "rest_days",
            "games_last_7d",
            "is_dgw",
            "price",
            "opp_strength",
            "games_since_return",
            "absence_length",
            "minutes_mean_5",
        )
        .drop_nulls()
    )


def _shifted_return_features(ordered: pl.DataFrame) -> pl.DataFrame:
    """Per-row return metrics from appearances strictly before that row."""
    rows: list[dict[str, float | int]] = []
    has_suspend = "is_suspension" in ordered.columns
    for key, group in ordered.group_by("player_key", maintain_order=True):
        player = int(key[0])
        flags = [bool(flag) for flag in group["is_absence"].to_list()]
        suspend = (
            [bool(flag) for flag in group["is_suspension"].to_list()]
            if has_suspend
            else None
        )
        fixtures = group["fixture_id"].to_list()
        for index, fixture in enumerate(fixtures):
            gsr, alen, from_suspension = return_metrics(
                flags[:index],
                suspend[:index] if suspend is not None else None,
            )
            rows.append(
                {
                    "player_key": player,
                    "fixture_id": int(fixture),
                    "games_since_return": gsr,
                    "absence_length": alen,
                    "from_suspension": 1.0 if from_suspension else 0.0,
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "player_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "games_since_return": pl.Float64(),
                "absence_length": pl.Float64(),
                "from_suspension": pl.Float64(),
            }
        )
    return pl.DataFrame(rows)


def _fixture_context(view: PITView) -> pl.DataFrame:
    fixtures = (
        view.latest("fixtures", ["season", "fixture_id"])
        .select(
            "season",
            "fixture_id",
            "gw",
            "team_h_key",
            "team_a_key",
            "kickoff_time",
        )
        .collect()
        .sort(["season", "kickoff_time", "fixture_id"])
    )
    if fixtures.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "fixture_id": pl.Int64(),
                "gw": pl.Int64(),
                "team_h_key": pl.Int64(),
                "team_a_key": pl.Int64(),
                "rest_home": pl.Float64(),
                "rest_away": pl.Float64(),
                "games_home": pl.Float64(),
                "games_away": pl.Float64(),
                "dgw_home": pl.Float64(),
                "dgw_away": pl.Float64(),
            }
        )
    # DGW flag: team appears twice in the same GW (home or away).
    home = fixtures.select(
        "season", "gw", pl.col("team_h_key").alias("team_key"), "fixture_id"
    )
    away = fixtures.select(
        "season", "gw", pl.col("team_a_key").alias("team_key"), "fixture_id"
    )
    appearances = pl.concat([home, away])
    dgw_teams = (
        appearances.group_by(["season", "gw", "team_key"])
        .len()
        .filter(pl.col("len") > 1)
        .select("season", "gw", "team_key", pl.lit(1.0).alias("is_dgw_team"))
    )
    schedule = team_schedule_features(fixtures, competition_team_kickoffs(view))
    rest = (
        fixtures.select(
            "season", "fixture_id", "gw", "team_h_key", "team_a_key", "kickoff_time"
        )
        .join(
            schedule.rename(
                {
                    "team_key": "team_h_key",
                    "rest_days": "rest_home",
                    "games_last_7d": "games_home",
                }
            ),
            on=["season", "fixture_id", "team_h_key"],
            how="left",
        )
        .join(
            schedule.rename(
                {
                    "team_key": "team_a_key",
                    "rest_days": "rest_away",
                    "games_last_7d": "games_away",
                }
            ),
            on=["season", "fixture_id", "team_a_key"],
            how="left",
        )
        .select(
            "season",
            "fixture_id",
            "team_h_key",
            "team_a_key",
            "rest_home",
            "rest_away",
            "games_home",
            "games_away",
            "gw",
        )
    )
    return (
        rest.join(
            dgw_teams.rename({"team_key": "team_h_key", "is_dgw_team": "dgw_home"}),
            on=["season", "gw", "team_h_key"],
            how="left",
        )
        .join(
            dgw_teams.rename({"team_key": "team_a_key", "is_dgw_team": "dgw_away"}),
            on=["season", "gw", "team_a_key"],
            how="left",
        )
        .with_columns(
            pl.col("dgw_home").fill_null(0.0),
            pl.col("dgw_away").fill_null(0.0),
            pl.col("rest_home").fill_null(1.0),
            pl.col("rest_away").fill_null(1.0),
            pl.col("games_home").fill_null(0.0),
            pl.col("games_away").fill_null(0.0),
        )
    )


def _opponent_ratings(view: PITView) -> pl.DataFrame:
    fixtures = (
        view.latest("fixtures", ["season", "fixture_id"])
        .collect()
        .filter(pl.col("team_h_score").is_not_null())
    )
    empty = pl.DataFrame(schema={"team_key": pl.Int64(), "opp_strength": pl.Float64()})
    if fixtures.is_empty():
        return empty
    home = fixtures.select(
        pl.col("team_h_key").alias("team_key"),
        pl.col("team_h_score").alias("gf"),
        pl.col("team_a_score").alias("ga"),
    )
    away = fixtures.select(
        pl.col("team_a_key").alias("team_key"),
        pl.col("team_a_score").alias("gf"),
        pl.col("team_h_score").alias("ga"),
    )
    return (
        pl.concat([home, away])
        .group_by("team_key")
        .agg(((pl.col("gf").mean() + pl.col("ga").mean()) / 2.0).alias("opp_strength"))
    )


def _predict_context(view: PITView, keys: pl.DataFrame) -> pl.DataFrame:
    players = (
        view.latest("players", ["season", "player_key"])
        .select("season", "player_key", "team_key", "position")
        .collect()
    )
    status = (
        view.latest("player_status", ["season", "player_key"])
        .select("season", "player_key", "price")
        .collect()
    )
    fixtures = _fixture_context(view)
    ratings = _opponent_ratings(view)
    season = view.as_of.season
    framed = keys.with_columns(pl.lit(season).alias("season"))
    joined = (
        framed.join(players, on=["season", "player_key"], how="left")
        .join(status, on=["season", "player_key"], how="left")
        .join(fixtures, on=["season", "fixture_id"], how="left")
        .with_columns(
            (pl.col("team_key") == pl.col("team_h_key")).alias("is_home"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("team_a_key"))
            .otherwise(pl.col("team_h_key"))
            .alias("opponent_key"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("rest_home"))
            .otherwise(pl.col("rest_away"))
            .alias("side_rest"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("games_home"))
            .otherwise(pl.col("games_away"))
            .alias("side_games"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("dgw_home"))
            .otherwise(pl.col("dgw_away"))
            .alias("side_dgw"),
        )
        .join(ratings, left_on="opponent_key", right_on="team_key", how="left")
        .with_columns(
            pl.col("is_home").cast(pl.Float64).fill_null(0.0).alias("was_home"),
            pl.col("side_rest").fill_null(1.0).alias("rest_days"),
            pl.col("side_games").fill_null(0.0).alias("games_last_7d"),
            pl.col("side_dgw").fill_null(0.0).alias("is_dgw"),
            (pl.col("price").cast(pl.Float64) / 100.0).fill_null(0.5).alias("price"),
            pl.col("opp_strength").fill_null(1.0),
            pl.col("position")
            .replace_strict(
                {member.name: float(member.value) for member in Position},
                default=None,
            )
            .cast(pl.Float64)
            .alias("position_enc"),
        )
        .select(
            "player_key",
            "fixture_id",
            "gw",
            "team_key",
            "was_home",
            "rest_days",
            "games_last_7d",
            "is_dgw",
            "price",
            "opp_strength",
            "position_enc",
        )
    )
    return joined


def _start_vector(features: dict[str, float], row: dict[str, object]) -> np.ndarray:
    return np.array(
        [
            1.0,
            features.get("starts_share_3", 0.0),
            features.get("starts_share_5", 0.0),
            features.get("starts_share_10", 0.0),
            features.get("minutes_std_5", 0.0) / 90.0,
            _as_float(row["was_home"]),
            _as_float(row["rest_days"]),
            _as_float(row["games_last_7d"]),
            _as_float(row["is_dgw"]),
            _as_float(row["price"]),
            _as_float(row["opp_strength"]),
            scale_return_feature(features.get("games_since_return", 0.0)),
            scale_return_feature(features.get("absence_length", 0.0)),
        ],
        dtype=np.float64,
    )


def _long_vector(features: dict[str, float], row: dict[str, object]) -> np.ndarray:
    return np.array(
        [
            1.0,
            features.get("minutes_mean_5", 0.0) / 90.0,
            features.get("starts_share_5", 0.0),
            features.get("minutes_std_5", 0.0) / 90.0,
            _as_float(row["was_home"]),
        ],
        dtype=np.float64,
    )


def _parse_windows(windows: object) -> list[int]:
    """Validate the multi-window list from the model spec."""
    if not isinstance(windows, list | tuple) or not windows:
        raise ValueError(f"windows must be a non-empty list, got {windows!r}")
    parsed: list[int] = []
    for value in cast(Sequence[object], windows):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"windows must be positive ints, got {windows!r}")
        parsed.append(value)
    if any(value < 1 for value in parsed):
        raise ValueError(f"windows must be positive ints, got {windows!r}")
    return parsed


def _as_float(value: object) -> float:
    """Coerce a scalar to float, treating missing values as zero."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, np.generic):
        return float(cast(int | float, value.item()))
    raise TypeError(f"expected a numeric scalar, got {type(value)!r}")


def _soft_project_starters(
    context: pl.DataFrame,
    p_start: np.ndarray,
    target: float,
    strength: float,
) -> np.ndarray:
    """Nudge per-team expected starters toward ``target`` without hard caps."""
    if strength <= 0 or context.is_empty():
        return p_start
    adjusted = p_start.copy()
    team = context["team_key"].to_list()
    fixture = context["fixture_id"].to_list()
    groups: dict[tuple[object, object], list[int]] = {}
    for index, key in enumerate(zip(fixture, team, strict=True)):
        groups.setdefault(key, []).append(index)
    for indexes in groups.values():
        current = float(adjusted[indexes].sum())
        if current <= 0:
            continue
        scale = (target / current) ** strength
        adjusted[indexes] *= scale
        capped = adjusted[indexes]
        overflow = capped > 1.0
        if np.any(overflow):
            capped = np.minimum(capped, 1.0)
            adjusted[indexes] = capped
    return adjusted


def _reweight_long(hist: np.ndarray, p_long: float) -> np.ndarray:
    """Tilt a start histogram so P(minutes ≥ 60) matches ``p_long``."""
    target = float(np.clip(p_long, 0.0, 1.0))
    long_mass = float(hist[_LONG].sum())
    short_mass = float(hist[~_LONG].sum())
    out = hist.copy()
    if long_mass <= 0 and short_mass <= 0:
        return START_FALLBACK.copy()
    if long_mass <= 0:
        # No long mass observed: put target on the 90 bin, keep short residual.
        out = np.zeros_like(hist)
        out[~_LONG] = hist[~_LONG] * (1.0 - target) / max(short_mass, 1e-12)
        out[int(np.flatnonzero(MINUTE_BINS == 90)[0])] = target
        return out
    if short_mass <= 0:
        out[_LONG] = hist[_LONG] * target / long_mass
        # Preserve a tiny short residual only when target < 1.
        if target < 1.0:
            out[int(np.flatnonzero(MINUTE_BINS == 45)[0])] = 1.0 - target
            out[_LONG] *= target / max(float(out[_LONG].sum()), 1e-12)
        return out
    out[_LONG] = hist[_LONG] * (target / long_mass)
    out[~_LONG] = hist[~_LONG] * ((1.0 - target) / short_mass)
    total = float(out.sum())
    return out / total if total > 0 else START_FALLBACK.copy()


def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logit(
    x: np.ndarray, y: np.ndarray, l2: float, max_iter: int = 25
) -> np.ndarray:
    """Fit L2-regularised logistic regression with Newton steps."""
    beta = np.zeros(x.shape[1], dtype=np.float64)
    if x.shape[0] == 0:
        return beta
    eye = np.eye(x.shape[1], dtype=np.float64)
    for _ in range(max_iter):
        probs = np.asarray(_sigmoid(x @ beta), dtype=np.float64)
        weights = probs * (1.0 - probs) + 1e-6
        grad = x.T @ (y - probs) - l2 * beta
        hessian = (x.T * weights) @ x + l2 * eye
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-6:
            break
    return beta


register_component("empirical_minutes_v2")(EmpiricalMinutesV2)
