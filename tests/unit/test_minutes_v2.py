"""Empirical minutes v2: multi-window rates, context, soft squad projection."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fplt.models.components.minutes  # noqa: F401 — registers the component
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.minutes.empirical_v2 import EmpiricalMinutesV2
from fplt.models.distributions import MINUTE_BINS
from fplt.models.spec import ModelSpec, get_component, load_model_specs
from fplt.types import GWKey

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)
_ZERO = {
    "goals_scored": 0,
    "assists": 0,
    "clean_sheets": 0,
    "goals_conceded": 0,
    "saves": 0,
    "bonus": 0,
    "yellow_cards": 0,
    "red_cards": 0,
    "own_goals": 0,
    "penalties_saved": 0,
    "penalties_missed": 0,
    "total_points": 0,
}


def _spec(**params: object) -> ModelSpec:
    return ModelSpec(
        name="empirical_minutes_v2",
        kind="empirical_minutes_v2",
        params={
            "windows": [3, 5, 10],
            "shrinkage": 5,
            "doubtful_chance": 0.5,
            "soft_squad": False,
            "target_starters": 11.0,
            "soft_squad_strength": 0.5,
            "logistic_blend": 0.5,
            "l2": 1.0,
            "return_ramp_games": 2.0,
            # Face-value gate unless a test opts into snapshot maps (P1/P2).
            "calibrate_chance_flags": False,
            **params,
        },
    )


def test_spec_registers_as_a_component() -> None:
    specs = load_model_specs()
    assert get_component("empirical_minutes_v2") is EmpiricalMinutesV2
    assert specs["empirical_minutes_v2"].params["windows"] == [3, 5, 10]
    assert specs["empirical_minutes_v2"].params["soft_squad"] is True
    assert specs["empirical_minutes_v2"].params["logistic_blend"] == 0.5
    assert specs["empirical_minutes_v2"].params["return_ramp_games"] == 2.0
    assert specs["empirical_minutes_v2"].params["availability_recovery_games"] == 3.0
    assert specs["empirical_minutes_v2"].params["calibrate_chance_flags"] is True


def test_a_regular_starter_puts_mass_on_sixty_plus(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        stats=[(1, "MID", 1, gw, 90, 1) for gw in range(1, 8)],
        status={1: ("a", 100.0)},
    )
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    p_60 = float(dist.pmf[0, MINUTE_BINS >= 60].sum())
    assert p_60 > 0.7
    assert dist.pmf.sum() == pytest.approx(1.0)


def test_chance_of_zero_forces_the_bench(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        stats=[(1, "MID", 1, gw, 90, 1) for gw in range(1, 6)],
        status={1: ("a", 0.0)},
    )
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    assert dist.pmf[0, 0] == pytest.approx(1.0)


def test_horizon_gate_uses_next_chance_not_this_round(tmp_path: Path) -> None:
    """This-round zero must not zero the whole plan when next-round is full (P3)."""
    store = _store(
        tmp_path,
        stats=[(1, "MID", 1, gw, 90, 1) for gw in range(1, 8)],
        # (status, chance_next, chance_this) — available history, gated now.
        status={1: ("a", 100.0, 0.0)},
        max_gw=10,
    )
    _expose_future_fixtures(store, from_gw=8)
    model = EmpiricalMinutesV2(_spec(soft_squad=False))
    view = PITData(store).view(GWKey(_SEASON, 8))
    model.fit(view)
    this_round = model.predict(
        view, pl.DataFrame({"player_key": [1], "fixture_id": [8]})
    )
    next_round = model.predict(
        view, pl.DataFrame({"player_key": [1], "fixture_id": [9]})
    )
    far = model.predict(view, pl.DataFrame({"player_key": [1], "fixture_id": [10]}))
    assert this_round.pmf[0, 0] == pytest.approx(1.0)
    assert next_round.pmf[0, 0] < 0.5
    assert far.pmf[0, 0] < 0.5


def test_horizon_gate_recovers_when_both_chances_zero(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        stats=[(1, "MID", 1, gw, 90, 1) for gw in range(1, 8)],
        status={1: ("a", 0.0, 0.0)},
        max_gw=12,
    )
    _expose_future_fixtures(store, from_gw=8)
    model = EmpiricalMinutesV2(_spec(soft_squad=False, availability_recovery_games=3.0))
    view = PITData(store).view(GWKey(_SEASON, 8))
    model.fit(view)
    this_round = model.predict(
        view, pl.DataFrame({"player_key": [1], "fixture_id": [8]})
    )
    far = model.predict(view, pl.DataFrame({"player_key": [1], "fixture_id": [12]}))
    assert this_round.pmf[0, 0] == pytest.approx(1.0)
    # Step 4 with recovery_games=3 → gate 1.0, so minutes mass returns.
    assert far.pmf[0, 0] < 0.5


def test_availability_gate_off_ignores_zero_chance(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        stats=[(1, "MID", 1, gw, 90, 1) for gw in range(1, 6)],
        status={1: ("a", 0.0)},
    )
    model = EmpiricalMinutesV2(_spec(use_availability_gate=False, soft_squad=False))
    view = PITData(store).view(GWKey(_SEASON, 8))
    model.fit(view)
    dist = model.predict(view, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    assert dist.pmf[0, 0] < 0.5


def test_multi_window_rates_use_recent_share(tmp_path: Path) -> None:
    # Early starts, then five blanks — windows 3/5 should dominate toward zero.
    rows = [(1, "MID", 1, gw, 90, 1) for gw in range(1, 6)]
    rows += [(1, "MID", 1, gw, 0, 0) for gw in range(6, 11)]
    store = _store(tmp_path, stats=rows, status={1: ("a", 100.0)}, max_gw=11)
    dist = _predict(
        store, pl.DataFrame({"player_key": [1], "fixture_id": [11]}), as_of=11
    )
    assert dist.pmf[0, 0] > 0.5


def test_soft_squad_reduces_overstuffed_start_mass(tmp_path: Path) -> None:
    # Fourteen teammates who always start → soft projection should cut P(start).
    players = list(range(1, 15))
    rows = [(player, "MID", 1, gw, 90, 1) for player in players for gw in range(1, 6)]
    status = {player: ("a", 100.0) for player in players}
    store = _store(tmp_path, stats=rows, status=status)
    ask = pl.DataFrame({"player_key": players, "fixture_id": [8] * len(players)})
    free = _predict(store, ask, soft_squad=False)
    soft = _predict(store, ask, soft_squad=True, soft_squad_strength=1.0)
    free_starts = _approx_p_start(free.pmf)
    soft_starts = _approx_p_start(soft.pmf)
    assert soft_starts.sum() < free_starts.sum()
    assert soft_starts.sum() == pytest.approx(11.0, abs=1.5)


def test_logistic_blend_moves_p_start_between_arms(tmp_path: Path) -> None:
    # Early starts then blanks: empirical rate is low; blend=0 tracks it,
    # blend=1 leans on the logistic (home/rest context still softens).
    rows = [(1, "MID", 1, gw, 90, 1) for gw in range(1, 6)]
    rows += [(1, "MID", 1, gw, 0, 0) for gw in range(6, 11)]
    store = _store(tmp_path, stats=rows, status={1: ("a", 100.0)}, max_gw=11)
    ask = pl.DataFrame({"player_key": [1], "fixture_id": [11]})
    empirical = _predict(store, ask, as_of=11, logistic_blend=0.0, soft_squad=False)
    logistic = _predict(store, ask, as_of=11, logistic_blend=1.0, soft_squad=False)
    mid = _predict(store, ask, as_of=11, logistic_blend=0.5, soft_squad=False)
    p_emp = float(1.0 - empirical.pmf[0, 0])
    p_log = float(1.0 - logistic.pmf[0, 0])
    p_mid = float(1.0 - mid.pmf[0, 0])
    assert p_emp != pytest.approx(p_log, abs=1e-6)
    assert min(p_emp, p_log) <= p_mid <= max(p_emp, p_log)


def test_a_result_after_the_deadline_is_invisible(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        stats=[(1, "MID", 1, gw, 90, 1) for gw in range(1, 6)],
        status={1: ("a", 100.0)},
    )
    ask = pl.DataFrame({"player_key": [1], "fixture_id": [8]})
    first = _predict(store, ask)
    late = store.read("player_fixture_stats").filter(pl.col("fixture_id") == 1)
    cutoff = PITData(store).deadline(GWKey(_SEASON, 8))
    store.write(
        "player_fixture_stats",
        _utc(
            late.with_columns(
                pl.lit(0).alias("minutes"),
                pl.lit(0).alias("starts"),
                pl.lit(cutoff + timedelta(days=1)).alias("available_at"),
            )
        ),
    )
    second = _predict(store, ask)
    np.testing.assert_allclose(first.pmf, second.pmf)


def test_predict_requires_a_fit_and_key_columns(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        stats=[(1, "MID", 1, 1, 90, 1)],
        status={1: ("a", 100.0)},
    )
    view = PITData(store).view(GWKey(_SEASON, 8))
    with pytest.raises(RuntimeError, match="fit"):
        EmpiricalMinutesV2(_spec()).predict(
            view, pl.DataFrame({"player_key": [1], "fixture_id": [1]})
        )
    fitted = EmpiricalMinutesV2(_spec())
    fitted.fit(view)
    with pytest.raises(ValueError, match="fixture_id"):
        fitted.predict(view, pl.DataFrame({"player_key": [1]}))
    empty = fitted.predict(
        view,
        pl.DataFrame(
            {"player_key": [], "fixture_id": []},
            schema={"player_key": pl.Int64, "fixture_id": pl.Int64},
        ),
    )
    assert empty.pmf.shape == (0, len(MINUTE_BINS))


def test_bad_params_are_refused() -> None:
    with pytest.raises(ValueError, match="windows"):
        EmpiricalMinutesV2(_spec(windows=[]))
    with pytest.raises(ValueError, match="windows"):
        EmpiricalMinutesV2(_spec(windows=[0]))
    with pytest.raises(ValueError, match="windows"):
        EmpiricalMinutesV2(_spec(windows=["5"]))  # type: ignore[list-item]
    with pytest.raises(ValueError, match="soft_squad"):
        EmpiricalMinutesV2(_spec(soft_squad="yes"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_starters"):
        EmpiricalMinutesV2(_spec(target_starters=0))
    with pytest.raises(ValueError, match="soft_squad_strength"):
        EmpiricalMinutesV2(_spec(soft_squad_strength=2.0))
    with pytest.raises(ValueError, match="logistic_blend"):
        EmpiricalMinutesV2(_spec(logistic_blend=1.5))
    with pytest.raises(ValueError, match="l2"):
        EmpiricalMinutesV2(_spec(l2=-1))
    with pytest.raises(ValueError, match="return_ramp_games"):
        EmpiricalMinutesV2(_spec(return_ramp_games=-1))
    with pytest.raises(ValueError, match="chance_cal_min_count"):
        EmpiricalMinutesV2(_spec(chance_cal_min_count=0))
    with pytest.raises(ValueError, match="shrinkage"):
        EmpiricalMinutesV2(_spec(shrinkage=-1))
    with pytest.raises(ValueError, match="doubtful"):
        EmpiricalMinutesV2(_spec(doubtful_chance=1.5))


def test_a_new_player_uses_the_position_prior(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        stats=[(1, "MID", 1, gw, 90, 1) for gw in range(1, 6)],
        status={1: ("a", 100.0), 2: ("a", 100.0)},
        extra_players=[(2, "MID", 1)],
    )
    dist = _predict(store, pl.DataFrame({"player_key": [2], "fixture_id": [8]}))
    assert dist.pmf[0].sum() == pytest.approx(1.0)
    assert dist.pmf[0, 0] < 0.9


def test_empty_history_still_predicts(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        stats=[],
        status={1: ("a", 100.0)},
        extra_players=[(1, "MID", 1)],
        max_gw=8,
    )
    # No finished appearances — fixtures still exist for the horizon.
    store.write(
        "player_fixture_stats",
        _utc(
            pl.DataFrame(
                {
                    "season": pl.Series("season", [], dtype=pl.Utf8),
                    "gw": pl.Series("gw", [], dtype=pl.Int64),
                    "player_key": pl.Series("player_key", [], dtype=pl.Int64),
                    "fixture_id": pl.Series("fixture_id", [], dtype=pl.Int64),
                    "minutes": pl.Series("minutes", [], dtype=pl.Int64),
                    "starts": pl.Series("starts", [], dtype=pl.Int64),
                    "available_at": pl.Series(
                        "available_at", [], dtype=pl.Datetime("us", "UTC")
                    ),
                    "source": pl.Series("source", [], dtype=pl.Utf8),
                    **{name: pl.Series(name, [], dtype=pl.Int64) for name in _ZERO},
                }
            )
        ),
    )
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    assert dist.pmf.shape == (1, len(MINUTE_BINS))
    assert dist.pmf[0].sum() == pytest.approx(1.0)


def test_helper_edge_cases() -> None:
    from fplt.models.components.minutes.empirical_v2 import (
        _as_float,
        _fit_logit,
        _reweight_long,
        _soft_project_starters,
    )

    assert _as_float(None) == 0.0
    assert _as_float(True) == 1.0
    assert _as_float(np.float64(2.5)) == 2.5
    with pytest.raises(TypeError):
        _as_float("x")

    zeros = np.zeros(len(MINUTE_BINS), dtype=np.float64)
    assert _reweight_long(zeros, 0.8)[MINUTE_BINS == 90][0] == pytest.approx(1.0)

    short_only = zeros.copy()
    short_only[MINUTE_BINS == 15] = 1.0
    tilted = _reweight_long(short_only, 0.6)
    assert tilted[MINUTE_BINS >= 60].sum() == pytest.approx(0.6)

    long_only = zeros.copy()
    long_only[MINUTE_BINS == 90] = 1.0
    tilted_long = _reweight_long(long_only, 0.7)
    assert tilted_long[MINUTE_BINS >= 60].sum() == pytest.approx(0.7)
    assert tilted_long[MINUTE_BINS == 45][0] == pytest.approx(0.3)

    mixed = zeros.copy()
    mixed[MINUTE_BINS == 90] = 0.5
    mixed[MINUTE_BINS == 15] = 0.5
    assert _reweight_long(mixed, 0.25).sum() == pytest.approx(1.0)

    context = pl.DataFrame(
        {
            "team_key": [1, 1, 1],
            "fixture_id": [8, 8, 8],
        }
    )
    starts = np.array([0.0, 0.0, 0.0])
    assert np.allclose(_soft_project_starters(context, starts, 11.0, 1.0), starts)
    under = np.array([0.4, 0.4, 0.2])
    lifted = _soft_project_starters(context, under, 11.0, 1.0)
    assert np.all(lifted <= 1.0)
    assert lifted.sum() == pytest.approx(3.0)

    empty = _fit_logit(np.zeros((0, 2)), np.zeros(0), 1.0)
    assert empty.shape == (2,)
    # Singular-ish design: constant features still return finite coeffs.
    x = np.ones((4, 2), dtype=np.float64)
    y = np.array([0.0, 1.0, 0.0, 1.0])
    beta = _fit_logit(x, y, 1.0)
    assert np.all(np.isfinite(beta))


def test_distributions_stay_non_negative(tmp_path: Path) -> None:
    rows = [(1, "MID", 1, gw, 90, 1) for gw in range(1, 5)]
    rows.append((1, "MID", 1, 5, 15, 0))
    store = _store(tmp_path, stats=rows, status={1: ("a", 100.0)})
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    assert np.all(dist.pmf >= 0)
    assert dist.pmf.sum() == pytest.approx(1.0)


def test_soft_uplift_keeps_pmf_valid(tmp_path: Path) -> None:
    # Few locked starters + soft uplift toward 11 can push start+sub over 1
    # before renormalisation; the PMF must still be valid.
    players = list(range(1, 4))
    rows = [(player, "MID", 1, gw, 90, 1) for player in players for gw in range(1, 6)]
    rows += [(player, "MID", 1, gw, 15, 0) for player in players for gw in range(6, 8)]
    status = {player: ("a", 100.0) for player in players}
    store = _store(tmp_path, stats=rows, status=status)
    ask = pl.DataFrame({"player_key": players, "fixture_id": [8] * len(players)})
    dist = _predict(store, ask, soft_squad=True, soft_squad_strength=1.0)
    assert np.all(dist.pmf >= 0)
    np.testing.assert_allclose(dist.pmf.sum(axis=1), 1.0)


def _approx_p_start(pmf: np.ndarray) -> np.ndarray:
    # Mass on 60+ is a stand-in for expected starters in these synthetic fixtures.
    return pmf[:, MINUTE_BINS >= 60].sum(axis=1)


def _predict(
    store: DataStore,
    player_fixtures: pl.DataFrame,
    *,
    as_of: int = 8,
    soft_squad: bool = False,
    soft_squad_strength: float = 0.5,
    logistic_blend: float = 0.5,
):
    model = EmpiricalMinutesV2(
        _spec(
            soft_squad=soft_squad,
            soft_squad_strength=soft_squad_strength,
            logistic_blend=logistic_blend,
        )
    )
    view = PITData(store).view(GWKey(_SEASON, as_of))
    model.fit(view)
    return model.predict(view, player_fixtures)


def _store(
    root: Path,
    *,
    stats: list[tuple[int, str, int, int, int, int]],
    status: dict[int, tuple[str, float | None]],
    max_gw: int = 8,
    extra_players: list[tuple[int, str, int]] | None = None,
) -> DataStore:
    """Build a minimal store.

    ``stats`` rows are ``(player_key, position, team_key, gw, minutes, starts)``.
    """
    store = DataStore(root)
    players = [(player, position, team) for player, position, team, _, _, _ in stats]
    players += extra_players or []
    unique = list(dict.fromkeys(players))
    store.write(
        "players",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * len(unique),
                    "player_key": [player for player, _, _ in unique],
                    "element_id": [player for player, _, _ in unique],
                    "team_key": [team for _, _, team in unique],
                    "position": [position for _, position, _ in unique],
                    "web_name": [str(player) for player, _, _ in unique],
                    "available_at": [_START] * len(unique),
                    "source": ["fpl_api"] * len(unique),
                }
            )
        ),
    )
    team_keys = sorted({team for _, _, team in unique} | {2})
    store.write(
        "teams",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * len(team_keys),
                    "team_key": team_keys,
                    "team_id": team_keys,
                    "name": [f"T{team}" for team in team_keys],
                    "short_name": [f"T{team}" for team in team_keys],
                    "available_at": [_START] * len(team_keys),
                    "source": ["fpl_api"] * len(team_keys),
                }
            )
        ),
    )
    deadlines = [_kick(gw) - timedelta(hours=21) for gw in range(1, max_gw + 1)]
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * max_gw,
                    "gw": list(range(1, max_gw + 1)),
                    "deadline_time": deadlines,
                    "finished": [gw < max_gw for gw in range(1, max_gw + 1)],
                    "data_checked": [gw < max_gw for gw in range(1, max_gw + 1)],
                    "available_at": [_START] * max_gw,
                    "source": ["fpl_api"] * max_gw,
                }
            )
        ),
    )
    fixture_ids = list(range(1, max_gw + 1))
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * len(fixture_ids),
                    "fixture_id": fixture_ids,
                    "gw": fixture_ids,
                    "team_h_key": [1] * len(fixture_ids),
                    "team_a_key": [2] * len(fixture_ids),
                    "kickoff_time": [_kick(gw) for gw in fixture_ids],
                    "team_h_score": [1 if gw < max_gw else None for gw in fixture_ids],
                    "team_a_score": [0 if gw < max_gw else None for gw in fixture_ids],
                    "finished": [gw < max_gw for gw in fixture_ids],
                    "available_at": [
                        _kick(gw) + timedelta(hours=3)
                        if gw < max_gw
                        else _kick(gw) - timedelta(days=14)
                        for gw in fixture_ids
                    ],
                    "source": ["fpl_api"] * len(fixture_ids),
                }
            )
        ),
    )
    listed = list(status)
    store.write(
        "player_status",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * len(listed),
                    "player_key": listed,
                    "status": [status[player][0] for player in listed],
                    "news": [""] * len(listed),
                    "price": [70 if player == 1 else 50 for player in listed],
                    "ep_next": pl.Series(
                        "ep_next", [None] * len(listed), dtype=pl.Float64
                    ),
                    "chance_of_playing": pl.Series(
                        "chance_of_playing",
                        [status[player][1] for player in listed],
                        dtype=pl.Float64,
                    ),
                    "chance_of_playing_this_round": pl.Series(
                        "chance_of_playing_this_round",
                        [
                            (
                                float(status[player][2])
                                if len(status[player]) > 2
                                else status[player][1]
                            )
                            for player in listed
                        ],
                        dtype=pl.Float64,
                    ),
                    "available_at": [_START] * len(listed),
                    "source": ["fpl_api"] * len(listed),
                }
            )
        ),
    )
    store.write(
        "player_fixture_stats",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * len(stats),
                    "gw": [gw for _p, _pos, _team, gw, _m, _s in stats],
                    "player_key": [
                        player for player, _pos, _team, _gw, _m, _s in stats
                    ],
                    "fixture_id": [gw for _p, _pos, _team, gw, _m, _s in stats],
                    "minutes": [minutes for _p, _pos, _team, _gw, minutes, _s in stats],
                    "starts": [starts for _p, _pos, _team, _gw, _m, starts in stats],
                    "available_at": [
                        _kick(gw) + timedelta(hours=3)
                        for _p, _pos, _team, gw, _m, _s in stats
                    ],
                    "source": ["fpl_api"] * len(stats),
                    **{name: [0] * len(stats) for name in _ZERO},
                }
            )
        ),
    )
    return store


def _kick(gw: int) -> datetime:
    return datetime(2024, 8, 10, 15, tzinfo=UTC) + timedelta(days=7 * (gw - 1))


def _expose_future_fixtures(store: DataStore, *, from_gw: int) -> None:
    """Stamp schedule rows so fixtures from ``from_gw`` are deadline-visible."""
    fixtures = store.read("fixtures")
    early = _START
    store.write(
        "fixtures",
        _utc(
            fixtures.with_columns(
                pl.when(pl.col("gw") >= from_gw)
                .then(pl.lit(early))
                .otherwise(pl.col("available_at"))
                .alias("available_at")
            )
        ),
    )


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(columns) if columns else frame


def test_availability_overlay_forks_load() -> None:
    """Board forks flip one minutes flag each; production stays gated."""
    specs = load_model_specs()
    parent = specs["dc_layered_v3"]
    noavail = specs["dc_layered_noavail_v1"]
    injuries = specs["dc_layered_injuries_v1"]
    parent_gate = parent.components["minutes"].params.get("use_availability_gate", True)
    assert parent_gate is True
    assert noavail.components["minutes"].name == "empirical_minutes_v2_noavail"
    assert noavail.components["minutes"].params["use_availability_gate"] is False
    assert parent.components["minutes"].params.get("use_injury_overlay", False) is False
    assert injuries.components["minutes"].name == "empirical_minutes_v2_injuries"
    assert injuries.components["minutes"].params["use_injury_overlay"] is True
    assert noavail.kind == parent.kind == injuries.kind


def test_predict_runs_with_competition_fixtures(tmp_path: Path) -> None:
    """Cup/Europe kickoffs in competition_fixtures must not break fit/predict."""
    store = _store(
        tmp_path,
        stats=[(1, "MID", 1, gw, 90, 1) for gw in range(1, 8)],
        status={1: ("a", 100.0)},
    )
    # Midweek Europe three days before GW8 PL kickoff for team 1.
    store.write(
        "competition_fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON],
                    "api_fixture_id": [9001],
                    "competition": ["ucl"],
                    "league_id": [2],
                    "team_h_key": [1],
                    "team_a_key": [None],
                    "api_team_h_id": [40],
                    "api_team_a_id": [541],
                    "kickoff_time": [_kick(8) - timedelta(days=3)],
                    "available_at": [_START],
                    "source": ["api_football"],
                }
            )
        ),
    )
    _expose_future_fixtures(store, from_gw=8)
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    assert dist.pmf.shape[0] == 1
    assert dist.pmf.sum() == pytest.approx(1.0)
