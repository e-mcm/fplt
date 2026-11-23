"""Empirical minutes use only appearances visible at the deadline."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fplt.models.components.minutes  # noqa: F401 — registers the component
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.minutes.empirical import EmpiricalMinutes
from fplt.models.distributions import MINUTE_BINS
from fplt.models.spec import (
    ModelSpec,
    get_component,
    load_model_specs,
    registered_kinds,
)
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
        name="empirical_minutes",
        kind="empirical_minutes",
        params={"window": 5, "shrinkage": 5, "doubtful_chance": 0.5, **params},
    )


def test_spec_registers_as_a_component() -> None:
    specs = load_model_specs()
    assert get_component("empirical_minutes") is EmpiricalMinutes
    assert specs["empirical_minutes_v1"].params["window"] == 5
    assert "empirical_minutes" not in registered_kinds()


def test_a_regular_starter_is_near_ninety_minutes(tmp_path: Path) -> None:
    store = _store(tmp_path, [(1, "MID", 1, 90, 1)], {1: ("a", 100.0)})
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    ninety = int(np.flatnonzero(MINUTE_BINS == 90)[0])
    assert dist.pmf[0, ninety] == pytest.approx(1.0)


def test_chance_of_zero_puts_every_starter_on_the_bench(tmp_path: Path) -> None:
    store = _store(tmp_path, [(1, "MID", 1, 90, 1)], {1: ("a", 0.0)})
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    assert dist.pmf[0, 0] == pytest.approx(1.0)


def test_an_omitted_chance_for_an_injury_is_zero(tmp_path: Path) -> None:
    store = _store(tmp_path, [(1, "MID", 1, 90, 1)], {1: ("i", None)})
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    assert dist.pmf[0, 0] == pytest.approx(1.0)


def test_doubtful_without_a_chance_uses_the_spec(tmp_path: Path) -> None:
    store = _store(tmp_path, [(1, "MID", 1, 90, 1)], {1: ("d", None)})
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    ninety = int(np.flatnonzero(MINUTE_BINS == 90)[0])
    assert dist.pmf[0, 0] == pytest.approx(0.5)
    assert dist.pmf[0, ninety] == pytest.approx(0.5)


def test_null_chance_with_news_is_not_treated_as_available(tmp_path: Path) -> None:
    store = _store(tmp_path, [(1, "MID", 1, 90, 1)], {1: ("a", None)})
    status = store.read("player_status").with_columns(
        pl.lit("Knock - 50% chance of playing").alias("news"),
        pl.lit(datetime(2024, 8, 10, tzinfo=UTC)).alias("news_added"),
    )
    store.path("player_status").unlink()
    store.write("player_status", status)
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    ninety = int(np.flatnonzero(MINUTE_BINS == 90)[0])
    assert dist.pmf[0, 0] == pytest.approx(0.5)
    assert dist.pmf[0, ninety] == pytest.approx(0.5)


def test_injury_overlay_soft_downgrades_when_fpl_still_available(
    tmp_path: Path,
) -> None:
    """API injury + FPL status=a/null chance → doubtful prior when overlay on."""
    store = _store(tmp_path, [(1, "MID", 1, 90, 1)], {1: ("a", None)})
    _write_injury(store, player_key=1, category="doubtful")
    off = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    ninety = int(np.flatnonzero(MINUTE_BINS == 90)[0])
    assert off.pmf[0, ninety] == pytest.approx(1.0)
    on = EmpiricalMinutes(_spec(use_injury_overlay=True))
    view = PITData(store).view(GWKey(_SEASON, 8))
    on.fit(view)
    dist = on.predict(view, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    assert dist.pmf[0, 0] == pytest.approx(0.5)
    assert dist.pmf[0, ninety] == pytest.approx(0.5)


def test_injury_overlay_zeros_suspension_without_raising_fpl_zero(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, [(1, "MID", 1, 90, 1)], {1: ("a", 100.0)})
    _write_injury(store, player_key=1, category="out")
    model = EmpiricalMinutes(_spec(use_injury_overlay=True))
    view = PITData(store).view(GWKey(_SEASON, 8))
    model.fit(view)
    dist = model.predict(view, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    assert dist.pmf[0, 0] == pytest.approx(1.0)
    # FPL already at 0% stays at 0 even if overlay somehow softer.
    store_zero = _store(tmp_path / "zero", [(1, "MID", 1, 90, 1)], {1: ("a", 0.0)})
    _write_injury(store_zero, player_key=1, category="doubtful")
    zero_model = EmpiricalMinutes(_spec(use_injury_overlay=True))
    zero_view = PITData(store_zero).view(GWKey(_SEASON, 8))
    zero_model.fit(zero_view)
    zero_dist = zero_model.predict(
        zero_view, pl.DataFrame({"player_key": [1], "fixture_id": [8]})
    )
    assert zero_dist.pmf[0, 0] == pytest.approx(1.0)


def test_fixture_scoped_suspension_overlay_only_hits_that_fixture(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, [(1, "MID", 1, 90, 1)], {1: ("a", 100.0)})
    _write_injury(store, player_key=1, category="out", fixture_id=8)
    model = EmpiricalMinutes(_spec(use_injury_overlay=True))
    view = PITData(store).view(GWKey(_SEASON, 8))
    model.fit(view)
    banned = model.predict(view, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    clear = model.predict(view, pl.DataFrame({"player_key": [1], "fixture_id": [9]}))
    ninety = int(np.flatnonzero(MINUTE_BINS == 90)[0])
    assert banned.pmf[0, 0] == pytest.approx(1.0)
    assert clear.pmf[0, ninety] == pytest.approx(1.0)


def test_a_substitute_snaps_positive_minutes_off_zero(tmp_path: Path) -> None:
    store = _store(tmp_path, [(1, "MID", 1, 1, 0)], {1: ("a", 100.0)})
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    fifteen = int(np.flatnonzero(MINUTE_BINS == 15)[0])
    assert dist.pmf[0, fifteen] == pytest.approx(1.0)


def test_only_the_latest_window_sets_the_start_rate(tmp_path: Path) -> None:
    rows = [(1, "MID", 1, 90, 1)]
    rows += [(1, "MID", gw, 0, 0) for gw in range(2, 7)]
    store = _store(tmp_path, rows, {1: ("a", 100.0)})
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    assert dist.pmf[0, 0] == pytest.approx(1.0)


def test_one_cameo_shrinks_toward_the_position_prior(tmp_path: Path) -> None:
    rows = [(1, "MID", 1, 15, 1)]
    rows += [(player, "MID", 1, 90, 1) for player in range(2, 7)]
    status = {player: ("a", 100.0) for player in range(1, 7)}
    store = _store(tmp_path, rows, status)
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    fifteen = int(np.flatnonzero(MINUTE_BINS == 15)[0])
    ninety = int(np.flatnonzero(MINUTE_BINS == 90)[0])
    assert dist.pmf[0, fifteen] == pytest.approx(11 / 36)
    assert dist.pmf[0, ninety] == pytest.approx(25 / 36)


def test_a_new_player_inherits_the_position(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [(1, "MID", 1, 90, 1)],
        {1: ("a", 100.0), 2: ("a", 100.0)},
        extra_players=[(2, "MID")],
    )
    dist = _predict(store, pl.DataFrame({"player_key": [2], "fixture_id": [8]}))
    ninety = int(np.flatnonzero(MINUTE_BINS == 90)[0])
    assert dist.pmf[0, ninety] == pytest.approx(1.0)


def test_a_missing_status_row_stays_available(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [(1, "MID", 1, 90, 1)],
        {2: ("i", None)},
        extra_players=[(2, "DEF")],
    )
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    ninety = int(np.flatnonzero(MINUTE_BINS == 90)[0])
    assert dist.pmf[0, ninety] == pytest.approx(1.0)


def test_no_history_uses_the_fallback_when_shrinkage_is_zero(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [(1, "MID", 1, 90, 1)],
        {1: ("a", 100.0), 2: ("a", 100.0)},
        extra_players=[(2, "MID")],
    )
    view = PITData(store).view(GWKey(_SEASON, 8))
    model = EmpiricalMinutes(_spec(shrinkage=0))
    model.fit(view)
    dist = model.predict(view, pl.DataFrame({"player_key": [2], "fixture_id": [8]}))
    ninety = int(np.flatnonzero(MINUTE_BINS == 90)[0])
    assert dist.pmf[0, ninety] == pytest.approx(1.0)


def test_a_full_appearance_window_stays_non_negative(tmp_path: Path) -> None:
    rows = [(1, "MID", gw, 90, 1) for gw in range(1, 5)]
    rows.append((1, "MID", 5, 15, 0))
    store = _store(tmp_path, rows, {1: ("a", 100.0)})
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    assert np.all(dist.pmf >= 0)
    assert dist.pmf.sum() == pytest.approx(1.0)


def test_a_squad_that_never_plays_stays_on_zero(tmp_path: Path) -> None:
    store = _store(tmp_path, [(1, "MID", 1, 0, 0)], {1: ("a", 100.0)})
    dist = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    assert dist.pmf[0, 0] == pytest.approx(1.0)


def test_a_result_after_the_deadline_is_invisible(tmp_path: Path) -> None:
    store = _store(tmp_path, [(1, "MID", 1, 90, 1)], {1: ("a", 100.0)})
    ask = pl.DataFrame({"player_key": [1], "fixture_id": [8]})
    first = _predict(store, ask)
    late = store.read("player_fixture_stats").filter(pl.col("fixture_id") == 1)
    cutoff = PITData(store).deadline(GWKey(_SEASON, 8))
    store.write(
        "player_fixture_stats",
        late.with_columns(
            pl.lit(0).alias("minutes"),
            pl.lit(0).alias("starts"),
            pl.lit(cutoff + timedelta(days=1)).alias("available_at"),
        ),
    )
    second = _predict(store, ask)
    np.testing.assert_allclose(first.pmf, second.pmf)


def test_predict_requires_a_fit_and_the_key_columns(tmp_path: Path) -> None:
    store = _store(tmp_path, [(1, "MID", 1, 90, 1)], {1: ("a", 100.0)})
    view = PITData(store).view(GWKey(_SEASON, 8))
    with pytest.raises(RuntimeError, match="fit"):
        EmpiricalMinutes(_spec()).predict(
            view, pl.DataFrame({"player_key": [1], "fixture_id": [1]})
        )
    fitted = EmpiricalMinutes(_spec())
    fitted.fit(view)
    with pytest.raises(ValueError, match="fixture_id"):
        fitted.predict(
            PITData(store).view(GWKey(_SEASON, 8)),
            pl.DataFrame({"player_key": [1]}),
        )
    empty = fitted.predict(
        PITData(store).view(GWKey(_SEASON, 8)),
        pl.DataFrame(
            {"player_key": [], "fixture_id": []},
            schema={"player_key": pl.Int64, "fixture_id": pl.Int64},
        ),
    )
    assert empty.pmf.shape == (0, len(MINUTE_BINS))


def test_bad_params_are_refused() -> None:
    with pytest.raises(ValueError, match="window"):
        EmpiricalMinutes(_spec(window=0))
    with pytest.raises(ValueError, match="shrinkage"):
        EmpiricalMinutes(_spec(shrinkage=-1))
    with pytest.raises(ValueError, match="doubtful"):
        EmpiricalMinutes(_spec(doubtful_chance=1.5))


def _predict(store: DataStore, player_fixtures: pl.DataFrame):
    model = EmpiricalMinutes(_spec())
    view = PITData(store).view(GWKey(_SEASON, 8))
    model.fit(view)
    return model.predict(view, player_fixtures)


def _store(
    root: Path,
    rows: list[tuple[int, str, int, int, int]],
    status: dict[int, tuple[str, float | None]],
    extra_players: list[tuple[int, str]] | None = None,
) -> DataStore:
    store = DataStore(root)
    players = [(player, position) for player, position, _, _, _ in rows]
    players += extra_players or []
    unique = list(dict.fromkeys(players))
    store.write(
        "players",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * len(unique),
                    "player_key": [player for player, _ in unique],
                    "element_id": [player for player, _ in unique],
                    "team_key": [1] * len(unique),
                    "position": [position for _, position in unique],
                    "web_name": [str(player) for player, _ in unique],
                    "available_at": [_START] * len(unique),
                    "source": ["fpl_api"] * len(unique),
                }
            )
        ),
    )
    store.write(
        "teams",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON],
                    "team_key": [1],
                    "team_id": [1],
                    "name": ["A"],
                    "short_name": ["A"],
                    "available_at": [_START],
                    "source": ["fpl_api"],
                }
            )
        ),
    )
    deadlines = [_kick(gw) - timedelta(hours=21) for gw in range(1, 9)]
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 8,
                    "gw": list(range(1, 9)),
                    "deadline_time": deadlines,
                    "finished": [gw < 8 for gw in range(1, 9)],
                    "data_checked": [gw < 8 for gw in range(1, 9)],
                    "available_at": [_START] * 8,
                    "source": ["fpl_api"] * 8,
                }
            )
        ),
    )
    fixtures = []
    stats = []
    for player, _, gw, minutes, starts in rows:
        fixtures.append(gw)
        stats.append((player, gw, minutes, starts, _kick(gw) + timedelta(hours=3)))
    fixture_ids = sorted(set(fixtures))
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
                    "team_h_score": [0] * len(fixture_ids),
                    "team_a_score": [0] * len(fixture_ids),
                    "finished": [True] * len(fixture_ids),
                    "available_at": [
                        _kick(gw) + timedelta(hours=3) for gw in fixture_ids
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
                    "price": [50] * len(listed),
                    "ep_next": pl.Series(
                        "ep_next", [None] * len(listed), dtype=pl.Float64
                    ),
                    "chance_of_playing": pl.Series(
                        "chance_of_playing",
                        [status[player][1] for player in listed],
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
                    "gw": [gw for _, gw, _, _, _ in stats],
                    "player_key": [player for player, _, _, _, _ in stats],
                    "fixture_id": [gw for _, gw, _, _, _ in stats],
                    "minutes": [minutes for _, _, minutes, _, _ in stats],
                    "starts": [starts for _, _, _, starts, _ in stats],
                    "available_at": [when for *_, when in stats],
                    "source": ["fpl_api"] * len(stats),
                    **{name: [0] * len(stats) for name in _ZERO},
                }
            )
        ),
    )
    return store


def _kick(gw: int) -> datetime:
    return datetime(2024, 8, 10, 15, tzinfo=UTC) + timedelta(days=7 * (gw - 1))


def _write_injury(
    store: DataStore,
    *,
    player_key: int,
    category: str,
    fixture_id: int | None = None,
) -> None:
    store.write(
        "player_injuries",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON],
                    "player_key": [player_key],
                    "api_player_id": [9000 + player_key],
                    "team_key": [1],
                    "api_team_id": [34],
                    "api_fixture_id": [1],
                    "fixture_id": pl.Series("fixture_id", [fixture_id], dtype=pl.Int64),
                    "injury_type": [
                        "Injury" if category == "doubtful" else "Suspension"
                    ],
                    "reason": ["Knock" if category == "doubtful" else "Suspended"],
                    "category": [category],
                    "available_at": [_START],
                    "source": ["api_football"],
                }
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
