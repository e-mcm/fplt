"""Near-deadline predicted-XI override on empirical minutes v2."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fplt.models.components.minutes  # noqa: F401
from fplt.data.pit import PITData
from fplt.data.predicted_lineups import ROLE_BENCH, ROLE_START
from fplt.data.store import DataStore
from fplt.models.components.minutes.empirical_v2 import EmpiricalMinutesV2
from fplt.models.distributions import MINUTE_BINS
from fplt.models.spec import ModelSpec, load_model_specs
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


def test_dc_layered_predicted_xi_fork_loads() -> None:
    specs = load_model_specs()
    fork = specs["dc_layered_predicted_xi_v1"]
    parent = specs["dc_layered_v3"]
    assert fork.components["minutes"].params["predicted_xi"] is True
    assert parent.components["minutes"].params.get("predicted_xi", False) is False
    assert fork.kind == parent.kind


def test_predicted_xi_raises_p_start_for_named_starter(tmp_path: Path) -> None:
    # Player rarely starts empirically; predicted XI marks them as start.
    rows = [(1, "MID", 1, gw, 0, 0) for gw in range(1, 6)]
    store = _store(tmp_path, stats=rows, status={1: ("a", 100.0)})
    _write_predicted(
        store,
        [
            {
                "player_key": 1,
                "fixture_id": 8,
                "team_key": 1,
                "role": ROLE_START,
                "confidence": 0.95,
            }
        ],
        available_at=_kick(8) - timedelta(hours=24),
    )
    ask = pl.DataFrame({"player_key": [1], "fixture_id": [8]})
    baseline = _predict(store, ask, predicted_xi=False)
    overridden = _predict(
        store,
        ask,
        predicted_xi=True,
        predicted_xi_strength=1.0,
        predicted_xi_window_hours=72.0,
    )
    base_start = float(baseline.pmf[0, MINUTE_BINS >= 60].sum())
    over_start = float(overridden.pmf[0, MINUTE_BINS >= 60].sum())
    assert over_start > base_start
    assert overridden.pmf[0].sum() == pytest.approx(1.0)


def test_predicted_xi_skips_far_fixtures(tmp_path: Path) -> None:
    rows = [(1, "MID", 1, gw, 0, 0) for gw in range(1, 6)]
    store = _store(tmp_path, stats=rows, status={1: ("a", 100.0)}, max_gw=10)
    # Deadline for gw 8 is ~21h before kickoff of gw8; window 1h → out of window.
    _write_predicted(
        store,
        [
            {
                "player_key": 1,
                "fixture_id": 8,
                "team_key": 1,
                "role": ROLE_START,
                "confidence": 0.99,
            }
        ],
        available_at=_kick(8) - timedelta(days=2),
    )
    ask = pl.DataFrame({"player_key": [1], "fixture_id": [8]})
    baseline = _predict(store, ask, predicted_xi=False)
    skipped = _predict(
        store,
        ask,
        predicted_xi=True,
        predicted_xi_strength=1.0,
        predicted_xi_window_hours=1.0,
    )
    np.testing.assert_allclose(baseline.pmf, skipped.pmf)


def test_predicted_xi_bench_lowers_start_mass(tmp_path: Path) -> None:
    rows = [(1, "MID", 1, gw, 90, 1) for gw in range(1, 6)]
    store = _store(tmp_path, stats=rows, status={1: ("a", 100.0)})
    _write_predicted(
        store,
        [
            {
                "player_key": 1,
                "fixture_id": 8,
                "team_key": 1,
                "role": ROLE_BENCH,
                "confidence": 0.9,
            }
        ],
        # Must be visible at the GW8 deadline (kickoff − 21h).
        available_at=_kick(8) - timedelta(hours=24),
    )
    ask = pl.DataFrame({"player_key": [1], "fixture_id": [8]})
    baseline = _predict(store, ask, predicted_xi=False)
    benched = _predict(
        store,
        ask,
        predicted_xi=True,
        predicted_xi_strength=1.0,
        predicted_xi_window_hours=72.0,
    )
    bench_long = float(benched.pmf[0, MINUTE_BINS >= 60].sum())
    base_long = float(baseline.pmf[0, MINUTE_BINS >= 60].sum())
    assert bench_long < base_long


def test_bad_predicted_xi_params_refused() -> None:
    with pytest.raises(ValueError, match="predicted_xi"):
        EmpiricalMinutesV2(_spec(predicted_xi="yes"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="predicted_xi_window"):
        EmpiricalMinutesV2(_spec(predicted_xi_window_hours=-1))
    with pytest.raises(ValueError, match="predicted_xi_strength"):
        EmpiricalMinutesV2(_spec(predicted_xi_strength=2.0))


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
            "predicted_xi": False,
            "predicted_xi_window_hours": 72.0,
            "predicted_xi_strength": 0.75,
            **params,
        },
    )


def _predict(
    store: DataStore,
    player_fixtures: pl.DataFrame,
    **params: object,
):
    model = EmpiricalMinutesV2(_spec(**params))
    view = PITData(store).view(GWKey(_SEASON, 8))
    model.fit(view)
    return model.predict(view, player_fixtures)


def _write_predicted(
    store: DataStore,
    rows: list[dict[str, object]],
    *,
    available_at: datetime,
) -> None:
    frame = pl.DataFrame(
        {
            "season": [_SEASON] * len(rows),
            "team_key": [int(row["team_key"]) for row in rows],
            "fixture_id": [int(row["fixture_id"]) for row in rows],
            "player_key": [int(row["player_key"]) for row in rows],
            "element_id": [int(row["player_key"]) for row in rows],
            "role": [str(row["role"]) for row in rows],
            "confidence": [float(row["confidence"]) for row in rows],
            "available_at": [available_at] * len(rows),
            "source": ["touchline_wire"] * len(rows),
        }
    )
    store.write("predicted_lineups", _utc(frame))


def _store(
    root: Path,
    *,
    stats: list[tuple[int, str, int, int, int, int]],
    status: dict[int, tuple[str, float | None]],
    max_gw: int = 8,
) -> DataStore:
    store = DataStore(root)
    players = [(player, position, team) for player, position, team, _, _, _ in stats]
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
                    "price": [70] * len(listed),
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


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(columns) if columns else frame
