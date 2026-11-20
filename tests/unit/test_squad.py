"""The single-gameweek squad meets the FPL constraints."""

import polars as pl
import pytest

from fplt.optimise.squad import aggregate_gameweek, select_squad

_POSITIONS = ["GK", "DEF", "MID", "FWD"]


def test_a_double_gameweek_sums_both_fixtures() -> None:
    predictions = pl.DataFrame(
        {
            "player_key": [1, 1, 2],
            "target_season": ["2024-25", "2024-25", "2024-25"],
            "target_gw": [1, 1, 2],
            "xpts": [3.0, 4.0, 9.0],
        }
    )
    points = aggregate_gameweek(predictions, "2024-25", 1)
    assert points.to_dicts() == [{"player_key": 1, "xpts": 7.0}]
    assert aggregate_gameweek(predictions, "2024-25", 3).is_empty()


def test_the_selected_squad_satisfies_the_rules() -> None:
    result = select_squad(_pool())
    squad = result.players
    assert squad.height == 15
    counts = dict(zip(squad["position"].to_list(), [0, 0, 0, 0], strict=False))
    for position in squad["position"].to_list():
        counts[position] = counts.get(position, 0) + 1
    assert counts == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    xi = squad.filter(pl.col("in_xi"))
    assert xi.height == 11
    assert set(xi["player_key"]) <= set(squad["player_key"])
    xi_counts = {position: 0 for position in _POSITIONS}
    for position in xi["position"].to_list():
        xi_counts[position] += 1
    assert xi_counts["GK"] == 1
    assert 3 <= xi_counts["DEF"] <= 5
    assert 2 <= xi_counts["MID"] <= 5
    assert 1 <= xi_counts["FWD"] <= 3
    assert squad.group_by("team_key").len()["len"].max() <= 3
    assert int(squad["price"].sum()) <= 1000
    assert int(squad["captain"].sum()) == 1
    assert int(squad["vice"].sum()) == 1
    captain = squad.filter(pl.col("captain"))
    vice = squad.filter(pl.col("vice"))
    assert bool(captain["in_xi"][0])
    assert bool(vice["in_xi"][0])
    assert captain["player_key"][0] != vice["player_key"][0]
    expected = float(xi["xpts"].sum()) + float(captain["xpts"][0])
    assert result.total_xpts == pytest.approx(expected)


def test_an_impossible_pool_is_rejected() -> None:
    tiny = _pool().head(10)
    with pytest.raises(ValueError, match="no squad"):
        select_squad(tiny)


def test_bad_player_frames_are_rejected() -> None:
    pool = _pool()
    with pytest.raises(ValueError, match="price"):
        select_squad(pool.drop("price"))
    repeated = pl.concat([pool.head(1), pool.head(1)])
    with pytest.raises(ValueError, match="unique"):
        select_squad(repeated)
    with pytest.raises(ValueError, match="xpts"):
        aggregate_gameweek(pl.DataFrame({"player_key": [1]}), "2024-25", 1)


def _pool() -> pl.DataFrame:
    """Thirty players across six clubs, so three per club can fill a squad."""
    rows: list[dict[str, object]] = []
    player = 1
    for club in range(1, 7):
        for position in ("GK", "DEF", "DEF", "MID", "FWD"):
            rows.append(
                {
                    "player_key": player,
                    "position": position,
                    "team_key": club,
                    "price": 40 + club,
                    "xpts": float(club + (1 if position == "DEF" else 0)),
                }
            )
            player += 1
    return pl.DataFrame(rows)
