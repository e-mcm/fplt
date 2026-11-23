"""Realised decision scoring: autosub, vice-on-blank, hits once."""

import polars as pl
import pytest

from fplt.optimise.realised import (
    assert_sane_mean_ppg,
    explain_gameweek,
    format_decision_scoreboard,
    score_gameweek,
    summarise_decisions,
)


def test_captain_doubles_and_hit_is_applied_once() -> None:
    picks = _squad(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 6],
        xi_mins=[90] * 11,
        captain_index=10,
        hit=4,
    )
    # XI sum 26 + captain extra 6 - hit 4 = 28
    assert score_gameweek(picks, hit=4) == 28
    dump = explain_gameweek(picks, hit=4, names={11: "Haul"})
    assert "XI sum (after autosub, mins>0): 26" in dump
    assert "captain extra (+6): Haul ×2 armband" in dump
    assert "hit: -4" in dump
    assert "net: 28" in dump


def test_assert_sane_mean_ppg_rejects_inflated_totals() -> None:
    summary = pl.DataFrame(
        {
            "model_id": ["hot", "ok"],
            "season": ["2024-25", "2024-25"],
            "total_points": [4062, 2000],
            "gameweeks": [38, 38],
            "per_gw": [[100] * 38, [50] * 38],
        }
    )
    with pytest.raises(ValueError, match="mean pts/GW"):
        assert_sane_mean_ppg(summary, max_mean=90.0)
    assert_sane_mean_ppg(summary.filter(pl.col("model_id") == "ok"), max_mean=90.0)


def test_vice_receives_armband_when_captain_blanks() -> None:
    picks = _squad(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 8],
        xi_mins=[90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 0],
        captain_index=10,
        vice_index=3,
        hit=0,
    )
    # captain blanked (no eligible autosub); vice (5 pts) already in XI → +5 for ×2
    assert score_gameweek(picks, hit=0) == 25


def test_bench_vice_scores_double_when_captain_blanks() -> None:
    picks = _squad(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 0],
        xi_mins=[90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 0],
        captain_index=10,
        vice_on_bench_pts=7,
        vice_on_bench_mins=90,
        hit=0,
    )
    # XI after autosub may replace captain; vice still gets 2× from bench
    net = score_gameweek(picks, hit=0)
    assert net >= 14  # at least 2× vice


def test_autosub_replaces_blank_outfielder() -> None:
    picks = _squad(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 0],
        xi_mins=[90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 0],
        captain_index=0,
        bench_pts=[9, 0, 0],
        bench_mins=[90, 0, 0],
        hit=0,
    )
    # blank FWD (0) replaced by bench MID (9); captain doubles first XI player (2)
    assert score_gameweek(picks, hit=0) == 31


def test_gk_blank_only_takes_gk_sub() -> None:
    picks = _squad(
        xi_pts=[0, 3, 4, 5, 1, 1, 1, 1, 1, 1, 1],
        xi_mins=[0, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90],
        captain_index=1,
        positions_xi=[
            "GK",
            "DEF",
            "DEF",
            "DEF",
            "MID",
            "MID",
            "MID",
            "MID",
            "FWD",
            "FWD",
            "FWD",
        ],
        bench_pts=[8, 0, 0],
        bench_mins=[90, 0, 0],
        bench_positions=["MID", "DEF", "FWD"],
        reserve_gk_pts=4,
        reserve_gk_mins=90,
        hit=0,
    )
    # GK blank → reserve GK (4) comes on; outfield bench ignored for GK slot
    # XI: 4+3+4+5+1+1+1+1+1+1+1 = 23; captain DEF doubles +3 → 26
    assert score_gameweek(picks, hit=0) == 26


def test_summarise_ranks_models_and_formats_table() -> None:
    frame = pl.concat(
        [
            _decision_rows("a", gw=1, net_xi=10, hit=0),
            _decision_rows("a", gw=2, net_xi=12, hit=4),
            _decision_rows("b", gw=1, net_xi=20, hit=0),
            _decision_rows("b", gw=2, net_xi=20, hit=0),
        ]
    )
    summary = summarise_decisions(frame)
    assert summary["model_id"].to_list() == ["b", "a"]
    assert summary["total_points"].to_list() == [40, 18]
    table = format_decision_scoreboard(summary)
    assert "| 1 | b |" in table
    assert "| 2 | a |" in table
    assert "Mean/GW" in table
    assert format_decision_scoreboard(summary.clear()) == "_No decision rows to score._"
    named = summary.with_columns(
        pl.when(pl.col("model_id") == "b")
        .then(pl.lit("dc_layered_v4"))
        .otherwise(pl.lit("dc_layered_v2"))
        .alias("model_name")
    )
    named_table = format_decision_scoreboard(named)
    assert "| 1 | dc_layered_v4 (b) |" in named_table
    assert "| 2 | dc_layered_v2 (a) |" in named_table


def test_triple_captain_adds_two_extras() -> None:
    picks = _squad(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 6],
        xi_mins=[90] * 11,
        captain_index=10,
        hit=0,
    )
    # XI 26 + TC extra 12 = 38
    assert score_gameweek(picks, hit=0, chip="3xc") == 38
    dump = explain_gameweek(picks, hit=0, chip="3xc", names={11: "Haul"})
    assert "triple captain extra (+12)" in dump
    assert "chip: 3xc" in dump


def test_bench_boost_adds_bench_points() -> None:
    picks = _squad(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 6],
        xi_mins=[90] * 11,
        captain_index=10,
        bench_pts=[4, 3, 2],
        bench_mins=[90, 90, 90],
        reserve_gk_pts=1,
        reserve_gk_mins=90,
        hit=0,
    )
    # base 26 + captain 6 = 32; bench 4+3+2+1 = 10 → 42
    assert score_gameweek(picks, hit=0, chip="bboost") == 42


def test_score_gameweek_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        score_gameweek(pl.DataFrame({"player_key": [1]}))
    with pytest.raises(ValueError, match="in_xi"):
        score_gameweek(
            pl.DataFrame(
                {
                    "player_key": [1],
                    "position": ["MID"],
                    "in_xi": [False],
                    "captain": [False],
                    "vice": [False],
                    "actual_pts": [1],
                    "actual_minutes": [90],
                }
            )
        )


def _squad(
    *,
    xi_pts: list[int],
    xi_mins: list[int],
    captain_index: int,
    vice_index: int | None = None,
    hit: float = 0.0,
    positions_xi: list[str] | None = None,
    bench_pts: list[int] | None = None,
    bench_mins: list[int] | None = None,
    bench_positions: list[str] | None = None,
    reserve_gk_pts: int = 0,
    reserve_gk_mins: int = 0,
    vice_on_bench_pts: int | None = None,
    vice_on_bench_mins: int = 0,
) -> pl.DataFrame:
    del hit  # applied by score_gameweek caller
    positions = positions_xi or [
        "GK",
        "DEF",
        "DEF",
        "DEF",
        "MID",
        "MID",
        "MID",
        "MID",
        "FWD",
        "FWD",
        "FWD",
    ]
    rows: list[dict[str, object]] = []
    for index, (pts, mins, position) in enumerate(
        zip(xi_pts, xi_mins, positions, strict=True)
    ):
        rows.append(
            {
                "player_key": index + 1,
                "position": position,
                "in_xi": True,
                "captain": index == captain_index,
                "vice": index == vice_index,
                "sold": False,
                "actual_pts": pts,
                "actual_minutes": mins,
            }
        )
    bench_pts = bench_pts or [0, 0, 0]
    bench_mins = bench_mins or [0, 0, 0]
    bench_positions = bench_positions or ["MID", "DEF", "FWD"]
    next_key = 12
    rows.append(
        {
            "player_key": next_key,
            "position": "GK",
            "in_xi": False,
            "captain": False,
            "vice": False,
            "sold": False,
            "actual_pts": reserve_gk_pts,
            "actual_minutes": reserve_gk_mins,
        }
    )
    next_key += 1
    for pts, mins, position in zip(bench_pts, bench_mins, bench_positions, strict=True):
        rows.append(
            {
                "player_key": next_key,
                "position": position,
                "in_xi": False,
                "captain": False,
                "vice": False,
                "sold": False,
                "actual_pts": pts,
                "actual_minutes": mins,
            }
        )
        next_key += 1
    if vice_on_bench_pts is not None:
        # overwrite last bench row as vice
        rows[-1]["vice"] = True
        rows[-1]["actual_pts"] = vice_on_bench_pts
        rows[-1]["actual_minutes"] = vice_on_bench_mins
        for row in rows:
            if row["in_xi"]:
                row["vice"] = False
    return pl.DataFrame(rows)


def _decision_rows(model_id: str, *, gw: int, net_xi: int, hit: float) -> pl.DataFrame:
    """Build a valid 15 with one scoring captain so net matches net_xi - hit."""
    # Put all points on captain in XI; others zero minutes/points.
    pts = [0] * 11
    mins = [0] * 11
    pts[0] = net_xi // 2
    mins[0] = 90
    # captain doubles → 2*(net_xi//2); pad remainder on another starter
    remainder = net_xi - 2 * (net_xi // 2)
    if remainder:
        pts[1] = remainder
        mins[1] = 90
    base = _squad(xi_pts=pts, xi_mins=mins, captain_index=0)
    return base.with_columns(
        pl.lit(model_id).alias("model_id"),
        pl.lit("2024-25").alias("season"),
        pl.lit(gw).alias("gw"),
        pl.lit(0.0).alias("xpts"),
        pl.lit(False).alias("bought"),
        pl.lit(hit).alias("hit"),
        pl.lit(1).alias("free_transfers"),
    )
