"""All-competition rest and congestion features for minutes v2.

PL-only ``rest_days`` misses midweek cups and Europe. Merge PIT-visible
``competition_fixtures`` (API-Football FA Cup / EFL Cup / UCL / UEL / UECL /
Community Shield) with FPL league fixtures so rest and recent-match counts
reflect the full calendar. When cup data is absent the helpers fall back to
PL kickoffs only (same behaviour as pre-FDB-252).
"""

from __future__ import annotations

import polars as pl

from fplt.data.pit import PITView

_REST_CAP_WEEKS = 3.0
_CONGESTION_WINDOW_DAYS = 7.0
_GAMES_CAP = 3.0


def competition_team_kickoffs(view: PITView) -> pl.DataFrame:
    """Expand PIT ``competition_fixtures`` to ``(season, team_key, kickoff)``.

    Args:
        view: PIT view; missing / empty ``competition_fixtures`` yields an
            empty typed frame.

    Returns:
        Rows with ``season``, ``team_key``, ``kickoff_time`` for mapped EPL
        clubs only (home and away sides that have a ``team_key``).
    """
    empty = pl.DataFrame(
        schema={
            "season": pl.Utf8(),
            "team_key": pl.Int64(),
            "kickoff_time": pl.Datetime(time_zone="UTC"),
        }
    )
    frame = (
        view.table("competition_fixtures")
        .select(
            "season",
            "team_h_key",
            "team_a_key",
            "kickoff_time",
        )
        .collect()
    )
    if frame.is_empty():
        return empty
    home = frame.filter(pl.col("team_h_key").is_not_null()).select(
        "season",
        pl.col("team_h_key").alias("team_key"),
        "kickoff_time",
    )
    away = frame.filter(pl.col("team_a_key").is_not_null()).select(
        "season",
        pl.col("team_a_key").alias("team_key"),
        "kickoff_time",
    )
    return pl.concat([home, away])


def team_schedule_features(
    fixtures: pl.DataFrame,
    extra_kickoffs: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per FPL fixture-team rest weeks and games in the prior 7 days.

    Rest is calendar weeks since the previous kickoff across PL + extras,
    clipped to ``[0, 3]`` (same scale as the legacy PL-only helper). Congestion
    ``games_last_7d`` counts prior kickoffs strictly before this fixture
    within seven days, clipped to ``[0, 3]``.

    Args:
        fixtures: FPL fixtures with ``season``, ``fixture_id``, ``team_h_key``,
            ``team_a_key``, ``kickoff_time``.
        extra_kickoffs: Optional cup/Europe rows with ``season``, ``team_key``,
            ``kickoff_time``. ``None`` / empty keeps PL-only rest.

    Returns:
        ``season``, ``fixture_id``, ``team_key``, ``rest_days``,
        ``games_last_7d``.
    """
    empty = pl.DataFrame(
        schema={
            "season": pl.Utf8(),
            "fixture_id": pl.Int64(),
            "team_key": pl.Int64(),
            "rest_days": pl.Float64(),
            "games_last_7d": pl.Float64(),
        }
    )
    if fixtures.is_empty():
        return empty
    home = fixtures.select(
        "season",
        "fixture_id",
        pl.col("team_h_key").alias("team_key"),
        "kickoff_time",
    )
    away = fixtures.select(
        "season",
        "fixture_id",
        pl.col("team_a_key").alias("team_key"),
        "kickoff_time",
    )
    pl_sides = pl.concat([home, away])
    calendar = pl_sides.select("season", "team_key", "kickoff_time")
    if extra_kickoffs is not None and not extra_kickoffs.is_empty():
        calendar = pl.concat(
            [
                calendar,
                extra_kickoffs.select("season", "team_key", "kickoff_time"),
            ]
        )
    calendar = calendar.unique(subset=["season", "team_key", "kickoff_time"]).sort(
        ["season", "team_key", "kickoff_time"]
    )
    rows: list[dict[str, float | int | str]] = []
    for key, group in pl_sides.group_by(["season", "team_key"], maintain_order=True):
        season = str(key[0])
        team = int(key[1])
        team_cal = calendar.filter(
            (pl.col("season") == season) & (pl.col("team_key") == team)
        ).sort("kickoff_time")
        cal_times = team_cal["kickoff_time"].to_list()
        for fixture_id, kickoff in zip(
            group["fixture_id"].to_list(),
            group["kickoff_time"].to_list(),
            strict=True,
        ):
            prior = [stamp for stamp in cal_times if stamp < kickoff]
            if prior:
                delta_hours = (kickoff - prior[-1]).total_seconds() / 3600.0
                rest = min(max(delta_hours / 24.0 / 7.0, 0.0), _REST_CAP_WEEKS)
            else:
                rest = 1.0
            # Strictly inside the prior 7 days (exactly-7d weekly PL ≠ congestion).
            window_start = kickoff.timestamp() - _CONGESTION_WINDOW_DAYS * 86400.0
            games = sum(1 for stamp in prior if stamp.timestamp() > window_start)
            rows.append(
                {
                    "season": season,
                    "fixture_id": int(fixture_id),
                    "team_key": team,
                    "rest_days": float(rest),
                    "games_last_7d": float(min(games, _GAMES_CAP)),
                }
            )
    if not rows:
        return empty
    return pl.DataFrame(rows)
