"""Build OpenFPL-compatible sample feature rows from a PIT view."""

from __future__ import annotations

from typing import Any

import polars as pl

from fplt.data.pit import PITView
from fplt.models.external.openfpl_columns import OPENFPL_FEATURE_NAMES

_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 38)

_PLAYER_FPL: tuple[tuple[str, str], ...] = (
    ("player fpl points", "total_points"),
    ("player minutes played", "minutes"),
    ("player influence", "influence"),
    ("player creativity", "creativity"),
    ("player threat", "threat"),
    ("player goals scored", "goals_scored"),
    ("player penalties missed", "penalties_missed"),
    ("player assists", "assists"),
    ("player goals conceded", "goals_conceded"),
    ("player own goals", "own_goals"),
    ("player saves", "saves"),
    ("player penalties saved", "penalties_saved"),
    ("player yellow cards", "yellow_cards"),
    ("player red cards", "red_cards"),
    ("player bps", "bps"),
    ("player fpl bonus points", "bonus"),
)

_PLAYER_UST: tuple[tuple[str, str], ...] = (
    ("player shots", "shots"),
    ("player xg", "xg"),
    ("player xgchain", "xg_chain"),
    ("player xgbuildup", "xg_buildup"),
    ("player key passes", "key_passes"),
    ("player xa", "xa"),
)


def assert_openfpl_features(features: dict[str, float]) -> None:
    """Raise if ``features`` is not exactly the OpenFPL xscaler column set.

    Args:
        features: Mapping of feature name to value for one sample row.

    Raises:
        ValueError: Names or count do not match ``OPENFPL_FEATURE_NAMES``.
    """
    names = tuple(features)
    if names != OPENFPL_FEATURE_NAMES:
        missing = [name for name in OPENFPL_FEATURE_NAMES if name not in features]
        extra = [name for name in names if name not in OPENFPL_FEATURE_NAMES]
        raise ValueError(
            "OpenFPL feature matrix mismatch: "
            f"missing={missing[:5]!r} extra={extra[:5]!r} "
            f"(expected {len(OPENFPL_FEATURE_NAMES)} columns, got {len(names)})"
        )


def build_openfpl_features(
    view: PITView, request: pl.DataFrame
) -> list[dict[str, float]]:
    """Build one OpenFPL feature dict per ``request`` row.

    Rolling means use the last 1/3/5/10/38 visible appearances before the
    view cutoff. Team/opponent Deep and PPDA come from
    ``understat_team_matches`` when staged; otherwise they stay 0.0. Team
    xG/xGA prefer that table too, else summed player xG. Availability is
    ``chance_of_playing / 100`` (default 1.0 when unknown).

    Args:
        view: Point-in-time tables at one deadline.
        request: Player-fixture rows with at least ``season``, ``gw``,
            ``player_key``, ``fixture_id``.

    Returns:
        Feature maps in the same order as ``request`` rows. Each map's
        key order matches ``OPENFPL_FEATURE_NAMES``.
    """
    if request.is_empty():
        return []
    context = _request_context(view, request)
    player_fpl = _player_fpl_history(view)
    player_ust = _player_understat_history(view)
    team_fpl = _team_fpl_history(view)
    team_ust = _team_understat_history(view)
    ranks = _league_ranks_by_fixture(view)
    status = _availability(view)
    as_of_ranks = _current_league_ranks(view)

    out: list[dict[str, float]] = []
    for row in context.iter_rows(named=True):
        features: dict[str, float] = {name: 0.0 for name in OPENFPL_FEATURE_NAMES}
        player = int(row["player_key"])
        team = int(row["team_key"])
        opponent = int(row["opponent_key"])
        home = bool(row["home"])
        hist = player_fpl.filter(pl.col("player_key") == player).sort(
            "kickoff_time", descending=True
        )
        _fill_player_fpl(features, hist, home)
        ust = player_ust.filter(pl.col("player_key") == player).sort(
            "available_at", descending=True
        )
        _fill_means(features, ust, _PLAYER_UST)
        team_hist = team_fpl.filter(pl.col("team_key") == team).sort(
            "kickoff_time", descending=True
        )
        opp_hist = team_fpl.filter(pl.col("team_key") == opponent).sort(
            "kickoff_time", descending=True
        )
        _fill_team_fpl(features, team_hist, prefix="team")
        _fill_team_fpl(features, opp_hist, prefix="opponent")
        team_u = team_ust.filter(pl.col("team_key") == team).sort(
            "available_at", descending=True
        )
        opp_u = team_ust.filter(pl.col("team_key") == opponent).sort(
            "available_at", descending=True
        )
        _fill_team_understat(features, team_u, prefix="team")
        _fill_team_understat(features, opp_u, prefix="opponent")
        _fill_team_ranks(features, team_hist, ranks, prefix="team")
        features["status player availability"] = status.get(player, 1.0)
        features["status team league rank"] = float(as_of_ranks.get(team, 0))
        features["status opponent league rank"] = float(as_of_ranks.get(opponent, 0))
        ordered = {name: float(features[name]) for name in OPENFPL_FEATURE_NAMES}
        assert_openfpl_features(ordered)
        out.append(ordered)
    return out


def _request_context(view: PITView, request: pl.DataFrame) -> pl.DataFrame:
    players = view.latest("players", by=["season", "player_key"]).collect()
    fixtures = view.latest("fixtures", by=["season", "fixture_id"]).collect()
    joined = request.join(
        players.select("season", "player_key", "team_key", "position"),
        on=["season", "player_key"],
        how="left",
    ).join(
        fixtures.select(
            "season",
            "fixture_id",
            "team_h_key",
            "team_a_key",
            "kickoff_time",
        ),
        on=["season", "fixture_id"],
        how="left",
    )
    if joined["team_key"].null_count() or joined["team_h_key"].null_count():
        raise ValueError("OpenFPL sample request is missing player or fixture context")
    return joined.with_columns(
        (pl.col("team_key") == pl.col("team_h_key")).alias("home"),
        pl.when(pl.col("team_key") == pl.col("team_h_key"))
        .then(pl.col("team_a_key"))
        .otherwise(pl.col("team_h_key"))
        .alias("opponent_key"),
    )


def _player_fpl_history(view: PITView) -> pl.DataFrame:
    stats = view.table("player_fixture_stats").collect()
    fixtures = view.latest("fixtures", by=["season", "fixture_id"]).collect()
    if stats.is_empty() or fixtures.is_empty():
        return pl.DataFrame(
            schema={
                "player_key": pl.Int64(),
                "kickoff_time": pl.Datetime(time_zone="UTC"),
                "home": pl.Boolean(),
                **{col: pl.Float64() for _, col in _PLAYER_FPL},
            }
        )
    finished = fixtures.filter(
        pl.col("finished") & pl.col("team_h_score").is_not_null()
    )
    players = view.latest("players", by=["season", "player_key"]).collect()
    keyed = stats.join(
        finished.select(
            "season",
            "fixture_id",
            "kickoff_time",
            "team_h_key",
            "team_a_key",
        ),
        on=["season", "fixture_id"],
        how="inner",
    ).join(
        players.select("season", "player_key", "team_key"),
        on=["season", "player_key"],
        how="left",
    )
    return keyed.with_columns(
        (pl.col("team_key") == pl.col("team_h_key")).alias("home"),
        *[pl.col(col).cast(pl.Float64) for _, col in _PLAYER_FPL],
    )


def _player_understat_history(view: PITView) -> pl.DataFrame:
    frame = (
        view.table("understat_player_matches")
        .filter(pl.col("player_key").is_not_null())
        .collect()
    )
    if frame.is_empty():
        return pl.DataFrame(
            schema={
                "player_key": pl.Int64(),
                "available_at": pl.Datetime(time_zone="UTC"),
                **{col: pl.Float64() for _, col in _PLAYER_UST},
            }
        )
    return frame.with_columns(
        pl.col("player_key").cast(pl.Int64),
        *[pl.col(col).cast(pl.Float64) for _, col in _PLAYER_UST],
    )


def _team_fpl_history(view: PITView) -> pl.DataFrame:
    fixtures = view.latest("fixtures", by=["season", "fixture_id"]).collect()
    finished = fixtures.filter(
        pl.col("finished") & pl.col("team_h_score").is_not_null()
    )
    if finished.is_empty():
        return pl.DataFrame(
            schema={
                "team_key": pl.Int64(),
                "opponent_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "season": pl.Utf8(),
                "kickoff_time": pl.Datetime(time_zone="UTC"),
                "goals_scored": pl.Float64(),
                "goals_conceded": pl.Float64(),
            }
        )
    home = finished.select(
        pl.col("team_h_key").alias("team_key"),
        pl.col("team_a_key").alias("opponent_key"),
        "fixture_id",
        "season",
        "kickoff_time",
        pl.col("team_h_score").cast(pl.Float64).alias("goals_scored"),
        pl.col("team_a_score").cast(pl.Float64).alias("goals_conceded"),
    )
    away = finished.select(
        pl.col("team_a_key").alias("team_key"),
        pl.col("team_h_key").alias("opponent_key"),
        "fixture_id",
        "season",
        "kickoff_time",
        pl.col("team_a_score").cast(pl.Float64).alias("goals_scored"),
        pl.col("team_h_score").cast(pl.Float64).alias("goals_conceded"),
    )
    return pl.concat([home, away])


def _team_understat_history(view: PITView) -> pl.DataFrame:
    """Team xG / Deep / PPDA per Understat match visible at the cutoff."""
    empty = pl.DataFrame(
        schema={
            "team_key": pl.Int64(),
            "match_id": pl.Int64(),
            "available_at": pl.Datetime(time_zone="UTC"),
            "xg": pl.Float64(),
            "xga": pl.Float64(),
            "deep": pl.Float64(),
            "deep_allowed": pl.Float64(),
            "ppda_att": pl.Float64(),
            "ppda_def": pl.Float64(),
            "ppda_allowed_att": pl.Float64(),
            "ppda_allowed_def": pl.Float64(),
        }
    )
    staged = view.table("understat_team_matches").collect()
    if not staged.is_empty():
        return staged.select(
            "team_key",
            "match_id",
            "available_at",
            pl.col("xg").cast(pl.Float64),
            pl.col("xga").cast(pl.Float64),
            pl.col("deep").cast(pl.Float64),
            pl.col("deep_allowed").cast(pl.Float64),
            pl.col("ppda_att").cast(pl.Float64),
            pl.col("ppda_def").cast(pl.Float64),
            pl.col("ppda_allowed_att").cast(pl.Float64),
            pl.col("ppda_allowed_def").cast(pl.Float64),
        )
    players = view.table("understat_player_matches").collect()
    if players.is_empty():
        return empty
    team = players.group_by("team_key", "match_id", "available_at", "season").agg(
        pl.col("xg").sum().alias("xg")
    )
    # xGA = opposing side's xG in the same match
    sides = team.join(
        team.select(
            pl.col("match_id"),
            pl.col("team_key").alias("other_team"),
            pl.col("xg").alias("xga"),
        ),
        on="match_id",
        how="left",
    ).filter(pl.col("team_key") != pl.col("other_team"))
    return sides.select(
        "team_key", "match_id", "available_at", "xg", "xga"
    ).with_columns(
        pl.lit(0.0).alias("deep"),
        pl.lit(0.0).alias("deep_allowed"),
        pl.lit(0.0).alias("ppda_att"),
        pl.lit(0.0).alias("ppda_def"),
        pl.lit(0.0).alias("ppda_allowed_att"),
        pl.lit(0.0).alias("ppda_allowed_def"),
    )


def _league_ranks_by_fixture(view: PITView) -> dict[tuple[str, int, int], float]:
    """Map (season, fixture_id, team_key) -> league rank before that match."""
    fixtures = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .filter(pl.col("finished") & pl.col("team_h_score").is_not_null())
        .sort("kickoff_time")
        .collect()
    )
    ranks: dict[tuple[str, int, int], float] = {}
    tables: dict[str, dict[int, list[int]]] = {}
    for row in fixtures.iter_rows(named=True):
        season = str(row["season"])
        table = tables.setdefault(season, {})
        fixture_id = int(row["fixture_id"])
        home = int(row["team_h_key"])
        away = int(row["team_a_key"])
        ranks[(season, fixture_id, home)] = float(_rank_of(table, home))
        ranks[(season, fixture_id, away)] = float(_rank_of(table, away))
        _apply_result(
            table,
            home,
            away,
            int(row["team_h_score"]),
            int(row["team_a_score"]),
        )
    return ranks


def _current_league_ranks(view: PITView) -> dict[int, int]:
    fixtures = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .filter(
            (pl.col("season") == view.as_of.season)
            & pl.col("finished")
            & pl.col("team_h_score").is_not_null()
        )
        .collect()
    )
    table: dict[int, list[int]] = {}
    for row in fixtures.sort("kickoff_time").iter_rows(named=True):
        _apply_result(
            table,
            int(row["team_h_key"]),
            int(row["team_a_key"]),
            int(row["team_h_score"]),
            int(row["team_a_score"]),
        )
    return {team: _rank_of(table, team) for team in table}


def _apply_result(
    table: dict[int, list[int]], home: int, away: int, home_goals: int, away_goals: int
) -> None:
    for team in (home, away):
        table.setdefault(team, [0, 0, 0])  # points, gd, gf
    table[home][2] += home_goals
    table[away][2] += away_goals
    table[home][1] += home_goals - away_goals
    table[away][1] += away_goals - home_goals
    if home_goals > away_goals:
        table[home][0] += 3
    elif home_goals < away_goals:
        table[away][0] += 3
    else:
        table[home][0] += 1
        table[away][0] += 1


def _rank_of(table: dict[int, list[int]], team: int) -> int:
    if team not in table:
        return 0
    ordered = sorted(
        table.items(),
        key=lambda item: (item[1][0], item[1][1], item[1][2]),
        reverse=True,
    )
    for index, (other, _) in enumerate(ordered, start=1):
        if other == team:
            return index
    return 0


def _availability(view: PITView) -> dict[int, float]:
    status = view.latest("player_status", by=["season", "player_key"]).collect()
    out: dict[int, float] = {}
    for row in status.iter_rows(named=True):
        chance = row["chance_of_playing"]
        if chance is None:
            out[int(row["player_key"])] = 1.0 if row["status"] == "a" else 0.0
        else:
            out[int(row["player_key"])] = float(chance) / 100.0
    return out


def _fill_player_fpl(
    features: dict[str, float], hist: pl.DataFrame, home: bool
) -> None:
    _fill_means(features, hist, _PLAYER_FPL)
    if hist.is_empty():
        return
    relevant = hist.filter(pl.col("home") == home)
    for window in _HORIZONS:
        features[f"player relevant fpl points {window}"] = _mean(
            relevant["total_points"].to_list(), window
        )


def _fill_team_fpl(
    features: dict[str, float], hist: pl.DataFrame, *, prefix: str
) -> None:
    for window in _HORIZONS:
        features[f"{prefix} goals scored {window}"] = _mean(
            hist["goals_scored"].to_list(), window
        )
        features[f"{prefix} goals conceded {window}"] = _mean(
            hist["goals_conceded"].to_list(), window
        )


def _fill_team_ranks(
    features: dict[str, float],
    hist: pl.DataFrame,
    ranks: dict[tuple[str, int, int], float],
    *,
    prefix: str,
) -> None:
    if prefix != "team":
        return
    values: list[float] = []
    opp_values: list[float] = []
    for row in hist.iter_rows(named=True):
        season = str(row["season"])
        fixture_id = int(row["fixture_id"])
        values.append(ranks.get((season, fixture_id, int(row["team_key"])), 0.0))
        opp_values.append(
            ranks.get((season, fixture_id, int(row["opponent_key"])), 0.0)
        )
    for window in _HORIZONS:
        features[f"team league rank {window}"] = _mean(values, window)
        features[f"team opponent league rank {window}"] = _mean(opp_values, window)


def _fill_team_understat(
    features: dict[str, float], hist: pl.DataFrame, *, prefix: str
) -> None:
    for window in _HORIZONS:
        features[f"{prefix} xg {window}"] = _mean(hist["xg"].to_list(), window)
        features[f"{prefix} xga {window}"] = _mean(hist["xga"].to_list(), window)
        features[f"{prefix} deep {window}"] = _mean(hist["deep"].to_list(), window)
        features[f"{prefix} deep allowed {window}"] = _mean(
            hist["deep_allowed"].to_list(), window
        )
        features[f"{prefix} ppda att {window}"] = _mean(
            hist["ppda_att"].to_list(), window
        )
        features[f"{prefix} ppda def {window}"] = _mean(
            hist["ppda_def"].to_list(), window
        )
        features[f"{prefix} ppda allowed att {window}"] = _mean(
            hist["ppda_allowed_att"].to_list(), window
        )
        features[f"{prefix} ppda allowed def {window}"] = _mean(
            hist["ppda_allowed_def"].to_list(), window
        )


def _fill_means(
    features: dict[str, float],
    hist: pl.DataFrame,
    mapping: tuple[tuple[str, str], ...],
) -> None:
    for label, column in mapping:
        values = hist[column].to_list() if column in hist.columns else []
        for window in _HORIZONS:
            features[f"{label} {window}"] = _mean(values, window)


def _mean(values: list[Any], window: int) -> float:
    if not values:
        return 0.0
    chosen = [float(value) for value in values[:window]]
    return sum(chosen) / len(chosen)
