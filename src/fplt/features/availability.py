"""First-party FPL availability features from staged ``player_status``."""

from __future__ import annotations

import re
from datetime import datetime

import polars as pl

from fplt.data.pit import PITView
from fplt.features.registry import feature
from fplt.models.components.minutes.common import (
    PlayerAvailability,
    gate_for,
)
from fplt.models.universe import player_fixtures
from fplt.schemas.features import AvailabilityFeatureSchema
from fplt.types import GWKey

_STATUS_ORD = {"a": 1.0, "d": 0.5, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}
_PCT = re.compile(r"(\d{1,3})\s*%\s*chance", re.IGNORECASE)


@feature(
    name="availability",
    version=1,
    grain="player_fixture",
    inputs=["player_status"],
    schema=AvailabilityFeatureSchema,
)
def availability_v1(view: PITView, horizon: list[GWKey]) -> pl.DataFrame:
    """Ordinal status, this/next chance, news flags, and a parsed news %."""
    keys = player_fixtures(view, horizon).select("player_key", "fixture_id")
    if keys.is_empty():
        return pl.DataFrame(
            schema={
                "player_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "status_ord": pl.Float64(),
                "chance_next": pl.Float64(),
                "chance_this": pl.Float64(),
                "chance_gate": pl.Float64(),
                "has_news": pl.Float64(),
                "has_news_added": pl.Float64(),
                "news_pct": pl.Float64(),
            }
        )
    status = (
        view.latest("player_status", by=["season", "player_key"])
        .collect()
        .sort("available_at")
        .unique(subset=["player_key"], keep="last")
    )
    by_player: dict[int, PlayerAvailability] = {}
    for row in status.iter_rows(named=True):
        chance_next = row.get("chance_of_playing")
        chance_this = row.get("chance_of_playing_this_round")
        news_added = row.get("news_added")
        by_player[int(row["player_key"])] = PlayerAvailability(
            status=None if row.get("status") is None else str(row["status"]),
            chance_next=None if chance_next is None else float(chance_next),
            chance_this=None if chance_this is None else float(chance_this),
            news=str(row.get("news") or ""),
            news_added=news_added if isinstance(news_added, datetime) else None,
        )
    rows: list[dict[str, float | int | None]] = []
    for player, fixture in zip(
        keys["player_key"].to_list(), keys["fixture_id"].to_list(), strict=True
    ):
        tags = by_player.get(int(player))
        if tags is None:
            rows.append(
                {
                    "player_key": int(player),
                    "fixture_id": int(fixture),
                    "status_ord": None,
                    "chance_next": None,
                    "chance_this": None,
                    "chance_gate": 1.0,
                    "has_news": 0.0,
                    "has_news_added": 0.0,
                    "news_pct": None,
                }
            )
            continue
        match = _PCT.search(tags.news)
        news_pct = float(match.group(1)) if match else None
        if news_pct is not None and not 0.0 <= news_pct <= 100.0:
            news_pct = None
        rows.append(
            {
                "player_key": int(player),
                "fixture_id": int(fixture),
                "status_ord": _STATUS_ORD.get(tags.status or "", None),
                "chance_next": tags.chance_next,
                "chance_this": tags.chance_this,
                "chance_gate": gate_for(tags, doubtful=0.5),
                "has_news": 1.0 if tags.news.strip() else 0.0,
                "has_news_added": 1.0 if tags.news_added is not None else 0.0,
                "news_pct": news_pct,
            }
        )
    return pl.DataFrame(rows)
