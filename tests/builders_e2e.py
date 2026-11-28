"""Builders for the weekly ops pipeline e2e fixture world (#307).

Seeds a local R2-shaped bucket with three prior vaastav seasons plus FPL API
captures around one deadline, sized for the MILP (≥5 clubs, 15-man squad) and
the training-history guard (``min_prior_seasons=3``).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from fplt.automation.production import OfficialLine, ProductionOps
from fplt.models.pipeline_spec import PlannerPool, live_pipeline_defaults
from fplt.schemas.vaastav_source import VaastavSourceFile, VaastavSourceManifest

# ---------------------------------------------------------------------------
# Clocks (UTC). Tick ``now`` is T−90m before GW2 deadline.
# ---------------------------------------------------------------------------

GW1_DEADLINE = datetime(2024, 8, 16, 17, 0, tzinfo=UTC)
GW1_KICKOFF = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)
GW1_RESULT_CAPTURE = datetime(2024, 8, 17, 18, 0, tzinfo=UTC)
GW2_DEADLINE = datetime(2024, 8, 23, 17, 0, tzinfo=UTC)
GW2_KICKOFF = datetime(2024, 8, 24, 14, 0, tzinfo=UTC)
NOW_T90M = GW2_DEADLINE - timedelta(minutes=90)

CURRENT_SEASON = "2024-25"
PRIOR_SEASONS = ("2021-22", "2022-23", "2023-24")
E2E_MODEL = "dc_layered_v4_e2e"
E2E_LINE = "e2e_primary"

_VAASTAV_STAMP = "2026-08-01T00-00-00Z"
_CAPTURE_PRE = "2024-08-10T12-00-00Z"
_CAPTURE_GW1 = "2024-08-17T18-00-00Z"
_CAPTURE_PRE_GW2 = "2024-08-23T10-00-00Z"

# Six clubs → legal 15 with ≤3/club. Codes are stable FPL team codes.
_TEAMS: list[tuple[int, int, str, str]] = [
    (1, 3, "Arsenal", "ARS"),
    (2, 7, "Aston Villa", "AVL"),
    (3, 11, "Bournemouth", "BOU"),
    (4, 14, "Brighton", "BHA"),
    (5, 21, "Fulham", "FUL"),
    (6, 43, "Man City", "MCI"),
]

# Per club: GK, DEF, DEF, MID, MID, FWD → 36 players.
_POS_LAYOUT: list[tuple[int, str, int]] = [
    (1, "GK", 45),
    (2, "DEF", 45),
    (2, "DEF", 50),
    (3, "MID", 55),
    (3, "MID", 70),
    (4, "FWD", 75),
]


def e2e_production_ops(*, min_prior_seasons: int = 3) -> ProductionOps:
    """Return a single-line Python-MC ProductionOps for the e2e.

    Args:
        min_prior_seasons: History-guard floor (default matches production).

    Returns:
        ``ProductionOps`` pointing at ``dc_layered_v4_e2e`` with H=1.
    """
    pipeline = live_pipeline_defaults(model=E2E_MODEL, horizon=1, mc_backend="python")
    pipeline = pipeline.model_copy(
        update={
            "planner": pipeline.planner.model_copy(
                update={
                    "horizon": 1,
                    "time_limit_s": 30.0,
                    "pool": PlannerPool(target=36),
                }
            ),
            "policy": pipeline.policy.model_copy(
                update={
                    "chips": pipeline.policy.chips.model_copy(
                        update={"arm": "off", "force_play": False}
                    )
                }
            ),
        }
    )
    return ProductionOps(
        models=[E2E_MODEL],
        plan_horizon=1,
        pipeline=pipeline,
        lines=[
            OfficialLine(
                line=E2E_LINE,
                spec=E2E_MODEL,
                horizon=1,
                role="primary",
                squad_mode="real",
            )
        ],
        min_prior_seasons=min_prior_seasons,
    )


def seed_e2e_bucket(bucket: Path) -> Path:
    """Write vaastav priors + FPL API captures under ``bucket/raw/``.

    Args:
        bucket: Local directory standing in for the data bucket root.

    Returns:
        The ``bucket`` path.
    """
    raw = bucket / "raw"
    _seed_vaastav(raw)
    _seed_fpl_api(raw)
    return bucket


def _seed_vaastav(raw: Path) -> None:
    capture = raw / "vaastav" / f"captured_at={_VAASTAV_STAMP}"
    root = capture / "data"
    for index, season in enumerate(PRIOR_SEASONS):
        _write_vaastav_season(
            root / season,
            season=season,
            kickoff=datetime(2021 + index, 8, 14, 14, 0, tzinfo=UTC),
        )
    # Completeness marker so readers treat this capture as finished (#353).
    _write_vaastav_source(capture, seasons=list(PRIOR_SEASONS))


def _write_vaastav_source(capture_dir: Path, *, seasons: list[str]) -> None:
    """Write a verified ``_source.json`` for every file under ``data/``."""
    data = capture_dir / "data"
    entries: list[VaastavSourceFile] = []
    for path in sorted(data.rglob("*")):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        entries.append(
            VaastavSourceFile(
                path=path.relative_to(capture_dir).as_posix(),
                bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    captured_at = datetime.strptime(
        capture_dir.name.removeprefix("captured_at="), "%Y-%m-%dT%H-%M-%SZ"
    ).replace(tzinfo=UTC)
    manifest = VaastavSourceManifest(
        repo="https://example.test/vaastav.git",
        commit_sha="b" * 40,
        captured_at=captured_at,
        seasons=seasons,
        files=entries,
    )
    (capture_dir / "_source.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def _write_vaastav_season(season_dir: Path, *, season: str, kickoff: datetime) -> None:
    """Write a two-club finished-GW vaastav season (history-guard fodder)."""
    del season  # label is the directory name
    (season_dir / "gws").mkdir(parents=True)
    # Two clubs from the six — enough for a finished GW + data_checked.
    teams = _TEAMS[:2]
    players = _players_for_teams(teams, element_base=1)
    pl.DataFrame(
        {
            "id": [p["element_id"] for p in players],
            "code": [p["player_key"] for p in players],
            "element_type": [p["element_type"] for p in players],
            "web_name": [p["web_name"] for p in players],
            "team": [p["team_id"] for p in players],
        }
    ).write_csv(season_dir / "players_raw.csv")
    pl.DataFrame(
        {
            "id": [t[0] for t in teams],
            "code": [t[1] for t in teams],
            "name": [t[2] for t in teams],
            "short_name": [t[3] for t in teams],
        }
    ).write_csv(season_dir / "teams.csv")
    kick = kickoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    pl.DataFrame(
        {
            "id": [1],
            "event": [1],
            "team_h": [1],
            "team_a": [2],
            "kickoff_time": [kick],
            "finished": [True],
            "team_h_score": [2],
            "team_a_score": [0],
        }
    ).write_csv(season_dir / "fixtures.csv")
    rows: list[dict[str, Any]] = []
    for player in players:
        home = player["team_id"] == 1
        rows.append(
            {
                "name": player["web_name"],
                "element": player["element_id"],
                "fixture": 1,
                "GW": 1,
                "was_home": home,
                "team": player["team_id"],
                "opponent_team": 2 if home else 1,
                "minutes": 90,
                "total_points": 6 if home else 2,
                "goals_scored": 1 if home and player["element_type"] >= 3 else 0,
                "assists": 0,
                "clean_sheets": 1 if home else 0,
                "goals_conceded": 0 if home else 2,
                "saves": 3 if player["element_type"] == 1 else 0,
                "bonus": 0,
                "yellow_cards": 0,
                "red_cards": 0,
                "own_goals": 0,
                "penalties_saved": 0,
                "penalties_missed": 0,
                "starts": 1,
                "value": player["now_cost"],
                "kickoff_time": kick,
                "position": player["position"],
            }
        )
    pl.DataFrame(rows).write_csv(season_dir / "gws" / "merged_gw.csv")


def _seed_fpl_api(raw: Path) -> None:
    season_root = raw / "fpl_api" / f"season={CURRENT_SEASON}"
    players = _players_for_teams(_TEAMS, element_base=10)
    teams_payload = [
        {"id": tid, "code": code, "name": name, "short_name": short}
        for tid, code, name, short in _TEAMS
    ]
    elements = [
        {
            "id": p["element_id"],
            "code": p["player_key"],
            "team": p["team_id"],
            "element_type": p["element_type"],
            "web_name": p["web_name"],
            "status": "a",
            "news": "",
            "now_cost": p["now_cost"],
            "ep_next": "4.0",
            "chance_of_playing_next_round": 100,
        }
        for p in players
    ]
    gw1_fx, gw2_fx = _fixtures_payload()
    # Pre-season schedule (GW1 not yet finished).
    _write_capture(
        season_root / f"captured_at={_CAPTURE_PRE}",
        bootstrap=_bootstrap(
            teams_payload,
            elements,
            events=[
                _event(1, GW1_DEADLINE, finished=False, data_checked=False),
                _event(2, GW2_DEADLINE, finished=False, data_checked=False),
            ],
        ),
        fixtures=[
            {**fx, "finished": False, "team_h_score": None, "team_a_score": None}
            for fx in gw1_fx + gw2_fx
        ],
    )
    # Post-GW1 results + event_live.
    _write_capture(
        season_root / f"captured_at={_CAPTURE_GW1}",
        bootstrap=_bootstrap(
            teams_payload,
            elements,
            events=[
                _event(1, GW1_DEADLINE, finished=True, data_checked=True),
                _event(2, GW2_DEADLINE, finished=False, data_checked=False),
            ],
        ),
        fixtures=gw1_fx
        + [
            {**fx, "finished": False, "team_h_score": None, "team_a_score": None}
            for fx in gw2_fx
        ],
    )
    live_dir = season_root / "event_live" / "gw=1" / f"captured_at={_CAPTURE_GW1}"
    live_dir.mkdir(parents=True)
    _write_json_gz(live_dir / "event_live.json.gz", _event_live(players, gw1_fx))
    # Latest pre-GW2 capture (tick feeds from this).
    _write_capture(
        season_root / f"captured_at={_CAPTURE_PRE_GW2}",
        bootstrap=_bootstrap(
            teams_payload,
            elements,
            events=[
                _event(1, GW1_DEADLINE, finished=True, data_checked=True),
                _event(2, GW2_DEADLINE, finished=False, data_checked=False),
            ],
        ),
        fixtures=gw1_fx
        + [
            {**fx, "finished": False, "team_h_score": None, "team_a_score": None}
            for fx in gw2_fx
        ],
    )


def _players_for_teams(
    teams: list[tuple[int, int, str, str]], *, element_base: int
) -> list[dict[str, Any]]:
    players: list[dict[str, Any]] = []
    element_id = element_base
    for team_id, team_code, _name, short in teams:
        for slot, (element_type, position, cost) in enumerate(_POS_LAYOUT):
            player_key = team_code * 100 + slot + 1
            players.append(
                {
                    "element_id": element_id,
                    "player_key": player_key,
                    "team_id": team_id,
                    "team_code": team_code,
                    "element_type": element_type,
                    "position": position,
                    "web_name": f"{short}{position}{slot}",
                    "now_cost": cost,
                }
            )
            element_id += 1
    return players


def _fixtures_payload() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Three GW1 finished fixtures and three GW2 upcoming pairings."""
    kick1 = GW1_KICKOFF.strftime("%Y-%m-%dT%H:%M:%SZ")
    kick2 = GW2_KICKOFF.strftime("%Y-%m-%dT%H:%M:%SZ")
    gw1 = [
        {
            "id": 100,
            "event": 1,
            "team_h": 1,
            "team_a": 2,
            "kickoff_time": kick1,
            "finished": True,
            "team_h_score": 2,
            "team_a_score": 0,
        },
        {
            "id": 101,
            "event": 1,
            "team_h": 3,
            "team_a": 4,
            "kickoff_time": kick1,
            "finished": True,
            "team_h_score": 1,
            "team_a_score": 1,
        },
        {
            "id": 102,
            "event": 1,
            "team_h": 5,
            "team_a": 6,
            "kickoff_time": kick1,
            "finished": True,
            "team_h_score": 0,
            "team_a_score": 3,
        },
    ]
    gw2 = [
        {
            "id": 200,
            "event": 2,
            "team_h": 2,
            "team_a": 1,
            "kickoff_time": kick2,
            "finished": False,
            "team_h_score": None,
            "team_a_score": None,
        },
        {
            "id": 201,
            "event": 2,
            "team_h": 4,
            "team_a": 3,
            "kickoff_time": kick2,
            "finished": False,
            "team_h_score": None,
            "team_a_score": None,
        },
        {
            "id": 202,
            "event": 2,
            "team_h": 6,
            "team_a": 5,
            "kickoff_time": kick2,
            "finished": False,
            "team_h_score": None,
            "team_a_score": None,
        },
    ]
    return gw1, gw2


def _event(
    gw: int, deadline: datetime, *, finished: bool, data_checked: bool
) -> dict[str, Any]:
    return {
        "id": gw,
        "deadline_time": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "finished": finished,
        "data_checked": data_checked,
    }


def _bootstrap(
    teams: list[dict[str, Any]],
    elements: list[dict[str, Any]],
    *,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {"teams": teams, "elements": elements, "events": events}


def _event_live(
    players: list[dict[str, Any]], gw1_fixtures: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-player GW1 stats keyed by the fixture their club played."""
    by_team_fixture: dict[int, int] = {}
    for fx in gw1_fixtures:
        by_team_fixture[int(fx["team_h"])] = int(fx["id"])
        by_team_fixture[int(fx["team_a"])] = int(fx["id"])
    elements: list[dict[str, Any]] = []
    for player in players:
        fixture_id = by_team_fixture[player["team_id"]]
        home = any(
            int(fx["id"]) == fixture_id and int(fx["team_h"]) == player["team_id"]
            for fx in gw1_fixtures
        )
        elements.append(
            {
                "id": player["element_id"],
                "stats": {
                    "minutes": 90,
                    "total_points": 6 if home else 2,
                    "goals_scored": (1 if home and player["element_type"] >= 3 else 0),
                    "assists": 0,
                    "clean_sheets": 1 if home else 0,
                    "goals_conceded": 0 if home else 2,
                    "saves": 2 if player["element_type"] == 1 else 0,
                    "bonus": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "own_goals": 0,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                    "starts": 1,
                    "influence": "20.0",
                    "creativity": "10.0",
                    "threat": "15.0",
                    "bps": 20,
                },
                "explain": [{"fixture": fixture_id}],
            }
        )
    return {"elements": elements}


def _write_capture(
    captured: Path, *, bootstrap: dict[str, Any], fixtures: list[dict[str, Any]]
) -> None:
    captured.mkdir(parents=True)
    _write_json_gz(captured / "bootstrap_static.json.gz", bootstrap)
    _write_json_gz(captured / "fixtures.json.gz", fixtures)


def _write_json_gz(path: Path, payload: object) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
