"""Predicted XI ingest, staging, rate limit, and accuracy scoring."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from fplt.data.predicted_lineups import (
    DEFAULT_MIN_INTERVAL_SECONDS,
    ROLE_BENCH,
    ROLE_START,
    PredictedLineupsDisabled,
    PredictedLineupsRateLimit,
    api_football_lineups_raw_path,
    capture_touchline_lineups,
    fetch_touchline_lineups,
    ingest_predicted_lineups,
    iter_api_football_lineups_raw,
    iter_touchline_raw,
    read_json_gz,
    reset_touchline_rate_limit,
    score_predicted_starts,
    stage_api_football_lineups_payload,
    stage_touchline_payload,
    touchline_raw_path,
    write_json_gz,
)
from fplt.data.store import DataStore
from fplt.schemas.staged import PredictedLineupsSchema
from tests.builders import build_league

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "touchline"
_SYNTH = _FIXTURE / "lineups_synthetic.json.gz"
_REALISH = _FIXTURE / "lineups.json.gz"
_CAPTURE = datetime(2024, 8, 16, 12, tzinfo=UTC)


def test_schema_accepts_predicted_lineup_row() -> None:
    frame = pl.DataFrame(
        {
            "season": ["2024-25"],
            "team_key": [3],
            "fixture_id": [2],
            "player_key": [100],
            "element_id": [10],
            "role": [ROLE_START],
            "confidence": [0.95],
            "available_at": [_CAPTURE],
            "source": ["touchline_wire"],
        }
    )
    checked = PredictedLineupsSchema.validate(frame)
    assert checked.height == 1


def test_stage_touchline_maps_element_and_fixture(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    payload = read_json_gz(_SYNTH)
    written = stage_touchline_payload(store, payload, "2024-25", _CAPTURE)
    assert written.height >= 2
    assert set(written["role"].to_list()) <= {ROLE_START, ROLE_BENCH}
    arsenal = written.filter(pl.col("team_key") == 3)
    assert arsenal.filter(pl.col("player_key") == 100)["role"].to_list() == [ROLE_START]
    # Earliest unfinished Arsenal fixture at capture is GW1 fixture_id=1.
    assert set(arsenal["fixture_id"].drop_nulls().to_list()) == {1}


def test_ingest_from_raw_touchline(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    raw_root = tmp_path / "raw"
    dest = touchline_raw_path(raw_root, "2024-25", _CAPTURE)
    assert write_json_gz(dest, read_json_gz(_SYNTH))
    assert write_json_gz(dest, {"meta": {}, "data": []}) is False  # write-once
    count = ingest_predicted_lineups(store, raw_root)
    assert count == 1
    staged = store.read("predicted_lineups")
    assert staged.height >= 2
    assert staged["source"].unique().to_list() == ["touchline_wire"]


def test_fetch_requires_flag_and_rate_limits() -> None:
    reset_touchline_rate_limit()
    with pytest.raises(PredictedLineupsDisabled):
        fetch_touchline_lineups(enabled=False)

    payload = {"meta": {"count": 0, "license": "test"}, "data": []}

    def handler(request: httpx.Request) -> httpx.Response:
        assert "fplt" in request.headers["User-Agent"].lower()
        assert "Touchline" in request.headers["User-Agent"]
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    first = fetch_touchline_lineups(
        client=client,
        enabled=True,
        min_interval_seconds=60,
        url="https://example.test/l",
    )
    assert first == payload
    with pytest.raises(PredictedLineupsRateLimit):
        fetch_touchline_lineups(
            client=client,
            enabled=True,
            min_interval_seconds=60,
            url="https://example.test/l",
        )
    reset_touchline_rate_limit()
    client.close()


def test_capture_touchline_writes_raw(tmp_path: Path) -> None:
    reset_touchline_rate_limit()
    payload = read_json_gz(_SYNTH)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    path, written = capture_touchline_lineups(
        tmp_path / "raw",
        "2024-25",
        enabled=True,
        min_interval_seconds=0,
        captured_at=_CAPTURE,
        client=client,
        url="https://example.test/l",
    )
    assert written is True
    assert path.exists()
    assert read_json_gz(path)["meta"]["count"] == 2
    reset_touchline_rate_limit()
    client.close()


def test_realish_fixture_parses(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    # Real-shaped dump may not map to builder element ids; staging must not raise.
    payload = read_json_gz(_REALISH)
    assert "meta" in payload
    assert "data" in payload
    assert payload["meta"].get("license")
    frame = stage_touchline_payload(store, payload, "2024-25", _CAPTURE)
    assert isinstance(frame, pl.DataFrame)


def test_api_football_adapter_stages_same_schema(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    payload = {
        "response": [
            {
                "fixture": {"id": 9001},
                "team": {"id": 42},
                "startXI": [{"player": {"id": 7, "name": "A"}}],
                "substitutes": [{"player": {"id": 8, "name": "B"}}],
            }
        ]
    }
    written = stage_api_football_lineups_payload(
        store,
        payload,
        "2024-25",
        _CAPTURE,
        player_id_map={7: 100, 8: 200},
        team_id_map={42: 3},
        fixture_id_map={9001: 2},
    )
    assert written.height == 2
    assert set(written["source"].to_list()) == {"api_football"}
    assert set(written["role"].to_list()) == {ROLE_START, ROLE_BENCH}


def test_score_predicted_starts_vs_labels(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    predicted = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "team_key": [3, 3],
            "fixture_id": [1, 1],
            "player_key": [100, 200],
            "element_id": [10, 20],
            "role": [ROLE_START, ROLE_START],
            "confidence": [0.9, 0.9],
            "available_at": [_CAPTURE, _CAPTURE],
            "source": ["touchline_wire", "touchline_wire"],
        }
    )
    store.write("predicted_lineups", predicted)
    actual = store.read("player_fixture_stats")
    scored = score_predicted_starts(store.read("predicted_lineups"), actual)
    assert scored.height == 2
    assert scored.filter(pl.col("player_key") == 100)["correct"].to_list() == [True]


def test_default_min_interval_is_documented() -> None:
    assert DEFAULT_MIN_INTERVAL_SECONDS == 300


def test_write_json_gz_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "x.json.gz"
    assert write_json_gz(path, {"a": 1})
    assert json.loads(__import__("gzip").open(path, "rt", encoding="utf-8").read()) == {
        "a": 1
    }


def test_api_football_lineups_raw_path_layout(tmp_path: Path) -> None:
    path = api_football_lineups_raw_path(tmp_path, "2024-25", _CAPTURE)
    assert "api_football" in path.parts
    assert path.parts[-5:] == (
        "api_football",
        "season=2024-25",
        "lineups",
        f"captured_at={_CAPTURE.strftime('%Y-%m-%dT%H-%M-%SZ')}",
        "lineups.json.gz",
    )


def test_fetch_rejects_non_object_payload() -> None:
    reset_touchline_rate_limit()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="JSON object"):
        fetch_touchline_lineups(
            client=client,
            enabled=True,
            min_interval_seconds=0,
            url="https://example.test/l",
        )
    client.close()
    reset_touchline_rate_limit()


def test_fetch_closes_owned_client(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_touchline_rate_limit()
    closed: list[bool] = []

    class TrackingClient(httpx.Client):
        def close(self) -> None:
            closed.append(True)
            super().close()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"meta": {"count": 0}, "data": []})

    def factory(*args: object, **kwargs: object) -> httpx.Client:
        opts = dict(kwargs)
        opts.setdefault("transport", httpx.MockTransport(handler))
        return TrackingClient(*args, **opts)

    monkeypatch.setattr("fplt.data.predicted_lineups.httpx.Client", factory)
    payload = fetch_touchline_lineups(
        enabled=True,
        min_interval_seconds=0,
        url="https://example.test/l",
    )
    assert payload["meta"]["count"] == 0
    assert closed == [True]
    reset_touchline_rate_limit()


def test_capture_naive_stamp_and_existing_path(tmp_path: Path) -> None:
    reset_touchline_rate_limit()
    payload = {"meta": {"count": 0}, "data": []}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    naive = datetime(2024, 8, 16, 12)  # no tzinfo → stamped as UTC
    path, written = capture_touchline_lineups(
        tmp_path / "raw",
        "2024-25",
        enabled=True,
        min_interval_seconds=0,
        captured_at=naive,
        client=client,
        url="https://example.test/l",
    )
    assert written is True
    assert path.exists()
    again, rewritten = capture_touchline_lineups(
        tmp_path / "raw",
        "2024-25",
        enabled=True,
        captured_at=naive,
        client=client,
        url="https://example.test/l",
    )
    assert again == path
    assert rewritten is False
    reset_touchline_rate_limit()
    client.close()


def test_iter_raw_empty_and_season_filter(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    assert iter_touchline_raw(raw) == []
    assert iter_api_football_lineups_raw(raw) == []

    touch = touchline_raw_path(raw, "2024-25", _CAPTURE)
    write_json_gz(touch, {"meta": {}, "data": []})
    other = touchline_raw_path(raw, "2023-24", _CAPTURE)
    write_json_gz(other, {"meta": {}, "data": []})
    found = iter_touchline_raw(raw, seasons=["2024-25"])
    assert len(found) == 1
    assert found[0][1] == "2024-25"

    api_path = api_football_lineups_raw_path(raw, "2024-25", _CAPTURE)
    write_json_gz(api_path, {"response": []})
    api_other = api_football_lineups_raw_path(raw, "2023-24", _CAPTURE)
    write_json_gz(api_other, {"response": []})
    api_found = iter_api_football_lineups_raw(raw, seasons=["2024-25"])
    assert len(api_found) == 1
    assert api_found[0][0] == api_path
    assert api_found[0][1] == "2024-25"


def test_stage_touchline_skips_and_empty_paths(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    empty = stage_touchline_payload(store, {}, "2024-25", _CAPTURE)
    assert empty.is_empty()

    # Unknown team_id, missing team_id, missing element id, non-numeric likelihood.
    payload = {
        "data": [
            {"team_id": None, "starters": [{"id": 10}]},
            {
                "team_id": 99,
                "starters": [{"id": 10, "likelihood": 90}],
                "tactical_confidence_rate": 0.5,
            },
            {
                "team_id": 1,
                "starters": [
                    {"id": None},
                    {"id": 10, "likelihood": "high"},
                    {"id": 999},  # no player_key → dropped
                ],
                "bench": [],
                "tactical_confidence_rate": 80,
            },
        ]
    }
    written = stage_touchline_payload(store, payload, "2024-25", _CAPTURE)
    # Only element 10 with club confidence survives mapping.
    assert written.height == 1
    assert written["confidence"].to_list() == [0.8]
    assert written["player_key"].to_list() == [100]

    # All rows lack player_key → empty schema frame after filter.
    only_unknown = {
        "data": [{"team_id": 1, "starters": [{"id": 999}], "bench": []}],
    }
    blank = stage_touchline_payload(store, only_unknown, "2024-25", _CAPTURE)
    assert blank.is_empty()

    # Club list with no starters/bench yields empty rows.
    no_players = {"data": [{"team_id": 1, "starters": [], "bench": []}]}
    assert stage_touchline_payload(store, no_players, "2024-25", _CAPTURE).is_empty()


def test_stage_api_football_skips_incomplete(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    payload = {
        "response": [
            {"fixture": {"id": 1}, "team": None, "startXI": []},
            {"fixture": {"id": 1}, "team": {"id": None}, "startXI": []},
            {"fixture": {"id": 1}, "team": {"id": 99}, "startXI": []},
            {
                "fixture": {"id": 1},
                "team": {"id": 42},
                "startXI": [
                    {"player": {"id": None}},
                    {"player": {"id": 77}},
                    {"player": {"id": 7}},
                ],
                "substitutes": [],
            },
        ]
    }
    written = stage_api_football_lineups_payload(
        store,
        payload,
        "2024-25",
        _CAPTURE,
        player_id_map={7: 100},
        team_id_map={42: 3},
        fixture_id_map={1: 2},
    )
    assert written.height == 1
    assert written["player_key"].to_list() == [100]

    empty = stage_api_football_lineups_payload(
        store,
        {"response": [{"team": {"id": 42}, "startXI": [], "substitutes": []}]},
        "2024-25",
        _CAPTURE,
        player_id_map={},
        team_id_map={42: 3},
    )
    assert empty.is_empty()


def test_ingest_rejects_non_object_touchline(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    raw = tmp_path / "raw"
    dest = touchline_raw_path(raw, "2024-25", _CAPTURE)
    write_json_gz(dest, [1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="JSON object"):
        ingest_predicted_lineups(store, raw)


def test_ingest_api_football_raw_with_empty_maps(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    raw = tmp_path / "raw"
    api_path = api_football_lineups_raw_path(raw, "2024-25", _CAPTURE)
    write_json_gz(
        api_path,
        [
            {
                "fixture": {"id": 1},
                "team": {"id": 42},
                "startXI": [{"player": {"id": 7}}],
                "substitutes": [],
            }
        ],
    )
    count = ingest_predicted_lineups(store, raw, seasons=["2024-25"])
    assert count == 1
    # Empty maps → no staged rows, but file still consumed.
    assert store.read("predicted_lineups").is_empty()
