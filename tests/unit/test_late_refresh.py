"""Late pre-deadline re-predict hooks when availability inputs move."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.automation.late_refresh import (
    latest_late_signal_at,
    latest_plan_finished_at,
    plans_stale_vs_signals,
)
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.types import GWKey
from tests.builders import build_league, write_test_plan_run

_WHEN = datetime(2024, 8, 1, tzinfo=UTC)
_DEADLINE = datetime(2024, 8, 16, 17, tzinfo=UTC)


def _tiny_plan(model_id: str, season: str, gw: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_id": [model_id],
            "as_of_season": [season],
            "as_of_gw": [gw],
            "target_gw": [gw],
            "step": [0],
            "player_key": [100],
            "position": ["MID"],
            "team_key": [3],
            "in_xi": [True],
            "captain": [True],
            "vice": [False],
            "bought": [False],
            "sold": [False],
            "xpts": [5.0],
            "hit": [0.0],
        }
    )


def test_latest_late_signal_caps_at_deadline(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    # Status after deadline must not count.
    store.write(
        "player_status",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "player_key": [100],
                "status": ["d"],
                "news": ["knock"],
                "price": [100],
                "ep_next": [3.0],
                "chance_of_playing": [25.0],
                "penalties_order": [1],
                "available_at": [_DEADLINE + timedelta(hours=1)],
                "source": ["fpl_api"],
            }
        ),
    )
    peak = latest_late_signal_at(
        store, GWKey("2024-25", 1), now=_DEADLINE - timedelta(hours=1)
    )
    assert peak is not None
    assert peak == _WHEN


def test_plans_stale_when_signal_newer_than_finished_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = build_league(DataStore(tmp_path))
    specs = load_model_specs()
    model = "rolling_mean_v1"
    model_id = specs[model].model_id
    write_test_plan_run(tmp_path, _tiny_plan(model_id, "2024-25", 1))
    assert latest_plan_finished_at(tmp_path, model, GWKey("2024-25", 1), 1) is not None

    old = _DEADLINE - timedelta(hours=6)
    newer = _DEADLINE - timedelta(hours=1)
    monkeypatch.setattr(
        "fplt.automation.late_refresh.latest_plan_finished_at",
        lambda *_a, **_k: old,
    )
    store.write(
        "predicted_lineups",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "team_key": [3],
                "fixture_id": [1],
                "player_key": [100],
                "element_id": [10],
                "role": ["start"],
                "confidence": [0.9],
                "available_at": [newer],
                "source": ["touchline_wire"],
            }
        ),
    )
    assert plans_stale_vs_signals(
        tmp_path,
        store,
        [model],
        GWKey("2024-25", 1),
        1,
        now=_DEADLINE - timedelta(minutes=30),
    )
