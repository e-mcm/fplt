"""The leaderboard is a slice of the ledger, including both baselines."""

from pathlib import Path

import fplt.models.baselines  # noqa: F401 — runs @register_model
from fplt.backtest.board import leaderboard, side_by_side
from fplt.backtest.ledger import (
    append_predictions,
    empty_ledger,
    fill_actuals,
    read_predictions,
)
from fplt.backtest.walk_forward import walk_forward
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import get_model, load_model_specs
from fplt.types import GWKey
from tests.builders import build_league


def test_both_baselines_appear_for_one_season(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    specs = load_model_specs()
    for spec in specs.values():
        model = get_model(spec.kind)(spec)
        frame = walk_forward(model, PITData(store), [GWKey("2024-25", 1)], steps=2)
        append_predictions(tmp_path, frame)
    fill_actuals(tmp_path, store)
    ledger = read_predictions(tmp_path)
    board = leaderboard(ledger, season="2024-25", horizon=1)
    expected = {spec.model_id for spec in specs.values()}
    assert set(board["model_id"].to_list()) == expected
    mids = leaderboard(ledger, season="2024-25", horizon=1, position="MID")
    assert mids["position"].unique().to_list() == ["MID"]
    compared = side_by_side(ledger, season="2024-25", gw=1, as_of_gw=1)
    assert set(compared["player_key"].to_list()) == {100, 200}
    assert expected <= set(compared.columns)
    assert leaderboard(ledger, season="1999-00").is_empty()


def test_empty_ledger_scores_nothing() -> None:
    assert leaderboard(empty_ledger()).is_empty()
    compared = side_by_side(empty_ledger(), season="2024-25", gw=1, as_of_gw=1)
    assert compared.is_empty()


def test_leaderboard_app_reads_the_ledger_only() -> None:
    source = Path("apps/leaderboard.py").read_text(encoding="utf-8")
    assert "DataStore" not in source
    assert "fplt.data" not in source
    assert "read_predictions" in source
