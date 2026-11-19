"""Streamlit leaderboard. Reads the prediction ledger and nothing else."""

from __future__ import annotations

import polars as pl
import streamlit as st

from fplt.backtest.board import leaderboard, side_by_side
from fplt.backtest.ledger import read_predictions
from fplt.settings import get_settings


def main() -> None:
    """Render metric slices and a side-by-side prediction table."""
    st.set_page_config(page_title="fplt leaderboard", layout="wide")
    st.title("fplt leaderboard")
    ledger = read_predictions(get_settings().data_dir)
    if ledger.is_empty():
        st.info("No predictions in the ledger yet.")
        return

    seasons = _choices(ledger, "target_season")
    horizons = _choices(ledger, "horizon")
    positions = ["All", *_choices(ledger, "position")]
    season = st.selectbox("Season", seasons)
    horizon = st.selectbox("Horizon", horizons)
    position = st.selectbox("Position", positions)
    chosen_position = None if position == "All" else str(position)
    st.dataframe(
        leaderboard(
            ledger,
            season=str(season),
            horizon=int(horizon),
            position=chosen_position,
        ),
        hide_index=True,
    )

    as_ofs = _choices(ledger, "as_of_gw")
    gameweeks = _choices(ledger, "target_gw")
    as_of = st.selectbox("As of gameweek", as_ofs)
    gameweek = st.selectbox("Target gameweek", gameweeks)
    st.dataframe(
        side_by_side(
            ledger,
            season=str(season),
            gw=int(gameweek),
            as_of_gw=int(as_of),
        ),
        hide_index=True,
    )


def _choices(ledger: pl.DataFrame, column: str) -> list[object]:
    return ledger.get_column(column).unique().sort().to_list()


if __name__ == "__main__":
    main()
