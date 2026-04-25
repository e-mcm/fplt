"""Prediction pipeline — loads models from MLflow and produces Xpts per player.

Usage:
    uv run python src/fplt/pipeline/predict.py pipeline.gw=38

Writes:
    outputs/predictions/xpts_gw{GW}_{YYYYMMDD}.parquet
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import hydra
import mlflow
import mlflow.sklearn
import pandas as pd
from omegaconf import DictConfig

from fplt.aggregator import compute_xpts
from fplt.data.load import load_gameweeks, load_teams
from fplt.features.engineer import build_features
from fplt.features.player_share import add_player_xg_share
from fplt.models.bonus.model import BonusModel
from fplt.models.cards.model import CardModel
from fplt.models.goal_allocation.model import GoalAllocationModel
from fplt.models.match_score.dixon_coles import DixonColesModel
from fplt.models.minutes.model import MinutesModel
from fplt.models.saves.model import SavesModel

log = logging.getLogger(__name__)


def _load_model(run_name: str) -> object:
    """Load the most recent logged model for a given run name."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("fplt_training")
    if experiment is None:
        raise RuntimeError("No 'fplt_training' experiment found. Run train.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No run found with name '{run_name}'")

    run_id = runs[0].info.run_id
    return mlflow.sklearn.load_model(f"runs:/{run_id}/{run_name}_obj")


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    gw = int(cfg.pipeline.get("gw", 1))
    season = str(cfg.pipeline.get("season", "2024-25"))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    log.info("Loading models from MLflow...")
    dixon_coles: DixonColesModel = _load_model("dixon_coles")  # type: ignore[assignment]
    minutes: MinutesModel = _load_model("minutes")  # type: ignore[assignment]
    goal_alloc: GoalAllocationModel = _load_model("goal_allocation")  # type: ignore[assignment]
    bonus: BonusModel = _load_model("bonus")  # type: ignore[assignment]
    cards: CardModel = _load_model("cards")  # type: ignore[assignment]
    saves: SavesModel = _load_model("saves")  # type: ignore[assignment]

    models = {
        "minutes": minutes,
        "goal_allocation": goal_alloc,
        "bonus": bonus,
        "cards": cards,
        "saves": saves,
    }

    log.info("Loading GW %d features for season %s...", gw, season)
    gw_df = load_gameweeks(season)
    teams_df = load_teams(season)
    features = build_features(gw_df, teams_df)
    features = add_player_xg_share(features)

    X = features[features["GW"] == gw].copy()
    if X.empty:
        log.warning("No rows for GW %d in season %s", gw, season)
        return

    log.info("Building score distribution for GW %d fixtures...", gw)
    gw_fixtures = X.drop_duplicates("fixture")[
        ["fixture", "team_h_score", "team_a_score"]
    ].copy()
    # For prediction, we need team IDs. Reconstruct from was_home/opponent_team.
    # Build a small fixtures DataFrame for predict_distribution
    fix_rows = []
    for _, row in gw_fixtures.iterrows():
        fid = row["fixture"]
        home_player = X[(X["fixture"] == fid) & (X["was_home"])].iloc[:1]
        away_player = X[(X["fixture"] == fid) & (~X["was_home"])].iloc[:1]
        if home_player.empty or away_player.empty:
            continue
        fix_rows.append(
            {
                "fixture_id": fid,
                "team_h": home_player["team"].iloc[0]
                if "team" in home_player.columns
                else fid,
                "team_a": away_player["team"].iloc[0]
                if "team" in away_player.columns
                else fid,
            }
        )

    if not fix_rows:
        log.error("Could not reconstruct fixture team IDs for GW %d", gw)
        return

    fix_df = pd.DataFrame(fix_rows)
    score_dist = dixon_coles.predict_distribution(fix_df)

    log.info("Computing Xpts for %d players...", len(X))
    xpts = compute_xpts(X, score_dist, models, fixture_col="fixture")

    out_df = (
        X[["name", "element", "position", "team", "GW"]].copy()
        if "position" in X.columns
        else X[["name", "element", "GW"]].copy()
    )
    out_df["xpts"] = xpts.values

    out_dir = Path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    out_path = out_dir / f"xpts_gw{gw:02d}_{date_str}.parquet"
    out_df.to_parquet(out_path, index=False)
    log.info("Wrote %d rows to %s", len(out_df), out_path)


if __name__ == "__main__":
    main()
