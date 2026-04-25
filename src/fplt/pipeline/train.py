"""Training pipeline — fits all models and logs to MLflow.

Usage:
    uv run python src/fplt/pipeline/train.py
    uv run python src/fplt/pipeline/train.py model=dixon_coles pipeline.seasons=[2022-23,2023-24]

Training and inference are fully decoupled: this script fits and serializes
models via MLflow; predict.py loads them by name without any shared state.
"""

from __future__ import annotations

import logging
import math

import hydra
import mlflow
import mlflow.pyfunc
import pandas as pd
from omegaconf import DictConfig

from fplt.data.load import ALL_SEASONS, load_all_seasons, load_teams
from fplt.features.engineer import build_features
from fplt.features.player_share import add_player_xg_share
from fplt.features.team_strength import build_fixture_records
from fplt.models.bonus.model import BonusModel
from fplt.models.cards.model import CardModel
from fplt.models.goal_allocation.model import GoalAllocationModel
from fplt.models.match_score.dixon_coles import DixonColesModel
from fplt.models.minutes.model import MinutesModel
from fplt.models.saves.model import SavesModel

log = logging.getLogger(__name__)


def _load_player_features(cfg: DictConfig) -> pd.DataFrame:
    seasons: list[str] = list(
        cfg.pipeline.seasons if hasattr(cfg, "pipeline") else ALL_SEASONS
    )
    gw = load_all_seasons(seasons)
    # Load teams for the first available season (strength ratings)
    teams = load_teams(seasons[-1])
    for s in reversed(seasons):
        try:
            teams = load_teams(s)
            break
        except FileNotFoundError:
            continue
    df = build_features(gw, teams)
    df = add_player_xg_share(df)
    # Drop GW1 rows — all rolling features are NaN
    return df[df["GW"] > 1].reset_index(drop=True)


def _load_fixture_records(cfg: DictConfig) -> pd.DataFrame:
    seasons: list[str] = list(
        cfg.pipeline.seasons if hasattr(cfg, "pipeline") else ALL_SEASONS
    )
    return build_fixture_records(seasons)


def _train_dixon_coles(fixtures: pd.DataFrame, cfg: DictConfig) -> DixonColesModel:
    dc_cfg = cfg.get("model/dixon_coles", cfg.get("model", {}))
    model = DixonColesModel(
        time_decay=float(dc_cfg.get("time_decay", 0.0065)),
        max_goals=int(dc_cfg.get("max_goals", 10)),
    )
    with mlflow.start_run(run_name="dixon_coles", nested=True):
        model.fit(fixtures)
        mlflow.log_params(model.params)
        finite_metrics = {k: v for k, v in model.metrics.items() if not math.isnan(v)}
        mlflow.log_metrics(finite_metrics)
        mlflow.sklearn.log_model(model, name="dixon_coles_obj")
    log.info("DixonColes fitted. log_loss=%.4f", model.metrics["log_loss"])
    return model


def _train_component(
    name: str,
    model: object,
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series | None,
) -> None:
    """Fit a PointModel and log it to MLflow."""
    from fplt.models.base import FPLModel

    assert isinstance(model, FPLModel)
    with mlflow.start_run(run_name=name, nested=True):
        model.fit(X, y)
        mlflow.log_params(model.params)
        finite_metrics = {k: v for k, v in model.metrics.items() if not math.isnan(v)}
        mlflow.log_metrics(finite_metrics)
        mlflow.sklearn.log_model(model, name=f"{name}_obj")
    log.info("%s fitted. metrics=%s", name, model.metrics)


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("fplt_training")

    with mlflow.start_run(run_name="full_pipeline"):
        log.info("Loading player features...")
        players = _load_player_features(cfg)
        log.info("Player rows: %d", len(players))

        log.info("Loading fixture records...")
        fixtures = _load_fixture_records(cfg)
        log.info("Fixture rows: %d", len(fixtures))

        # --- Walk-forward split: last season for validation ---
        seasons_sorted = sorted(players["season"].unique())
        val_season = seasons_sorted[-1]
        train_mask = players["season"] != val_season

        X_train = players[train_mask].copy()
        X_val = players[~train_mask].copy()
        log.info(
            "Train: %d rows, Val (%s): %d rows", len(X_train), val_season, len(X_val)
        )

        # --- Stage 1: Dixon-Coles ---
        log.info("Training DixonColesModel...")
        _train_dixon_coles(fixtures[fixtures["season"] != val_season], cfg)

        # --- Stage 3: Independent component models ---
        component_configs: list[tuple[str, object, pd.DataFrame, pd.Series]] = [
            (
                "minutes",
                MinutesModel(),
                X_train,
                X_train[["starts", "minutes"]]
                if "starts" in X_train.columns
                else X_train[["minutes"]],
            ),  # type: ignore[list-item]
            ("bonus", BonusModel(), X_train, X_train["bonus"]),
            ("cards", CardModel(), X_train, X_train[["yellow_cards", "red_cards"]]),
            ("saves", SavesModel(), X_train, X_train["saves"]),
            (
                "goal_allocation",
                GoalAllocationModel(),
                X_train,
                X_train["goals_scored"],
            ),
        ]

        for name, model, X, y in component_configs:
            log.info("Training %s...", name)
            _train_component(name, model, X, y)  # type: ignore[arg-type]

        log.info("Training complete.")


if __name__ == "__main__":
    main()
