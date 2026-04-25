# FPL Predictor — Repo Design Document

This document describes the intended architecture, pipeline structure, tooling, and code design for a Fantasy Premier League points prediction and team selection system. Use it as the authoritative reference when scaffolding the repo.

---

## Purpose

The system has three primary outputs:
1. **Gameweek points predictions** — expected points (Xpts) per player per GW
2. **Optimal starting XI and captain pick** — given predicted points and squad constraints
3. **Transfer recommendations** — 1–2 move suggestions maximising net Xpts gain

---

## Tooling

| Concern | Tool |
|---|---|
| Dependency management | `uv` — use `pyproject.toml` and `uv.lock` |
| Pipeline DAG | `Makefile` — stages with dependency tracking |
| Experiment tracking + model serialization | `MLflow` — local tracking server |
| Config management | `Hydra` — YAML configs with CLI overrides |
| Orchestration (deferred) | `Prefect` — add later for scheduled GW runs |

Do not use Poetry, DVC, or Kedro.

MLflow is used for:
- Logging hyperparameters and metrics per training run
- Serializing fitted model objects as artifacts (`mlflow.pyfunc.log_model`)
- Providing a model registry so the prediction pipeline loads models by name and version

The training pipeline and prediction pipeline are **fully decoupled** — training fits and logs to MLflow; prediction loads registered models by name. They never share in-memory state.

---

## Pipeline Stages

The pipeline has four sequential stages:

```
Stage 1: Match Score Model
  Input:  team-level features (attack/defence strength, home advantage, fixture history)
  Output: joint probability distribution over (home goals, away goals) — Dixon-Coles model
  → P(H=h, A=a) for all h, a

Stage 2: Score-Derived Components
  Computed from the score distribution (no separate model):
  → P(clean sheet)              derived from P(opponent scored 0)
  → E[goals conceded penalty]   derived from P(opponent scored ≥ 3)
  Computed via allocation models (conditional on score distribution):
  → Goal points                 P(player scored | team scored k goals)
  → Assist points               P(player assisted | goal occurred)

Stage 3: Independent Component Models
  Fit separately, not conditioned on match score:
  → Minutes model               P(starts) × E[minutes | starts]
  → Bonus points model          E[BPS rank → bonus] — per-position prior initially
  → Card model                  P(yellow), P(red)
  → GK save points              E[saves] from opponent shot volume features

Stage 4: Aggregation
  Combines all components → Xpts per player per GW
  Feeds Xpts into the optimisation layer
```

**Important:** Do not compute P(clean sheet) from E[goals]. Integrate over the full score distribution. See Aggregator section below.

---

## Optimisation Layer

This is a **constrained integer linear programme**, separate from the ML layer. Use `PuLP` or `OR-Tools`.

Constraints to encode:
- £100m budget
- 15-player squad (2 GK, 5 DEF, 5 MID, 3 FWD)
- Max 3 players per club
- Starting XI formation validity
- Captain selection (doubles points)
- Free transfer count
- Chip flags: wildcard, free hit, bench boost, triple captain

The optimiser takes Xpts per player as input. It has no dependency on any model internals.

---

## Data Sourcing

Primary: **FPL official API** — player metadata, prices, ownership, GW results, fixture list.

Secondary (wire up later): **Understat** — xG, xA, xGC per player per match. Single biggest feature lift. Design the ingestion layer to support both from day one.

Historical FPL data: use `vaastrom` datasets or `fplreview` historical CSVs for training.

---

## Repo Structure

```
fpl-predictor/
│
├── pyproject.toml              # uv — dependencies
├── uv.lock                     # committed to git
├── Makefile                    # pipeline DAG
│
├── conf/                       # Hydra config
│   ├── config.yaml
│   ├── model/
│   │   ├── xgboost.yaml
│   │   └── dixon_coles.yaml
│   └── pipeline.yaml
│
├── data/                       # gitignored; stamped by GW and date
│   ├── raw/                    # e.g. fpl_gw31_20260423.parquet
│   ├── processed/
│   └── features/
│
├── mlruns/                     # MLflow local tracking — gitignored
│
├── src/
│   ├── ingestion/
│   │   ├── fpl_client.py
│   │   └── understat.py
│   │
│   ├── processing/
│   │   ├── players.py
│   │   ├── fixtures.py
│   │   └── results.py
│   │
│   ├── features/
│   │   ├── team_strength.py    # attack/defence ratings for match model
│   │   ├── player_share.py     # xG/shot share per player per team
│   │   ├── rotation_risk.py    # minutes/appearance features
│   │   └── pipeline.py         # assembles full feature matrix
│   │
│   ├── models/
│   │   ├── base.py             # abstract base classes — see below
│   │   ├── score_distribution.py  # ScoreDistribution wrapper class
│   │   ├── match_score/
│   │   │   └── dixon_coles.py
│   │   ├── goal_allocation/
│   │   │   └── model.py
│   │   ├── minutes/
│   │   │   └── model.py
│   │   └── bonus/
│   │       └── model.py
│   │
│   ├── aggregator.py           # combines all component predictions → Xpts
│   │
│   ├── optimisation/
│   │   ├── team_selector.py    # optimal XI + captain (ILP)
│   │   └── transfers.py        # 1–2 transfer recommendations
│   │
│   └── pipeline/
│       ├── train.py            # fits all models, logs to MLflow
│       ├── validate.py         # walk-forward CV per model, logs metrics to MLflow
│       └── predict.py          # loads models from MLflow, runs GW inference
│
├── notebooks/
│   ├── eda/                    # exploratory data analysis
│   └── validation.ipynb        # loads MLflow runs, plots predicted vs actual Xpts
│
├── outputs/
│   ├── predictions/            # Xpts per player per GW
│   └── selections/             # recommended XI + transfers
│
└── tests/
    ├── unit/
    │   ├── test_score_distribution.py   # highest priority — axis conventions
    │   ├── test_optimiser.py            # constraint satisfaction on known problems
    │   ├── test_features.py             # known input → known output
    │   └── test_model_interface.py      # parametrised over all model classes
    └── integration/
        └── test_pipeline_smoke.py       # end-to-end with tiny synthetic dataset
```

---

## Model Base Classes (`src/models/base.py`)

There are two types of model in this system. They differ in what their prediction interface returns.

**Key principle:** serialization and the prediction interface are orthogonal. MLflow serializes the model object's fitted state (via pickle) regardless of what methods the class exposes. The prediction methods exist because the pipeline needs them, not because MLflow requires them.

```python
from abc import ABC, abstractmethod
import pandas as pd


class FPLModel(ABC):
    """
    Abstract base for all FPL pipeline models.
    All models must implement fit(), params, and metrics.
    Serialization is handled by MLflow (pickle of __dict__).
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "FPLModel":
        """Fit the model. Returns self for chaining."""
        ...

    @property
    @abstractmethod
    def params(self) -> dict:
        """Hyperparameters — logged to MLflow on each training run."""
        ...

    @property
    @abstractmethod
    def metrics(self) -> dict:
        """Evaluation metrics — populated after fit(), logged to MLflow."""
        ...


class PointModel(FPLModel):
    """
    Models that output point estimates or probabilities — one row per input row.
    Examples: MinutesModel, GoalAllocationModel, BonusModel.
    """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        ...


class DistributionalModel(FPLModel):
    """
    Models that output a full probability distribution per input row.
    Examples: DixonColesModel.
    The aggregator calls predict_distribution() to get the full distribution.
    """

    @abstractmethod
    def predict_distribution(self, X: pd.DataFrame) -> "ScoreDistribution":
        ...
```

---

## ScoreDistribution (`src/models/score_distribution.py`)

Wraps the joint PMF output of Dixon-Coles. Provides clean accessors so the aggregator can compute derived quantities by integrating over the full distribution rather than using point estimates.

```python
import numpy as np


class ScoreDistribution:
    """
    Joint probability mass function over (home goals, away goals) per fixture.

    pmf shape: (n_fixtures, max_goals+1, max_goals+1)
    pmf[i, h, a] = P(home scored h, away scored a) for fixture i
    """

    def __init__(self, pmf: np.ndarray, fixture_ids: list):
        self.pmf = pmf
        self.fixture_ids = fixture_ids

    def p_clean_sheet(self, side: str) -> np.ndarray:
        """P(opponent scored 0) for each fixture — correct CS probability."""
        # home CS: P(away scored 0) = sum over h of pmf[i, h, 0]
        # away CS: P(home scored 0) = sum over a of pmf[i, 0, a]
        if side == "home":
            return self.pmf[:, :, 0].sum(axis=1)
        else:
            return self.pmf[:, 0, :].sum(axis=1)

    def p_team_scored_k(self, side: str, k: int) -> np.ndarray:
        """P(team scored exactly k goals) for each fixture."""
        if side == "home":
            marginal = self.pmf.sum(axis=2)   # sum over away goals
        else:
            marginal = self.pmf.sum(axis=1)   # sum over home goals
        if k >= marginal.shape[1]:
            return np.zeros(len(self.fixture_ids))
        return marginal[:, k]

    def expected_goals(self, side: str) -> np.ndarray:
        """E[goals] for each fixture — for reference/logging only."""
        k = np.arange(self.pmf.shape[1])
        if side == "home":
            return (self.pmf.sum(axis=2) * k).sum(axis=1)
        else:
            return (self.pmf.sum(axis=1) * k).sum(axis=1)
```

---

## Training Harness (`src/pipeline/train.py`)

Generic — works for all models because they share the base class interface.

```python
import mlflow

# instantiate all models
models = {
    "dixon_coles": DixonColesModel(time_decay=0.0065),
    "goal_allocation": GoalAllocationModel(),
    "minutes": MinutesModel(),
    "bonus": BonusModel(),
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X[name], y.get(name))
        mlflow.log_params(model.params)
        mlflow.log_metrics(model.metrics)
        mlflow.pyfunc.log_model(artifact_path=name, python_model=model)
```

---

## Aggregator (`src/aggregator.py`)

Combines all component predictions into Xpts per player. Integrates over the score distribution rather than chaining point estimates.

```python
def compute_xpts(player, score_dist: ScoreDistribution, models: dict) -> float:
    side = player.side   # "home" or "away"

    # minutes — gates all other contributions
    p_starts, e_minutes = models["minutes"].predict(player)

    # score-derived components — integrate over full distribution
    p_cs = score_dist.p_clean_sheet(side)
    p_goal = sum(
        score_dist.p_team_scored_k(side, k)
        * models["goal_allocation"].predict(player, k)
        for k in range(1, 5)
    )

    # independent components
    e_bonus = models["bonus"].predict(player)

    return (
        appearance_points(e_minutes)
        + p_starts * (
            p_cs * cs_points(player.position)
            + p_goal * goal_points(player.position)
            + e_bonus
        )
    )
```

---

## Makefile (Pipeline DAG)

```makefile
data/raw/fpl.parquet:
	uv run python src/ingestion/fpl_client.py

data/features/feature_matrix.parquet: data/raw/fpl.parquet
	uv run python src/features/pipeline.py

train: data/features/feature_matrix.parquet
	uv run python src/pipeline/train.py

validate:
	uv run python src/pipeline/validate.py --model all

predict:
	uv run python src/pipeline/predict.py --gw $(GW)

test:
	uv run pytest tests/ -v

mlflow-ui:
	uv run mlflow ui

test-and-validate: test validate
```

`make test` is fast — no real data needed, runs on every commit. `make validate` is slow — run when a model has meaningfully changed and you want to see whether it improved.

---

## Testing

Testing covers two distinct activities that require different tooling. Do not conflate them.

**Software tests (pytest)** — does the code do what it's supposed to do? Fast, synthetic data, no model quality judgement. Run on every commit.

**Model validation (MLflow + walk-forward CV)** — does the model perform well on held-out data? Slow, requires historical data, produces metrics visible in the MLflow UI. Run when a model has changed.

### Software Tests

Each stage has a distinct testable contract. Test the interface and correctness, not model quality.

**`ScoreDistribution`** — the highest priority unit to test. Axis convention errors propagate silently into every downstream Xpts calculation:

```python
def test_pmf_sums_to_one():
    pmf = np.random.dirichlet(np.ones(25), size=3).reshape(3, 5, 5)
    dist = ScoreDistribution(pmf, fixture_ids=[1, 2, 3])
    assert np.allclose(pmf.sum(axis=(1, 2)), 1.0)

def test_clean_sheet_probability_in_range():
    assert ((dist.p_clean_sheet("home") >= 0) &
            (dist.p_clean_sheet("home") <= 1)).all()

def test_marginals_consistent():
    # p_team_scored_k summed over all k must equal 1
    total = sum(dist.p_team_scored_k("home", k) for k in range(6))
    assert np.allclose(total, 1.0, atol=1e-3)
```

**Optimiser** — test constraint satisfaction on a known small problem, not solution quality:

```python
def test_budget_constraint():
    result = team_selector.solve(mock_players, budget=100.0)
    assert sum(p.price for p in result.squad) <= 100.0

def test_club_limit():
    result = team_selector.solve(mock_players, budget=100.0)
    assert max(Counter(p.club for p in result.squad).values()) <= 3

def test_position_limits():
    xi = result.starting_xi
    assert sum(1 for p in xi if p.position == "GK") == 1
    assert sum(1 for p in xi if p.position == "DEF") >= 3
```

**Feature engineering** — test on known inputs with known outputs:

```python
def test_rolling_form_window():
    # player scored in GW1, GW3, GW5 — assert 3-GW rolling mean is correct
    ...

def test_player_xg_share_sums_to_one_per_team():
    shares = compute_player_share(mock_team_df)
    assert np.isclose(shares.groupby("team")["xg_share"].sum(), 1.0).all()
```

**Model interface** — parametrised across all concrete model classes:

```python
@pytest.mark.parametrize("model_cls", [
    DixonColesModel, GoalAllocationModel, MinutesModel, BonusModel
])
def test_model_has_required_interface(model_cls):
    model = model_cls()
    assert hasattr(model, "fit")
    assert isinstance(model.params, dict)
    # metrics only populated post-fit, so just check it's a property
    assert isinstance(model.__class__.metrics, property)

def test_distributional_model_returns_score_distribution():
    model = DixonColesModel()
    model.fit(mock_match_df)
    dist = model.predict_distribution(mock_fixture_df)
    assert isinstance(dist, ScoreDistribution)
    assert dist.pmf.shape[1] == dist.pmf.shape[2]   # square goal matrix
```

**Integration smoke test** — end-to-end with a tiny synthetic dataset, asserts the pipeline runs without error and output shapes are correct:

```python
def test_full_pipeline_smoke():
    # synthetic 5-team, 10-GW dataset
    # assert predict.py produces one Xpts row per player
    # assert optimiser returns a valid 15-player squad
    ...
```

### Model Validation (`src/pipeline/validate.py`)

Walk-forward CV per model. Every fold is logged as a separate MLflow run so results are visible and comparable across model versions in the MLflow UI (`mlflow ui`).

```python
def walk_forward_validate(model, X, y, model_name, train_gws, val_gws):
    with mlflow.start_run(run_name=f"{model_name}_validation"):
        mlflow.log_params(model.params)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(
            walk_forward_splits(X, train_gws, val_gws)
        ):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[val_idx])
            metrics = evaluate(preds, y.iloc[val_idx])
            fold_metrics.append(metrics)
            mlflow.log_metrics({f"fold_{fold}_{k}": v for k, v in metrics.items()})

        mlflow.log_metrics({
            "mean_mae":  np.mean([m["mae"]  for m in fold_metrics]),
            "mean_rmse": np.mean([m["rmse"] for m in fold_metrics]),
        })
```

**Metrics per model:**

| Model | Primary Metric | Secondary |
|---|---|---|
| Dixon-Coles | Log-likelihood vs null Poisson | Brier score on CS prediction |
| Goal allocation | Log-loss on scorer prediction | Calibration of P(scored) |
| Minutes model | MAE on minutes played | Accuracy of P(starts) |
| Bonus model | MAE on bonus points | — |
| Aggregator (end-to-end) | MAE on total Xpts vs actual | Spearman rank correlation |

The end-to-end Xpts MAE is the most important single number — it determines optimiser quality. Log it as a top-level MLflow metric.

For visual inspection, `notebooks/validation.ipynb` loads MLflow runs programmatically and plots predicted vs actual Xpts per GW, calibration curves, and fold-level metric variance:

```python
import mlflow
runs = mlflow.search_runs(experiment_names=["fpl_validation"])
# plot mean_mae across model versions, fold variance, etc.
```

---

## Key Design Principles

1. **Training and inference are decoupled.** `train.py` fits and serializes; `predict.py` loads by model name. They never share in-memory state.

2. **Serialization is orthogonal to the prediction interface.** MLflow pickles the model object's `__dict__`. The `predict()` and `predict_distribution()` methods exist for the pipeline, not for MLflow.

3. **Always integrate over the score distribution.** Never compute P(clean sheet) from E[goals]. The `ScoreDistribution` class enforces this by providing the correct integral accessors.

4. **The optimisation layer is fully separate from the ML layer.** The ILP takes Xpts per player as a flat input. It has no dependency on any model.

5. **Data files are GW- and date-stamped.** e.g. `fpl_gw31_20260423.parquet`. MLflow logs which data file was used in each training run. No DVC needed.

6. **Per-position models.** At minimum split GK from outfield. Reasonable to go GK / DEF / MID / FWD given the different point structures.

7. **Walk-forward cross-validation only.** Train on GWs 1–N, validate on N+1 onwards. Never shuffle time-series splits.

8. **Software tests and model validation are separate.** pytest tests correctness with synthetic data — fast, no real data, runs on every commit. MLflow validation tests model quality with historical data — slow, run when a model changes. Never put model quality assertions in pytest.