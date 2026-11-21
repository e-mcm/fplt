# fplt — Design Document

Authoritative reference for the architecture of `fplt`, a system that predicts Fantasy Premier League (FPL) points per player per gameweek and turns those predictions into squad, captaincy and transfer decisions.

Location in repo: `docs/design.md`. If code and this document disagree, either the code is wrong or this document must be updated in the same PR.

---

## 1. Goals

The system must:

1. **Compose models.** A layered predictor is built from components (team score model, minutes model, goal/assist involvement model, other-points model) that are swappable independently.
2. **Expose one interface.** Every predictor — layered, all-in-one, baseline or external — implements the same `PointsPredictor` contract, so any of them can be trained, backtested, compared and fed to the optimiser.
3. **Be versioned and reproducible.** Any prediction can be traced to the exact code, config, features and data that produced it, and re-run.
4. **Run automatically.** After each gameweek: ingest, refit, evaluate, predict, re-optimise. Before each deadline: refresh availability, re-predict, re-optimise.
5. **Benchmark easily.** Compare any set of predictors (internal or external, e.g. OpenFPL) on identical walk-forward backtests, at prediction level and at decision level.
6. **Validate data visibly.** Schema and domain checks at every stage boundary, visible in a UI.

Non-goals (for now): mini-league rival modelling, real-time in-match updates. Gated opt-in apply of a saved plan to FPL (`fplt apply`, dry-run by default, `--confirm` to POST) is supported; Dagster sensors never auto-submit.

---

## 2. Core principles

1. **Point-in-time (PIT) everything.** Every `fit` and `predict` takes an `as_of` gameweek and may only see information available at that gameweek's deadline. This is enforced by the data access layer (§4.4), not by convention.
2. **Two-level model interface.** `PointsPredictor` at the top (what the backtester and optimiser see); typed component protocols underneath (what composite predictors are built from). See §5.
3. **Simulation is the composition glue.** Layered predictors combine components by Monte Carlo simulation of match scenarios, not by hand-derived closed-form integrals. See §5.6.
4. **Raw data is immutable.** Raw API responses are written once and never edited. Everything downstream is derived and can be rebuilt.
5. **Schemas are contracts.** Every table crossing a module boundary has a `pandera.polars` schema. No bare DataFrames between stages.
6. **Scoring rules are data, keyed by season.** FPL rules change between seasons; backtests must apply the rules of the season being tested.
7. **Library first, orchestrator second.** All logic lives in plain Python functions in the `fplt` package and runs without Dagster (tests, notebooks, CLI). Dagster assets are thin wrappers that call library functions and record metadata.
8. **Determinism.** All randomness uses seeded generators derived from (model_id, as_of, fixture_id). Same inputs → identical outputs.

---

## 3. Technology stack

| Concern | Choice | Notes |
|---|---|---|
| Environment & deps | `uv` | `pyproject.toml` + committed `uv.lock` |
| Dataframe engine | `polars` | pandas only inside external adapters if a dependency forces it |
| Storage | Parquet (derived), gzipped JSON (raw API responses) | DuckDB for ad hoc SQL over parquet |
| Schemas / validation | `pandera.polars` | Wired into Dagster asset checks via a small factory (§8.4) |
| Configs | `pydantic` models, loaded from YAML | `pydantic-settings` for environment settings. No Hydra. |
| Orchestration, lineage, validation UI | Dagster (OSS, run locally) | Webserver + daemon as launchd services (§8.6) |
| Score model | `scipy` | Dixon-Coles via MLE |
| ML models | `lightgbm` | |
| Optimiser | `highspy` (HiGHS) | |
| Experiment comparison | Parquet backtest ledger + Streamlit leaderboard app | No MLflow initially (§7.5) |
| Tests | `pytest`, `hypothesis`, `pytest-cov` | Coverage Gutters in VS Code reads `coverage.xml` |
| Lint / types | `ruff` (lint + format), `pyright` | |
| Task tracking | GitHub Issues + milestones (§11) | |

---

## 4. Data layer

### 4.1 Sources

| Source | Use | Notes |
|---|---|---|
| FPL API `bootstrap-static/` | Players, teams, gameweeks (deadlines, `finished`, `data_checked`), prices, status, news, `ep_next` | Overwritten continuously — must be snapshotted |
| FPL API `fixtures/` | Fixtures, kickoff times, results, per-fixture stats | |
| FPL API `event/{gw}/live/` | Per-player GW stats and the `explain` points breakdown per fixture | Ground truth for actual points |
| FPL API `element-summary/{id}/` | Per-player history and upcoming fixtures | Optional; heavy (one call per player) |
| FPL API `entry/{id}/...` | The user's squad, picks, transfers, history | Public finished-GW picks only; live GW is 403 without auth |
| FPL API `my-team/{id}/` | Live squad with selling prices, bank, FT limit; lineup POST | Requires `x-api-authorization: Bearer` (`FPLT_FPL_API_TOKEN`); `pl_profile` (`FPLT_FPL_SESSION`) is legacy. See `fplt.data.fpl_client` |
| FPL API `transfers/` (POST) | Submit paired element in/out transfers | Unofficial write API; gated via `fplt apply --confirm` only |
| vaastav/Fantasy-Premier-League | Historical seasons | No per-deadline availability snapshots — see §4.5 |
| Understat | xG / xA / shots / key passes / xGChain / xGBuildup per player per match | Staged as `understat_player_matches`. `player_key` is null until the Understat id is mapped. Team-level Deep/PPDA for OpenFPL samples still need match-stats scrape (not in `getMatchData`) |

### 4.2 Identifiers

- `season`: string, e.g. `"2026-27"`.
- `gw`: int 1–38. `GWKey = (season, gw)`.
- **FPL element `id` and team `id` change between seasons.** Use the element `code` and team `code` (stable across seasons) as the cross-season keys: `player_key`, `team_key`. Keep the season-specific `element_id` alongside for API joins.
- `fixture_id`: FPL fixture id, unique within a season; use `(season, fixture_id)` globally.
- Predictions are made at **player–fixture** grain and aggregated to player–GW. This handles double and blank gameweeks naturally.

### 4.3 Storage layout

All data lives under `$FPLT_DATA_DIR` (default `./data`, gitignored).

```
data/
├── raw/                         # immutable, write-once
│   ├── fpl_api/season=2026-27/
│   │   └── captured_at=2026-11-21T08-00-00Z/
│   │       ├── bootstrap_static.json.gz
│   │       └── fixtures.json.gz
│   │   └── event_live/gw=12/captured_at=.../event_live.json.gz
│   └── understat/season=2026-27/
│       ├── dates/captured_at=.../dates.json.gz
│       └── match_id=<id>/match.json.gz   # getMatchData AJAX payload
├── staged/                      # typed parquet, one table per entity, derived from raw
│   ├── players/  teams/  fixtures/  gameweeks/
│   ├── player_status/           # status/news/price/ep_next per snapshot (time series)
│   ├── player_fixture_stats/    # actuals from event_live
│   └── understat_player_matches/
├── features/<feature_name>/v=<version>/season=.../gw=.../part.parquet
├── artifacts/<model_id>/as_of=<season>_gw<gw>/   # fitted model files + spec.json
├── predictions/<model_id>/as_of=<season>_gw<gw>/part.parquet
├── ledger/                      # backtest ledger (§7.2)
└── plans/<model_id>/as_of=.../  # optimiser outputs
```

Raw is stored as gzipped JSON rather than parquet because it is lossless against API schema changes; the first transform produces typed parquet.

Understat ids are joined to FPL codes via `configs/understat/id_maps.yaml` (`player_key` / `team_key`). Unmapped players stage with a null `player_key`; unmapped teams are rejected. `scripts/fetch_understat.py` (and `.github/workflows/understat.yml`) write match JSON onto the `snapshots` branch; the Dagster `understat_ingest` asset stages them with `available_at = kickoff + 3h`.

### 4.4 Point-in-time access (`PITData`)

Every staged table carries an `available_at` timestamp (UTC):

- Snapshotted attributes (status, news, price, `ep_next`): `available_at = captured_at`.
- Match results and stats: `available_at` = time the GW was marked `data_checked` (or kickoff + 3h as a historical proxy).
- Fixture schedule: `available_at` = earliest snapshot in which the fixture appears with that date.

`PITData` is the only way models read data:

```python
class PITData:
    def __init__(self, store: DataStore) -> None: ...
    def deadline(self, gw: GWKey) -> datetime: ...
    def view(self, as_of: GWKey) -> "PITView": ...

class PITView:
    as_of: GWKey
    cutoff: datetime                       # deadline of as_of
    def table(self, name: str) -> pl.LazyFrame:
        """Rows with available_at <= cutoff only. For time-series tables,
        also offers latest-as-of semantics via latest(name, by=...)."""
    def latest(self, name: str, by: list[str]) -> pl.LazyFrame: ...
```

Models receive a `PITView`, never a `DataStore`. The leakage test (§9.2) verifies this end to end.

### 4.5 Snapshot capture

Availability and news cannot be recovered after the fact, so capture runs independently of the local Dagster instance:

- A standalone script `scripts/snapshot.py` (stdlib + `httpx` only) runs hourly on **GitHub Actions cron**. It snapshots `bootstrap-static` and `fixtures` if (a) the next deadline is within 6 hours, or (b) the last snapshot is more than 12 hours old. After a GW is `data_checked` it also captures `event/{gw}/live`.
- Snapshots are committed to an orphan branch `snapshots` of this repo (path layout as `data/raw/fpl_api/...`).
- Local Dagster has a sensor that fetches the `snapshots` branch and ingests new files into `data/raw`.

Historical seasons (vaastav) lack per-deadline status/news/chance snapshots. Mark their provenance (`source = "vaastav"`) and treat availability features for those seasons as degraded; component backtests of the minutes model should report results separately for snapshot-backed seasons. Per-gameweek prices are staged from vaastav `merged_gw.csv` `value` into `player_status.price` with `available_at` at that gameweek's deadline so historical transfer budgets bind.

### 4.6 Schemas

All schemas live in `src/fplt/schemas/`, one module per domain (`staged.py`, `features.py`, `predictions.py`, `ledger.py`, `plans.py`). Example:

```python
import pandera.polars as pa
import polars as pl

class PredictionSchema(pa.DataFrameModel):
    model_id: str
    as_of_season: str
    as_of_gw: int = pa.Field(ge=1, le=38)
    target_season: str
    target_gw: int = pa.Field(ge=1, le=38)
    player_key: int
    fixture_id: int
    xpts: float = pa.Field(ge=-10, le=40)
    xpts_sd: float = pa.Field(ge=0, nullable=True)
    p_haul: float = pa.Field(ge=0, le=1, nullable=True)   # P(points >= 10)
    p_60: float = pa.Field(ge=0, le=1, nullable=True)     # P(minutes >= 60)

    class Config:
        strict = True
        unique = ["model_id", "as_of_season", "as_of_gw", "player_key", "fixture_id"]
```

---

## 5. Model architecture

### 5.1 Top-level contract: `PointsPredictor`

```python
class PointsPredictor(Protocol):
    spec: ModelSpec

    def fit(self, view: PITView) -> None:
        """Fit on data visible at view.as_of. No-op for pre-trained/external models."""

    def predict(self, view: PITView, horizon: Sequence[GWKey]) -> pl.DataFrame:
        """Return rows conforming to PredictionSchema for every player-fixture
        in the horizon. Must only use data from `view`."""

    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path, spec: ModelSpec) -> Self: ...
```

Optionally, predictors that can produce scenario samples implement:

```python
    def sample(self, view: PITView, horizon: Sequence[GWKey], n: int) -> pl.DataFrame:
        """Long frame: player_key, fixture_id, sim, points. Used by the optimiser
        for captaincy and risk-aware choices."""
```

Implementations:

| Kind | Examples |
|---|---|
| `CompositePredictor` | Dixon-Coles + minutes + involvement + other-points, combined by simulation |
| Direct predictors | LightGBM on total points per player-fixture |
| Baselines | Rolling mean points; FPL `ep_next` (from snapshots) |
| External | OpenFPL adapter (§7.4) |

### 5.2 Model identity: `ModelSpec`

```python
class ModelSpec(BaseModel):
    name: str                      # human label, e.g. "dc_layered_v1"
    kind: str                      # registry key of the implementing class
    params: dict[str, Any]
    components: dict[str, "ModelSpec"] = {}   # for composites
    features: list[str] = []       # "name@version"
    train_window: TrainWindow      # e.g. seasons back, time-decay
    refit: Literal["every_gw", "every_n_gw", "never"] = "every_gw"
    seed: int = 0

    @property
    def model_id(self) -> str:
        """Short stable hash of the canonical JSON of this spec + package version.
        Changing any param, component, feature version or window changes the id."""
```

- Specs live in `configs/models/*.yaml` and are loaded into a registry at import time.
- Implementing classes register with `@register_model("kind")`.
- Artifacts are written to `artifacts/<model_id>/as_of=<...>/` with `spec.json` and `git_sha.txt` alongside. Nothing is ever overwritten.

### 5.3 Component contracts

All components take a `PITView` and a context frame describing what to predict.

```python
class TeamScoreModel(Protocol):
    def fit(self, view: PITView) -> None: ...
    def predict(self, view: PITView, fixtures: pl.DataFrame) -> ScorelineDist: ...

class MinutesModel(Protocol):
    def fit(self, view: PITView) -> None: ...
    def predict(self, view: PITView, player_fixtures: pl.DataFrame) -> MinutesDist: ...

class InvolvementModel(Protocol):
    def fit(self, view: PITView) -> None: ...
    def predict(self, view: PITView, player_fixtures: pl.DataFrame) -> pl.DataFrame:
        """InvolvementSchema: player_key, fixture_id, goal_weight, assist_weight —
        per-90 relative weights within the player's team."""

class OtherPointsModel(Protocol):
    def fit(self, view: PITView) -> None: ...
    def predict(self, view: PITView, player_fixtures: pl.DataFrame) -> pl.DataFrame:
        """OtherPointsSchema: expected bonus, saves, cards, defensive-contribution
        and other points, conditional on minutes bucket (0, 1-59, 60+)."""
```

Distribution objects:

```python
class ScorelineDist:
    fixture_ids: np.ndarray            # (F,)
    pmf: np.ndarray                    # (F, G+1, G+1); pmf[i, h, a] = P(home=h, away=a)
    def sample(self, rng, n: int) -> np.ndarray      # (F, n, 2) ints
    def p_clean_sheet(self, side: Literal["home", "away"]) -> np.ndarray
    def marginal(self, side) -> np.ndarray           # (F, G+1)

class MinutesDist:
    keys: pl.DataFrame                 # player_key, fixture_id
    bins: np.ndarray                   # e.g. [0, 15, 30, 45, 60, 75, 90]
    pmf: np.ndarray                    # (P, len(bins))
    def sample(self, rng, n: int) -> np.ndarray      # (P, n) minutes
```

Truncate scorelines at `G` = 32 and renormalise; assert mass lost < 1e-6. Attack and defence are bounded by 3 before rescaling, and that rescaling keeps their product, so with home advantage 1.3 a rate can reach about 12. Thirty-two keeps that tail under the limit. The joint is computed past `G` so the discarded mass is the real tail, not just the last stored bin.

### 5.4 Initial component implementations

| Component | v1 implementation |
|---|---|
| `TeamScoreModel` | Dixon-Coles. ξ always comes from the model spec. v1 also fixes home advantage and ρ in YAML and identifies only attack/defence. v2 (`dixon_coles_v2`) jointly estimates ρ and/or home advantage when `fit_rho` / `fit_home_advantage` are set, and shrinks sparse or promoted-team ratings toward the league mean via `new_team_prior_strength` / `promoted_prior_strength`. ρ = 0 is the independent Poisson baseline. |
| `MinutesModel` | v1: P(start) from the last `window` appearances, status, and chance of playing. Minutes given a start or a substitute appearance come from a per-player histogram shrunk toward the position prior. Support is 0, 15, 30, 45, 60, 75, 90, and draws are independent across players. Rotation features are later |
| `InvolvementModel` | `involvement_shares_v1`: shrunk per-90 shares of team goals/assists; penalty-taker override. `involvement_xg_v1`: blend Understat xG/xA per-90 with FPL goals/assists (`xg_blend`), same penalty override; fits `own_goal_weight` and `p_assist` from visible history for the simulator |
| `OtherPointsModel` | v1: position-level priors by minutes bucket. `other_player_v1`: player-level means shrunk toward that prior, with a BPS→bonus proxy and defensive contribution when staged action counts are present; later GBM |

### 5.5 Scoring rules

`configs/scoring/<season>.yaml` holds the FPL scoring table for that season: points per goal/assist/clean sheet by position, appearance thresholds, goals-conceded penalties, saves, cards, defensive contributions (introduced 2025/26), bonus, penalties saved/missed, own goals. Fill values from the official rules for each season.

`fplt.scoring.Rules.for_season(season)` loads and validates them. `Rules.score` applies the table to one player's integer match stats. Clean sheets and goals conceded are the official counts (already gated by minutes and red cards), not a scoreline recomputation. Bonus is the published bonus multiplied by the season's `bonus` rate (1). Defensive contributions are empty before 2025-26. From 2025-26 a defender scores 2 points for 10 clearances, blocks, interceptions and tackles, and a midfielder or forward scores 2 for 12 of those plus recoveries. Points do not stack. A per-season test reconstructs actual FPL points from `event/{gw}/live` stats for a sample of player-GWs and asserts exact equality with the official totals.

### 5.6 Simulation engine (composition)

`CompositePredictor.predict` runs, per target fixture, `n_sims` scenarios (default 5,000):

1. **Scoreline**: draw (home_goals, away_goals) from `ScorelineDist`.
2. **Minutes**: draw minutes for each player in both squads from `MinutesDist`. Draws are independent across players in v1 (known simplification; a later version can enforce 11 starters per team).
3. **Goal allocation**: for each team goal, pick a scorer from players with minutes > 0 with probability ∝ `goal_weight × minutes/90`, plus an "unattributed" bucket for own goals (weight from history).
4. **Assist allocation**: each goal is assisted with team-level probability `p_assist`; the assister is drawn ∝ `assist_weight × minutes/90`, excluding the scorer.
5. **Clean sheet / conceded**: v1 approximation — clean sheet if the opponent scored 0 and the player played 60+; goals-conceded penalty applied scaled by minutes/90. Document the approximation in code. Optional fidelity params (defaults preserve v1): `cs_red_void_prob` voids a would-be clean sheet with that probability (red-card stand-in); `conceded_mode: played` applies the full goals-conceded band whenever minutes > 0 instead of scaling by minutes/90.
6. **Other points**: add the `OtherPointsModel` expectation for the player's drawn minutes bucket. With `sample_other: true`, draw Poisson / signed-Poisson noise around that mean so `xpts_sd` and `p_haul` see residual variance.
7. **Score**: apply `Rules` for the season to get points per player per sim.

Output: `xpts` = mean, `xpts_sd`, `p_haul`, `p_60`, plus optional sample frame. RNG seed = hash(model_id, as_of, fixture_id). Vectorise across sims with numpy; target < 2 s per GW for all fixtures. `dc_layered_v1` keeps the v1 defaults; a later layered spec (e.g. `dc_layered_v2`) enables the fidelity knobs.

A unit test checks simulation against analytics: mean simulated team goals equals the `ScorelineDist` expectation, and simulated P(clean sheet) equals `p_clean_sheet` within Monte Carlo error.

---

## 6. Features

### 6.1 Registry

```python
@feature(name="rolling_points", version=2, grain="player_fixture",
         inputs=["player_fixture_stats"])
def rolling_points(view: PITView, window: int = 5) -> pl.LazyFrame:
    ...
```

- `grain` is one of `player_fixture`, `team_fixture`, `player_gw`.
- A feature function receives a `PITView` and must use only its tables.
- **Changing a feature's logic requires bumping its version.** Old versions remain available until no spec references them.
- Feature outputs are materialised to `features/<name>/v=<version>/...` and validated by a per-feature schema (key columns + declared value columns).
- Models declare features as `name@version` in their `ModelSpec`; `build_feature_matrix(view, specs, horizon)` joins them onto the player-fixture rows for that horizon.

### 6.2 Initial feature set

Team-level: Dixon-Coles attack/defence ratings (as a feature for other models), rolling goals for/against, rest days, home/away.
Player-level: rolling minutes, starts share, rolling points, per-90 goal and assist involvement, status/chance_of_playing at deadline, price, position, penalty-taker flag, opponent strength. `understat_share` v1 is each mapped player's xG and xA per 90, and their share of team xG and xA, from matches visible at the deadline.

---

## 7. Backtesting, evaluation and benchmarks

### 7.1 Walk-forward protocol

For each `as_of` in the evaluation window: build `view = pit.view(as_of)`, `fit(view)` according to the spec's refit policy, `predict(view, horizon)` for horizons 1..H (default H = 5). Never shuffle. Training data for `as_of` is everything with `available_at <= cutoff(as_of)` within the spec's `train_window`. A holdout is `refit: never` from the first deadline of the test season, so that one fit is reused for every later gameweek. `train_window.seasons_back` chooses how many prior seasons that fit can see.

Fair multi-model compare (`fplt compare` / `make compare-models`) runs the same selected specs over a shared test window in two modes — walk-forward (`refit=every_gw`) and fixed parameters (`refit=never`, fit at `--fit-gw` which defaults to the start of the test window) — then writes MAE/RMSE/Spearman/top-10/Brier summary JSON + parquet under `$FPLT_DATA_DIR/compare/` for the leaderboard or a later Streamlit view.

### 7.2 Ledger

`ledger/predictions` — every backtest and live prediction, joined later to actuals:

| Column | |
|---|---|
| `model_id`, `run_kind` (`backtest` / `live`) | |
| `as_of_season`, `as_of_gw`, `target_season`, `target_gw`, `horizon` | |
| `player_key`, `position`, `fixture_id` | |
| `xpts`, `xpts_sd`, `p_haul`, `p_60` | |
| `actual_pts`, `actual_minutes` | filled when the target GW is `data_checked` |

`ledger/components` — one row per model, deadline, and fixture. Scorelines store `p_home`, `p_draw`, `p_away`, `expected_home`, `expected_away`, and, once the target gameweek is `data_checked`, `actual_home`, `actual_away`, and ranked probability score over home/draw/away. Minutes rows join this table later.
`ledger/decisions` — one row per model, gameweek, and player kept or sold that week: `position`, `in_xi`, `captain`, `vice`, `bought`, `sold`, `xpts`, `actual_pts`, `actual_minutes`, `hit`, and `free_transfers` available at the deadline. A re-run replaces rows for the same model and gameweek.

### 7.3 Metrics

Prediction level (by model × horizon × position × season):
- MAE, RMSE on points
- Spearman rank correlation within position per GW
- Top-k precision (were the top 10 predicted players in the actual top 10?)
- Calibration of `p_haul` and `p_60` (reliability curves, Brier score)

Decision level — what actually matters:
- Run the same optimiser over a season using each model's predictions, starting from an optimal GW1 squad, with simplified rules (free transfers, banking, -4 hits; chips off in v1). Selling price stays at purchase price (no FPL price-rise / 50% profit modelling). Realised scoring applies autosubs and vice-captain-on-blank, doubles the captain (or vice), and subtracts each gameweek's hit once. `fplt decisions` / `make decisions-board` prints the multi-model season-total scoreboard from `ledger/decisions`.

Component level (layered models):
- Scorelines: ranked probability score and log-loss vs a no-skill Poisson baseline
- Minutes: Brier score on P(played) and P(60+), reported by stats `source` (snapshot vs vaastav)
- Involvement: MAE of predicted team goal share vs realised share among players with minutes
- Monte Carlo: `n_sims` stability sweep (mean/max \|Δxpts\| vs a high-`n` reference, wall time, analytic SE)
- CLI: `fplt eval-components` / `make eval-components` (P0 harness)

### 7.4 Benchmarks

Internal: `rolling_mean` baseline, `fpl_ep_next` (from snapshots; only for snapshot-backed GWs), direct LightGBM, each layered variant.

External — **OpenFPL** (github.com/daniegr/OpenFPL, MIT):
- Position-specific ensembles trained on FPL + Understat data for 2020-21 to 2023-24. **Only compare on 2024-25 onwards**; earlier seasons are in its training data.
- Sample features matching OpenFPL's published `xscaler` schema (~228 columns) are built in `fplt.models.external.openfpl_samples` from PIT FPL + Understat tables. Team/opponent Deep and PPDA stay 0 until a match-stats scrape exists. See [`docs/openfpl.md`](openfpl.md).
- Runs in its own uv environment (`external/openfpl/`) as a subprocess. The adapter (`fplt.models.external.openfpl`) invokes it, reads its output, and joins each score back by request `row_id` (which already carries `player_key`). `web_name` is sent as a label and does not need to be unique. It returns a `PredictionSchema` frame. `fit` is a no-op. A target season before 2024-25 raises. `params.model_dir` is `external/openfpl/OpenFPL`. That ensemble is about 750MB (files near GitHub's 100MB limit), so the first real predict sparse-clones it there and `uv sync`s the runner env when `.venv` is missing. An empty `model_dir` or a checkout without `models/xscaler.save` raises; the runner does not substitute zeros. OpenFPL stays opt-in on `fplt compare` / `eval_board` (`--with-openfpl` / `--openfpl-model-dir` / a `model_dir` other than this default), not in `CORE_MODELS`. The contract suite drives the adapter with an offline echo runner.
- Operator steps (`uv sync`, season floor, feature gap): [`docs/openfpl.md`](openfpl.md).

### 7.5 Why no MLflow (yet)

The comparisons we need are slices over model × GW × horizon × position. A long parquet ledger queried with polars/DuckDB does this directly. Dagster already provides lineage, run history and per-materialisation metadata; `ModelSpec` hashing provides model identity and artifacts are content-addressed. Revisit MLflow only if large hyperparameter sweeps start.

### 7.6 Leaderboard app

`apps/leaderboard.py` (Streamlit), reading only the ledger: model leaderboard by metric, filters for season/horizon/position, calibration plots, per-GW decision-quality curves, and side-by-side player predictions for a chosen GW.

---

## 8. Orchestration (Dagster)

### 8.1 Layout

`src/fplt/orchestration/` contains only Dagster definitions; all logic is imported from library modules.

### 8.2 Partitions

`MultiPartitionsDefinition(season=StaticPartitionsDefinition([...]), gw=StaticPartitionsDefinition(["1".."38"]))`.

### 8.3 Assets

| Asset | Partitioned | Description |
|---|---|---|
| `raw_snapshots` | no (observable source) | Files fetched from the `snapshots` branch |
| `staged_*` (players, teams, fixtures, gameweeks, player_status, player_fixture_stats) | season-gw | Typed parquet from raw |
| `feature__<name>__v<version>` | no (as-of via run config / `resolve_as_of`) | One asset per registered feature version (`orchestration/factories.py`) |
| `model__<name>` | no (same) | Fitted artifact per points-predictor `ModelSpec` (component YAML skipped) |
| `predictions__<name>` | no (same) | Predictions for horizon 1..H (default 5); CLI/catch-up stay authoritative |
| `ledger_predictions` | season-gw | Appends predictions; joins actuals when available |
| `squad_plan` | season-gw | Optimiser output for the chosen production model |
| `recommendation_report` | season-gw | Markdown summary attached as Dagster metadata |

Factories mean adding a YAML spec or a new feature version makes a new asset appear in the UI without new orchestration code. Asset `code_version` = feature version / `model_id`, so stale assets are flagged.

### 8.4 Asset checks (data validation UI)

- A factory `pandera_check(asset, Schema, blocking=True)` validates the materialised frame with `pandera.polars` and returns failure cases as metadata. (`dagster-pandera` targets pandas, hence the factory.)
- Domain checks, e.g.: team goals in `player_fixture_stats` equal the fixture score; every finished fixture has stats for ≥ 11 players per side; predictions have no NaN; sum of `xpts` per team per fixture is within plausible bounds; every player in the next GW has a prediction from the production model.
- Blocking checks stop downstream materialisation.

### 8.5 Automation

| Trigger | Type | Action |
|---|---|---|
| New files on `snapshots` branch | sensor (every 10 min) | Ingest into `data/raw`, materialise `staged_*` |
| GW `data_checked` and not yet processed | sensor (every 30 min) | Join actuals → ledger; refit models; predict next H GWs; re-optimise. Processes **all** missing GWs, so downtime self-heals. |
| Next deadline within 3 h and no pre-deadline run yet | sensor (every 15 min) | Re-stage latest snapshot; re-predict (no refit); re-optimise; emit report |
| Run failure | run-failure sensor | Notify (email or push via ntfy) |

The pre-deadline run can always be launched manually from the UI.

A checked gameweek is waiting until `rolling_mean_v1` has a plan parquet for that deadline. The catch-up sensor asks the `recommendation_report` asset to plan every waiting week, and the markdown is stored as asset metadata. After a successful catch-up or pre-deadline plan, the concatenated `report.md` body is pushed via `FPLT_NTFY_URL` and/or optional SMTP (`FPLT_SMTP_HOST`, `FPLT_SMTP_FROM`, `FPLT_SMTP_TO`, plus user/password/port/TLS) when those settings are present; missing config is a no-op. Run failures POST to `FPLT_NTFY_URL` when it is set.

### 8.6 Running locally

- Development: `make dagster-dev` (`dagster dev`).
- Always-on: `dagster-webserver` and `dagster-daemon` as launchd services with fixed `DAGSTER_HOME` and SQLite storage (templates in `deploy/launchd/`). Secrets (`FPLT_ENTRY_ID`, `FPLT_FPL_API_TOKEN`, `FPLT_NTFY_URL`, SMTP) load from `~/.config/fplt/env` via `run-with-env.sh` — never from the committed plists. Operator steps: `deploy/README.md`.
- A `deploy/docker-compose.yml` keeps the option of moving to a small VPS without code changes; the same `FPLT_*` vars are passed through from a gitignored `deploy/.env` or the shell.
- Multi-model eval (manual / `eval_board` Dagster job): `make eval-board SEASON=2024-25` runs walk-forward backtests for `rolling_mean_v1`, `ep_next_v1`, `dc_layered_v1`, and `lgbm_direct_v1` into the prediction ledger for `make leaderboard`. OpenFPL (`openfpl_v1`) is optional — include it with `--with-openfpl`, `--openfpl-model-dir`, or a `model_dir` other than the fetched default checkout, and only for seasons from 2024-25 onwards. Live `fplt predict` refuses to record an all-zero frame. Walk-forward backtests still record a cold start, such as a rolling mean before any appearances.
- Fair compare (CLI): `make compare-models SEASON=2024-25` / `uv run fplt compare …` runs the same core set under shared windows in walk-forward and fixed-parameter modes and writes `$FPLT_DATA_DIR/compare/.../summary.json` (see §7.1).
- Component eval (CLI): `make eval-components SEASON=2024-25` / `uv run fplt eval-components …` walk-forwards scoreline RPS/log-loss (Dixon-Coles vs Poisson), minutes Brier on P(played) and P(≥60) sliced by `source`, involvement goal-share MAE, and an optional `n_sims` stability sweep for `dc_layered_v1`. Writes `$FPLT_DATA_DIR/component_eval/.../summary.json` (see §7.3).
- Operator logging: CLI and scripts call `fplt.utils.logging.configure_logging` (default INFO to stderr; `-v`/`--verbose` for DEBUG including FPL HTTP paths; `-q` for WARNING+). Library modules use `logging.getLogger(__name__)`.

---

## 9. Testing

### 9.1 Principles

- Unit and contract tests use synthetic data or recorded API fixtures under `tests/fixtures/`. **No network access in tests.**
- Never assert model quality (MAE etc.) in pytest; that is the backtester's job.
- `pytest -m "not slow"` must run in under 60 s.

### 9.2 Test suites

| Suite | What it checks |
|---|---|
| Contract suite (`tests/contracts/`) | Parametrised over every registered `PointsPredictor` and component kind: output conforms to schema, covers every player-fixture in the horizon, deterministic under fixed seed, save/load round-trips to identical predictions |
| **Leakage test** | For each predictor: predict at `as_of = n`; then perturb every row with `available_at > cutoff(n)`; predict again; assert identical output |
| Scoring rules | Per season, reconstruct official points from recorded `event/{gw}/live` fixtures exactly |
| Simulation | Hypothesis property tests: simulated marginals match `ScorelineDist`; allocated goals sum to team goals; no player assigned goals with 0 minutes |
| Distributions | pmf sums to 1; clean-sheet probabilities in [0, 1]; home/away axis conventions |
| Optimiser | Constraint satisfaction on small known problems: budget, 2/5/5/3, max 3 per club, valid formation, transfer accounting, hit costs |
| Features | Known input → known output; PIT respected |
| Integration | Synthetic mini-league (6 teams, 10 GWs) materialised end to end with Dagster's in-process `materialize()` |

### 9.3 Coverage

`pytest-cov` writes `coverage.xml` (for Coverage Gutters) and a terminal report. `fail_under = 70` initially; the target is meaningful coverage of contracts, scoring, simulation and optimiser, not a vanity number.

---

## 10. Optimiser

### 10.1 Inputs

- Predictions for horizon H from one predictor (expected points; optionally samples).
- Squad state: current 15, purchase/selling prices, bank, free transfers available, chips available. `fplt.optimise.decisions.resolve_squad_state` is the shared entry point: YAML draft by default; `entry_json` when provided; otherwise `fplt.data.fpl_client.load_current_team` → `apply_entry_picks` when `FPLT_ENTRY_ID` is set (authenticated `my-team/{id}` needs `FPLT_FPL_API_TOKEN`, the Bearer value from `x-api-authorization`; else public finished-GW picks, no selling prices). `run_plan` (and thus `make plan` / CLI / automation) calls this whenever a pre-built `squad` is not passed. FT from `my-team` replaces the YAML count when it fits the banking cap; chips stay in `configs/squad_state.yaml`.
- Rules for the season (squad limits, FT banking cap, hit cost).

### 10.2 Formulation (multi-period ILP)

Decision variables per GW t in horizon and player p: in squad, in XI, captain, vice, transferred in/out. Constraints: budget with selling prices, 2 GK / 5 DEF / 5 MID / 3 FWD, ≤ 3 per club, valid XI formation (1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD), FT accounting with banking cap and -4 per extra transfer. Objective: Σ_t decay^t × (XI points + captain bonus + small bench weight) − hit costs. Solve with HiGHS (`highspy`).

Library entry point: `fplt.optimise.optimize_transfers(candidates, expected_points, state, gameweeks=...)`. Predictions are opaque (`ExpectedPointsSchema`: `player_key`, `gw`, `xpts`); the engine does not call predictors or the data store. `fplt plan` is the CLI wrapper that loads candidates and ledger rows, then calls the same solver.

v1: single-GW XI + captain. v2: multi-period transfers. v3: chips. Captaincy can use `p_haul` / samples as a tiebreak.

### 10.3 Output

`optimize_transfers` returns a `TransferPlan` (first-week transfers in/out, XI, captain, vice, hit, plus full-horizon picks). Persistence uses `PlanSchema` parquet at `data/plans/<model_id>/as_of=<season>_gwNN_hHH/part.parquet`, one row per player kept or sold in each horizon week (`target_gw`, `in_xi`, `captain`, `vice`, `bought`, `sold`, `xpts`, `hit`), plus a markdown report in the same directory. `<model_id>` is the model-spec hash. `hHH` is the horizon actually covered, capped at gameweek 38. The report labels players with the staged `web_name`, and falls back to `player_key` when that name is missing. The same model, deadline, and horizon are never overwritten. A repeat run still prints the transfer summary, then reports that the parquet already exists. `fplt plan` raises before solving when this deadline has no ledger rows, when a horizon week is missing or all zero, or when a week covers fewer than half the candidates. Missing rows are not treated as zero.

`fplt apply` (library: `fplt.optimise.apply.apply_plan`) maps step-0 buys/sells onto paired FPL transfer rows and the planned XI/captain onto a lineup body. Dry-run is the default; `--confirm` POSTs with `FPLT_FPL_API_TOKEN`. Initial drafts (15 buys, no sells) are refused. Automation never calls this path.

Squad state lives in `configs/squad_state.yaml`: bank, free transfers, the banking cap, hit cost, decay, a bench weight below 0.01, and owned selling prices. An empty squad is an initial draft. `resolve_squad_state` (and `fplt plan --entry-json`) replace owned players and bank from an entry picks payload, mapping element ids through staged players. `fplt plan --rebuild` clears ownership after that resolve and sets `bank` to today's available funds (`bank + Σ selling_price`), so the optimiser drafts fifteen from scratch under the current budget (wildcard-style; no transfer hits). Needs selling prices from authenticated `my-team` or `--entry-json`. The decision backtest ignores that squad and starts from an optimal draft with a 1000 budget. Chips stay off.

---

## 11. Repository layout

```
fplt/
├── AGENTS.md  CLAUDE.md  README.md
├── pyproject.toml  uv.lock  Makefile
├── .github/
│   ├── ISSUE_TEMPLATE/task.md
│   └── workflows/ci.yml  snapshot.yml
├── configs/
│   ├── models/*.yaml            # ModelSpecs
│   ├── scoring/<season>.yaml    # FPL rules per season
│   └── squad_state.yaml         # optimiser overrides
├── docs/design.md
├── scripts/snapshot.py          # standalone, runs on GitHub Actions
├── src/fplt/
│   ├── settings.py              # pydantic-settings (paths, FPL entry id, FPL session)
│   ├── types.py                 # GWKey, enums
│   ├── data/                    # FPL client, raw writer, staging, DataStore, PITData
│   ├── schemas/                 # pandera.polars schemas
│   ├── features/                # registry + feature definitions
│   ├── scoring/                 # Rules
│   ├── models/
│   │   ├── contracts.py         # PointsPredictor + component protocols
│   │   ├── spec.py              # ModelSpec, registry
│   │   ├── distributions.py     # ScorelineDist, MinutesDist
│   │   ├── simulation.py        # composition engine
│   │   ├── composite.py
│   │   ├── components/{score,minutes,involvement,other}/
│   │   ├── direct/              # all-in-one predictors
│   │   ├── baselines/
│   │   └── external/openfpl.py
│   ├── backtest/                # walk-forward, ledger, metrics, decision sim
│   ├── optimise/
│   ├── orchestration/           # Dagster definitions, factories, sensors, checks
│   └── cli.py                   # `fplt backtest`, `fplt compare`, `fplt predict`, ...
├── apps/leaderboard.py
├── external/openfpl/            # isolated env; ensemble fetched into OpenFPL/
├── deploy/{launchd/, docker-compose.yml}
├── notebooks/
└── tests/{unit/, contracts/, integration/, fixtures/}
```

---

## 12. Build plan

Milestones map to GitHub milestones; each bullet becomes one or more issues.

| Milestone | Scope | Done when |
|---|---|---|
| **M0 Scaffold** | uv project, ruff, pyright, pytest + cov, Makefile, CI workflow, settings, package skeleton | `make check` passes in CI |
| **M1 Snapshots** | `scripts/snapshot.py`, `snapshot.yml` workflow, `snapshots` branch | Snapshots appear hourly near deadlines; **ship first — missed snapshots are unrecoverable** |
| **M2 Data layer** | Raw ingest, staging tables, `available_at`, `DataStore`, `PITData`, schemas, vaastav import, Dagster data assets + checks | Staged tables for ≥ 3 seasons + current season validate in the Dagster UI |
| **M3 Evaluation spine** | Contracts, `ModelSpec`, registry, baselines (rolling mean, ep_next), walk-forward backtester, ledger, metrics, contract + leakage tests, leaderboard v0 | Two baselines on the leaderboard |
| **M4 Score model** | Dixon-Coles, `ScorelineDist`, component ledger and diagnostics | DC beats Poisson baseline on RPS |
| **M5 Layered predictor** | Scoring rules + reconstruction tests, minutes, involvement, other-points, simulation engine, `CompositePredictor` | Layered model on the leaderboard |
| **M6 Optimiser** | Single-GW XI/captain, then multi-period transfers; decision-quality backtest | Decision ledger populated for all models |
| **M7 Automation** | Sensors, pre-deadline run, failure notifications, launchd, recommendation report | One full GW cycle runs unattended |
| **M8 External benchmarks** | Understat ingestion, OpenFPL adapter | OpenFPL on the leaderboard for 2024-25+ |
| **M9 Direct models** | LightGBM all-in-one predictor | Layered vs direct comparison available |
| **M10 Cleanup** | Delete Hydra/MLflow/pandas legacy stack (`pipeline/`, `predict/`, `optimisation/`, old model packages, `conf/`, …) once the new path is self-contained | No imports of deleted modules; `make check` green without those deps |

---

## 13. Decisions

These are settled. Change them here before changing the code.

1. **Training window.** Train from the 2021-22 season onward.
2. **Snapshot storage.** An orphan git branch named `snapshots`. Object storage (for example Cloudflare R2) is only worth revisiting if that branch grows unwieldy.
3. **Leaderboard.** Streamlit (`apps/leaderboard.py`).
4. **Experiment tracking.** No MLflow. The parquet ledger, Dagster metadata, and `ModelSpec` hashing cover comparison and lineage (§7.5).
5. **Minutes model.** v1 draws each player's minutes independently. Enforcing 11 starters per team is a later version.
