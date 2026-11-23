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
| FPL API `bootstrap-static/` | Players, teams, gameweeks (deadlines, `finished`, `data_checked`), prices, status, news, `ep_next`, `penalties_order` | Overwritten continuously — must be snapshotted |
| FPL API `fixtures/` | Fixtures, kickoff times, results, per-fixture stats | |
| FPL API `event/{gw}/live/` | Per-player GW stats (incl. Opta `expected_goals` / `expected_assists` / `expected_goal_involvements` / `expected_goals_conceded`) and the `explain` points breakdown per fixture | Ground truth for actual points; xG fields since 2022/23 |
| FPL API `element-summary/{id}/` | Per-player history and upcoming fixtures | Optional; heavy (one call per player) |
| FPL API `entry/{id}/...` | The user's squad, picks, transfers, history | Public finished-GW picks only; live GW is 403 without auth |
| FPL API `my-team/{id}/` | Live squad with selling prices, bank, FT limit; lineup POST | Requires `x-api-authorization: Bearer` (`FPLT_FPL_API_TOKEN`, ~8h JWT from browser `oidc.user` localStorage). Prefer `fplt auth from-browser` → `~/.config/fplt/env` — never the repo `.env`. `pl_profile` cookies are legacy. See `fplt.browser_auth` / `fplt.auth` |
| FPL API `transfers/` (POST) | Submit paired element in/out transfers | Unofficial write API; gated via `fplt apply --confirm` only |
| vaastav/Fantasy-Premier-League | Historical seasons | No per-deadline availability snapshots — see §4.5 |
| Understat | xG / xA / shots / key passes / xGChain / xGBuildup per player per match; team Deep / PPDA / xG per match | Player rows in `understat_player_matches` (`player_key` null until mapped). Team Deep/PPDA/xG in `understat_team_matches` from league-page `teamsData` (not `getMatchData`) |
| football-data.co.uk E0 CSV | Closing + scheduled-prematch 1X2 (+ O/U 2.5) → staged `match_odds` | Free history. `odds_kind=closing_proxy` (`available_at=kickoff`) and `odds_kind=prematch_scheduled` (non-closing columns; conservative Fri/Tue 17:00 London collection). **Closing ≠ FPL deadline.** Prematch needs the deadline coverage gate when deadline &lt; collection. Team names → `team_key` via `configs/football_data/team_names.yaml` |
| API-Football (API-Sports) `/injuries` | Structured injury / suspension rows → staged `player_injuries` | Write-once raw under `raw/api_football/...`; `available_at = captured_at`. Player/team ids → FPL codes via `configs/api_football/id_maps.yaml` (unmapped `player_key` null). Auth: `FPLT_API_FOOTBALL_KEY` (fetch also accepts `API_FOOTBALL_KEY` / `APISPORTS_KEY`). Free tier ~100 req/day — capture skips if &lt;4h old. Minutes overlay via `use_injury_overlay` (default off; fork `empirical_minutes_v2_injuries`) |
| Touchline Wire `/api/v1/lineups` (experimental) | Predicted XI + bench → staged `predicted_lineups` | Zero-auth JSON; MIT-with-attribution. **Flagged** (`FPLT_TOUCHLINE_LINEUPS_ENABLED`); rate-limited; measure vs official XIs via `score_predicted_starts` before promoting. Not a production SoR until accuracy is audited. Optional API-Football `/fixtures/lineups` stages into the same table under `raw/api_football/lineups/` (maps land separately from injuries). **No FFS HTML / X scrapes.** |

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
│   │   └── overall_rank/captured_at=.../points_at_rank.json.gz  # Overall league points at key ranks (§7.8)
│   ├── understat/season=2026-27/
│   │   ├── dates/captured_at=.../dates.json.gz
│   │   ├── teams/captured_at=.../teams.json.gz  # league teamsData (Deep/PPDA)
│   │   └── match_id=<id>/match.json.gz   # getMatchData AJAX payload
│   ├── football_data/season=2024-25/
│   │   ├── E0.csv                 # optional legacy once-per-season path
│   │   └── captured_at=2025-11-01/E0.csv  # monthly GHA write-once refreshes
│   ├── touchline/season=…/captured_at=…/lineups.json.gz   # experimental predicted XI
│   └── api_football/season=2026-27/
│       ├── injuries/captured_at=.../injuries.json.gz
│       └── lineups/captured_at=…/   # optional predicted XIs (separate from injuries)
├── staged/                      # typed parquet, one table per entity, derived from raw
│   ├── players/  teams/  fixtures/  gameweeks/
│   ├── player_status/           # status/news/news_added/price/ep_next/chance this+next/penalties_order per snapshot
│   ├── player_fixture_stats/    # actuals from event_live (+ Opta expected_*)
│   ├── understat_player_matches/
│   ├── understat_team_matches/  # Deep / PPDA / team xG per match
│   ├── player_injuries/         # API-Football injuries; available_at=captured_at
│   ├── match_odds/              # closing_proxy (+ prematch_scheduled); see §4.4
│   └── predicted_lineups/       # near-deadline XI/bench/out overrides (PIT)
├── features/<feature_name>/v=<version>/season=.../gw=.../part.parquet
├── artifacts/<model_id>/as_of=<season>_gw<gw>/   # fitted model files + spec.json
├── predictions/<model_id>/as_of=<season>_gw<gw>/part.parquet
├── benchmark_projections/as_of=<season>_gw<gw>/part.parquet  # N1 deadline archives (§7.5)
├── archive_compare/<stamp>/     # N2 paired board vs archives (§7.5)
├── ledger/                      # backtest ledger (§7.2)
└── plans/<model_id>/as_of=.../  # optimiser outputs
```

Raw is stored as gzipped JSON rather than parquet because it is lossless against API schema changes; the first transform produces typed parquet. football-data E0 files are plain CSV (write-once) as published upstream.

Understat ids are joined to FPL codes via `configs/understat/id_maps.yaml` (`player_key` / `team_key`). Unmapped players stage with a null `player_key`; unmapped teams are rejected. `scripts/fetch_understat.py` (and `.github/workflows/understat.yml`) write match JSON plus league `teamsData` onto the `snapshots` branch; the Dagster `understat_ingest` asset stages player and team tables with `available_at = kickoff + 3h`.

football-data team names map to `team_key` via `configs/football_data/team_names.yaml`. `scripts/fetch_football_data.py` (and monthly `.github/workflows/football_data.yml`, 1st of month 06:00 UTC + manual) writes write-once E0 CSVs onto the `snapshots` branch under `raw/football_data/season=…/captured_at=YYYY-MM-DD/E0.csv` (local `make fetch-football-data` may still use the legacy `season=…/E0.csv` path). The Dagster `match_odds_sensor` (30 min) requests `match_odds_ingest` when a new E0 CSV appears under raw after `snapshots_sensor` copies it. Staging writes `match_odds` with both `odds_kind=closing_proxy` (`available_at=kickoff`; closing lines are pre-KO, **not** deadline PIT) and `odds_kind=prematch_scheduled` from non-closing columns only (`AvgH`/`B365H`/…, never relabelled closes) with a conservative collection `available_at` (Friday 17:00 London for Fri–Sun kickoffs; Tuesday 17:00 London for Mon–Thu). Unique key includes `odds_kind` so both rows coexist; later `captured_at` files win on key collision. Optional join fills `fixture_id` from staged FPL fixtures on season + teams + match date. Prematch boards must run `check_prematch_deadline_coverage` / `require_prematch_deadline_coverage` so fixtures whose FPL deadline is before that stamp drop out clearly.

API-Football player/team ids map via `configs/api_football/id_maps.yaml`. `scripts/fetch_api_football.py` (and `.github/workflows/api_football.yml`) writes league-season `/injuries` JSON onto the `snapshots` branch under `raw/api_football/...`; the Dagster `api_football_ingest` asset stages `player_injuries` with `available_at = captured_at`. Unmapped players keep a null `player_key`. Minutes components may set `use_injury_overlay: true` to soft-downgrade (doubtful) or zero (suspension/missing) when FPL tags still look available — never raising an FPL zero. Default remains off so production `dc_layered_v2` is unchanged until eval.

### 4.4 Point-in-time access (`PITData`)

Every staged table carries an `available_at` timestamp (UTC):

- Snapshotted attributes (status, news, `news_added`, price, `ep_next`, chance this/next, `penalties_order`): `available_at = captured_at`.
- Match results and stats: `available_at` = time the GW was marked `data_checked` (or kickoff + 3h as a historical proxy).
- Fixture schedule: `available_at` = earliest snapshot in which the fixture appears with that kickoff/gw/teams identity. FPL API staging stamps each capture at `captured_at` (no synthetic near-kickoff backdate — that would leak late rearrangements). Vaastav has no mid-season fixture archive, so schedule rows use `season_start` (1 July) as the earliest-known proxy and results stay at kickoff + 3h; rearrangements that exist only in the final `fixtures.csv` can appear early — prefer FPL API snapshot history when present. Mid-season deadlines then see season-ahead GWs (enough for H=8–10 planning) while PIT filtering still hides unpublished rearrangements from later snapshots.
- Closing match odds (`match_odds`, `odds_kind=closing_proxy`): `available_at = kickoff`. Bookmaker close is pre-KO, not the FPL deadline.
- Scheduled prematch odds (`match_odds`, `odds_kind=prematch_scheduled`): `available_at` = conservative football-data collection estimate (Fri/Tue 17:00 London). Still not automatically deadline-legal — gate on `available_at <= deadline`.
- API-Football injuries (`player_injuries`): `available_at = captured_at` of the raw snapshot.

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
    def closing_proxy_match_odds(self) -> pl.LazyFrame:
        """match_odds with odds_kind=closing_proxy, ignoring cutoff.
        Closing lines are stamped available_at=kickoff — labelled probe only,
        not deadline-safe PIT. Season windows from within_seasons still apply."""
```

Models receive a `PITView`, never a `DataStore`. The leakage test (§9.2) verifies this end to end.

### 4.5 Snapshot capture

Availability and news cannot be recovered after the fact, so capture runs independently of the local Dagster instance:

- A standalone script `scripts/snapshot.py` (stdlib + `httpx` only) runs every **15 minutes** on **GitHub Actions cron**, throttling writes: every 15 minutes in the last 3 hours before a deadline, every hour in the 3–6 hour window, otherwise when the last snapshot is more than 12 hours old. After a GW is `data_checked` it also captures `event/{gw}/live`.
- Snapshots are committed to an orphan branch `snapshots` of this repo (path layout as `data/raw/fpl_api/...`).
- Local Dagster has a sensor that fetches the `snapshots` branch and ingests new files into `data/raw`.

Historical seasons (vaastav) lack per-deadline status/news/`news_added`/chance/`penalties_order` snapshots. Mark their provenance (`source = "vaastav"`) and treat availability features for those seasons as degraded; component backtests of the minutes model should report results separately for snapshot-backed seasons (`make minutes-brier` / `fplt eval-components --minutes-only`, read `slice=source=fpl_api`). Fixture schedules are still staged at `season_start` so long-horizon planning works without inventing near-kickoff visibility. Per-gameweek prices are staged from vaastav `merged_gw.csv` `value` into `player_status.price` with `available_at` at that gameweek's deadline so historical transfer budgets bind. `penalties_order`, both chance fields, and `news_added` are null for vaastav rows.

After restaging fixtures with the season-start / snapshot stamps, an H=8 chip board can be re-run without flipping `DEFAULT_PLAN_HORIZON` (still 3):

```bash
# Restage if the store still has kickoff−14d fixture stamps
uv run python -c "from fplt.data.store import DataStore; from fplt.data.vaastav import import_available; from fplt.settings import get_settings; s=get_settings(); print(import_available(DataStore(s.data_dir), s.data_dir))"
# Prefer FPL API snapshots when present:
# uv run python -c "from fplt.data.raw import ingest_raw; from fplt.data.store import DataStore; from fplt.settings import get_settings; s=get_settings(); print(ingest_raw(DataStore(s.data_dir), s.data_dir/'raw'))"

rm -f data/ledger/predictions/part.parquet
uv run fplt backtest --models dc_layered_v2 --season 2025-26 --from-gw 1 --to-gw 38 --steps 8
uv run fplt decisions --chip-board --season 2025-26 --horizon 8 \
  --models dc_layered_v2 --from-gw 1 --to-gw 38
```

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
    p_played: float = pa.Field(ge=0, le=1, nullable=True)  # P(minutes > 0)
    p_60: float = pa.Field(ge=0, le=1, nullable=True)      # P(minutes >= 60)
    p_goal: float = pa.Field(ge=0, le=1, nullable=True)    # P(goals >= 1)
    p_assist: float = pa.Field(ge=0, le=1, nullable=True)  # P(assists >= 1)
    p_cs: float = pa.Field(ge=0, le=1, nullable=True)      # P(clean sheet)
    p_haul: float = pa.Field(ge=0, le=1, nullable=True)   # P(points >= 10)

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
| Forecast combination | `forecast_combine`: convex blend of member PointsPredictors (`params.members` expanded into `components` at load). Equal weights or fixed per-position weights. OpenFPL omitted until Deep/PPDA. |
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
    train_window: TrainWindow      # seasons_back + optional time_decay
    refit: Literal["every_gw", "every_n_gw", "never"] = "every_gw"
    seed: int = 0

    @property
    def model_id(self) -> str:
        """Short stable hash of the canonical JSON of this spec + package version.
        Changing any param, component, feature version or window changes the id."""
```

`TrainWindow.time_decay` is an optional calendar rate λ (inverse days). Trainers that honour it weight each training row by `exp(-λ × days_before_as_of)` relative to the PIT cutoff (`PITView.cutoff`). Default `null` keeps equal weights so bakeoffs stay unchanged until a sweep. **LightGBM (`lgbm_direct`) applies it as LightGBM sample weights.** Dixon–Coles keeps its own `xi` (days before latest finished kickoff in the fit) and **ignores** `TrainWindow.time_decay` — do not set both intending a single decay. Minutes / involvement / other layered fits still use equal weights within their last-N (or unweighted logistic) windows; wiring decay there is a follow-up. Override via YAML `train_window.time_decay` or `fplt backtest/compare --time-decay`.

- Specs live in `configs/models/*.yaml` and are loaded into a registry at import time.
- A fork may set `extends: <parent_name>` with an `overrides` mapping. The loader deep-merges the override dict onto a copy of the parent YAML (one parent only; no chains, no OmegaConf), then validates with `ModelSpec`. `extends` / `overrides` are loader keys, not `ModelSpec` fields. Resolved `model_id` must match an equivalent full-copy YAML.
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
| `TeamScoreModel` | Dixon-Coles. ξ (`xi`) always comes from the model spec and is the only DC calendar decay — `TrainWindow.time_decay` is ignored here. v1 also fixes home advantage and ρ in YAML and identifies only attack/defence. v2 (`dixon_coles_v2`) jointly estimates ρ and/or home advantage when `fit_rho` / `fit_home_advantage` are set, and shrinks sparse or promoted-team ratings toward the league mean via `new_team_prior_strength` / `promoted_prior_strength`. ρ = 0 is the independent Poisson baseline. Odds-prior specs (`dixon_coles_odds_v1`, used by `dc_layered_odds_v1`) set `market_prior_strength` > 0 to shrink per-match λ toward PIT `match_odds` via `odds_to_lambda` (missing odds → pure DC). History rating prior (`dixon_coles_history_v1` / `dc_layered_history_v1`) sets `market_prior_source: history` so `fit` maps past PIT-visible closes with `odds_to_lambda`, fits market ratings `(a0, d0)` under the same ξ weights, and adds a Gaussian penalty on attack and defence only (HA and ρ unpenalised; absent source keeps the predict-time λ blend). The 1X2-region fork (`dixon_coles_1x2_v1` / `dc_layered_1x2_v1`) sets `market_1x2_strength` > 0 to rescale the joint's home/draw/away regions toward de-vigged bookmaker 1X2 while keeping DC's conditional scoreline shape (the right *target* vs the λ prior when blending match outcomes); that arm uses `use_closing_proxy: true` (closing oracle). The deadline-PIT redesign (`dixon_coles_1x2_prematch_v1` / `dc_layered_1x2_prematch_v1`) keeps the same strength but sets `use_closing_proxy: false` so only `prematch_scheduled` rows are visible (EvalSpec `prematch_1x2_prior_ab`). The CS-only fork (`dixon_coles_cs_v1` / `dc_layered_cs_v1`) sets `market_cs_strength` > 0 (prefer strengths below 1) to blend clean-sheet / zero-goal margins toward market λ then restore the joint's 1X2 — λ prior and 1X2 strengths stay off; `use_closing_proxy: false` so deadline-legal `prematch_scheduled` rows are used under strict PIT. Vanilla `dixon_coles_v2` / `dc_layered_v2` keep all market strengths off (production default). Closing-proxy odds use `available_at=kickoff`, so they are **not** deadline-visible under strict PIT. `odds_1x2_v1` de-vigs 1X2 directly (no λ fit) onto a neutral scoreline shape; `odds_poisson_v1` still uses `odds_to_lambda` then independent Poisson. Both may read kickoff-stamped closing lines via `PITView.closing_proxy_match_odds` when `use_closing_proxy` is set (labelled **closing_proxy**, not deadline PIT). |
| `MinutesModel` | v1: P(start) from the last `window` appearances, status, and chance of playing. Minutes given a start or a substitute appearance come from a per-player histogram shrunk toward the position prior. Support is 0, 15, 30, 45, 60, 75, 90, and draws are independent across players. v2 (`empirical_minutes_v2`): multi-window start shares, logistic P(start) / P(≥60\|start) with home/rest/DGW/price/opponent context, histogram residuals, and optional soft projection of expected starters toward ~11 per team-fixture. Logistic P(start) blends with the multi-window empirical rate via `logistic_blend` (default 0.5). FPL status/chance/news gating is on by default (`use_availability_gate: true`; eval-only off-fork `dc_layered_noavail_v1`). Optional `use_injury_overlay` (default off; forks `empirical_minutes_v2_injuries` / `dc_layered_injuries_v1`) applies staged `player_injuries` as a soft/hard downgrade that never raises an FPL zero. Optional `predicted_xi` (default off; fork `dc_layered_predicted_xi_v1`) mixes P(start) toward staged `predicted_lineups` near kickoff |
| `InvolvementModel` | `involvement_shares_v1`: shrunk per-90 shares of team goals/assists; penalty-taker override from latest PIT `penalties_order == 1`, falling back to `penalties_missed` when order is null (vaastav); opt out with `use_penalties_order: false` for A/B (miss-only). `involvement_xg_v1`: blend Opta/FPL-native `expected_goals`/`expected_assists` (falling back to Understat xG/xA) per-90 with FPL goals/assists (`xg_blend`), same penalty override; fits `own_goal_weight` and `p_assist` from visible history for the simulator |
| `OtherPointsModel` | v1: position-level priors by minutes bucket. `other_player_v1`: player-level means shrunk toward that prior, with a BPS→bonus proxy and defensive contribution from staged `defensive_contribution` when the season rules award it (no-op before 2025-26; opt out with `use_defensive_contribution: false` for A/B). `defcon_mode: rate` (`other_player_defcon_rate_v1`) replaces the shrunk 0/2 DC prior with Poisson `P(award \| minutes, position, rate)` from actions-per-90; **production `dc_layered_v2` uses rate** after FDB-211 keep (opt-out fork `dc_layered_defcon_mean_v1`). Later GBM |

### 5.5 Scoring rules

`configs/scoring/<season>.yaml` holds the FPL scoring table for that season: points per goal/assist/clean sheet by position, appearance thresholds, goals-conceded penalties, saves, cards, defensive contributions (introduced 2025/26), bonus, penalties saved/missed, own goals. Fill values from the official rules for each season.

`fplt.scoring.Rules.for_season(season)` loads and validates them. `Rules.score` applies the table to one player's integer match stats. Clean sheets and goals conceded are the official counts (already gated by minutes and red cards), not a scoreline recomputation. Bonus is the published bonus multiplied by the season's `bonus` rate (1). Defensive contributions are empty before 2025-26. From 2025-26 a defender scores 2 points for 10 clearances, blocks, interceptions and tackles, and a midfielder or forward scores 2 for 12 of those plus recoveries. Points do not stack. A per-season test reconstructs actual FPL points from `event/{gw}/live` stats for a sample of player-GWs and asserts exact equality with the official totals.

### 5.5.1 Chip rules

`configs/chips/<season>.yaml` holds the FPL chip allotment for that season (alongside scoring configs; never hardcode season rules in Python). Each file lists:

- `one_chip_per_gameweek` (always true under current FPL rules)
- `free_hit_keeps_free_transfers` (Free Hit preserves banked free transfers)
- `windows`: named closed gameweek intervals with `carries_over` (first-half chips that expire at a deadline set `carries_over: false`)
- `grants`: one entry per chip use (`wildcard` / `freehit` / `bboost` / `3xc`), each tied to a window

From 2025-26 there are two of each chip: a first window through GW19 that does not carry, and a second from GW20–38. Earlier seasons keep two wildcards (one per half) and one each of Free Hit, Bench Boost, and Triple Captain for the season. `fplt.scoring.ChipRules.for_season(season)` loads and validates them. Squad remaining/played counts live in `configs/squad_state.yaml` (`SquadState.chips`). When `plan_squad` is given `season` and `gameweeks`, the MILP uses all four chips under those rules (one chip per GW, half-season caps from windows + remaining inventory). Wildcard suppresses hits that week and resets the next free-transfer count to 1. Free Hit fields a temporary squad/XI/captain within selling value of the permanent squad plus bank; step `t+1` continues from the pre-FH squad; free-transfer banking passes through when `free_hit_keeps_free_transfers` is true.

Optional `ChipUseWindows` (`allowed` / `forced` / `no_chip`, mirroring FPL-Optimization-Tools) further constrain chip binaries by gameweek. `schedule_chip_windows` (Wave C5) fills them from a coarse half-season model: fixture doubles/blanks, team-level expected points, and squad continuity, with an **option-value** cut keyed by `ChipGrant` (chip × window id — never a bare chip name) so play-now only beats the best remaining opportunity later in the **same** half; first- and second-half scores are never mixed. Remaining unused chips are **force-packed** onto the last eligible gameweeks of each half-window (one chip per GW) when the window `stop_event` is inside the short horizon or remaining chips outnumber remaining weeks, so first-half grants cannot expire unused; mid-window deadlines stay optional. Pass `force_play=False` to `schedule_chip_windows` / `simulate_decisions` / `run_plan` / `run_chip_arm_sim_season` (CLI: `fplt decisions --chip-board --no-force-play`, `fplt plan --no-force-play`, `fplt sim-season --no-force-play`) to disable packing for ablations while keeping the option-value cut. Live / automation still default to force-play on; do not flip the default from an ablation. The rule-based baseline still forces BB / TC / FH / WC onto double and blank landmarks. `plan_squad(..., chip_windows=..., chip_rules_season=...)` applies those hooks; live `run_plan` / automation and the optimised decision arm rebuild scheduled windows each deadline. Each deadline's `ChipUseWindows` are logged to `ledger/chip_windows` and surfaced by `fplt decisions --explain-gw`. The three-arm chip decision board (`fplt decisions --chip-board` / `make chip-decision-board`) compares chips-off, rule-based landmarks, and optimised MILP chips, scoring **season points** under the chosen chip-rules year. Running 2024-25 fixtures under 2025/26 chip rules is labelled `2024-25-under-2025/26-rules` (counterfactual). Decision-ledger rows carry a nullable `chip` column; realised scoring applies Triple Captain ×3 and Bench Boost bench points.

### 5.6 Simulation engine (composition)

`CompositePredictor.predict` runs, per target fixture, `n_sims` scenarios (default 5,000):

1. **Scoreline**: draw (home_goals, away_goals) from `ScorelineDist`.
2. **Minutes**: draw minutes for each player in both squads from `MinutesDist`. Draws are independent across players in v1. v2 may soft-project expected starters toward ~11 in the PMF. When `hard_starters: true` (on `dc_layered_v2`), `enforce_hard_starters` adjusts each team-fixture sim toward exactly 11 starters (minutes ≥ 60), promoting/demoting by P(≥60) weight, with exactly one GK among those starters when any GK is eligible.
3. **Goal allocation**: for each team goal, pick a scorer from players with minutes > 0 with probability ∝ `goal_weight × minutes/90`, plus an "unattributed" bucket for own goals (weight from history).
4. **Assist allocation**: each goal is assisted with team-level probability `p_assist`; the assister is drawn ∝ `assist_weight × minutes/90`, excluding the scorer.
5. **Clean sheet / conceded**: v1 approximation — clean sheet if the opponent scored 0 and the player played 60+; goals-conceded penalty applied scaled by minutes/90. Document the approximation in code. Optional fidelity params (defaults preserve v1): `cs_red_void_prob` voids a would-be clean sheet with that probability (red-card stand-in); `conceded_mode: played` applies the full goals-conceded band whenever minutes > 0 instead of scaling by minutes/90.
6. **Other points**: add the `OtherPointsModel` expectation for the player's drawn minutes bucket. With `sample_other: true`, draw Poisson / signed-Poisson noise around that mean so `xpts_sd` and `p_haul` see residual variance.
7. **Score**: apply `Rules` for the season to get points per player per sim.

Output: `xpts` = mean, `xpts_sd`, event probs (`p_played`, `p_60`, `p_goal`, `p_assist`, `p_cs`, `p_haul`), plus optional sample frame. RNG seed = hash(model_id, as_of, fixture_id). Vectorise across sims with numpy; target < 2 s per GW for all fixtures. `dc_layered_v1` keeps the v1 defaults. `dc_layered_v2` composes `dixon_coles_v2`, `empirical_minutes_v2`, `involvement_xg_v1`, and `other_player_defcon_rate_v1` (`defcon_mode: rate`) with fidelity knobs on (`cs_red_void_prob`, `conceded_mode: played`, `sample_other`) and `n_sims: 10000`. `dc_layered_odds_v1` is the same stack with `dixon_coles_odds_v1` (`market_prior_strength: 1.0`); `dc_layered_history_v1` uses `dixon_coles_history_v1` (`market_prior_source: history`); `dc_layered_1x2_v1` uses `dixon_coles_1x2_v1` (`market_1x2_strength: 1.0`, `use_closing_proxy: true`); `dc_layered_1x2_prematch_v1` uses `dixon_coles_1x2_prematch_v1` (`market_1x2_strength: 1.0`, `use_closing_proxy: false`); `dc_layered_cs_v1` uses `dixon_coles_cs_v1` (`market_cs_strength: 0.5`; closing proxy left at the off default). None change the production default `dc_layered_v2`.

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
Player-level: rolling minutes, starts share, rolling points, per-90 goal and assist involvement, status/chance (this + next) at deadline, `availability@1` (news flags + gate), price, position, penalty-taker flag, opponent strength. `understat_share` v1 is each mapped player's xG and xA per 90, and their share of team xG and xA, from matches visible at the deadline.

---

## 7. Backtesting, evaluation and benchmarks

### 7.1 Walk-forward protocol

For each `as_of` in the evaluation window: build `view = pit.view(as_of)`, `fit(view)` according to the spec's refit policy, `predict(view, horizon)` for horizons 1..H (default H = 10 via `DEFAULT_PROJECTION_HORIZON`, so H=8–10 planning has projections without raising `DEFAULT_PLAN_HORIZON`). Never shuffle. Training data for `as_of` is everything with `available_at <= cutoff(as_of)` within the spec's `train_window`. A holdout is `refit: never` from the first deadline of the test season, so that one fit is reused for every later gameweek. `train_window.seasons_back` chooses how many prior seasons that fit can see. Optional `train_window.time_decay` down-weights older rows inside that window for trainers that implement it (see §5.2); it does not expand or shrink the season set.

Fair multi-model compare (`fplt compare` / `make compare-models`) runs the same selected specs over a shared test window in two modes — walk-forward (`refit=every_gw`) and fixed parameters (`refit=never`, fit at `--fit-gw` which defaults to the start of the test window) — then writes RMSE (headline) / MAE (secondary) / Spearman / mean per-GW top-10 / event Brier+log summary JSON + parquet, plus calibration tables by xPts decile and position, under `$FPLT_DATA_DIR/compare/`, scored on the decision-relevant player pool (see §7.3).

### 7.2 Ledger

`ledger/predictions` — every backtest and live prediction, joined later to actuals:

| Column | |
|---|---|
| `model_id`, `run_kind` (`backtest` / `live`) | |
| `as_of_season`, `as_of_gw`, `target_season`, `target_gw`, `horizon` | |
| `player_key`, `position`, `fixture_id` | |
| `xpts`, `xpts_sd` | |
| `p_played`, `p_60`, `p_goal`, `p_assist`, `p_cs`, `p_haul` | nullable event probs; layered Monte Carlo fills them from the same sims; baselines leave null |
| `actual_pts`, `actual_minutes`, `actual_goals`, `actual_assists`, `actual_cs` | filled when the target GW is `data_checked` |

`ledger/components` — one row per model, deadline, and fixture. Scorelines store `p_home`, `p_draw`, `p_away`, `expected_home`, `expected_away`, and, once the target gameweek is `data_checked`, `actual_home`, `actual_away`, and ranked probability score over home/draw/away. Minutes rows join this table later.
`ledger/decisions` — one row per model, gameweek, and player kept or sold that week: `position`, `in_xi`, `captain`, `vice`, `bought`, `sold`, `xpts`, `actual_pts`, `actual_minutes`, `hit`, and `free_transfers` available at the deadline. A re-run replaces rows for the same model and gameweek.
`ledger/chip_windows` — one row per model, season, and deadline with the JSON-serialized `allowed` / `forced` / `no_chip` maps from that deadline's `ChipUseWindows` (written by `run_decisions` / `run_plan` when chips are on; shown by `--explain-gw`).

### 7.3 Metrics

Prediction level (by model × horizon × position × season), on the **decision-relevant player pool** by default for compare / promote-eval / bakeoff (as-of price ≥ £4.5m and either price ≥ £7.0m, ≥ 90 prior season minutes, or as_of GW ≤ 2; PIT via `available_at ≤ deadline` — see `fplt.backtest.decision_pool`):
- **RMSE** is the headline point metric for mean-seeking / expected-points boards (what the optimiser consumes). **MAE** stays as a secondary continuity metric so older A/Bs remain readable; do not delete it. When MAE and RMSE disagree (W9-style), prefer the RMSE side for mean-seeking calls.
- Spearman rank correlation within position per GW
- Top-k precision: mean of per-gameweek top-10 precision (not a single pooled top-10 over the window)
- Event Brier and log score on played, 60+, goal, assist, clean sheet, haul whenever the arm emits those probs
- Calibration tables by xPts decile and by position (`calibration.parquet` / `summary.md` from `fplt compare`; promote-eval recommendation embeds a slice)

Decision level — what actually matters:
- Run the same optimiser over a season using each model's predictions, starting from an optimal GW1 squad, with simplified rules (free transfers, banking, -4 hits; chips off by default). Selling prices follow FPL's 50% sell-on (`fplt.optimise.prices.fpl_selling_price`): sticky `purchase_price` on `OwnedPlayer`, refreshed each deadline from as-of staged `player_status.price` (historical actuals in backtests; live `my-team` supplies purchase/selling). Multi-step plan horizons may vary buy costs and owned sell credits by step via `plan_squad(..., price_forecast=)` (`PriceForecastSchema` / `fplt.optimise.price_forecast`) from **deadline-visible** history only (step 0 must match as-of). Live `run_plan` / automation defaults to `mean_gw_delta` (`DEFAULT_LIVE_PRICE_FORECAST_METHOD`, FDB-218 keep); decision boards still omit the forecast for control A/B (`--price-forecast mean_gw_delta` for treatment). Realised scoring applies autosubs and vice-captain-on-blank, doubles the captain (or vice; Triple Captain ×3; Bench Boost adds bench points), and subtracts each gameweek's hit once. Actuals are loaded from the prediction ledger **deduplicated per fixture** before summing a gameweek (walk-forward `steps>1` re-states the same fixture under later `as_of` rows; summing those raw rows double-counts). `fplt decisions` / `make decisions-board` prints the multi-model season-total scoreboard from `ledger/decisions`, fails by default when mean pts/GW exceeds 90, and supports `fplt decisions --explain-gw` / `make decisions-explain` for per-GW XI arithmetic. `fplt decisions --chip-board` / `make chip-decision-board` runs the three-arm chip board (off / rule-based / optimised).
- **Explainability (forecast board + transfer deltas).** Prefer explaining *decisions* over model internals: `fplt explain` / `make explain` (and the sections appended to `plans/.../report.md` by `fplt plan`) render (1) a per-player forecast board — first-order xPts split from ledger event probs × `Rules` (appearance / goals / assists / CS, with residual absorbing GC / other / multi-event mass; opaque models show xPts with null components) and (2) counterfactual transfer deltas vs keep-and-line-up the current fifteen (`select_squad` on the owned pool; captain doubled; hit subtracted). No natural-language transfer narratives; no MILP duals/shadow prices required. Does not change `PRODUCTION_MODELS`.
- **Share of attainable points** (`fplt.optimise.attainable`): a constrained hindsight oracle runs the same decision sim with realised points as planner `xpts` (same FT banking, hits, budget, 50% sell-on selling prices; chips **off** by default for the N5 board). Pass `chip_arm="optimised"` / `force_play` / fixtures into `simulate_hindsight_oracle` / `run_hindsight_oracle` for a chips-on ceiling (`hindsight_oracle_v1::chip_optimised`); `actuals_from_staged` builds actuals without a prediction ledger. Share = realised ÷ oracle. Component regret covers captaincy (best armband in the chosen XI), XI/bench order (best lineup from the fifteen vs chosen XI with optimal captain), transfers (H=1 hindsight action from the pre-transfer squad), and chip timing (nullable hook while chips are off). Board hook: `fplt decisions --with-oracle` enriches the scoreboard via `enrich_decision_summary` (still chips-off).
- **Forecast × optimiser 2×2 swap (N4).** Separate forecast quality from optimiser quality: `(ours | benchmark projections) × (ours | reference optimiser)`. Library: `fplt.backtest.forecast_optimiser_swap.run_forecast_optimiser_swap` over opaque `SwapProjectionSchema` frames (no live FPL Review). Season totals land in `SeasonPointsSchema` (nullable component-regret columns filled by N5 / `fplt.optimise.attainable`). The reference arm today is `IdentityReferenceOptimiser` — the same MILP with frozen hit/decay/banking/bench knobs — because FPL-Optimization-Tools / open-fpl-solver is a CLI/CSV script tree, not an installable season-sim library for `SeasonOptimiser`. `FplOptimizationToolsAdapter` is a documented stub. Does not change `PRODUCTION_MODELS`.
- **Simulated-season decision evaluation (N6).** Decision differences need standard errors: draw many outcome paths from a held-out truth intensity (or reuse pre-drawn sim actuals), run each policy on the **same** path (common random numbers), and report Monte Carlo SE on paired diffs for season points and optional share of attainable (N5). Library: `fplt.backtest.simulated_season.run_simulated_season_eval` / `paired_policy_se`. Chip arms reuse the C6 board contract via `DecisionPolicy.chip_arm` / `force_play` / optional per-policy `horizon` / `chip_arm_policies` / `run_chip_arm_sim_season` (off / rule_based / optimised + scheduler windows); CLI `fplt sim-season` / `make sim-season-smoke` for short CRN smokes (`--no-force-play` matches the chip-board ablation). Horizon × force-play CRN pairs (e.g. H=3 force-on vs H=8 force-off) use two `DecisionPolicy` rows with distinct `policy_id`s in one eval. Reuses `simulate_decisions` and the hindsight oracle; does not change `PRODUCTION_MODELS`, `DEFAULT_PLAN_HORIZON`, or scrape FPL Review. Required before trusting chips / horizon / scenario-lineup claims. Luck bands for live percentile (N7) also come from this variance.
- **Ensemble sensitivity / move frequency (Wave D2).** Shared-squad solve of the current bakeoff pair (`dc_layered_v2` control vs `lgbm_direct_v1` by default): each deadline plans once per forecast member from the same pre-transfer state, then ownership advances on the **control** plan. Report week-1 agree rates (transfers / XI / captain / vice) and **move frequency** (`1 − week1_agree_rate`) so optimiser thrashing that is forecast-driven is visible. Library: `fplt.optimise.ensemble_sensitivity`; CLI `fplt ensemble-sensitivity` / `make ensemble-sensitivity`. Reuses `Week1Actions` from the pruning-loss harness. Chips off; does not change `PRODUCTION_MODELS` or the MILP.

Component level (layered models):
- Scorelines: ranked probability score and log-loss vs a no-skill Poisson baseline
- Minutes: Brier score on P(played) and P(60+), reported by stats `source` (snapshot vs vaastav)
- Involvement: MAE of predicted team goal share vs realised share among players with minutes
- Monte Carlo: `n_sims` stability sweep (mean/max \|Δxpts\| vs a high-`n` reference, wall time, analytic SE)
- **Skill by horizon (Wave C4 / M9):** player xPts MAE/RMSE on the decision-relevant pool for lead times H=1…N (`DEFAULT_SKILL_STEPS` = 10). Reuses `compare_models` / `overall_summary`. CLI: `fplt eval-components --points-models … --steps 10` or `fplt horizon-skill` / `make horizon-skill`. Does **not** flip `DEFAULT_PLAN_HORIZON`; chips-on H retune is a season-points decision board (FDB-120).
- CLI: `fplt eval-components` / `make eval-components` (P0 harness)

### 7.4 Promote-eval (control vs treatment)

Any lever — new staged data, feature version, hyperparam / `train_window` fork, layered **component registry swap** (e.g. minutes empirical → LightGBM / handwritten), or full **predictor fork** (e.g. `lgbm_direct` → XGBoost) — is two `ModelSpec` YAMLs plus an EvalSpec under `configs/evals/`. CLI: `fplt promote-eval --eval <name>` (dry-run plans; `--execute` runs preflight + compare, and when `suites.components` is on names the two `components.<slot>.name` specs on `eval-components` with `--skip-sims` unless `n_sims` is the lever). Scoreline levers also write `per_match.parquet` with a raw de-vig 1X2 market row (`odds_poisson` is not that book). Full rules: [`docs/feature-eval-pipeline.md`](feature-eval-pipeline.md).

Control is the off state and treatment is the on state. Production stays whichever of those it already is; fork the other arm, and never edit that production YAML to manufacture the contrast. `use_penalties_order` and `use_defensive_contribution` default on, so the control is `dc_layered_nopens_v1` / `dc_layered_nodefcon_v1` and the treatment is `dc_layered_v2`. `market_prior_strength` defaults to 0, so the control is `dc_layered_v2` and the treatment is `dc_layered_odds_v1`. That landed blend is a recorded arm. `market_cs_strength` also defaults to 0: control `dc_layered_v2`, treatment `dc_layered_cs_v1` (CS-only; EvalSpec `cs_only_market_lambda_ab` gates on `coverage.read: prematch_scheduled`). `market_1x2_strength` defaults to 0 with a closing-proxy default when on: control `dc_layered_v2`, prematch treatment `dc_layered_1x2_prematch_v1` (EvalSpec `prematch_1x2_prior_ab` gates on `coverage.read: prematch_scheduled`); the closing-oracle `dc_layered_1x2_v1` is a recorded arm, not a deadline board. History rating prior (`market_prior_source: history`) remains a separate recorded arm. `odds_poisson` is a model column, not that book. Lockfile upgrades and no-behaviour refactors are equivalence checks (prediction identity, then an interval), not improvement A/Bs. Forks gain a one-parent `extends`. Coverage is the model's real PIT read (`available_at <= deadline`). A post-deadline feature on a fixture being scored fails the run. Closing odds (`available_at = kickoff`) are a labelled side rate; `prematch_scheduled` coverage uses the football-data deadline gate. Every run writes `audit.json` (git SHA, lockfile hash, data hashes, resolved spec). `--allow-dirty` forces `inconclusive`. A recommendation does not flip `PRODUCTION_MODELS`. Decision bakeoffs are opt-in and are not required for these config promos. The metric loop calls `compare` / `eval-components` / `decisions`. Paired intervals use `statsmodels` and `arch` (§7.6).

**Multi-arm bakeoff (championship ranking).** For several ModelSpecs on one shared window — a dose of one knob via EvalSpec `arms:`, or a prediction championship via `fplt bakeoff` / `BakeoffSpec` — reuse the same `compare` produce path and ledger cache; report SPA/MCS and an **RMSE** rank table (MAE still in the metric table). That readout is descriptive and is not a `keep` (confirmation needs a fresh window). Decision season totals stay on `fplt decisions`. Do not invent a second eval framework.

### 7.5 Benchmarks

Internal: `rolling_mean` baseline, `fpl_ep_next` (from snapshots; only for snapshot-backed GWs), direct LightGBM, each layered variant.

**Deadline archive (`benchmark_projections`).** Canonical N1 table for prospective paired boards. Schema `BenchmarkProjectionSchema` (`source` ∈ {`ep_next`, `openfpl`, `fpl_review`}, `player_key`, `fixture_id`, `xpts`, `available_at` strictly before the deadline, optional `quality_flag`). Write-once parquet at `benchmark_projections/as_of=<season>_gwNN/part.parquet`. Pre-deadline automation (`pre_deadline` + Dagster `benchmark_archive`) saves `ep_next` and OpenFPL each GW and fails loudly if a required source is missing. OpenFPL rows without staged Deep/PPDA are labelled `quality_flag=openfpl_degraded`. FPL Review enters only via manual CSV/JSON drop (`fplt ingest-review` / `fplt.data.fpl_review`) after a terms check — **no scrape**; do not invent a second table. Ops: [`docs/benchmark-archives.md`](benchmark-archives.md), [`docs/fpl_review.md`](fpl_review.md).

**Paired archive compare (N2).** `fplt archive-compare` / `fplt.backtest.archive_compare.compare_to_archives` joins a candidate ledger arm to archived sources on identical player–fixtures. Coverage gate: only rows where every compared arm has a pre-deadline snapshot (`available_at` strictly before the deadline for archives; ledger presence for ours). Default board: RMSE headline, MAE secondary, Spearman by position, OpenFPL groups (zeros / blanks / tickers / haulers). Paired significance reuses Diebold–Mariano / HAC + stationary bootstrap (`fplt.backtest.intervals`). Does not scrape FPL Review.

External — **OpenFPL** (github.com/daniegr/OpenFPL, MIT):
- Position-specific ensembles trained on FPL + Understat data for 2020-21 to 2023-24. **Only compare on 2024-25 onwards**; earlier seasons are in its training data.
- Sample features matching OpenFPL's published `xscaler` schema (~228 columns) are built in `fplt.models.external.openfpl_samples` from PIT FPL + Understat tables (`understat_team_matches` supplies Deep/PPDA). See [`docs/openfpl.md`](openfpl.md).
- Runs in its own uv environment (`external/openfpl/`) as a subprocess. The adapter (`fplt.models.external.openfpl`) invokes it, reads its output, and joins each score back by request `row_id` (which already carries `player_key`). `web_name` is sent as a label and does not need to be unique. It returns a `PredictionSchema` frame. `fit` is a no-op. A target season before 2024-25 raises. `params.model_dir` is `external/openfpl/OpenFPL`. That ensemble is about 750MB (files near GitHub's 100MB limit), so the first real predict sparse-clones it there and `uv sync`s the runner env when `.venv` is missing. An empty `model_dir` or a checkout without `models/xscaler.save` raises; the runner does not substitute zeros. OpenFPL stays opt-in on `fplt compare` / `eval_board` (`--with-openfpl` / `--openfpl-model-dir` / a `model_dir` other than this default), not in `CORE_MODELS`. The contract suite drives the adapter with an offline echo runner.
- Operator steps (`uv sync`, season floor, feature gap): [`docs/openfpl.md`](openfpl.md).

### 7.6 Eval libraries

Prefer an existing library when it already does about 90% of the job. For promote-eval, none does. [`mlflow.models.evaluate`](https://mlflow.org/docs/latest/ml/evaluation/) is the closest: one fitted pyfunc, one static frame, optional `baseline_model`, and `MetricThreshold`. Deepchecks' `ModelComparisonSuite` is the same shape (fitted scikit-learn models, one test set, MAE/RMSE/R²). Neither walks gameweeks, refits on a deadline, nor knows coverage or `available_at`. The walk-forward, the slices, and the component metrics already live in `fplt compare`, `eval-components`, and `fplt decisions`; the runner calls those. The ledger stays the metric store. The walk-forward stays ours. The paired test does not: Diebold–Mariano / Giacomini–White via `statsmodels` (OLS, HAC) and a stationary bootstrap, SPA, and the model confidence set via `arch`. Those two dependencies are the statistics layer in [`docs/feature-eval-pipeline.md`](feature-eval-pipeline.md) §9. They are not a runner and not a second store. A scoreline-lab design that tracked runs in local MLflow was discarded; its follow-up is folded into that doc (§17) as equivalence levers, `extends`, and the history rating prior. M10 removed the `mlflow` dependency; do not add it back for a scalar UI. Do not add OmegaConf or `joblib` for forks or the ledger cache.

### 7.7 Leaderboard app

`apps/leaderboard.py` (Streamlit), reading only the ledger: model leaderboard by metric, filters for season/horizon/position, calibration plots, per-GW decision-quality curves, and side-by-side player predictions for a chosen GW.

### 7.8 Live percentile (overall-league points-at-rank)

Season rank alone is a noisy live check (evaluation north star §3). Cache Overall classic-league (id ``314``) points totals at key ranks (at least 10k and 100k; defaults also sample 1, 1k, 500k, 1M) via ``fplt.data.overall_rank`` / ``make fetch-overall-rank``. Convert any season total to an equivalent rank and top-fraction percentile with ``fplt.backtest.live_percentile``; roll up with ``mean_percentile`` across seasons. **Single-season live rank is not a promote gate** — promote-eval stays on paired forecast/decision metrics; use multi-season percentile only as a consistency check. Luck bands from simulator variance belong with simulated-season evaluation, not this ladder.

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
| `staged_*` (players, teams, fixtures, gameweeks, player_status, player_fixture_stats, player_injuries) | season-gw | Typed parquet from raw |
| `feature__<name>__v<version>` | no (as-of via run config / `resolve_as_of`) | One asset per registered feature version (`orchestration/factories.py`) |
| `model__<name>` | no (same) | Fitted artifact per points-predictor `ModelSpec` (component YAML skipped) |
| `predictions__<name>` | no (same) | Predictions for horizon 1..H (default 10 via `DEFAULT_PROJECTION_HORIZON`); CLI/catch-up stay authoritative |
| `ledger_predictions` | season-gw | Appends predictions; joins actuals when available |
| `squad_plan` | season-gw | Optimiser output for the chosen production model |
| `recommendation_report` | season-gw | Markdown summary attached as Dagster metadata |
| `benchmark_archive` | no (due GW via `due_gameweek`) | Write-once `benchmark_projections` for `ep_next` + OpenFPL (§7.5) |

Factories mean adding a YAML spec or a new feature version makes a new asset appear in the UI without new orchestration code. Asset `code_version` = feature version / `model_id`, so stale assets are flagged.

### 8.4 Asset checks (data validation UI)

- A factory `pandera_check(asset, Schema, blocking=True)` validates the materialised frame with `pandera.polars` and returns failure cases as metadata. (`dagster-pandera` targets pandas, hence the factory.)
- Domain checks, e.g.: team goals in `player_fixture_stats` equal the fixture score; every finished fixture has stats for ≥ 11 players per side; predictions have no NaN; sum of `xpts` per team per fixture is within plausible bounds; every player in the next GW has a prediction from the production model.
- Blocking checks stop downstream materialisation.

### 8.5 Automation

| Trigger | Type | Action |
|---|---|---|
| New files on `snapshots` branch | sensor (every 10 min) | Ingest into `data/raw`, materialise `staged_*` (football-data CSVs are copied but restage is skipped — odds sensor handles them) |
| New Understat raw under `raw/understat/` | sensor (every 10 min) | Materialise `understat_ingest` (fetch + prefix filter; cursor fallback if `snapshots_sensor` already copied) |
| New API-Football injury raw under `raw/api_football/` (excludes `lineups/`) | sensor (every 10 min) | Materialise `api_football_ingest` |
| New football-data `E0.csv` under raw | sensor (every 30 min) | Materialise `match_odds_ingest` |
| Monthly / manual football-data capture | GHA `football_data.yml` | Current-season E0 → `snapshots` under `captured_at=YYYY-MM-DD` |
| GW `data_checked` and not yet processed | sensor (every 30 min) | Join actuals → ledger; refit models; predict next H GWs; re-optimise. Processes **all** missing GWs, so downtime self-heals. |
| Next deadline within 3 h and plan or `benchmark_projections` still missing | sensor (every 15 min) | Archive `ep_next` + OpenFPL; re-predict (no refit); re-optimise; emit report |
| Run failure | run-failure sensor | Notify (email or push via ntfy) |

The pre-deadline run can always be launched manually from the UI.

Live unattended models and planner horizon come from `configs/ops/production.yaml` (defaults: `dc_layered_v2`, horizon 3), loaded by `fplt.automation.production.load_production_ops`. Operators may override without a code change via `FPLT_PRODUCTION_MODELS` (comma-separated) and/or `FPLT_PLAN_HORIZON`; env wins over YAML. Keep the live list short — never silently multi-arm. CLI / `make plan` still default to `DEFAULT_PLAN_HORIZON` in `fplt.optimise.plan` (also 3); the ops file only feeds catch-up / pre-deadline. A checked gameweek is waiting until every configured production model has a plan parquet for that deadline at the ops horizon. The catch-up sensor asks the `recommendation_report` asset to plan every waiting week, and the markdown is stored as asset metadata. After a successful catch-up or pre-deadline plan, the concatenated `report.md` body is pushed via `FPLT_NTFY_URL` and/or optional SMTP (`FPLT_SMTP_HOST`, `FPLT_SMTP_FROM`, `FPLT_SMTP_TO`, plus user/password/port/TLS) when those settings are present; missing config is a no-op. Run failures POST to `FPLT_NTFY_URL` when it is set.

### 8.6 Running locally

- Development: `make dagster-dev` (`dagster dev`).
- Always-on: `dagster-webserver` and `dagster-daemon` as launchd services with fixed `DAGSTER_HOME` and SQLite storage (templates in `deploy/launchd/`). Secrets (`FPLT_ENTRY_ID`, `FPLT_FPL_API_TOKEN`, `FPLT_NTFY_URL`, SMTP) load from `~/.config/fplt/env` via `run-with-env.sh` — never from the committed plists. The same file is loaded by `Settings` for local CLI (`fplt auth set` / `status` / `clear`). Operator steps: `deploy/README.md`.
- Post-downtime one-shot (no UI): `make ops-catchup` / `fplt ops-catchup` runs fetch → restage → Understat/API-Football ingest → `catch_up` → status (library steps mirrored by the Dagster jobs). See `deploy/README.md`.
- A `deploy/docker-compose.yml` keeps the option of moving to a small VPS without code changes; the same `FPLT_*` vars are passed through from a gitignored `deploy/.env` or the shell.
- Multi-model eval (manual / `eval_board` Dagster job): `make eval-board SEASON=2024-25` runs walk-forward backtests for `rolling_mean_v1`, `ep_next_v1`, `dc_layered_v1`, `dc_layered_v2`, and `lgbm_direct_v1` into the prediction ledger for `make leaderboard`. OpenFPL (`openfpl_v1`) is optional — include it with `--with-openfpl`, `--openfpl-model-dir`, or a `model_dir` other than the fetched default checkout, and only for seasons from 2024-25 onwards. Live `fplt predict` refuses to record an all-zero frame. Walk-forward backtests still record a cold start, such as a rolling mean before any appearances.
- Fair compare (CLI): `make compare-models SEASON=2024-25` / `uv run fplt compare …` runs the same core set under shared windows in walk-forward and fixed-parameter modes and writes `$FPLT_DATA_DIR/compare/.../summary.json` (see §7.1).
- Component eval (CLI): `make eval-components SEASON=2024-25` / `uv run fplt eval-components …` walk-forwards scoreline RPS/log-loss/CS Brier (Dixon-Coles vs Poisson vs `odds_poisson_v1` / `odds_1x2_v1`), minutes Brier on P(played) and P(≥60) sliced by `source`, involvement goal-share MAE, and an optional `n_sims` stability sweep for `dc_layered_v1`. Writes `$FPLT_DATA_DIR/component_eval/.../summary.json` (see §7.3).
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

- Predictions for horizon H from one predictor (expected points; optionally samples). Live `fplt predict` / lineage assets default to **H = 10** (`DEFAULT_PROJECTION_HORIZON`) so planners can request 8–10 steps. CLI / `make plan` / `make decisions-board` / `fplt apply` still default to **H = 3** (`DEFAULT_PLAN_HORIZON`), from the 2024-25 chips-off horizon sweep (H=3 best cross-model season points; H=4 matched actions at higher solve cost). Match backtest `--steps` to the planner horizon. Fixture PIT + candidate prune make honest H=5–8 boards runnable; do **not** flip `DEFAULT_PLAN_HORIZON` until the three-arm chip board at the longer H wins on season points (`make horizon-decisions` / FDB-131).
- Squad state: current 15, purchase/selling prices, bank, free transfers available, chips available. `fplt.optimise.decisions.resolve_squad_state` is the shared entry point: YAML draft by default; `entry_json` when provided; otherwise `fplt.data.fpl_client.load_current_team` → `apply_entry_picks` when `FPLT_ENTRY_ID` is set (authenticated `my-team/{id}` needs `FPLT_FPL_API_TOKEN`, the Bearer value from `x-api-authorization`; else public finished-GW picks, no selling prices). `run_plan` (and thus `make plan` / CLI / automation) calls this whenever a pre-built `squad` is not passed. FT from `my-team` replaces the YAML count when it fits the banking cap; chips stay in `configs/squad_state.yaml`.
- **Live chip policy:** `run_plan` / automation default to **`DEFAULT_LIVE_CHIP_ARM = optimised`** (season `schedule_chip_windows` → short-horizon MILP under `ChipRules`), matching the C6 board winner (`fplt decisions --chip-board --chip-arms optimised`). Override with `fplt plan --chip-arm off|rule_based|optimised`. Staged fixtures are required when chips are on.
- Rules for the season (squad limits, FT banking cap, hit cost).

### 10.2 Formulation (multi-period ILP)

Decision variables per GW t in horizon and player p: in squad, in XI, captain, vice, transferred in/out, plus chip binaries `wc_t` / `fh_t` / `bb_t` / `tc_t` (TC extra-captain / BB bench-score player vars; Free Hit temporary squad/XI/captain). Constraints: budget with selling prices, 2 GK / 5 DEF / 5 MID / 3 FWD, ≤ 3 per club, valid XI formation (1 GK, ≥3 DEF, ≥2 MID, ≥1 FWD), FT accounting with banking cap and -4 per extra transfer (hits bounded by `M·(1−wc_t)`; FH freezes permanent transfers), one chip per GW, half-season chip caps from `ChipRules` + `SquadState.chips`. Objective: Σ_t decay^t × (XI points + captain bonus + TC extra + BB bench + FH temporary lineup) − hit costs. `bench_weight` defaults to **0** (retired as primary bench valuation; the field remains only as an optional tiny legacy tie-break below 0.01). Solve with HiGHS (`highspy`).

**Candidate prune (long horizons).** Binary count scales with priced players × H. For `horizon >= 3` (`PRUNE_AUTO_MIN_HORIZON`), `plan_squad` applies `CandidatePrunePolicy` via `prune_candidates` before building the MILP. Order: (1) **exact dominance** pre-filter — drop player *i* when ≥ *q* same-position others each have price ≤ *i* and xPts ≥ *i* at every step (strict on price or some step), with a club-feasible selection of size *q* (≤3 per club; *q* = GK2/DEF5/MID5/FWD3); owned never drop; restore any drops needed so the remaining pool still admits a club-feasible fifteen; (2) heuristic union per position of top **decayed** Σ xPts (GK16 / DEF36 / MID46 / FWD23), top **points-above-replacement per £** `(xPts_i − xPts_r) / (price_i − price_r)` with decayed xPts (GK10 / DEF23 / MID29 / FWD13; *r* = cheapest playable at the position, `p_60 >= min_p_60`; skip players priced ≤ *r*), top **raw** xPts at each step (GK2 / DEF5 / MID6 / FWD3), top raw Σ xPts over rolling 2– and 3-step windows (same quotas), the cheapest 4 *playing* fillers (`p_60 >= min_p_60`, default 0.5; fall back to plain cheapest when too few), plus owned. Horizon / PAR ranking uses the planner's `SquadState.decay` (`Σ_t decay^t · xPts_t`, 0-indexed `t`) — the same weight the MILP objective applies; per-step and short-window tops use undecayed step xPts so Free Hit / one-week / short-stretch decisions keep that week's best players. Optional `p_60` on the horizon points frame is joined for fillers and the PAR replacement baseline; `HorizonPointsSchema` stays `{player_key, step, xpts}`. Callers may pass `prune_policy=` or force prune on/off with `prune=True|False`. HiGHS knobs live in `HighsSolveOptions` (`highs_options=`; default 600s time limit, `mip_rel_gap=1e-3`, accept time-limit incumbents). `plan_squad(..., warm_start=prior_plan)` seeds HiGHS from a prior plan shifted forward one step (last week's step 1 → this week's step 0); `simulate_decisions` passes the previous deadline's plan automatically. Logs raw→kept pool size and HiGHS wall time per solve. **Pruning-loss harness** (`fplt.optimise.pruning_loss` / `fplt pruning-loss` / `make pruning-loss`): sample squad states, solve a full-pool reference (long time limit, `mip_rel_gap=0`), sweep target N ∈ {80,100,130,160,200,300} plus `current`, and record week-1 action agreement (primary), objective gap, solve time, and MIP status. A prune-policy change is an equivalence check via `check_week1_equivalence` (identical week-1 actions). **Default N (FDB-159):** `DEFAULT_TARGET_POOL = 130` (was ~200). Measured on the synthetic pruning-loss harness (640-player universe, 24 scenarios, H=5, seed 0): N∈{80,100,130,160,200,300} all had mean objective gap 0; week-1 agree rates were 91.7% / 83.3% / 87.5% / 87.5% / 87.5% / 87.5% (current=87.5%). Exact dominance already caps kept pools near ~90–110, so larger N does not recover the 95% agree bar — remaining misses are zero-gap alternate optima. N=130 is the smallest size in the §3.4 130–170 band that matches the previous default's agreement; N=80 scored slightly higher on this synthetic set but sits below that band. Re-check on real ledgers before cutting further. **Long-horizon MILP shrink (H≥6).** `LongHorizonPolicy` (`fplt.optimise.long_horizon`; `long_horizon_policy=` on `plan_squad`) defaults to `auto_min_horizon=6`, `relax_and_fix=True` (`binary_steps=2`), and `coarsen_from_step=3`: (1) keep steps 0–1 integer, relax later steps to continuous, solve, fix week-1 variables to the incumbent, then re-solve with mid-horizon steps restored to integer (far steps stay continuous); (2) from step 3 onward drop captain/vice (and TC) detail — pin armband vars off and zero their objective, while XI/squad/BB remain — **except** steps whose gameweek has a **forced** chip (`ChipUseWindows.forced`), which keep full armband/TC integrality so rule-based / force-play TC or FH cannot make the MILP infeasible inside a long look-ahead. `coarsen_steps(...)` computes the effective set. Week-1 action agreement vs a full-detail solve is the acceptance check (pruning-loss harness / unit tests); no quality assertions in pytest. Pass `coarsen_from_step=None` to disable coarsening.

Library entry point: `fplt.optimise.optimize_transfers(candidates, expected_points, state, gameweeks=...)`. Predictions are opaque (`ExpectedPointsSchema`: `player_key`, `gw`, `xpts`); the engine does not call predictors or the data store. `fplt plan` is the CLI wrapper that loads candidates and ledger rows, then calls the same solver.

v1: single-GW XI + captain. v2: multi-period transfers. v3: chips in MILP (BB/TC/WC/FH) plus chip-use windows, the season-level chip scheduler, and the three-arm decision board. Captaincy can use `p_haul` / samples as a tiebreak.

**Scenario lineup (Wave D1).** For a locked fifteen, `fplt.optimise.lineup.select_lineup_scenarios` maximises *mean realised points* across a shared `ScenarioBundle` (`player_key × sim` points and minutes), applying autosub and captain/vice-on-blank per draw and returning explicit `bench_slot` order. This is the sample-average approximation for the single-GW lineup problem; it does not use `bench_weight`. When joint samples are unavailable, `select_lineup_expected` uses a one-sim EV bundle and optional `p_haul` as a captaincy tie-break. **`plan_squad` defaults to `lineup_reopt=True`**: after the transfer MILP, each *locked* fifteen is re-lined via EV (+ optional `p_haul`); pass `scenarios=` to use SAA on step 0. Set `lineup_reopt=False` to keep the raw MILP lineup. Under H≥6 relax-and-fix, far steps stay continuous — 0.5-thresholded pick rows may not be a legal formation; that is look-ahead only. `lineup_reopt` skips those steps (keeps the MILP XI) and **requires** step 0 to be a legal fifteen (`fplt.optimise.squad_integrity`). `select_squad` also tilts captaincy by `p_haul` when that column is present.

### 10.3 Output

`optimize_transfers` returns a `TransferPlan` (first-week transfers in/out, XI, captain, vice, hit, plus full-horizon picks). Persistence uses `PlanSchema` parquet at `data/plans/<model_id>/as_of=<season>_gwNN_hHH/part.parquet`, one row per player kept or sold in each horizon week (`target_gw`, `in_xi`, `captain`, `vice`, `bought`, `sold`, `xpts`, `hit`), plus a markdown report in the same directory. `<model_id>` is the model-spec hash. `hHH` is the horizon actually covered, capped at gameweek 38. The report labels players with the staged `web_name`, and falls back to `player_key` when that name is missing. The same model, deadline, and horizon are never overwritten. A repeat run still prints the transfer summary, then reports that the parquet already exists. `fplt plan` raises before solving when this deadline has no ledger rows, when a horizon week is missing or all zero, or when a week covers fewer than half the candidates. Missing rows are not treated as zero.

`fplt apply` (library: `fplt.optimise.apply.apply_plan`) maps step-0 buys/sells onto paired FPL transfer rows and the planned XI/captain onto a lineup body. Dry-run is the default; `--confirm` POSTs with `FPLT_FPL_API_TOKEN`. Initial drafts (15 buys, no sells) are refused. Automation never calls this path.

Squad state lives in `configs/squad_state.yaml`: bank, free transfers, the banking cap, hit cost, decay, `bench_weight` (default **0**; retired as primary bench valuation — use scenario / EV lineup re-opt instead; values below 0.01 still allowed as a legacy MILP tie-break), owned purchase/selling prices, and `chips` (`remaining` counts plus `played` `{name, gw}` history). An empty squad is an initial draft. `resolve_squad_state` (and `fplt plan --entry-json`) replace owned players and bank from an entry picks payload, mapping element ids through staged players (``purchase_price`` when present, else defaults to ``selling_price``); chip inventory stays in YAML. Before each `plan_squad` / decision-sim deadline, selling prices are refreshed from as-of candidate list prices via `fplt.optimise.prices` (50% of rises kept, full falls). Live `run_plan` defaults to within-horizon `mean_gw_delta` forecasts from PIT-visible history (`DEFAULT_LIVE_PRICE_FORECAST_METHOD`); chip-board control omits `--price-forecast`. See `fplt.optimise.price_forecast`. `fplt plan --rebuild` clears ownership after that resolve and sets `bank` to today's available funds (`bank + Σ selling_price`), so the optimiser drafts fifteen from scratch under the current budget (wildcard-style; no transfer hits). Needs selling prices from authenticated `my-team` or `--entry-json`. Live `fplt plan` / automation use optimised chips by default (`DEFAULT_LIVE_CHIP_ARM`); the default decision backtest (`fplt decisions` without `--chip-board`) stays chips-off for continuity; `fplt decisions --chip-board` / `make chip-decision-board` runs the three arms and wires `season` / `gameweeks` / `chip_windows` into `plan_squad`. Season allotments are in `configs/chips/<season>.yaml`.

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
│   ├── scoring/<season>.yaml    # FPL points table per season
│   ├── chips/<season>.yaml      # FPL chip allotments per season
│   └── squad_state.yaml         # optimiser overrides (incl. chip inventory)
├── docs/design.md
├── scripts/snapshot.py          # standalone, runs on GitHub Actions
├── src/fplt/
│   ├── settings.py              # pydantic-settings (paths, FPL entry id; loads ~/.config/fplt/env)
│   ├── auth.py                  # fplt auth: store Bearer token outside the repo
│   ├── browser_auth.py          # fplt auth from-browser: local Chrome/Firefox localStorage
│   ├── types.py                 # GWKey, enums
│   ├── data/                    # FPL client, raw writer, staging, DataStore, PITData
│   ├── schemas/                 # pandera.polars schemas
│   ├── features/                # registry + feature definitions
│   ├── scoring/                 # Rules, ChipRules
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
4. **Experiment tracking.** No MLflow, Deepchecks, Giskard, Evidently, Feast, or TFX as the promote-eval runner. Reconsidered against a "use it at ~90%" bar: those APIs score one fitted model on one static frame. The walk-forward and the parquet ledger stay the implementation. Paired intervals use `statsmodels` and `arch`; they are not the runner (§7.6).
5. **Minutes model.** v1 draws each player's minutes independently. v2 may soft-project expected starters toward ~11 in the distribution. `logistic_blend` (default 0.5) weights logistic P(start) vs the multi-window empirical rate. `dc_layered_v2` also applies a hard 11-starter draw constraint per team-fixture via `enforce_hard_starters`, keeping exactly one GK when any GK is eligible. FPL availability gating stays on in production (`use_availability_gate`; board-only off-fork `dc_layered_noavail_v1`). Optional `use_injury_overlay` (forks `empirical_minutes_v2_injuries` / `dc_layered_injuries_v1`) soft/hard-downgrades from staged `player_injuries`. Optional near-deadline predicted-XI override (`predicted_xi`; fork `dc_layered_predicted_xi_v1`) mixes P(start) toward staged `predicted_lineups` inside a kickoff window. Injury and predicted-XI overlays stay off in production — neither changes `PRODUCTION_MODELS`.
