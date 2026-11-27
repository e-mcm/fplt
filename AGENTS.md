# AGENTS.md — Rules for coding agents working in this repo
 
This file is the standing brief for any AI coding agent (Claude Code, Codex, etc.) working on `fplt`. Read it fully before making changes. `CLAUDE.md` points here.
 
## 1. Read the design first

`docs/design.md` is the authoritative architecture. Before implementing anything:

1. Read the sections of `docs/design.md` relevant to the task (the issue should name them).
2. If the task conflicts with the design, **stop and say so** in the PR/issue rather than silently diverging.
3. If you make a design-relevant change (new contract, schema, asset, config shape), update `docs/design.md` in the same PR.

Doc map and per-directory context: [`docs/README.md`](docs/README.md); short READMEs under `configs/`, `scripts/`, `src/fplt/`, `tests/`, `deploy/`, `external/`, `data/`.
## 2. Hard rules
 
These are non-negotiable. A PR that breaks one of these is wrong even if tests pass.
 
### Point-in-time correctness
- Models and features read data **only** through a `PITView` (`fplt.data.pit`). Never import `DataStore`, read parquet paths directly, or call the FPL API from model or feature code.
- Every staged table must carry `available_at` (UTC). New tables without it will be rejected.
- Every new `PointsPredictor` or component must pass the contract suite **including the leakage test** (`tests/contracts/`). Register it so the parametrised suite picks it up.
### Data and schemas
- Use **polars**, not pandas. pandas is allowed only inside `src/fplt/models/external/` adapters when a third-party dependency requires it, and must be converted to polars at the adapter boundary.
- Every DataFrame crossing a module boundary (function argument or return in a public function) must have a `pandera.polars` `DataFrameModel` in `src/fplt/schemas/`. Validate at boundaries: when producing a staged/feature/prediction/plan table, and in Dagster asset checks.
- Add the schema **before** the function that produces the table.
- Raw data under `data/raw/` is write-once. Never modify or delete raw files in code.
### Models
- Top-level predictors implement `PointsPredictor` (`src/fplt/models/contracts.py`). Components implement the relevant component protocol. Do not invent parallel interfaces.
- Every model is described by a `ModelSpec` in `configs/models/*.yaml` and registered with `@register_model("<kind>")`. No hyperparameters hardcoded in source.
- Artifacts are written under `artifacts/<model_id>/as_of=.../` and never overwritten. Use the model's own `save`/`load`; no ad hoc pickling elsewhere.
- All randomness uses `numpy.random.Generator` seeded via `fplt.utils.rng.seed_for(model_id, as_of, fixture_id)`. No global RNG, no unseeded randomness.
- Layered models combine components through the simulation engine (`fplt.models.simulation`). Do not chain point estimates (e.g. never compute clean-sheet probability from expected goals).
### Features
- Features are registered with `@feature(name=..., version=..., grain=..., inputs=[...])`.
- **Changing an existing feature's output requires bumping its version.** Never edit a released feature version in place. Pure refactors that provably leave output unchanged keep the version (add a test that shows it).
### Scoring rules
- FPL scoring lives in `configs/scoring/<season>.yaml`, loaded via `fplt.scoring.Rules.for_season`. Never hardcode points values in code.
### Orchestration
- Dagster code (`src/fplt/orchestration/`) is a thin wrapper. All logic lives in library modules and must be callable without Dagster.
- New tables get a blocking pandera asset check via `pandera_check(...)`.
- Prefer factories (assets generated from feature registry / model specs) over hand-written per-model assets.
### Engineering habits
- **Bug → regression test.** Every time a bug is found, write a unit test that would have exposed it. Ship that test with (or before) the fix — never a bare fix.
- **Hard investigation → utility.** Every time an ad-hoc investigation is hard, reflect on why and add helpers, fixtures, scripts, or docs so the next time is easier. Do not only solve the one-off.
## 3. Testing rules
 
**Everything is tested.** No code is merged without tests that exercise it.
 
- Every function and class you add or change needs tests in the same PR. Write the tests alongside the code. Do not leave them for a later PR.
- Bug fixes must include a unit test that would have failed before the fix (see Engineering habits above).
- Aim for full line and branch coverage of each function and class. Cover the normal path, edge cases (empty frames, double/blank gameweeks, missing players, zero minutes) and error paths (invalid input raises the documented exception).
- CI (`test-linux` in `.github/workflows/ci.yml`) runs the **full** suite (including `@pytest.mark.slow`) under `pytest-xdist` (`-n auto`) with `--cov=fplt --cov-branch --cov-report=xml --durations=25`. Do not use testmon or path-based skipping of Python tests on PRs.
- Coverage gates, enforced in CI (`test-linux` in `.github/workflows/ci.yml`):
  - **Diff coverage ≥ 95%** on changed lines: `uv run diff-cover coverage.xml --compare-branch=origin/main --fail-under=95` (PRs only; `actions/checkout` with `fetch-depth: 0`).
  - **Overall line + branch coverage ≥ 90%**: `[tool.coverage.report] fail_under = 90` in `pyproject.toml` (coverage.py combined `percent_covered` with `branch = true`). The pytest step fails the job when under the floor.
- Exclusions are rare. Use `# pragma: no cover` only for code that genuinely cannot run in tests, such as `if TYPE_CHECKING:` blocks or `__main__` entry points. Add a one-line comment giving the reason. Never exclude code just to hit the number.
- Coverage is the floor, not the goal: assert on behaviour and values, not merely that code ran.
- Tests never touch the network. Enforced by `pytest-socket` (`--disable-socket --allow-unix-socket` in `[tool.pytest.ini_options] addopts`; unix sockets stay allowed for xdist). Opt in with `@pytest.mark.enable_socket` only for an explicit local stub. Use recorded fixtures in `tests/fixtures/` (gzipped API JSON) or synthetic data builders in `tests/builders.py`.
- Test order is randomised by `pytest-randomly` (also reseeds `random` / numpy per test). CI prints the seed; reproduce with `uv run pytest --randomly-seed=<n>`.
- Per-test timeouts via `pytest-timeout` (global 60s; unit tests 30s unless a test sets its own `@pytest.mark.timeout`). A hang fails that test with a stack dump instead of burning the job timeout.
- Hypothesis profiles in `tests/conftest.py`: `ci` (`HYPOTHESIS_PROFILE=ci` in CI — `deadline=None`, `print_blob=True`, `max_examples=100`) and `dev` (default locally — fewer examples). Per-test `@settings(max_examples=…)` still wins.
- Never assert model quality (MAE, RMSE, log-loss) in pytest. Quality is measured by the backtester and recorded in the ledger.
- Unit tests in `tests/unit/`, contract tests in `tests/contracts/`, Dagster/end-to-end in `tests/integration/`. Mark anything slower than ~2 s with `@pytest.mark.slow`.
- Optimiser tests check **constraint satisfaction** on small known problems, not solution optimality on real data.
- Add a property test (hypothesis) for anything probabilistic: distributions sum to 1, allocations conserve goals, etc.
## 4. Workflow
 
- Commands (see `Makefile`):
  - `make check` — ruff lint (including docstring rules) + format check, pyright, workflow lints, then fast tests (`pytest -n auto -m "not slow"`). Must pass before finishing any task. Diff-cover runs in CI on PRs, not in `make check`.
  - `make test` — full suite (incl. slow) with xdist, coverage XML/term, `--durations=25`, and `fail_under = 90`.
  - `make pytest-profile` — rank slowest pytest cases (CI-shaped by default; see `scripts/pytest_profile.py`).
  - `make dagster-dev` — local Dagster UI.
  - `make mc-dev` / `mc-test` / `mc-format` / `mc-bench` — native `fplt-mc` kernel under `mc/` (C++ editors only).
- **`fplt-mc` / cloud agents:** `uv sync --frozen` must **never** compile the kernel. Wheels publish to the **public** GitHub Pages simple index (`https://e-mcm.github.io/fplt/simple/`; no `UV_INDEX` credentials). Root `pyproject.toml` pins `fplt-mc` with `no-build-package = ["fplt-mc"]` so a missing wheel fails fast. Live PRODUCTION is `dc_layered_v4_native` (`mc_backend: native`) — missing import raises; no silent Python MC fallback. Experiment VMs and cloud agents install the locked wheel only; do not run `make mc-dev` there. Prefer GHA for release wheels. See `mc/README.md`.
- **R2 data access (cloud agents):** pull raw / curated / official reads from bucket **`fplt-data`** with the read-only agent token; write runs, experiments and lab specs only to **`fplt-lab`**. Never push to `fplt-data` from an agent. Library code still reads the local `$FPLT_DATA_DIR` mirror after pull — no model or feature code talks to the network. Layout and tokens: `docs/design.md` §4.3; ops: `docs/proposals/r2-data-bucket-plan.md`.
- Use `uv add` / `uv add --dev` to add dependencies; never edit `uv.lock` by hand. Justify any new dependency in the PR description.
- One issue → one branch → one PR. Reference the issue (`Closes #N`).
- One milestone finishes in its own PR, and so does each sub-milestone such as M2a. Open that PR and merge it before starting the next one. Do not stack later milestone work onto an unmerged branch.
### Small, minimal changes
 
- **Make the smallest change that completes the task.** Touch only what the task requires.
- No drive-by refactors, renames, reformatting or "while I'm here" fixes. If you notice something else that needs doing, note it in the PR description as a suggested follow-up issue instead.
- **Prefer several small PRs to one large one.** Each PR does one thing and leaves `main` working.
- Size guide: aim for under ~200 changed lines of non-test code per PR. If a task looks bigger, split it before starting and propose the split in the issue. For example: schema → producing function → Dagster asset → asset check, each as its own PR.
- Land foundations first (contracts, schemas, types), then implementations, then wiring. Don't bundle all three.
- Each commit should be a coherent step with a clear message, so the PR is easy to review commit by commit.
## 5. Code style
 
- Python 3.12+. Type-annotate all public functions; pyright in standard mode must pass.
- `ruff` handles lint and format. Do not hand-format.
- Enabled ruff lint groups (see `[tool.ruff.lint]` in `pyproject.toml`): `E`, `F`, `I`, `UP`, `D` (pydocstyle, Google convention). Further groups (`B`, `BLE`, `S`, `RUF012`, `TRY400`, `PT`) are ratcheted in #311.
- **Every public free function and every public class has a docstring.** "Public" means the name doesn't start with `_`. This is enforced by ruff's pydocstyle rules (`D101`, `D103`; Google convention). Other `D*` codes and all `D` in `tests/` are ignored until separately ratcheted.
  - Start with a one-line summary in the imperative mood ("Fit the Dixon-Coles parameters.").
  - Then document `Args`, `Returns` and `Raises` wherever they are not obvious from the signature.
  - For functions that take or return DataFrames, name the pandera schema.
  - Explain *why* for non-obvious modelling choices, such as approximations in the simulator.
- Public methods should also have docstrings where their behaviour isn't obvious from the name and the class docstring.
- Use `pydantic` for configs, `dataclasses`/`attrs` for internal value objects, `Protocol` for contracts.
- Paths come from `fplt.settings`; never hardcode `./data` or absolute paths.
- Log with `logging` (Dagster captures it); no `print` in library code (CLI and `fplt.ui` only). Timestamps are ISO-8601 UTC via `fplt.utils.logging.configure_logging`. Levels: **DEBUG** = diagnostic detail (HTTP paths, per-GW steps); **INFO** = operator progress; **WARNING** = recoverable problems; **ERROR** = failures that need attention. Bind run/ops identity with `log_context` / `RunWriter` (`run_id`, `experiment_id`, `checkpoint`, `line`, `spec`, `season`, `gw`). Use `--log-format json` or `FPLT_LOG_FORMAT=json` in workflows; humans keep text. Run tees INFO+ to `runs/run=<id>/logs/run.log.jsonl`; ops checkpoints write write-once `official/…/ops_log.jsonl` (redaction applies before write).
## 6. Domain notes that trip people up
 
- FPL element and team `id`s change between seasons. Use `player_key` (element `code`) and `team_key` (team `code`) across seasons.
- Predictions are at **player–fixture** grain. Double gameweeks produce two rows; blank gameweeks produce none. Aggregate to player–GW only at the optimiser boundary.
- Gameweek deadlines come from `bootstrap-static` `events[].deadline_time` (UTC). Results are final only after `events[].data_checked` is true.
- vaastav historical data lacks per-deadline availability snapshots; minutes-model features from those seasons are degraded and must be flagged via the `source` column.
- OpenFPL was trained on 2020-21 to 2023-24. Never benchmark it on those seasons.
## 7. When unsure
 
Prefer asking (in the issue or PR) over guessing on anything that touches: contracts, schemas, `ModelSpec` shape, PIT semantics, scoring rules, or the ledger format. These are shared foundations; changes ripple everywhere.
 