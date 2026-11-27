.PHONY: test check lint typecheck leaderboard ui eval-board compare-models eval-components minutes-brier minutes-measure other-measure predicted-xi-audit horizon-skill pruning-loss ensemble-sensitivity promote-eval bakeoff decisions-board chip-decision-board regret-board weekly-report decisions-explain horizon-decisions sim-season-smoke plan explain apply archive-benchmarks archive-compare ops-catchup ops-tick release dagster-dev clean bootstrap-vaastav fetch-understat fetch-api-football fetch-football-data fetch-overall-rank mc-dev mc-test mc-format mc-bench mc-ci mc-golden perf-report pytest-profile

# ── Historical data (vaastav) → raw/vaastav/captured_at=… ─────────────────────
# First seed: make bootstrap-vaastav ARGS='--full'
# Weekly / local: make bootstrap-vaastav
bootstrap-vaastav:
	uv run python scripts/bootstrap_vaastav.py $(ARGS)

# ── Understat raw capture (matches + league teamsData Deep/PPDA) ──────────────
# Usage: make fetch-understat SEASON=2024-25
# Writes write-once gzipped JSON under $$FPLT_DATA_DIR/raw/understat/...
fetch-understat:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make fetch-understat SEASON=2024-25"; exit 1; fi
	uv run python scripts/fetch_understat.py --out "$${FPLT_DATA_DIR:-./data}/raw" --season $(SEASON)

# ── API-Football injuries + cup/Europe fixtures (API-Sports) ──────────────────
# Usage: make fetch-api-football SEASON=2024-25
# Needs FPLT_API_FOOTBALL_KEY (or API_FOOTBALL_KEY / APISPORTS_KEY) in the env.
# Writes write-once gzipped JSON under $$FPLT_DATA_DIR/raw/api_football/...
fetch-api-football:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make fetch-api-football SEASON=2024-25"; exit 1; fi
	uv run python scripts/fetch_api_football.py --out "$${FPLT_DATA_DIR:-./data}/raw" --season $(SEASON)

# ── football-data.co.uk E0 odds (closing + scheduled prematch) ────────────────
# Usage: make fetch-football-data SEASON=2024-25
# Monthly refresh (write-once): CAPTURED_AT=YYYY-MM-DD
# GHA: .github/workflows/football_data.yml (1st of month 06:00 UTC) → snapshots;
# Dagster match_odds_sensor stages when a new E0.csv lands under raw.
fetch-football-data:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make fetch-football-data SEASON=2024-25 [CAPTURED_AT=YYYY-MM-DD]"; exit 1; fi
	uv run python scripts/fetch_football_data.py --out "$${FPLT_DATA_DIR:-./data}/raw" --season $(SEASON) \
		$(if $(CAPTURED_AT),--captured-at $(CAPTURED_AT),)

# ── FPL overall-league points-at-rank (live percentile mapping) ───────────────
# Usage: make fetch-overall-rank
# Optional: SEASON=2026-27 ARGS='--rank 10000 --rank 100000'
# Writes write-once gzipped JSON under $$FPLT_DATA_DIR/raw/fpl_api/.../overall_rank/
fetch-overall-rank:
	uv run python scripts/fetch_overall_rank.py --out "$${FPLT_DATA_DIR:-./data}/raw" \
		$(if $(SEASON),--season $(SEASON),) $(ARGS)

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	uv run pytest -n auto --cov=fplt --cov-branch --cov-report=term-missing --cov-report=xml --durations=25

# Rank slowest pytest cases (CI-shaped by default). Optional: ARGS='--parse-only …'
pytest-profile:
	uv run python scripts/pytest_profile.py $(ARGS)

check:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts
	uv run pyright
	uv run python scripts/check_workflow_timeouts.py
	uv run python scripts/check_workflow_r2_environment.py
	uv run python scripts/check_workflow_concurrency.py
	uv run pytest -n auto -m "not slow"

# ── Monte Carlo micro-bench (native-mc-build-spec §1.1 / §5.5) ─────────────────
# Synthetic 40-player fixture, n_sims=10000 by default. Best-of-3 wall times.
mc-bench:
	uv run python scripts/mc_bench.py $(ARGS)

# ── Lint + type-check ─────────────────────────────────────────────────────────
lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

typecheck:
	uv run pyright src/

# ── Leaderboard ───────────────────────────────────────────────────────────────
leaderboard:
	uv run streamlit run apps/leaderboard.py

# ── Local results UI (localhost only; reads $FPLT_DATA_DIR; no writes) ────────
# Prefer: uv run fplt ui   or   make ui
ui:
	uv run streamlit run src/fplt/ui/app.py --server.address localhost --browser.gatherUsageStats false

# ── Multi-model eval (ledger for make leaderboard) ────────────────────────────
# Usage: make eval-board SEASON=2024-25
# Optional: ARGS='--with-openfpl --from-gw 1 --to-gw 10'
eval-board:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make eval-board SEASON=2024-25"; exit 1; fi
	uv run python scripts/eval_board.py --season $(SEASON) $(ARGS)

# ── Fair model compare (walk-forward + fixed params) ──────────────────────────
# Usage: make compare-models SEASON=2024-25
# Optional: ARGS='--models rolling_mean_v1 ep_next_v1 --test-from-gw 1 --test-to-gw 10'
# Optional OpenFPL (2024-25+): ARGS='--with-openfpl' or '--openfpl-model-dir /path'
compare-models:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make compare-models SEASON=2024-25"; exit 1; fi
	uv run fplt compare --season $(SEASON) $(ARGS)

# ── Layered component eval (RPS / minutes Brier / involvement / n_sims) ────────
# Usage: make eval-components SEASON=2024-25
# Optional: ARGS='--from-gw 1 --to-gw 10 --skip-sims'
# Skill-by-horizon (Wave C4): ARGS='--skip-sims --points-models rolling_mean_v1 --steps 10'
eval-components:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make eval-components SEASON=2024-25"; exit 1; fi
	uv run fplt eval-components --season $(SEASON) $(ARGS)

# ── Snapshot-backed minutes v2 Brier (P(played), P(≥60)) ───────────────────────
# Usage: make minutes-brier SEASON=2024-25
# Optional: ARGS='--from-gw 1 --to-gw 10 --minutes-model empirical_minutes_v2'
# Summary rows are sliced by stats source; read slice source=fpl_api (not vaastav).
minutes-brier:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make minutes-brier SEASON=2024-25"; exit 1; fi
	uv run fplt eval-components --season $(SEASON) --minutes-only $(ARGS)

# ── Minutes measurement split + misses (snapshot GWs; FDB-246) ────────────────
# Usage: make minutes-measure SEASON=2025-26
# Optional: ARGS='--from-gw 1 --to-gw 19 --minutes-model empirical_minutes_v2'
# Writes availability/rotation/early-sub Brier+log, xPts-weighted decision-pool
# Brier, and weekly |ΔP(start)|×xPts misses under $$FPLT_DATA_DIR/minutes_measurement/.
minutes-measure:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make minutes-measure SEASON=2025-26"; exit 1; fi
	uv run fplt minutes-measure --season $(SEASON) $(ARGS)

# ── Other-points / haul measurement (FDB-240 Rank-3 plumbing) ─────────────────
# Usage: make other-measure SEASON=2025-26
# Optional: ARGS='--from-gw 1 --to-gw 19 --points-model dc_layered_v4 --n-sims 500'
# Writes haul Brier/log, top xPts-decile haul cal, match-context other residual,
# and unconstrained captain-regret proxy under $$FPLT_DATA_DIR/other_points_measurement/.
# Measurement only — no PRODUCTION flip.
other-measure:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make other-measure SEASON=2025-26"; exit 1; fi
	uv run fplt other-measure --season $(SEASON) $(ARGS)

# ── Predicted XI accuracy audit (FDB-253 / PR8) ────────────────────────────────
# Usage: make predicted-xi-audit SEASON=2025-26
# Optional: ARGS='--from-gw 1 --to-gw 19 --source touchline_wire --min-accuracy 0.55'
# Writes start precision/recall, bench→start rate, full-XI coverage under
# $$FPLT_DATA_DIR/predicted_xi_audit/. Exit 1 when start precision < min-accuracy.
predicted-xi-audit:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make predicted-xi-audit SEASON=2025-26"; exit 1; fi
	uv run fplt predicted-xi-audit --season $(SEASON) $(ARGS)

# ── Forecast skill by lead time (MAE/RMSE for H=1…STEPS; wraps compare) ────────
# Usage: make horizon-skill SEASON=2024-25
# Optional: STEPS=10 FROM_GW=1 TO_GW=20 ARGS='--models rolling_mean_v1 dc_layered_v2'
# Does not change DEFAULT_PLAN_HORIZON (still 3). Report curve before retuning H.
horizon-skill:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make horizon-skill SEASON=2024-25"; exit 1; fi
	uv run fplt horizon-skill --season $(SEASON) \
		--from-gw $(or $(FROM_GW),1) \
		--to-gw $(or $(TO_GW),20) \
		--steps $(or $(STEPS),10) \
		$(ARGS)

# Rank wall-clock timing hotspots from $$FPLT_DATA_DIR/timing/*.json
# Usage: make perf-report
# Optional: ARGS='--kind walk_forward --limit 5 --json'
perf-report:
	uv run fplt perf-report $(ARGS)

# Candidate-prune loss curve (week-1 action agreement vs pool size N).
# Usage: make pruning-loss
# Optional: N_SCENARIOS=24 SEED=0 HORIZON=5 ARGS='--targets 80 100 130 160 200 300'
# Writes $$FPLT_DATA_DIR/pruning_loss/ (override with ARGS='--output-dir …').
# DEFAULT_PRUNE_POLICY N was retuned from this harness (FDB-159 → ~130).
pruning-loss:
	uv run fplt pruning-loss \
		--n-scenarios $(or $(N_SCENARIOS),24) \
		--seed $(or $(SEED),0) \
		--horizon $(or $(HORIZON),5) \
		$(ARGS)

# Ensemble sensitivity / move-frequency (Wave D2): shared-state DC vs LGBM.
# Requires prediction ledger for control + challengers at matching --horizon.
# Usage: make ensemble-sensitivity SEASON=2024-25
# Optional: HORIZON=3 FROM_GW=1 TO_GW=38 CONTROL=dc_layered_v2
#   ARGS='--challengers lgbm_direct_v1 --output-dir …'
# Does not change PRODUCTION_MODELS; chips-off reporting board only.
ensemble-sensitivity:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make ensemble-sensitivity SEASON=2024-25"; exit 1; fi
	uv run fplt ensemble-sensitivity --season $(SEASON) \
		--from-gw $(or $(FROM_GW),1) \
		--to-gw $(or $(TO_GW),38) \
		--horizon $(or $(HORIZON),3) \
		--control $(or $(CONTROL),dc_layered_v2) \
		$(ARGS)

# ── Promote-eval (control vs treatment EvalSpec; dry-run scaffold) ─────────────
# Usage: make promote-eval EVAL=example_model_lgbm_vs_rolling
# Optional: ARGS='--dry-run --tier S1'
promote-eval:
	@if [ -z "$(EVAL)" ]; then echo "Usage: make promote-eval EVAL=<name>"; exit 1; fi
	uv run fplt promote-eval --eval $(EVAL) $(ARGS)

# ── Multi-arm prediction bakeoff (compare + SPA/MCS ranking; not a keep) ───────
# Usage: make bakeoff BAKEOFF=example_prediction_championship
# Optional: ARGS='--execute' or CLI mode ARGS='--models a b c --control a --season 2024-25 --execute'
bakeoff:
	@if [ -z "$(BAKEOFF)" ] && [ -z "$(ARGS)" ]; then \
		echo "Usage: make bakeoff BAKEOFF=<name>  or  make bakeoff ARGS='--models … --control … --season …'"; \
		exit 1; \
	fi
	@if [ -n "$(BAKEOFF)" ]; then \
		uv run fplt bakeoff --bakeoff $(BAKEOFF) $(ARGS); \
	else \
		uv run fplt bakeoff $(ARGS); \
	fi

# ── Decision bakeoff (season totals after backtest fills the prediction ledger)
# Usage: make decisions-board SEASON=2024-25
# Optional: ARGS='--models rolling_mean_v1 dc_layered_v1 --from-gw 1 --to-gw 10'
# Optional: ARGS='--summarise-only' to score ledger/decisions without re-simulating
# Horizon should match the backtest --steps used when writing predictions.
# Sanity: fails if mean pts/GW > 90 unless ARGS includes --max-mean-ppg 0
decisions-board:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make decisions-board SEASON=2024-25"; exit 1; fi
	uv run fplt decisions --season $(SEASON) --horizon $(or $(HORIZON),3) $(ARGS)

# Three-arm chip decision board (Wave C3): chips-off / rule-based / optimised.
# Score season points under chip rules; label counterfactual when RULES_SEASON
# differs (e.g. SEASON=2024-25 RULES_SEASON=2025-26).
# Usage: make chip-decision-board SEASON=2025-26
# Optional: RULES_SEASON=2025-26 MODELS='rolling_mean_v1' FROM_GW=1 TO_GW=10
# Requires a prediction ledger with --steps matching HORIZON (default 3).
# Does not run the full post-chips bakeoff (FDB-120 / Wave C6).
chip-decision-board:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make chip-decision-board SEASON=2025-26"; exit 1; fi
	uv run fplt decisions --chip-board --season $(SEASON) \
		--horizon $(or $(HORIZON),3) \
		$(if $(RULES_SEASON),--chip-rules-season $(RULES_SEASON),) \
		$(if $(MODELS),--models $(MODELS),) \
		$(if $(FROM_GW),--from-gw $(FROM_GW),) \
		$(if $(TO_GW),--to-gw $(TO_GW),) \
		$(ARGS)

# Regret-by-component board (north-star §2): captain / XI-bench / transfers /
# chip timing vs constrained attainable. Live-stack defaults (H=3, force-play,
# lineup_reopt, mean_gw_delta, optimised chips, dc_layered_v2).
# Usage: make regret-board SEASON=2025-26
# Optional: OUT=.../regret-board/2025-26 ARGS='--policy-parquet path.parquet'
#           MODEL=dc_layered_v2 HORIZON=3 FROM_GW=1 TO_GW=38
regret-board:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make regret-board SEASON=2025-26"; exit 1; fi
	uv run fplt regret-board --season $(SEASON) \
		--horizon $(or $(HORIZON),3) \
		--model $(or $(MODEL),dc_layered_v2) \
		$(if $(FROM_GW),--from-gw $(FROM_GW),) \
		$(if $(TO_GW),--to-gw $(TO_GW),) \
		$(if $(OUT),--output-dir $(OUT),) \
		$(ARGS)

# Post-GW decision scoring + weekly report (ops-08). Writes
# $$FPLT_DATA_DIR/reports/gw=<n>/weekly.md + parquet and updates
# season_to_date.parquet. Notify once per distinct body (ntfy optional).
# Usage: make weekly-report SEASON=2025-26 GW=12
# Optional: MODEL_ID=… HINDSIGHT_BEST=… N_FIXTURES=… ARGS='--no-notify'
weekly-report:
	@if [ -z "$(SEASON)" ] || [ -z "$(GW)" ]; then \
		echo "Usage: make weekly-report SEASON=2025-26 GW=12"; exit 1; fi
	uv run fplt weekly-report --season $(SEASON) --gw $(GW) \
		$(if $(MODEL_ID),--model-id $(MODEL_ID),) \
		$(if $(HINDSIGHT_BEST),--hindsight-best $(HINDSIGHT_BEST),) \
		$(if $(N_FIXTURES),--n-fixtures $(N_FIXTURES),) \
		$(ARGS)

# Chip-arm simulated-season SE smoke (N6). Short GW window × few CRN paths —
# not a full 38-GW × large-N board. Needs prediction ledger with actuals and
# --steps matching HORIZON. Does not flip DEFAULT_PLAN_HORIZON or live chips /
# force_play defaults. Same-H force ablation: ARGS='--chip-arms optimised
# --no-force-play'. Promote case E (H=3 force-on vs H=8 force-off) needs two
# DecisionPolicy rows in one run_simulated_season_eval — see design §7.4 N6.
# Usage: make sim-season-smoke SEASON=2025-26
# Optional: FROM_GW=1 TO_GW=3 N_PATHS=3 SEED=0 MODELS='dc_layered_v2'
#           RULES_SEASON=2025-26 HORIZON=3 ARGS='--chip-arms off optimised'
sim-season-smoke:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make sim-season-smoke SEASON=2025-26"; exit 1; fi
	uv run fplt sim-season --season $(SEASON) \
		--from-gw $(or $(FROM_GW),1) \
		--to-gw $(or $(TO_GW),3) \
		--horizon $(or $(HORIZON),3) \
		--n-paths $(or $(N_PATHS),5) \
		--seed $(or $(SEED),0) \
		$(if $(RULES_SEASON),--chip-rules-season $(RULES_SEASON),) \
		$(if $(MODELS),--models $(MODELS),) \
		$(ARGS)

# Multi-horizon decision bakeoff. Prediction ledger must already cover max H
# (backtest --steps >= max HORIZONS; predict default is now 10). Archives each
# run because ledger/decisions overwrites by model×gw (no horizon key).
# Usage: make horizon-decisions SEASON=2024-25
# Optional: HORIZONS='1 2 3 4 8 10' ARGS='--models openfpl_v1 dc_layered_v2 --from-gw 1 --to-gw 38'
# Chips-on retune (Wave C4 → FDB-120): add ARGS='--chip-board' (or use
# chip-decision-board in a loop) — do not flip DEFAULT_PLAN_HORIZON from the
# skill curve alone; season points on the full decision sim is the gate.
horizon-decisions:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make horizon-decisions SEASON=2024-25"; exit 1; fi
	@data="$${FPLT_DATA_DIR:-./data}"; \
	mkdir -p "$$data/sweep"; \
	for H in $(or $(HORIZONS),1 2 3 4); do \
		echo "=== horizon $$H ==="; \
		rm -f "$$data/ledger/decisions/part.parquet"; \
		uv run fplt decisions --season $(SEASON) --horizon $$H $(ARGS); \
		cp "$$data/ledger/decisions/part.parquet" "$$data/sweep/decisions_h$$H.parquet"; \
	done

# Dump one model × GW realised arithmetic from ledger/decisions
# Usage: make decisions-explain SEASON=2024-25 GW=2 MODEL=openfpl_v1
decisions-explain:
	@if [ -z "$(SEASON)" ] || [ -z "$(GW)" ]; then \
		echo "Usage: make decisions-explain SEASON=2024-25 GW=2 MODEL=openfpl_v1"; exit 1; fi
	uv run fplt decisions --season $(SEASON) --summarise-only \
		--explain-gw $(GW) --explain-model $(or $(MODEL),rolling_mean_v1) $(ARGS)

# ── Explainability (forecast board + plan/transfer deltas) ────────────────────
# Usage: make explain SEASON=2025-26 GW=10
# Optional: MODEL=dc_layered_v2 HORIZON=3 TOP=30 ARGS='--forecast-only'
# Needs ledger predictions for the deadline; decision section needs a saved plan
# (make plan). Also appended onto plans/.../report.md when you run make plan.
explain:
	@if [ -z "$(SEASON)" ] || [ -z "$(GW)" ]; then \
		echo "Usage: make explain SEASON=2025-26 GW=10"; exit 1; fi
	uv run fplt explain \
		--model $(or $(MODEL),dc_layered_v2) \
		--season $(SEASON) \
		--gw $(GW) \
		--horizon $(or $(HORIZON),3) \
		--top $(or $(TOP),30) \
		$(ARGS)

# ── Transfer plan (uses FPLT_ENTRY_ID when set) ────────────────────────────────
# Usage: make plan SEASON=2025-26 GW=10
# Optional: MODEL=dc_layered_v4 HORIZON=3 ARGS='--entry-json picks.json'
# Default MODEL matches configs/ops/production.yaml (blank-week: leave
# FPLT_ENTRY_ID unset + owned: [] — see docs/blank-week-prod-plan.md).
# Default chip policy is optimised (scheduler → MILP). Override with
# ARGS='--chip-arm off' (or rule_based). Horizon default stays 3.
plan:
	@if [ -z "$(SEASON)" ] || [ -z "$(GW)" ]; then \
		echo "Usage: make plan SEASON=2025-26 GW=10"; exit 1; fi
	uv run fplt plan \
		--model $(or $(MODEL),dc_layered_v4) \
		--season $(SEASON) \
		--gw $(GW) \
		--horizon $(or $(HORIZON),3) \
		$(ARGS)

# ── Apply saved plan to FPL (dry-run unless CONFIRM=1) ─────────────────────────
# Usage: make apply SEASON=2025-26 GW=10
# Optional: MODEL=dc_layered_v4 HORIZON=3 CONFIRM=1 ARGS='--transfers-only'
apply:
	@if [ -z "$(SEASON)" ] || [ -z "$(GW)" ]; then \
		echo "Usage: make apply SEASON=2025-26 GW=10"; exit 1; fi
	uv run fplt apply \
		--model $(or $(MODEL),dc_layered_v4) \
		--season $(SEASON) \
		--gw $(GW) \
		--horizon $(or $(HORIZON),3) \
		$(if $(filter 1 yes true,$(CONFIRM)),--confirm,) \
		$(ARGS)

# ── Benchmark archives (N1: ep_next + OpenFPL → benchmark_projections) ────────
# Live (due GW inside 3h): make archive-benchmarks
# Backfill: make archive-benchmarks SEASON=2024-25 GW=10
# Optional: ARGS='--skip-openfpl' (local only; not for production)
archive-benchmarks:
	@if [ -n "$(SEASON)" ] || [ -n "$(GW)" ]; then \
		if [ -z "$(SEASON)" ] || [ -z "$(GW)" ]; then \
			echo "Usage: make archive-benchmarks SEASON=2024-25 GW=10"; exit 1; \
		fi; \
		uv run fplt archive-benchmarks --season $(SEASON) --gw $(GW) $(ARGS); \
	else \
		uv run fplt archive-benchmarks $(ARGS); \
	fi

# ── Paired archive compare (N2: ledger vs benchmark_projections) ─────────────
# Usage: make archive-compare SEASON=2024-25 MODEL=dc_layered_v2
# Optional: FROM_GW=1 TO_GW=38 ARGS='--sources ep_next openfpl'
archive-compare:
	@if [ -z "$(SEASON)" ] || [ -z "$(MODEL)" ]; then \
		echo "Usage: make archive-compare SEASON=2024-25 MODEL=dc_layered_v2"; exit 1; \
	fi
	uv run fplt archive-compare --season $(SEASON) --model $(MODEL) \
		$(if $(FROM_GW),--from-gw $(FROM_GW),) \
		$(if $(TO_GW),--to-gw $(TO_GW),) \
		$(ARGS)

# ── Post-downtime ops catch-up (no Dagster UI) ────────────────────────────────
# After travel / long sleep: fetch snapshots → restage FPL → understat +
# api_football ingest → catch_up missing checked GWs → print status (+ auth).
# Same library steps as the Dagster refresh_staged / understat_ingest /
# api_football_ingest / catch_up jobs. Optional: ARGS='--no-notify --horizon 3'
ops-catchup:
	uv run fplt ops-catchup $(ARGS)

# ── Idempotent ops tick (GHA cron / local) ────────────────────────────────────
# fetch → restage → catch_up → pre_deadline. Extra ticks no-op when idle.
# Deadline checkpoints (ops-02) build on this later. Optional: ARGS='--no-notify'
ops-tick:
	uv run fplt ops-tick $(ARGS)

# ── Promote / roll back the moving prod tag (ops-04) ──────────────────────────
# Gates: CI green, outside freeze (T−24h → kickoff), smoke plan loads production.
# Usage: make release ARGS='--ref main'
# Rollback: make release ARGS='--ref prod-YYYYMMDD-<sha>'
# Dry-run: make release ARGS='--ref main --dry-run'
# Prefer GHA: gh workflow run release.yml -f ref=main
release:
	uv run fplt release promote $(ARGS)

# ── Dagster UI ────────────────────────────────────────────────────────────────
dagster-dev:
	uv run dagster dev -m fplt.orchestration

# ── fplt-mc native kernel (tree: mc/; never compile on experiment VMs) ───────
# Cloud agents / board VMs: uv sync --frozen only (no-build-package once pinned).
# C++ editors: make mc-dev. Prefer GHA for release wheels.
mc-dev: ## editable C++ build into the venv (needs cmake, ninja, C++20)
	uv sync --frozen
	uv pip install --no-deps -e ./mc

mc-test: ## ctest L0 + fplt-mc Python smoke
	cmake -S mc -B mc/build -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DFPLT_MC_BUILD_PYTHON=OFF \
		-DFPLT_MC_BUILD_TESTS=ON
	cmake --build mc/build
	ctest --test-dir mc/build --output-on-failure
	uv run pytest mc/tests -q

mc-format: ## clang-format -i on mc C++ sources
	uv pip install -q 'clang-format==18.1.8'
	find mc \( -name '*.hpp' -o -name '*.cpp' -o -name '*.h' -o -name '*.cc' \) \
		-print0 | xargs -0 uv run clang-format -i

mc-golden: ## regenerate tests/fixtures/mc_golden (deliberate; review in PR)
	@echo "mc-golden: edit mc/tests/test_simulate_parity.py::test_l5_golden_hashes and re-run to refresh hashes; do not auto-overwrite."

mc-ci: ## CI: L0 ctest + wheel into dist/ (manylinux/macOS runners)
	cmake -S mc -B mc/build-ci -G Ninja \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CXX_COMPILER="$${CMAKE_CXX_COMPILER:-g++}" \
		-DFPLT_MC_BUILD_PYTHON=OFF \
		-DFPLT_MC_BUILD_TESTS=ON
	cmake --build mc/build-ci
	ctest --test-dir mc/build-ci --output-on-failure
	bash mc/scripts/build_wheel.sh

# ── Clean generated artefacts ─────────────────────────────────────────────────
clean:
	rm -rf data/features/ outputs/ .pytest_cache/ mc/build mc/build-ci mc/build-san dist/
