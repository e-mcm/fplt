.PHONY: test check lint typecheck leaderboard eval-board compare-models eval-components promote-eval bakeoff decisions-board decisions-explain horizon-decisions plan apply archive-benchmarks dagster-dev clean bootstrap-vaastav fetch-understat fetch-football-data fetch-overall-rank

# ── Historical data (vaastav) ─────────────────────────────────────────────────
bootstrap-vaastav:
	uv run python scripts/bootstrap_vaastav.py

# ── Understat raw capture (matches + league teamsData Deep/PPDA) ──────────────
# Usage: make fetch-understat SEASON=2024-25
# Writes write-once gzipped JSON under $$FPLT_DATA_DIR/raw/understat/...
fetch-understat:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make fetch-understat SEASON=2024-25"; exit 1; fi
	uv run python scripts/fetch_understat.py --out "$${FPLT_DATA_DIR:-./data}/raw" --season $(SEASON)

# ── football-data.co.uk E0 odds (closing + scheduled prematch) ────────────────
# Usage: make fetch-football-data SEASON=2024-25
# Writes write-once CSV under $$FPLT_DATA_DIR/raw/football_data/season=.../E0.csv
fetch-football-data:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make fetch-football-data SEASON=2024-25"; exit 1; fi
	uv run python scripts/fetch_football_data.py --out "$${FPLT_DATA_DIR:-./data}/raw" --season $(SEASON)

# ── FPL overall-league points-at-rank (live percentile mapping) ───────────────
# Usage: make fetch-overall-rank
# Optional: SEASON=2026-27 ARGS='--rank 10000 --rank 100000'
# Writes write-once gzipped JSON under $$FPLT_DATA_DIR/raw/fpl_api/.../overall_rank/
fetch-overall-rank:
	uv run python scripts/fetch_overall_rank.py --out "$${FPLT_DATA_DIR:-./data}/raw" \
		$(if $(SEASON),--season $(SEASON),) $(ARGS)

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	uv run pytest --cov=fplt --cov-report=term-missing --cov-report=xml

check:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts
	uv run pyright
	uv run pytest -m "not slow"

# ── Lint + type-check ─────────────────────────────────────────────────────────
lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

typecheck:
	uv run pyright src/

# ── Leaderboard ───────────────────────────────────────────────────────────────
leaderboard:
	uv run streamlit run apps/leaderboard.py

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
eval-components:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make eval-components SEASON=2024-25"; exit 1; fi
	uv run fplt eval-components --season $(SEASON) $(ARGS)

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

# Multi-horizon decision bakeoff. Prediction ledger must already cover max H
# (backtest --steps >= max HORIZONS). Archives each run because ledger/decisions
# overwrites by model×gw (no horizon key).
# Usage: make horizon-decisions SEASON=2024-25
# Optional: HORIZONS='1 2 3 4' ARGS='--models openfpl_v1 dc_layered_v2 --from-gw 1 --to-gw 38'
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

# ── Transfer plan (uses FPLT_ENTRY_ID when set) ────────────────────────────────
# Usage: make plan SEASON=2025-26 GW=10
# Optional: MODEL=rolling_mean_v1 HORIZON=3 ARGS='--entry-json picks.json'
plan:
	@if [ -z "$(SEASON)" ] || [ -z "$(GW)" ]; then \
		echo "Usage: make plan SEASON=2025-26 GW=10"; exit 1; fi
	uv run fplt plan \
		--model $(or $(MODEL),rolling_mean_v1) \
		--season $(SEASON) \
		--gw $(GW) \
		--horizon $(or $(HORIZON),3) \
		$(ARGS)

# ── Apply saved plan to FPL (dry-run unless CONFIRM=1) ─────────────────────────
# Usage: make apply SEASON=2025-26 GW=10
# Optional: MODEL=rolling_mean_v1 HORIZON=3 CONFIRM=1 ARGS='--transfers-only'
apply:
	@if [ -z "$(SEASON)" ] || [ -z "$(GW)" ]; then \
		echo "Usage: make apply SEASON=2025-26 GW=10"; exit 1; fi
	uv run fplt apply \
		--model $(or $(MODEL),rolling_mean_v1) \
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

# ── Dagster UI ────────────────────────────────────────────────────────────────
dagster-dev:
	uv run dagster dev -m fplt.orchestration

# ── Clean generated artefacts ─────────────────────────────────────────────────
clean:
	rm -rf data/features/ outputs/ .pytest_cache/
