.PHONY: test check lint typecheck leaderboard eval-board compare-models eval-components decisions-board plan apply dagster-dev clean bootstrap-vaastav

# ── Historical data (vaastav) ─────────────────────────────────────────────────
bootstrap-vaastav:
	uv run python scripts/bootstrap_vaastav.py

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

# ── Decision bakeoff (season totals after backtest fills the prediction ledger)
# Usage: make decisions-board SEASON=2024-25
# Optional: ARGS='--models rolling_mean_v1 dc_layered_v1 --from-gw 1 --to-gw 10'
# Optional: ARGS='--summarise-only' to score ledger/decisions without re-simulating
# Horizon should match the backtest --steps used when writing predictions.
decisions-board:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make decisions-board SEASON=2024-25"; exit 1; fi
	uv run fplt decisions --season $(SEASON) --horizon $(or $(HORIZON),2) $(ARGS)

# ── Transfer plan (uses FPLT_ENTRY_ID when set) ────────────────────────────────
# Usage: make plan SEASON=2025-26 GW=10
# Optional: MODEL=rolling_mean_v1 HORIZON=2 ARGS='--entry-json picks.json'
plan:
	@if [ -z "$(SEASON)" ] || [ -z "$(GW)" ]; then \
		echo "Usage: make plan SEASON=2025-26 GW=10"; exit 1; fi
	uv run fplt plan \
		--model $(or $(MODEL),rolling_mean_v1) \
		--season $(SEASON) \
		--gw $(GW) \
		--horizon $(or $(HORIZON),2) \
		$(ARGS)

# ── Apply saved plan to FPL (dry-run unless CONFIRM=1) ─────────────────────────
# Usage: make apply SEASON=2025-26 GW=10
# Optional: MODEL=rolling_mean_v1 HORIZON=2 CONFIRM=1 ARGS='--transfers-only'
apply:
	@if [ -z "$(SEASON)" ] || [ -z "$(GW)" ]; then \
		echo "Usage: make apply SEASON=2025-26 GW=10"; exit 1; fi
	uv run fplt apply \
		--model $(or $(MODEL),rolling_mean_v1) \
		--season $(SEASON) \
		--gw $(GW) \
		--horizon $(or $(HORIZON),2) \
		$(if $(filter 1 yes true,$(CONFIRM)),--confirm,) \
		$(ARGS)

# ── Dagster UI ────────────────────────────────────────────────────────────────
dagster-dev:
	uv run dagster dev -m fplt.orchestration

# ── Clean generated artefacts ─────────────────────────────────────────────────
clean:
	rm -rf data/features/ outputs/ .pytest_cache/
