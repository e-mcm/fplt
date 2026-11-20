.PHONY: test check lint typecheck leaderboard eval-board compare-models dagster-dev clean bootstrap-vaastav

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
compare-models:
	@if [ -z "$(SEASON)" ]; then echo "Usage: make compare-models SEASON=2024-25"; exit 1; fi
	uv run fplt compare --season $(SEASON) $(ARGS)

# ── Dagster UI ────────────────────────────────────────────────────────────────
dagster-dev:
	uv run dagster dev -m fplt.orchestration

# ── Clean generated artefacts ─────────────────────────────────────────────────
clean:
	rm -rf data/features/ outputs/ .pytest_cache/
