.PHONY: test check lint typecheck leaderboard dagster-dev clean

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

# ── Dagster UI ────────────────────────────────────────────────────────────────
dagster-dev:
	uv run dagster dev -m fplt.orchestration

# ── Clean generated artefacts ─────────────────────────────────────────────────
clean:
	rm -rf data/features/ outputs/ .pytest_cache/
