.PHONY: train predict test check lint typecheck clean

# ── Training ──────────────────────────────────────────────────────────────────
train:
	uv run python src/fplt/pipeline/train.py

# ── Prediction  (GW is required: make predict GW=38) ─────────────────────────
predict:
	@if [ -z "$(GW)" ]; then echo "Usage: make predict GW=<number>"; exit 1; fi
	uv run python src/fplt/pipeline/predict.py pipeline.gw=$(GW)

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

# ── MLflow UI ─────────────────────────────────────────────────────────────────
mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# ── Clean generated artefacts ─────────────────────────────────────────────────
clean:
	rm -rf data/features/ outputs/ mlruns/ mlflow.db .hydra/
