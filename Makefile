.PHONY: train predict test lint typecheck clean

# ── Training ──────────────────────────────────────────────────────────────────
train:
	uv run python src/fplt/pipeline/train.py

# ── Prediction  (GW is required: make predict GW=38) ─────────────────────────
predict:
	@if [ -z "$(GW)" ]; then echo "Usage: make predict GW=<number>"; exit 1; fi
	uv run python src/fplt/pipeline/predict.py pipeline.gw=$(GW)

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v

# ── Lint + type-check ─────────────────────────────────────────────────────────
lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

typecheck:
	uv run pyright src/

# ── MLflow UI ─────────────────────────────────────────────────────────────────
mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# ── Clean generated artefacts ─────────────────────────────────────────────────
clean:
	rm -rf data/features/ outputs/ mlruns/ mlflow.db .hydra/
