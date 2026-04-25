# Claude Code — Project Rules

## Repo Design

See `docs/design.md` for the full architecture, pipeline stages, model base
classes, and key design decisions. Read it before implementing anything.

## DataFrames — Pandera Required

Every function that accepts or returns a DataFrame must:

- Define a `pandera.DataFrameModel` schema in `src/schemas/`
- Use `@pa.check_types` decorator with typed `DataFrame[SchemaName]` annotations
- Never accept or return a bare `pd.DataFrame` without a schema

Schemas are the contracts between pipeline stages. If you are adding a new
pipeline output, add the schema first, then implement the function.

## Model Base Classes

All models must inherit from `PointModel` or `DistributionalModel` in
`src/models/base.py`. See design doc for the distinction.
Serialization is handled by MLflow — do not use pickle directly.

## Testing

- Unit tests in `tests/unit/` use synthetic data only — no real FPL data
- Never assert on model quality (MAE, RMSE) in pytest
- Model validation goes in `src/pipeline/validate.py` and logs to MLflow

## Walk-Forward CV

Never shuffle time-series splits. Train on GWs 1–N, validate on N+1 onwards.

## Config

All hyperparameters go in `conf/model/`. No hardcoded values in source files.