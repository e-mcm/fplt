"""Dagster asset factories for features, fitted models, and predictions.

Design §8.3: one asset per registered feature version and per points-predictor
``ModelSpec``. These assets expose lineage in the Dagster UI. CLI ``fplt
predict`` / ``fplt backtest`` and catch-up remain the authoritative runners;
assets call the same library helpers and refuse to replace catch-up sensors.

Assets are unpartitioned (same as staged tables today). Optional run config
selects the as-of deadline; otherwise ``resolve_as_of`` picks the next one.
"""

# Dagster does not ship type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingTypeArgument=false
#
# Do not enable ``from __future__ import annotations`` here: Dagster resolves
# ``config: LineageConfig`` at decoration time.

from collections.abc import Sequence
from pathlib import Path

import dagster as dg
import polars as pl
from dagster import Config
from pandera.errors import SchemaError

# Register points predictors before the factories walk YAML specs.
import fplt.models.baselines  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.composite  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.direct  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.ensemble  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.external.openfpl  # noqa: F401  # pyright: ignore[reportUnusedImport]
from fplt.backtest.predict_live import resolve_as_of
from fplt.backtest.walk_forward import allowed_seasons, horizon_for
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.features.registry import (
    Feature,
    materialise_feature,
    registered_features,
)
from fplt.models.contracts import PointsPredictor
from fplt.models.predictions import predictions_path, write_predictions
from fplt.models.spec import (
    ModelSpec,
    artifact_dir,
    get_model,
    load_model_specs,
)
from fplt.orchestration.assets import staged_ingest, understat_ingest
from fplt.schemas.predictions import PredictionSchema
from fplt.settings import get_settings
from fplt.types import GWKey

_DEFAULT_HORIZON = 5


class LineageConfig(Config):
    """Optional as-of deadline and prediction horizon for lineage assets."""

    season: str | None = None
    gw: int | None = None
    horizon: int = _DEFAULT_HORIZON


def feature_asset_name(name: str, version: int) -> str:
    """Dagster asset name for one feature version."""
    return f"feature__{name}__v{version}"


def model_asset_name(spec_name: str) -> str:
    """Dagster asset name for one fitted ``ModelSpec``."""
    return f"model__{spec_name}"


def predictions_asset_name(spec_name: str) -> str:
    """Dagster asset name for one model's prediction parquet."""
    return f"predictions__{spec_name}"


def points_predictor_specs(
    directory: Path | None = None,
) -> list[ModelSpec]:
    """YAML specs whose ``kind`` is a registered ``PointsPredictor``.

    Component-only YAML files (score, minutes, …) are skipped.
    """
    specs = load_model_specs(directory)
    chosen: list[ModelSpec] = []
    for spec in specs.values():
        try:
            get_model(spec.kind)
        except KeyError:
            continue
        chosen.append(spec)
    return chosen


def ensure_features_loaded() -> list[Feature]:
    """Import feature modules so the registry is populated, then list them."""
    import fplt.features.form as form_features
    import fplt.features.understat as understat_rows

    if form_features.context.__name__ != "context":
        raise RuntimeError("context did not register")
    if understat_rows.understat_share.__name__ != "understat_share":
        raise RuntimeError("understat_share did not register")
    return registered_features()


def resolve_lineage_as_of(
    data_dir: Path,
    config: LineageConfig,
) -> GWKey:
    """Resolve the as-of deadline from run config or staged gameweeks."""
    return resolve_as_of(
        DataStore(data_dir),
        season=config.season,
        gw=config.gw,
    )


def fit_or_load_model(
    data_dir: Path,
    spec: ModelSpec,
    as_of: GWKey,
    view: PITView,
) -> PointsPredictor:
    """Load a fitted artifact for ``as_of``, or fit and save one."""
    dest = artifact_dir(data_dir, spec, as_of)
    cls = get_model(spec.kind)
    if (dest / "spec.json").exists():
        loaded: PointsPredictor = cls.load(dest, spec)
        return loaded
    model: PointsPredictor = cls(spec)
    model.fit(view)
    model.save(dest)
    return model


def write_feature_or_existing(
    view: PITView,
    feature: Feature,
    data_dir: Path,
) -> Path:
    """Materialise a feature parquet, or return the path when it already exists."""
    stamp = f"as_of={view.as_of.season}_gw{view.as_of.gw:02d}"
    path = (
        data_dir
        / "features"
        / feature.name
        / f"v={feature.version}"
        / stamp
        / "part.parquet"
    )
    if path.exists():
        return path
    try:
        return materialise_feature(view, feature.name, feature.version, data_dir)
    except FileExistsError:
        return path


def write_predictions_or_existing(
    data_dir: Path,
    spec: ModelSpec,
    as_of: GWKey,
    frame: pl.DataFrame,
) -> Path:
    """Write predictions once, or return the existing parquet path."""
    path = predictions_path(data_dir, spec, as_of)
    if path.exists():
        return path
    try:
        return write_predictions(data_dir, spec, as_of, frame)
    except FileExistsError:
        return path


def _pit_view(data_dir: Path, as_of: GWKey, spec: ModelSpec | None = None) -> PITView:
    pit = PITData(DataStore(data_dir))
    view = pit.view(as_of)
    if spec is None:
        return view
    seasons = allowed_seasons(as_of, spec.train_window.seasons_back)
    return view.within_seasons(seasons)


def _feature_deps(feature: Feature) -> list[dg.AssetsDefinition]:
    deps: list[dg.AssetsDefinition] = [staged_ingest]
    if "understat_player_matches" in feature.inputs:
        deps.append(understat_ingest)
    return deps


def _feature_keys(specs: Sequence[str]) -> list[str]:
    keys: list[str] = []
    for item in specs:
        name, version = item.split("@", maxsplit=1)
        keys.append(feature_asset_name(name, int(version)))
    return keys


def feature_asset(feature: Feature) -> dg.AssetsDefinition:
    """Build ``feature__<name>__v<version>`` for one registered feature."""

    asset_name = feature_asset_name(feature.name, feature.version)

    def _materialize(config: LineageConfig) -> dg.MaterializeResult:
        data_dir = get_settings().data_dir
        as_of = resolve_lineage_as_of(data_dir, config)
        view = _pit_view(data_dir, as_of)
        path = write_feature_or_existing(view, feature, data_dir)
        frame = pl.read_parquet(path)
        return dg.MaterializeResult(
            metadata={
                "as_of": f"{as_of.season} gw{as_of.gw}",
                "path": dg.MetadataValue.path(str(path)),
                "rows": frame.height,
            }
        )

    _materialize.__name__ = asset_name
    _materialize.__doc__ = (
        f"Materialise feature {feature.name}@{feature.version} for one deadline."
    )
    return dg.asset(
        name=asset_name,
        deps=_feature_deps(feature),
        code_version=str(feature.version),
        description=(
            f"Feature {feature.name} version {feature.version} (grain={feature.grain})."
        ),
    )(_materialize)


def model_asset(spec: ModelSpec) -> dg.AssetsDefinition:
    """Build ``model__<name>`` that fits or loads one points predictor."""

    asset_name = model_asset_name(spec.name)
    feature_deps = _feature_keys(spec.features)
    deps: list[dg.AssetsDefinition | str] = [staged_ingest, *feature_deps]

    def _materialize(config: LineageConfig) -> dg.MaterializeResult:
        data_dir = get_settings().data_dir
        as_of = resolve_lineage_as_of(data_dir, config)
        view = _pit_view(data_dir, as_of, spec)
        model = fit_or_load_model(data_dir, spec, as_of, view)
        del model
        dest = artifact_dir(data_dir, spec, as_of)
        return dg.MaterializeResult(
            metadata={
                "as_of": f"{as_of.season} gw{as_of.gw}",
                "model_id": spec.model_id,
                "path": dg.MetadataValue.path(str(dest)),
            }
        )

    _materialize.__name__ = asset_name
    _materialize.__doc__ = f"Fit or load model spec {spec.name}."
    return dg.asset(
        name=asset_name,
        deps=deps,
        code_version=spec.model_id,
        description=f"Fitted artifact for ModelSpec {spec.name} ({spec.kind}).",
    )(_materialize)


def predictions_asset(spec: ModelSpec) -> dg.AssetsDefinition:
    """Build ``predictions__<name>`` from a fitted model artifact."""

    asset_name = predictions_asset_name(spec.name)
    model_name = model_asset_name(spec.name)

    def _materialize(config: LineageConfig) -> dg.MaterializeResult:
        if config.horizon < 1:
            raise ValueError(f"horizon must be positive, got {config.horizon}")
        data_dir = get_settings().data_dir
        as_of = resolve_lineage_as_of(data_dir, config)
        view = _pit_view(data_dir, as_of, spec)
        model = fit_or_load_model(data_dir, spec, as_of, view)
        predicted = model.predict(view, horizon_for(as_of, config.horizon))
        path = write_predictions_or_existing(data_dir, spec, as_of, predicted)
        frame = pl.read_parquet(path)
        return dg.MaterializeResult(
            metadata={
                "as_of": f"{as_of.season} gw{as_of.gw}",
                "model_id": spec.model_id,
                "path": dg.MetadataValue.path(str(path)),
                "rows": frame.height,
                "horizon": config.horizon,
            }
        )

    _materialize.__name__ = asset_name
    _materialize.__doc__ = f"Write predictions for model spec {spec.name}."
    return dg.asset(
        name=asset_name,
        deps=[model_name],
        code_version=spec.model_id,
        description=f"Prediction parquet for ModelSpec {spec.name}.",
    )(_materialize)


def feature_schema_check(
    asset: dg.AssetsDefinition, feature: Feature
) -> dg.AssetChecksDefinition:
    """Blocking pandera check over materialised feature parquet files."""

    def _check() -> dg.AssetCheckResult:
        root = (
            get_settings().data_dir / "features" / feature.name / f"v={feature.version}"
        )
        paths = sorted(root.rglob("part.parquet")) if root.exists() else []
        if not paths:
            return dg.AssetCheckResult(
                passed=True,
                metadata={"rows": dg.MetadataValue.int(0)},
            )
        rows = 0
        for path in paths:
            frame = pl.read_parquet(path)
            try:
                feature.schema.validate(frame)
            except (SchemaError, pl.exceptions.PolarsError) as exc:
                message = str(exc).splitlines()[0][:500]
                return dg.AssetCheckResult(
                    passed=False,
                    metadata={"failure": dg.MetadataValue.text(message)},
                )
            rows += frame.height
        return dg.AssetCheckResult(
            passed=True,
            metadata={"rows": dg.MetadataValue.int(rows)},
        )

    check_name = f"{feature_asset_name(feature.name, feature.version)}_schema"
    _check.__name__ = check_name
    return dg.asset_check(asset=asset, blocking=True, name=check_name)(_check)


def predictions_schema_check(
    asset: dg.AssetsDefinition, spec: ModelSpec
) -> dg.AssetChecksDefinition:
    """Blocking pandera check over materialised prediction parquet files."""

    def _check() -> dg.AssetCheckResult:
        root = get_settings().data_dir / "predictions" / spec.model_id
        paths = sorted(root.rglob("part.parquet")) if root.exists() else []
        if not paths:
            return dg.AssetCheckResult(
                passed=True,
                metadata={"rows": dg.MetadataValue.int(0)},
            )
        rows = 0
        for path in paths:
            frame = pl.read_parquet(path)
            try:
                PredictionSchema.validate(frame)
            except (SchemaError, pl.exceptions.PolarsError) as exc:
                message = str(exc).splitlines()[0][:500]
                return dg.AssetCheckResult(
                    passed=False,
                    metadata={"failure": dg.MetadataValue.text(message)},
                )
            rows += frame.height
        return dg.AssetCheckResult(
            passed=True,
            metadata={"rows": dg.MetadataValue.int(rows)},
        )

    check_name = f"{predictions_asset_name(spec.name)}_schema"
    _check.__name__ = check_name
    return dg.asset_check(asset=asset, blocking=True, name=check_name)(_check)


def build_feature_assets() -> list[dg.AssetsDefinition]:
    """One asset per registered feature version."""
    return [feature_asset(feature) for feature in ensure_features_loaded()]


def build_model_assets(
    specs: Sequence[ModelSpec] | None = None,
) -> list[dg.AssetsDefinition]:
    """One asset per points-predictor ``ModelSpec``."""
    chosen = list(specs) if specs is not None else points_predictor_specs()
    return [model_asset(spec) for spec in chosen]


def build_predictions_assets(
    specs: Sequence[ModelSpec] | None = None,
) -> list[dg.AssetsDefinition]:
    """One predictions asset per points-predictor ``ModelSpec``."""
    chosen = list(specs) if specs is not None else points_predictor_specs()
    return [predictions_asset(spec) for spec in chosen]


def build_factory_checks(
    feature_assets: Sequence[dg.AssetsDefinition],
    features: Sequence[Feature],
    prediction_assets: Sequence[dg.AssetsDefinition],
    specs: Sequence[ModelSpec],
) -> list[dg.AssetChecksDefinition]:
    """Pandera checks for feature and prediction factory assets."""
    feature_checks = [
        feature_schema_check(asset, feature)
        for asset, feature in zip(feature_assets, features, strict=True)
    ]
    prediction_checks = [
        predictions_schema_check(asset, spec)
        for asset, spec in zip(prediction_assets, specs, strict=True)
    ]
    return [*feature_checks, *prediction_checks]


# Built once at import so Definitions and the lineage job share the same objects.
feature_assets = build_feature_assets()
_predictor_specs = points_predictor_specs()
model_assets = build_model_assets(_predictor_specs)
predictions_assets = build_predictions_assets(_predictor_specs)
factory_checks = build_factory_checks(
    feature_assets,
    ensure_features_loaded(),
    predictions_assets,
    _predictor_specs,
)

lineage_job = dg.define_asset_job(
    name="lineage_features_models",
    selection=dg.AssetSelection.assets(
        *feature_assets,
        *model_assets,
        *predictions_assets,
    ),
    description=(
        "Materialise feature, model, and prediction lineage assets. "
        "Does not replace catch-up or CLI predict."
    ),
)
