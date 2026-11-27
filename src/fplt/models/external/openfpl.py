"""OpenFPL predictions from an isolated subprocess."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, cast

import polars as pl

from fplt.data.pit import PITView
from fplt.models.external.openfpl_samples import build_openfpl_features
from fplt.models.predictions import prediction_frame
from fplt.models.spec import ModelSpec, register_model, write_identity
from fplt.models.universe import player_fixtures
from fplt.types import GWKey
from fplt.utils.process import run_command

_RUNNER = Path("external") / "openfpl" / "predict.py"
DEFAULT_MODEL_DIR = "external/openfpl/OpenFPL"
_OPENFPL_ORIGIN = "https://github.com/daniegr/OpenFPL.git"

logger = logging.getLogger(__name__)


@register_model("openfpl")
class OpenFPLPredictor:
    """Call the OpenFPL runner and keep each score on its request ``row_id``.

    ``web_name`` is sent as ``openfpl_player_id`` and is not unique. The
    returned score is joined back by ``row_id``, which already carries
    ``player_key``.

    ``fit`` does nothing: the published ensemble is already fitted. Target
    seasons before ``compare_from`` (2024-25 in the YAML spec) raise, because
    those seasons are in OpenFPL's training data. ``model_dir`` defaults to
    ``external/openfpl/OpenFPL``, fetched on first predict. An empty
    ``model_dir`` or a checkout without ``models/xscaler.save`` raises.
    Each row carries an OpenFPL feature map built from PIT FPL + Understat
    tables (including Deep/PPDA when ``understat_team_matches`` is staged).
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def fit(self, view: PITView) -> None:
        """Do nothing. OpenFPL ships a fitted ensemble."""
        del view

    def predict(self, view: PITView, horizon: list[GWKey]) -> pl.DataFrame:
        """Score every player-fixture in ``horizon``. Returns ``PredictionSchema``."""
        _require_comparable(view.as_of, horizon, _compare_from(self.spec))
        rows = player_fixtures(view, horizon)
        if rows.is_empty():
            zero = pl.lit(0.0)
            return prediction_frame(self.spec, view.as_of, rows, zero, zero, zero, zero)
        seasons = rows["season"].unique().to_list()
        names = _player_names(view, [str(season) for season in seasons])
        joined = rows.join(names, on=["season", "player_key"], how="left")
        if joined["openfpl_player_id"].null_count():
            raise ValueError("a player-fixture has no OpenFPL player id")
        request = joined.with_row_index("row_id")
        features = build_openfpl_features(view, request)
        response = _invoke(self.spec, _payload(request, features))
        scored = _attach(request, response)
        return prediction_frame(
            self.spec,
            view.as_of,
            scored,
            pl.col("prediction"),
            pl.lit(None),
            pl.lit(None),
            pl.lit(None),
        )

    def save(self, path: Path) -> None:
        """Record the spec. Refuses to overwrite an existing artifact."""
        write_identity(path, self.spec)

    @classmethod
    def load(cls, path: Path, spec: ModelSpec) -> OpenFPLPredictor:
        """Load a spec written by ``save``. There is no fitted state."""
        if not (path / "spec.json").is_file():
            raise FileNotFoundError(path)
        return cls(spec)


def _compare_from(spec: ModelSpec) -> str:
    value = spec.params.get("compare_from", "2024-25")
    if not isinstance(value, str):
        raise ValueError("compare_from must be a season such as '2024-25'")
    _season_start(value)
    return value


def _require_comparable(as_of: GWKey, horizon: list[GWKey], compare_from: str) -> None:
    start = _season_start(compare_from)
    seasons = [as_of.season, *[key.season for key in horizon]]
    early = [season for season in seasons if _season_start(season) < start]
    if early:
        raise ValueError(
            f"OpenFPL is only comparable from {compare_from} onwards, got {early[0]}"
        )


def _season_start(season: str) -> int:
    head, _, tail = season.partition("-")
    if len(head) != 4 or not head.isdigit() or len(tail) != 2:
        raise ValueError(f"season must look like 2024-25, got {season!r}")
    return int(head)


def _player_names(view: PITView, seasons: list[str]) -> pl.DataFrame:
    """Label each player with ``web_name``. Shared names stay distinct rows."""
    return (
        view.latest("players", by=["season", "player_key"])
        .filter(pl.col("season").is_in(seasons))
        .select(
            "season",
            "player_key",
            "position",
            pl.col("web_name").alias("openfpl_player_id"),
        )
        .collect()
    )


def _payload(
    request: pl.DataFrame, features: list[dict[str, float]]
) -> dict[str, list[dict[str, object]]]:
    if len(features) != request.height:
        raise ValueError("OpenFPL feature rows must align with player-fixtures")
    rows: list[dict[str, object]] = []
    for row, feature_row in zip(request.iter_rows(named=True), features, strict=True):
        rows.append(
            {
                "row_id": int(row["row_id"]),
                "season": row["season"],
                "gw": int(row["gw"]),
                "position": row["position"],
                "openfpl_player_id": row["openfpl_player_id"],
                "features": feature_row,
            }
        )
    return {"rows": rows}


def _invoke(
    spec: ModelSpec, payload: dict[str, list[dict[str, object]]]
) -> pl.DataFrame:
    completed = run_command(
        _command(spec),
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        timeout=_timeout(spec),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "openfpl runner failed"
        raise RuntimeError(detail)
    return _response_frame(completed.stdout)


def _command(spec: ModelSpec) -> list[str]:
    command = _string_list(spec.params.get("command"))
    if command is not None:
        return command
    model_dir = _resolve_model_dir(spec)
    python = spec.params.get("python")
    executable = python if isinstance(python, str) and python else _scoring_python()
    return [executable, str(_runner_script()), "--model-dir", str(model_dir)]


def _resolve_model_dir(spec: ModelSpec) -> Path:
    """Return the ensemble directory, fetching the default checkout if needed.

    Args:
        spec: OpenFPL ``ModelSpec``. ``params.model_dir`` is a path, absolute
            or relative to the repo root.

    Returns:
        Directory that contains ``models/xscaler.save``.

    Raises:
        RuntimeError: ``model_dir`` is missing.
        FileNotFoundError: A non-default directory has no ensemble.
    """
    raw = spec.params.get("model_dir")
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError(
            "OpenFPL model_dir is not set. The spec should point at "
            f"{DEFAULT_MODEL_DIR}, which is fetched on first predict."
        )
    path = Path(raw.strip())
    if not path.is_absolute():
        path = _repo_root() / path
    if path.resolve() == (_repo_root() / DEFAULT_MODEL_DIR).resolve():
        return ensure_openfpl_checkout(path)
    if not (path / "models" / "xscaler.save").is_file():
        raise FileNotFoundError(f"OpenFPL models not found in {path}")
    return path


def ensure_openfpl_checkout(dest: Path) -> Path:
    """Fetch the published OpenFPL ensemble into ``dest`` when it is missing.

    A directory that already contains ``models/xscaler.save`` is left alone.
    The ensemble is about 750MB, so it is gitignored rather than committed.

    Args:
        dest: Sparse clone of daniegr/OpenFPL. Relative paths are not resolved
            here; callers pass the absolute checkout path.

    Returns:
        ``dest``.

    Raises:
        FileNotFoundError: ``dest`` exists but is not a complete ensemble, or
            the clone finished without ``models/xscaler.save``.
        RuntimeError: ``git`` could not fetch the ensemble.
    """
    marker = dest / "models" / "xscaler.save"
    if marker.is_file():
        return dest
    if dest.exists() and any(dest.iterdir()):
        raise FileNotFoundError(
            f"OpenFPL ensemble at {dest} is incomplete (missing {marker.name})"
        )
    if dest.exists():
        dest.rmdir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("fetching OpenFPL ensemble into %s", dest)
    try:
        run_command(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                _OPENFPL_ORIGIN,
                str(dest),
            ],
            check=True,
        )
        run_command(
            ["git", "sparse-checkout", "set", "models"],
            cwd=dest,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"could not fetch OpenFPL ensemble into {dest}") from exc
    if not marker.is_file():
        raise FileNotFoundError(f"OpenFPL models not found in {dest}")
    return dest


def _scoring_python() -> str:
    """Return the isolated runner interpreter, installing it on first use."""
    root = _repo_root() / "external" / "openfpl"
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    logger.info("installing OpenFPL runner environment in %s", root)
    try:
        run_command(["uv", "sync"], cwd=root, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"could not install the OpenFPL runner in {root}") from exc
    if not venv_python.is_file():
        raise FileNotFoundError(venv_python)
    return str(venv_python)


def _string_list(value: object) -> list[str] | None:
    if not isinstance(value, list) or not value:
        return None
    command: list[str] = []
    for part in cast(list[object], value):
        if not isinstance(part, str):
            return None
        command.append(part)
    return command


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _runner_script() -> Path:
    script = _repo_root() / _RUNNER
    if not script.is_file():
        raise FileNotFoundError(script)
    return script


def _timeout(spec: ModelSpec) -> int:
    value = spec.params.get("timeout_seconds", 120)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("timeout_seconds must be a number")
    return int(value)


def _response_frame(stdout: str) -> pl.DataFrame:
    try:
        parsed: Any = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("openfpl runner did not return JSON") from exc
    records = _records(parsed)
    if not records:
        return pl.DataFrame(
            schema={
                "row_id": pl.Int64(),
                "season": pl.Utf8(),
                "openfpl_player_id": pl.Utf8(),
                "prediction": pl.Float64(),
            }
        )
    return pl.DataFrame(
        {
            "row_id": [_as_int(row.get("row_id"), "row_id") for row in records],
            "season": [_as_str(row.get("season"), "season") for row in records],
            "openfpl_player_id": [
                _as_str(row.get("openfpl_player_id"), "openfpl_player_id")
                for row in records
            ],
            "prediction": [
                _as_float(row.get("prediction"), "prediction") for row in records
            ],
        }
    )


def _records(parsed: object) -> list[dict[str, Any]]:
    if not isinstance(parsed, dict):
        raise RuntimeError("openfpl runner returned an unexpected payload")
    rows = cast(dict[str, Any], parsed).get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("openfpl runner returned an unexpected payload")
    records: list[dict[str, Any]] = []
    for item in cast(list[object], rows):
        if not isinstance(item, dict):
            raise RuntimeError("openfpl runner returned a row that is not an object")
        records.append(cast(dict[str, Any], item))
    return records


def _attach(request: pl.DataFrame, response: pl.DataFrame) -> pl.DataFrame:
    """Join runner output back to fixtures by ``row_id``.

    ``player_key`` comes from the request. ``openfpl_player_id`` is only
    checked so a runner cannot relabel a row.
    """
    if response.height != request.height:
        raise ValueError("openfpl runner did not return one row per player-fixture")
    aligned = request.select(
        "row_id",
        "season",
        "gw",
        "fixture_id",
        "player_key",
        pl.col("openfpl_player_id").alias("sent_id"),
    ).join(response, on=["row_id", "season"], how="inner")
    if aligned.height != request.height:
        raise ValueError("openfpl runner returned a row id that was not requested")
    mismatched = aligned.filter(pl.col("openfpl_player_id") != pl.col("sent_id"))
    if not mismatched.is_empty():
        raise ValueError("openfpl runner changed the player id for a row")
    return aligned


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"openfpl {field} is not an integer")
    return int(value)


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"openfpl {field} is not a number")
    return float(value)


def _as_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"openfpl {field} is not a string")
    return value
