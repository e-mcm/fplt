"""OpenFPL predictions from an isolated subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import polars as pl

from fplt.data.pit import PITView
from fplt.models.predictions import prediction_frame
from fplt.models.spec import ModelSpec, register_model, write_identity
from fplt.models.universe import player_fixtures
from fplt.types import GWKey

_RUNNER = Path("external") / "openfpl" / "predict.py"


@register_model("openfpl")
class OpenFPLPredictor:
    """Call the OpenFPL runner and map its player names onto ``player_key``.

    ``fit`` does nothing: the published ensemble is already fitted. Target
    seasons before ``compare_from`` (2024-25 in the YAML spec) raise, because
    those seasons are in OpenFPL's training data. Without ``model_dir`` the
    runner returns zeros; those zeros are not OpenFPL scores. A non-empty
    ``model_dir`` still cannot produce real scores until the adapter builds
    OpenFPL's sample feature matrix (see ``docs/openfpl.md``).
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
        _require_feature_path(self.spec)
        response = _invoke(self.spec, _payload(request))
        scored = _attach(request, response, names)
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
    """Map ``web_name`` to ``player_key`` for the seasons being scored."""
    players = (
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
    counts = players.group_by("season", "openfpl_player_id").len()
    duplicated = counts.filter(pl.col("len") > 1)
    if not duplicated.is_empty():
        season = duplicated["season"][0]
        name = duplicated["openfpl_player_id"][0]
        raise ValueError(
            f"OpenFPL id {name!r} matches more than one player in {season}"
        )
    return players


def _payload(request: pl.DataFrame) -> dict[str, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for row in request.iter_rows(named=True):
        rows.append(
            {
                "row_id": int(row["row_id"]),
                "season": row["season"],
                "gw": int(row["gw"]),
                "position": row["position"],
                "openfpl_player_id": row["openfpl_player_id"],
            }
        )
    return {"rows": rows}


def _invoke(
    spec: ModelSpec, payload: dict[str, list[dict[str, object]]]
) -> pl.DataFrame:
    completed = subprocess.run(
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


def _require_feature_path(spec: ModelSpec) -> None:
    """Refuse a real OpenFPL checkout until sample features are wired.

    Empty ``model_dir`` keeps the zero path for the contract suite. A custom
    ``params.command`` (tests) bypasses this check. A missing checkout still
    raises as before; a present checkout raises until the adapter builds the
    OpenFPL sample feature matrix.
    """
    if _string_list(spec.params.get("command")) is not None:
        return
    model_dir = spec.params.get("model_dir")
    if not isinstance(model_dir, str) or not model_dir.strip():
        return
    scaler = Path(model_dir) / "models" / "xscaler.save"
    if not scaler.is_file():
        raise RuntimeError(f"OpenFPL models not found in {model_dir}")
    raise ValueError(
        "OpenFPL model_dir is set but the adapter does not build the OpenFPL "
        "sample feature matrix yet (xscaler expects ~228 columns). Leave "
        "model_dir empty for offline zeros, or see docs/openfpl.md."
    )


def _command(spec: ModelSpec) -> list[str]:
    command = _string_list(spec.params.get("command"))
    if command is not None:
        return command
    python = spec.params.get("python")
    executable = python if isinstance(python, str) and python else _default_python()
    argv = [executable, str(_runner_script())]
    model_dir = spec.params.get("model_dir")
    if isinstance(model_dir, str) and model_dir:
        argv.extend(["--model-dir", model_dir])
    return argv


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


def _default_python() -> str:
    """Prefer the isolated OpenFPL venv when ``uv sync`` has been run there."""
    venv_python = _repo_root() / "external" / "openfpl" / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


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


def _attach(
    request: pl.DataFrame, response: pl.DataFrame, names: pl.DataFrame
) -> pl.DataFrame:
    """Join runner output back to fixtures by mapping the returned player id."""
    if response.height != request.height:
        raise ValueError("openfpl runner did not return one row per player-fixture")
    keyed = response.join(
        names.select("season", "openfpl_player_id", "player_key"),
        on=["season", "openfpl_player_id"],
        how="left",
    )
    if keyed["player_key"].null_count():
        raise ValueError("openfpl runner returned a player id that is not mapped")
    aligned = request.select(
        "row_id",
        "season",
        "gw",
        "fixture_id",
        pl.col("openfpl_player_id").alias("sent_id"),
    ).join(keyed, on=["row_id", "season"], how="inner")
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
