"""Score OpenFPL samples in an isolated interpreter.

Inference follows ``play.ipynb`` in github.com/daniegr/OpenFPL (MIT).
``--model-dir`` must be a checkout that contains ``models/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> None:
    """Read a JSON request on stdin and write predictions on stdout."""
    parser = argparse.ArgumentParser(description="Score OpenFPL sample rows.")
    parser.add_argument(
        "--model-dir",
        default="",
        help="Clone of daniegr/OpenFPL containing models/",
    )
    args = parser.parse_args(argv)
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise SystemExit("request must be a JSON object with a rows list")
    rows = [row for row in payload["rows"] if isinstance(row, dict)]
    if len(rows) != len(payload["rows"]):
        raise SystemExit("every request row must be an object")
    if not args.model_dir:
        raise SystemExit("OpenFPL needs --model-dir with models/xscaler.save")
    scored = _score_with_openfpl(rows, Path(args.model_dir))
    json.dump({"rows": scored}, sys.stdout)


def _score_with_openfpl(
    rows: list[dict[str, Any]], model_dir: Path
) -> list[dict[str, object]]:
    """Median ensemble from an OpenFPL checkout. Returns one row per request."""
    models_root = model_dir / "models"
    scaler_path = models_root / "xscaler.save"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"OpenFPL models not found in {model_dir}")

    import joblib
    import numpy as np
    import pandas as pd

    xscaler = joblib.load(scaler_path)
    yscaler = joblib.load(models_root / "yscaler.save")
    features = joblib.load(models_root / "features.save")
    feature_names = [str(name) for name in xscaler.feature_names_in_]
    positions = ["GK", "DEF", "MID", "FWD", "AM"]
    loaded = _load_models(models_root, positions)
    frame = pd.DataFrame(rows)
    if "features" not in frame.columns:
        raise SystemExit("OpenFPL samples need a features object on each row")
    wide = pd.json_normalize(frame["features"])
    missing = [name for name in feature_names if name not in wide.columns]
    if missing:
        raise SystemExit(f"OpenFPL samples are missing {missing[0]}")
    predictions: dict[int, float] = {}
    for position in positions:
        mask = frame["position"] == position
        if not bool(mask.any()):
            continue
        chosen = wide.loc[mask, feature_names]
        scaled = np.nan_to_num(
            xscaler.transform(np.nan_to_num(chosen.to_numpy()).astype("float32"))
        ).astype("float32")
        indices = [feature_names.index(name) for name in features[position]]
        scaled = scaled[:, indices]
        members: list[Any] = []
        for fold in loaded.values():
            members.extend(fold[position])
        stacked = [
            yscaler.inverse_transform(model.predict(scaled).reshape(-1, 1)).reshape(-1)
            for model in members
        ]
        median = np.median(np.vstack(stacked), axis=0)
        for row_id, score in zip(
            frame.loc[mask, "row_id"].tolist(), median.tolist(), strict=True
        ):
            predictions[int(row_id)] = float(score)
    scored: list[dict[str, object]] = []
    for row in rows:
        row_id = int(row["row_id"])
        if row_id not in predictions:
            raise SystemExit(f"no OpenFPL model for position {row.get('position')!r}")
        scored.append(
            {
                "row_id": row_id,
                "season": row["season"],
                "openfpl_player_id": row["openfpl_player_id"],
                "prediction": predictions[row_id],
            }
        )
    return scored


def _load_models(
    models_root: Path, positions: list[str]
) -> dict[int, dict[str, list[Any]]]:
    """Load the five-fold candidate ensembles published with OpenFPL."""
    import joblib

    loaded: dict[int, dict[str, list[Any]]] = {}
    for fold in range(1, 6):
        loaded[fold] = {}
        for position in positions:
            search_dir = models_root / f"cv{fold}_{position}"
            search_log = (search_dir / "search.txt").read_text(encoding="utf-8")
            population = search_log.split("The population is:")[-1]
            candidates = [
                part.split(" ")[0]
                for part in population.split("Candidate ")[1:]
                if part.split(" ")[0]
            ]
            models: list[Any] = []
            for candidate in candidates:
                files = [
                    path
                    for path in (search_dir / candidate).iterdir()
                    if path.is_file()
                ]
                models.append(joblib.load(files[0]))
            loaded[fold][position] = models
    return loaded


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
