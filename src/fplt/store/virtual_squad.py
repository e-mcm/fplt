"""Virtual-line squad chain: ``squad_state`` in/out + ``prev_run_id``.

Candidate lines carry bank, FTs and chips from their previous plan run
(data model §4.1 / step 6b). Null ``prev_run_id`` initialises from the real
squad (mid-season start).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from fplt.optimise.state import ChipInventory, ChipPlay, OwnedPlayer, SquadState
from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.squad_state_table import SquadStateTableSchema
from fplt.scoring.chips import ChipName
from fplt.store.runs import run_dir
from fplt.types import GWKey

logger = logging.getLogger(__name__)

_IN = "plans/squad_state_in.parquet"
_OUT = "plans/squad_state_out.parquet"


def squad_state_to_frame(
    state: SquadState,
    *,
    run_id: str,
    as_of: GWKey,
    deadline_utc: datetime,
) -> pl.DataFrame:
    """Serialize one ``SquadState`` to a validated one-row table."""
    owned = [
        {
            "player_key": p.player_key,
            "selling_price": p.selling_price,
            "purchase_price": p.purchase_price,
        }
        for p in state.owned
    ]
    remaining = {name: int(n) for name, n in state.chips.remaining.items()}
    played = [{"name": play.name, "gw": play.gw} for play in state.chips.played]
    frame = pl.DataFrame(
        {
            "run_id": [run_id],
            "as_of_season": [as_of.season],
            "as_of_gw": [as_of.gw],
            "deadline_utc": [deadline_utc],
            "schema_version": [SCHEMA_VERSION],
            "bank": [state.bank],
            "free_transfers": [state.free_transfers],
            "ft_bank_cap": [state.ft_bank_cap],
            "hit_cost": [state.hit_cost],
            "decay": [state.decay],
            "bench_weight": [state.bench_weight],
            "owned": [json.dumps(owned, sort_keys=True)],
            "chips_remaining": [json.dumps(remaining, sort_keys=True)],
            "chips_played": [json.dumps(played, sort_keys=True)],
        }
    )
    return SquadStateTableSchema.validate(frame)


def frame_to_squad_state(frame: pl.DataFrame) -> SquadState:
    """Deserialize the first row of a squad-state table to ``SquadState``."""
    if frame.is_empty():
        raise ValueError("squad_state frame is empty")
    row = frame.row(0, named=True)
    owned = [OwnedPlayer.model_validate(x) for x in json.loads(str(row["owned"]))]
    played = [ChipPlay.model_validate(x) for x in json.loads(str(row["chips_played"]))]
    remaining = json.loads(str(row["chips_remaining"]))
    return SquadState(
        bank=int(row["bank"]),
        free_transfers=int(row["free_transfers"]),
        ft_bank_cap=int(row["ft_bank_cap"]),
        hit_cost=int(row["hit_cost"]),
        decay=float(row["decay"]),
        bench_weight=float(row["bench_weight"]),
        owned=owned,
        chips=ChipInventory(remaining=remaining, played=played),
    )


def load_squad_state_out(data_dir: Path, prev_run_id: str) -> SquadState:
    """Load ``plans/squad_state_out.parquet`` from a succeeded plan run."""
    path = run_dir(data_dir, prev_run_id)
    if not (path / "_SUCCESS").is_file():
        raise FileNotFoundError(f"run {prev_run_id} has no _SUCCESS")
    table = path / _OUT
    if not table.is_file():
        raise FileNotFoundError(f"missing {table}")
    return frame_to_squad_state(pl.read_parquet(table))


def resolve_virtual_squad(
    data_dir: Path, *, prev_run_id: str | None, real_squad: SquadState
) -> SquadState:
    """Input squad for a virtual line: prev out-table, else ``real_squad``."""
    if prev_run_id is None:
        logger.info("virtual squad: initialising from real squad")
        return real_squad
    return load_squad_state_out(data_dir, prev_run_id)


def write_squad_state_pair(
    writer: Any,
    *,
    state_in: SquadState,
    state_out: SquadState,
    as_of: GWKey,
    deadline_utc: datetime,
) -> tuple[Path, Path]:
    """Write ``squad_state_in`` / ``out`` on an open ``RunWriter``."""
    run_id = str(writer.run_id)
    path_in = writer.write_parquet(
        _IN,
        squad_state_to_frame(
            state_in, run_id=run_id, as_of=as_of, deadline_utc=deadline_utc
        ),
    )
    path_out = writer.write_parquet(
        _OUT,
        squad_state_to_frame(
            state_out, run_id=run_id, as_of=as_of, deadline_utc=deadline_utc
        ),
    )
    return path_in, path_out


def chip_remaining(state: SquadState, name: ChipName) -> int:
    """Remaining uses of ``name`` on ``state``."""
    return int(state.chips.remaining[name])
