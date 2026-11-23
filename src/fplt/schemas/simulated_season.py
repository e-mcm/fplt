"""Pandera schemas for simulated-season decision evaluation (N6)."""

from __future__ import annotations

import pandera.polars as pa


class TruthRateSchema(pa.DataFrameModel):
    """Independent Poisson intensities (smoke / legacy only).

    ``rate`` is the Poisson mean for ``actual_pts``. Prefer joint model draws
    via ``JointTruthSimSchema`` / ``path_actuals_from_joint_sims`` for SE
    claims; independent Poisson around realised points is not a generative
    truth model (see Project doc ``sim-season-joint-truth.md``).
    """

    gw: int = pa.Field(ge=1, le=38)
    player_key: int
    rate: float = pa.Field(ge=0)
    minutes_rate: float = pa.Field(ge=0, le=90, nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["gw", "player_key"]


class JointTruthSimSchema(pa.DataFrameModel):
    """One joint truth draw at player–gameweek grain.

    Rows that share ``sim`` are a common-random-number outcome across the
    player pool (scoreline-correlated within each ``gw``). Stack sims into
    season paths with ``path_actuals_from_joint_sims``.
    """

    sim: int = pa.Field(ge=0)
    gw: int = pa.Field(ge=1, le=38)
    player_key: int
    points: float
    minutes: int = pa.Field(ge=0, le=90)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["sim", "gw", "player_key"]


class SimulatedPathActualsSchema(pa.DataFrameModel):
    """One CRN path of realised points at player–gameweek grain."""

    path_id: int = pa.Field(ge=0)
    gw: int = pa.Field(ge=1, le=38)
    player_key: int
    actual_pts: int
    actual_minutes: int = pa.Field(ge=0, le=90)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["path_id", "gw", "player_key"]


class SimulatedSeasonPathSchema(pa.DataFrameModel):
    """One policy's season total on one shared outcome path."""

    path_id: int = pa.Field(ge=0)
    policy_id: str
    season: str
    total_points: int
    gameweeks: int = pa.Field(ge=0)
    attainable_points: int = pa.Field(ge=0, nullable=True)
    share_of_attainable: float = pa.Field(ge=0, nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["path_id", "policy_id"]


class PolicyDiffSESchema(pa.DataFrameModel):
    """Paired mean difference and Monte Carlo SE between two policies.

    Differences are ``treatment − control`` on the same CRN paths. ``se`` is
    the sample standard deviation of the paired diffs divided by ``sqrt(n)``.
    """

    metric: str
    control_id: str
    treatment_id: str
    n_paths: int = pa.Field(ge=1)
    mean_diff: float
    se: float = pa.Field(ge=0)
    sd: float = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["metric", "control_id", "treatment_id"]
