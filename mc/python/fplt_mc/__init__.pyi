from typing import Any

import numpy as np
from numpy.typing import NDArray

KERNEL_API: int

def build_info() -> dict[str, Any]: ...
def philox_u01(seed: int, stream: int, sim: int, slot: int) -> float: ...
def mathx_exp(x: float) -> float: ...
def simulate_fixture(
    home_goals: NDArray[np.int64],
    away_goals: NDArray[np.int64],
    side: NDArray[np.uint8],
    position: NDArray[np.uint8],
    minutes: NDArray[np.float64],
    goal_weight: NDArray[np.float64],
    assist_weight: NDArray[np.float64],
    other_points: NDArray[np.float64],
    appearance_long_minutes: int,
    appearance_short: float,
    appearance_long: float,
    goals_by_pos: NDArray[np.float64],
    assists_pts: float,
    cs_by_pos: NDArray[np.float64],
    conceded_every: int,
    conceded_by_pos: NDArray[np.float64],
    own_goal_weight: float,
    p_assist: float,
    cs_red_void_prob: float,
    conceded_scaled: bool,
    sample_other: bool,
    kernel_seed: int,
    delta_goals: NDArray[np.float64] | None = ...,
    delta_cs: NDArray[np.float64] | None = ...,
    delta_scored: NDArray[np.float64] | None = ...,
    return_samples: bool = ...,
) -> dict[str, Any]: ...
