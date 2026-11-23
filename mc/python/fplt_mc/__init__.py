"""fplt-mc: native Monte Carlo kernel for fplt.

Cloud agents and VMs must install a prebuilt wheel via ``uv sync --frozen``.
Never compile this package during experiment runs; see ``mc/README.md``.
"""

from __future__ import annotations

from typing import Any

from fplt_mc._core import KERNEL_API as _KERNEL_API
from fplt_mc._core import build_info as _build_info
from fplt_mc._core import mathx_exp, philox_u01, simulate_fixture

KERNEL_API: int = int(_KERNEL_API)

__all__ = [
    "KERNEL_API",
    "build_info",
    "mathx_exp",
    "philox_u01",
    "simulate_fixture",
]


def build_info() -> dict[str, Any]:
    """Return build provenance (version, git SHA, compiler, flags)."""
    return dict(_build_info())
