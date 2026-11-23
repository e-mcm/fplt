"""Python smoke tests for the fplt-mc skeleton."""

from __future__ import annotations

import math

import fplt_mc


def test_kernel_api_is_positive() -> None:
    """KERNEL_API must be a positive int for agent preflight."""
    assert isinstance(fplt_mc.KERNEL_API, int)
    assert fplt_mc.KERNEL_API >= 1


def test_build_info_keys() -> None:
    """build_info exposes provenance fields used in audit.json."""
    info = fplt_mc.build_info()
    for key in ("version", "git_sha", "kernel_api", "compiler", "flags", "rng"):
        assert key in info
        assert isinstance(info[key], str)
        assert info[key]  # non-empty (git_sha may be "unknown" in isolated builds)
    assert info["kernel_api"] == str(fplt_mc.KERNEL_API)
    assert info["version"] == "0.2.0"


def test_philox_u01_bounds() -> None:
    """Philox uniforms land in [0, 1)."""
    u = fplt_mc.philox_u01(1, 0, 0, 0)
    assert 0.0 <= u < 1.0


def test_mathx_exp_stub() -> None:
    """mathx_exp stub matches libm for common Poisson arguments."""
    assert fplt_mc.mathx_exp(0.0) == 1.0
    assert math.isclose(fplt_mc.mathx_exp(-2.0), math.exp(-2.0))
