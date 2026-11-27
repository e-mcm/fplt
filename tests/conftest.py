"""Shared pytest hooks."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import settings as hypothesis_settings

from fplt.settings import get_settings

_DATA = (
    Path(__file__).resolve().parents[1] / "data" / "2024-25" / "gws" / "merged_gw.csv"
)
_NEEDS_VAASTAV = {
    "test_data.py",
    "test_features.py",
    "test_predict.py",
    "test_predict_strategies.py",
}

# Mutable flag set by the autouse strip fixture; the guard test asserts this so
# removing the fixture fails even when the host shell has no GITHUB_* / CI (#287).
HOST_CI_ENV_STRIP_STATE: dict[str, bool] = {"ok": False}

# Hypothesis profiles (#306). CI sets HYPOTHESIS_PROFILE=ci; local defaults to
# ``dev`` (fewer examples). Per-test ``@settings(max_examples=…)`` still wins.
hypothesis_settings.register_profile(
    "ci",
    deadline=None,
    print_blob=True,
    derandomize=False,
    max_examples=100,
)
hypothesis_settings.register_profile(
    "dev",
    deadline=None,
    print_blob=False,
    derandomize=False,
    max_examples=25,
)
hypothesis_settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

# Unit tests get a tighter wall than the global 60s (#306).
_UNIT_TIMEOUT_S = 30


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip missing-vaastav tests and apply a tighter unit-test timeout.

    CI does not have ``data/<season>/``. Local runs that have downloaded it
    still execute those tests. Unit tests get ``timeout=30`` unless a test
    already sets its own timeout marker (global default remains 60s).
    """
    del config
    unit_timeout = pytest.mark.timeout(_UNIT_TIMEOUT_S)
    skip_vaastav = None
    if not _DATA.exists():
        skip_vaastav = pytest.mark.skip(reason="vaastav CSVs are not in this checkout")
    for item in items:
        path = Path(str(item.fspath))
        if skip_vaastav is not None and path.name in _NEEDS_VAASTAV:
            item.add_marker(skip_vaastav)
        if path.parent.name == "unit" and not any(
            marker.name == "timeout" for marker in item.iter_markers()
        ):
            item.add_marker(unit_timeout)


@pytest.fixture(autouse=True)
def _strip_host_ci_and_fplt_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove host CI / FPLT env so unit tests do not inherit GHA or secrets.

    Push-to-main jobs set ``GITHUB_REF=refs/heads/main``, which made soft-skip
    R2 tests raise after #278/#284 while the same tests passed on PRs (#287).
    Tests that need a ref or ``FPLT_*`` value must ``monkeypatch.setenv`` it.
    """
    for key in list(os.environ):
        if key == "CI" or key.startswith("GITHUB_") or key.startswith("FPLT_"):
            monkeypatch.delenv(key, raising=False)
    HOST_CI_ENV_STRIP_STATE["ok"] = True
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
