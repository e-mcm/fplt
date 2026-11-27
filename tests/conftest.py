"""Shared pytest hooks."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

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


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip tests that read the gitignored vaastav checkout when it is absent.

    CI does not have ``data/<season>/``. Local runs that have downloaded it
    still execute these tests.
    """
    del config
    if _DATA.exists():
        return
    skip = pytest.mark.skip(reason="vaastav CSVs are not in this checkout")
    for item in items:
        if Path(str(item.fspath)).name in _NEEDS_VAASTAV:
            item.add_marker(skip)


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
