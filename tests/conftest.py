"""Shared pytest hooks."""

from pathlib import Path

import pytest

_DATA = (
    Path(__file__).resolve().parents[1] / "data" / "2024-25" / "gws" / "merged_gw.csv"
)
_NEEDS_VAASTAV = {
    "test_data.py",
    "test_features.py",
    "test_predict.py",
    "test_predict_strategies.py",
}


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
