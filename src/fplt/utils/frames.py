"""DataFrame type guards (assert replacements for bandit S101)."""

from __future__ import annotations

import polars as pl


def require_dataframe(value: object, *, what: str = "schema.validate") -> pl.DataFrame:
    """Return ``value`` as a DataFrame or raise ``TypeError``.

    Pandera's ``validate`` is typed as returning ``Any``; call sites used
    ``assert isinstance(...)`` which disappears under ``python -O``. Raise
    instead so the check stays in optimised builds.

    Args:
        value: Object expected to be a polars DataFrame.
        what: Label for the error message (usually the producing call).

    Returns:
        ``value`` when it is a ``pl.DataFrame``.

    Raises:
        TypeError: ``value`` is not a DataFrame.
    """
    if not isinstance(value, pl.DataFrame):
        raise TypeError(f"{what} returned {type(value).__name__}, expected DataFrame")
    return value


def require_not_none[T](value: T | None, *, what: str) -> T:
    """Return ``value`` or raise ``ValueError`` when it is ``None``.

    Args:
        value: Possibly-null value.
        what: Label for the error message.

    Returns:
        The non-``None`` value.

    Raises:
        ValueError: ``value`` is ``None``.
    """
    if value is None:
        raise ValueError(f"{what} is required")
    return value
