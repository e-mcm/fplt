"""Shared utilities."""

import functools
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def timed(fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator that prints how long a function took to run.

    Example:
        >>> @timed
        ... def slow():
        ...     time.sleep(1)
        >>> slow()
        slow  1.00s
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{fn.__name__}  {elapsed:.2f}s")
        return result

    return wrapper
