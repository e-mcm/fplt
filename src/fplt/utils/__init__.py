"""Shared utilities."""

import functools
import time
from collections.abc import Callable


def timed[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator that prints how long a function took to run."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{fn.__name__}  {elapsed:.2f}s")
        return result

    return wrapper
