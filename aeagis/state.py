"""
In-Memory thread-safe mechanism to manage temporary, user-specific configurations,
primarily for dynamically selecting AI models based on the application's self-healing.
"""

import time
import functools

from typing import Callable, Awaitable, ParamSpec, TypeVar


P = ParamSpec("P")  # Capture parameter specification # 3.12+
R = TypeVar("R")  # Capture return type               # 3.12+


def latency() -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """
    Decorator to measure latency of a coroutine execution time.
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"{func.__name__}' in {elapsed_time:.4f} seconds")
        return wrapper

    return decorator
