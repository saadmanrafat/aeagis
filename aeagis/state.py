"""
In-Memory thread-safe mechanism to manage temporary, user-specific configurations,
primarily for dynamically selecting AI models based on the application's self-healing.
"""

import time
import asyncio
import functools
import logging

from dataclasses import dataclass
from typing import Callable, Awaitable, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

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
                logger.info(f"{func.__name__}' in {elapsed_time:.4f} seconds")

        return wrapper

    return decorator


@dataclass
class _UserOverride:
    model: str
    expires_at: float

    def is_active(self) -> bool:
        """
        Checks if override is still within TTL
        :return: bool
        """
        return time.time() < self.expires_at


class StateManager:
    """
    In-Memory State Store for managing user-specific configurations with TTL.

    This class is designed as an async context manager to ensure that
    background cleanup tasks are started and stopped gracefully.

    Usage:
        async with StateManager() as state:
            await state.upgrade_user("user_1", "gpt-4-turbo")
            model = await state.get_model("user_1", "gpt-3.5")
    """

    def __init__(self, cleanup_interval_sec: int = 60):
        self._user_overrides: dict[str, _UserOverride] = {}
        self._lock = asyncio.Lock()
        self._cleanup_interval = cleanup_interval_sec
        self._cleanup_task: asyncio.Task | None = None

    async def __aenter__(self):
        """Starts the background cleanup task when entering the context."""
        logger.info("StateManager starting.")
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Gracefully stops the background cleanup task upon exit."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass  # Cancellation is expected
        logger.info("StateManager stopped.")

    async def _periodic_cleanup(self):
        """Periodically removes expired user overrides from the store."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                async with self._lock:
                    now = time.time()
                    expired_users = [
                        user_id
                        for user_id, data in self._user_overrides.items()
                        if now >= data.expires_at
                    ]
                    if expired_users:
                        logger.info(
                            f"Background cleanup: removing {len(expired_users)} expired entries."
                        )
                        for user_id in expired_users:
                            del self._user_overrides[user_id]
            except asyncio.CancelledError:
                logger.info("Background cleanup task cancelled.")
                break  # Exit the loop cleanly on cancellation

    @latency()
    async def get_model(self, user_id: str, default: str) -> str:
        """
        Returns the premium model if an active override exists, else the default.
        Also performs on-access cleanup for expired entries.
        """
        async with self._lock:
            if override := self._user_overrides.get(user_id):
                if override.is_active():
                    return override.model
                else:
                    # Lazy cleanup on access
                    logger.info(f"Cleaning up expired entry for {user_id} on access.")
                    del self._user_overrides[user_id]
            return default

    @latency()
    async def upgrade_user(self, user_id: str, model: str, duration_sec: int = 300):
        """Sets or updates a temporary model override for a user."""
        async with self._lock:
            override = _UserOverride(model=model, expires_at=time.time() + duration_sec)
            self._user_overrides[user_id] = override
            logger.info(f"User '{user_id}' upgraded to '{model}' for {duration_sec}s.")


state = StateManager()
