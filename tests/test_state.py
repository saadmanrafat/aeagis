"""
Tests for the state module.
"""

import asyncio
import time
import pytest
from asyncio import Task
from unittest.mock import patch, MagicMock

from aeagis.state import StateManager, _UserOverride, latency


# Enable asyncio support for pytest
@pytest.mark.asyncio
class TestAsyncioEnabled:
    """Base class to enable asyncio support."""

    pass


class TestUserOverride:
    """Test the _UserOverride dataclass."""

    def test_user_override_creation(self):
        """Test creating a _UserOverride instance."""
        override = _UserOverride(model="gpt-4", expires_at=time.time() + 300)
        assert override.model == "gpt-4"
        assert isinstance(override.expires_at, float)

    def test_is_active_active(self):
        """Test that an active override returns True."""
        future_time = time.time() + 100  # 100 seconds in the future
        override = _UserOverride(model="gpt-4", expires_at=future_time)
        assert override.is_active() is True

    def test_is_active_expired(self):
        """Test that an expired override returns False."""
        past_time = time.time() - 100  # 100 seconds in the past
        override = _UserOverride(model="gpt-4", expires_at=past_time)
        assert override.is_active() is False


class TestLatencyDecorator:
    """Test the latency decorator."""

    @latency()
    async def async_test_function(self):
        """Test function for latency decorator."""
        await asyncio.sleep(0.01)  # Sleep briefly to measure time
        return "result"

    @pytest.mark.asyncio
    async def test_latency_decorator(self, caplog):
        """Test that the latency decorator measures execution time."""
        with caplog.at_level("INFO"):
            result = await self.async_test_function()

        assert result == "result"
        # Check that the log message was recorded
        assert (
            len(
                [
                    record
                    for record in caplog.records
                    if "async_test_function" in record.message
                ]
            )
            > 0
        )


class TestStateManager:
    """Test the StateManager class."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test StateManager initialization."""
        state_manager = StateManager(cleanup_interval_sec=30)
        assert state_manager._user_overrides == {}
        assert isinstance(state_manager._lock, asyncio.Lock)
        assert state_manager._cleanup_interval == 30
        assert state_manager._cleanup_task is None

    @pytest.mark.asyncio
    async def test_upgrade_user(self):
        """Test the upgrade_user method."""
        async with StateManager() as state:
            await state.upgrade_user("user_1", "gpt-4-turbo", duration_sec=300)

            assert "user_1" in state._user_overrides
            override = state._user_overrides["user_1"]
            assert override.model == "gpt-4-turbo"
            assert time.time() < override.expires_at  # Should not be expired

    @pytest.mark.asyncio
    async def test_get_model_with_override(self):
        """Test get_model returns override when active."""
        async with StateManager() as state:
            # Set up an active override
            await state.upgrade_user("user_1", "gpt-4-turbo", duration_sec=300)

            # Get model should return the override, not the default
            model = await state.get_model("user_1", "gpt-3.5")
            assert model == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_get_model_with_expired_override(self):
        """Test get_model returns default when override is expired."""
        async with StateManager() as state:
            # Set up an expired override
            await state.upgrade_user(
                "user_1", "gpt-4-turbo", duration_sec=-1
            )  # Negative duration makes it expired

            # Get model should return the default since override is expired
            model = await state.get_model("user_1", "gpt-3.5")
            assert model == "gpt-3.5"

            # Expired override should be cleaned up
            assert "user_1" not in state._user_overrides

    @pytest.mark.asyncio
    async def test_get_model_no_override(self):
        """Test get_model returns default when no override exists."""
        async with StateManager() as state:
            # No override for this user
            model = await state.get_model("user_1", "gpt-3.5")
            assert model == "gpt-3.5"

    @pytest.mark.asyncio
    async def test_get_model_lazy_cleanup(self):
        """Test that expired overrides are cleaned up on access."""
        async with StateManager() as state:
            # Set up an expired override
            await state.upgrade_user(
                "user_1", "gpt-4-turbo", duration_sec=-1
            )  # Expired

            # There should be an entry now
            assert "user_1" in state._user_overrides

            # Access the model, which should clean up the expired entry
            model = await state.get_model("user_1", "gpt-3.5")
            assert model == "gpt-3.5"

            # Entry should now be removed
            assert "user_1" not in state._user_overrides

    @pytest.mark.asyncio
    async def test_periodic_cleanup_removes_expired(self):
        """Test that periodic cleanup removes expired overrides."""
        state = StateManager(cleanup_interval_sec=1)  # Short interval for testing

        # Add an expired override
        await state.upgrade_user(
            "user_1", "gpt-4-turbo", duration_sec=-10
        )  # Already expired
        await state.upgrade_user(
            "user_2", "claude-3", duration_sec=300
        )  # Active override

        assert len(state._user_overrides) == 2
        assert "user_1" in state._user_overrides  # Expired
        assert "user_2" in state._user_overrides  # Active

        # Start the state manager
        await state.__aenter__()

        # Wait for cleanup to occur
        await asyncio.sleep(1.5)  # Wait longer than cleanup interval

        # Stop the state manager to cancel the cleanup task
        await state.__aexit__(None, None, None)

        # Only active override should remain
        assert "user_1" not in state._user_overrides  # Should be cleaned up
        assert "user_2" in state._user_overrides  # Should remain

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test the async context manager functionality."""
        state = StateManager(cleanup_interval_sec=60)

        # Before entering context
        assert state._cleanup_task is None

        # Enter context should start cleanup task
        await state.__aenter__()
        assert state._cleanup_task is not None
        assert isinstance(state._cleanup_task, Task)

        # Exit context should cancel cleanup task
        await state.__aexit__(None, None, None)
        assert state._cleanup_task.done() or state._cleanup_task.cancelled()

    @pytest.mark.asyncio
    async def test_multiple_users(self):
        """Test that multiple users can have different overrides."""
        async with StateManager() as state:
            # Add overrides for multiple users
            await state.upgrade_user("user_1", "gpt-4-turbo", duration_sec=300)
            await state.upgrade_user("user_2", "claude-3", duration_sec=300)

            # Check their models
            model1 = await state.get_model("user_1", "gpt-3.5")
            model2 = await state.get_model("user_2", "gpt-3.5")

            assert model1 == "gpt-4-turbo"
            assert model2 == "claude-3"

            # Default for unknown user
            model3 = await state.get_model("user_3", "gpt-3.5")
            assert model3 == "gpt-3.5"

    @pytest.mark.asyncio
    async def test_override_update(self):
        """Test that updating a user's override works correctly."""
        async with StateManager() as state:
            # Initial override
            await state.upgrade_user("user_1", "gpt-4-turbo", duration_sec=300)
            model = await state.get_model("user_1", "gpt-3.5")
            assert model == "gpt-4-turbo"

            # Update the override
            await state.upgrade_user("user_1", "claude-3", duration_sec=300)
            model = await state.get_model("user_1", "gpt-3.5")
            assert model == "claude-3"

    @pytest.mark.asyncio
    async def test_short_cleanup_interval(self):
        """Test behavior with a very short cleanup interval."""
        state = StateManager(cleanup_interval_sec=0.1)  # Very short interval

        # Add an expired override
        await state.upgrade_user("user_1", "gpt-4-turbo", duration_sec=-1)

        # Start the manager
        await state.__aenter__()

        # Wait a bit for cleanup to occur
        await asyncio.sleep(0.2)

        # Stop the manager
        await state.__aexit__(None, None, None)

        # Expired override should have been cleaned up
        assert "user_1" not in state._user_overrides

    @pytest.mark.asyncio
    async def test_thread_safety_concurrent_upgrade_get(self):
        """Test thread safety with concurrent upgrade and get operations."""
        async with StateManager() as state:
            num_operations = 10

            async def concurrent_operations(user_id: str):
                # Perform multiple upgrade and get operations concurrently
                for i in range(num_operations):
                    await state.upgrade_user(
                        user_id, f"model-{user_id}-{i}", duration_sec=300
                    )
                    model = await state.get_model(user_id, "default-model")
                    assert model.startswith(f"model-{user_id}-")

            # Run concurrent operations for multiple users
            await asyncio.gather(
                concurrent_operations("user_1"),
                concurrent_operations("user_2"),
                concurrent_operations("user_3"),
            )

    @pytest.mark.asyncio
    async def test_thread_safety_concurrent_access(self):
        """Test that concurrent access to the same user doesn't cause issues."""
        async with StateManager() as state:
            # Set an initial override
            await state.upgrade_user("shared_user", "initial-model", duration_sec=300)

            async def access_model(user_id: str, default: str):
                return await state.get_model(user_id, default)

            # Run many concurrent accesses to the same user
            results = await asyncio.gather(
                *[access_model("shared_user", "default-model") for _ in range(20)]
            )

            # All should return the same active override
            assert all(result == "initial-model" for result in results)

    @pytest.mark.asyncio
    async def test_thread_safety_concurrent_upgrades_same_user(self):
        """Test concurrent upgrades to the same user."""
        async with StateManager() as state:
            # Run many concurrent upgrades to the same user
            tasks = []
            for i in range(10):
                task = asyncio.create_task(
                    state.upgrade_user("same_user", f"model-{i}", duration_sec=300)
                )
                tasks.append(task)

            # Wait for all upgrades to complete
            await asyncio.gather(*tasks)

            # Get the final model - should be one of the models that was set
            final_model = await state.get_model("same_user", "default-model")
            assert final_model.startswith("model-")
