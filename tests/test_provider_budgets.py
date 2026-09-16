"""Retry attempts and failed batches must obey the query's remaining budget."""

import asyncio
import time

import pytest
from tenacity import wait_none

from rnsr.errors import BudgetExhausted
from rnsr.llm.batch import map_prompts
from rnsr.llm.mock import MockLLM


async def test_failed_provider_attempts_consume_budget(monkeypatch):
    monkeypatch.setattr("rnsr.llm.batch.wait_exponential", lambda **_: wait_none())
    attempts = 0

    def reserve():
        nonlocal attempts
        if attempts >= 2:
            raise BudgetExhausted("max_sub_calls", 2, attempts)
        attempts += 1

    client = MockLLM(fail_times=10)
    with pytest.raises(BudgetExhausted):
        await map_prompts(client, ["q"], model="m", on_attempt=reserve)
    assert attempts == 2


async def test_deadline_cancels_provider_work():
    finished = asyncio.Event()

    class SlowClient:
        async def complete(self, *_args, **_kwargs):
            try:
                await asyncio.sleep(10)
            finally:
                finished.set()

    with pytest.raises(TimeoutError):
        await map_prompts(SlowClient(), ["q"], model="m", deadline=time.monotonic() + 0.02)
    assert finished.is_set()


@pytest.mark.parametrize("kwargs", [{"concurrency": 0}, {"attempts": 0}])
async def test_invalid_batch_bounds_fail_immediately(kwargs):
    with pytest.raises(ValueError):
        await map_prompts(MockLLM(), ["q"], model="m", **kwargs)
