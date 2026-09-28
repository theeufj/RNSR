"""Retry attempts and failed batches must obey the query's remaining budget."""

import asyncio
import time

import pytest
from tenacity import wait_none

from rnsr.errors import BudgetExhausted, PermanentProviderError
from rnsr.llm.batch import map_prompts
from rnsr.llm.governor import SpendCeilingExceeded
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


async def test_uncapped_worker_retries_past_former_attempt_limit(monkeypatch):
    monkeypatch.setattr("rnsr.llm.batch.wait_exponential", lambda **_: wait_none())
    client = MockLLM(fail_times=7, default="classified")
    output = await map_prompts(client, ["row"], model="m", attempts=None,
                               request_timeout_s=1, raise_permanent_errors=True)
    assert output[0].text == "classified" and len(client.calls) == 8


async def test_uncapped_worker_request_timeouts_retry_without_losing_question(monkeypatch):
    monkeypatch.setattr("rnsr.llm.batch.wait_exponential", lambda **_: wait_none())

    class InitiallyStuck(MockLLM):
        started = 0
        timed_out = 0

        async def complete(self, *args, **kwargs):
            self.started += 1
            if self.started <= 5:
                try:
                    await asyncio.Event().wait()
                finally:
                    self.timed_out += 1
            return await super().complete(*args, **kwargs)

    client = InitiallyStuck(default="done")
    output = await map_prompts(client, ["row"], model="m", attempts=None,
                               request_timeout_s=0.01, raise_permanent_errors=True)
    assert output[0].text == "done" and client.timed_out == 5


@pytest.mark.parametrize("message", ["HTTP 401 invalid API key", "429 insufficient_quota",
                                      "credit balance is too low"])
async def test_uncapped_worker_surfaces_permanent_provider_failure(message):
    class Rejected(MockLLM):
        attempts = 0

        async def complete(self, *args, **kwargs):
            self.attempts += 1
            raise ValueError(message)

    client = Rejected()
    with pytest.raises(PermanentProviderError, match=message):
        await map_prompts(client, ["row"], model="m", attempts=None,
                          request_timeout_s=1, raise_permanent_errors=True)
    assert client.attempts == 1


async def test_uncapped_worker_retries_remain_cancellable(monkeypatch):
    monkeypatch.setattr("rnsr.llm.batch.wait_exponential", lambda **_: wait_none())
    retrying = asyncio.Event()

    class Offline(MockLLM):
        attempts = 0

        async def complete(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts >= 6:
                retrying.set()
            raise ConnectionError("provider temporarily offline")

    client = Offline()
    task = asyncio.create_task(map_prompts(client, ["row"], model="m", attempts=None,
                                          request_timeout_s=1, raise_permanent_errors=True))
    try:
        await asyncio.wait_for(retrying.wait(), 2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert client.attempts >= 6


@pytest.mark.parametrize("failure", [BudgetExhausted("max_sub_calls", 500, 500),
                                     SpendCeilingExceeded(0.5, 0.5)])
async def test_explicit_limits_never_enter_unlimited_retry_loop(failure, monkeypatch):
    monkeypatch.setattr("rnsr.llm.batch.wait_exponential", lambda **_: wait_none())
    attempts = 0

    def reserve():
        nonlocal attempts
        attempts += 1
        raise failure

    with pytest.raises(type(failure)):
        await asyncio.wait_for(map_prompts(MockLLM(), ["q"], model="m", attempts=None,
                                          on_attempt=reserve), 1)
    assert attempts == 1


@pytest.mark.parametrize("status", [408, 409, 429, 500, 529])
async def test_worker_retries_transient_http_status_without_matching_text(status, monkeypatch):
    monkeypatch.setattr("rnsr.llm.batch.wait_exponential", lambda **_: wait_none())

    class HttpFailure(Exception):
        status_code = status

    class TemporarilyRejected(MockLLM):
        attempts = 0

        async def complete(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts <= 5:
                raise HttpFailure("try again")
            return await super().complete(*args, **kwargs)

    client = TemporarilyRejected(default="done")
    output = await map_prompts(client, ["row"], model="m", attempts=None,
                               request_timeout_s=1, raise_permanent_errors=True)
    assert output[0].text == "done" and client.attempts == 6
