"""Root timeout regressions, using real cancellation and no paid providers."""

import asyncio
from types import SimpleNamespace

import pytest

from rnsr.config import Settings
from rnsr.harness.budget import BudgetLedger
from rnsr.harness.root_call import complete_root
from rnsr.llm.base import LLMResponse
from rnsr.llm.governor import Governor, SpendCeilingExceeded, governed


class Events:
    def __init__(self):
        self.rows = []

    def event(self, kind, **fields):
        self.rows.append({"kind": kind, **fields})

    def of_kind(self, kind):
        return [row for row in self.rows if row["kind"] == kind]


def request(client, events, ledger=None, **kwargs):
    return complete_root(client, "question", model="mock", system="system",
                         ledger=ledger or BudgetLedger(), trajectory=events, **kwargs)


def no_backoff(monkeypatch):
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("rnsr.harness.root_call.asyncio.sleep", sleep)
    return delays


async def test_governor_queue_does_not_use_active_request_timeout():
    gov = Governor(max_in_flight=1)
    await gov.acquire()
    calls = []

    async def complete(*args, **kwargs):
        calls.append(kwargs)
        return LLMResponse("done", "mock")

    client = governed(SimpleNamespace(complete=complete), gov)
    # Reproduce the old boundary: a healthy provider is never called, just
    # because a sibling held the permit longer than the request timeout.
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.02):
            await client.complete("q", model="mock")
    assert calls == []
    assert gov.snapshot()["in_flight"] == 1

    events = Events()
    task = asyncio.create_task(request(client, events, timeout_s=0.02))
    await asyncio.sleep(0.04)
    assert not task.done() and not calls
    gov.release()
    response = await asyncio.wait_for(task, timeout=1)
    assert response.text == "done"
    assert len(calls) == 1
    assert events.of_kind("root_call_failed") == []
    assert events.of_kind("root_call_started")[0]["queue_wait_s"] >= 0.02
    assert gov.snapshot()["in_flight"] == 0


async def test_admission_wait_is_still_bounded_by_query_deadline():
    gov = Governor(max_in_flight=1)
    await gov.acquire()
    events = Events()
    calls = []

    async def complete(*args, **kwargs):
        calls.append(kwargs)
        return LLMResponse("done", "mock")

    ledger = BudgetLedger(max_wall_s=0.1)
    try:
        response = await asyncio.wait_for(
            request(governed(SimpleNamespace(complete=complete), gov), events,
                    ledger, timeout_s=10), timeout=1)
        assert response is None and not calls
        [failure] = events.of_kind("root_call_failed")
        assert failure["phase"] == "queue"
        assert failure["timeout_scope"] == "overall"
        assert 0 < failure["timeout_s"] <= 0.09
        assert gov.snapshot()["in_flight"] == 1  # original holder only
    finally:
        gov.release()


async def test_active_timeout_cancels_provider_and_retry_can_finish(monkeypatch):
    delays = no_backoff(monkeypatch)
    gov = Governor(max_in_flight=1)
    calls = 0
    cancelled = asyncio.Event()

    async def complete(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return LLMResponse("recovered", "mock")

    events = Events()
    response = await request(governed(SimpleNamespace(complete=complete), gov),
                             events, timeout_s=0.01)
    assert cancelled.is_set() and response.text == "recovered"
    assert calls == 2 and delays == [5.0]
    starts = events.of_kind("root_call_started")
    assert [row["timeout_s"] for row in starts] == [0.01, 0.02]
    [failure] = events.of_kind("root_call_failed")
    assert failure["phase"] == "provider" and failure["timeout_scope"] == "request"
    assert gov.snapshot()["in_flight"] == 0
    assert gov.snapshot()["attempts"] == 2


async def test_slow_response_gets_a_bounded_longer_retry(monkeypatch):
    no_backoff(monkeypatch)
    received = []

    class SlowClient:
        async def complete_with_timeout(self, *args, timeout_s, on_admitted, **kwargs):
            received.append(timeout_s)
            on_admitted(0.0)
            # Deterministic replay of a request requiring 150 seconds: the
            # former 120/120/120 policy necessarily fails all three times.
            if timeout_s < 150:
                raise TimeoutError("request needs 150 seconds")
            return LLMResponse("done", "mock")

    response = await request(SlowClient(), Events())
    assert response.text == "done"
    assert received == [120.0, 240.0]


async def test_three_failures_do_not_sleep_after_final_attempt(monkeypatch):
    delays = no_backoff(monkeypatch)
    calls = 0

    async def complete(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError("network timeout")

    events = Events()
    assert await request(SimpleNamespace(complete=complete), events) is None
    assert calls == 3 and delays == [5.0, 10.0]
    assert [row["timeout_s"] for row in events.of_kind("root_call_started")] == [
        120.0, 240.0, 240.0,
    ]


async def test_plain_client_is_cancelled_at_remaining_wall_bound():
    events = Events()
    cancelled = asyncio.Event()

    async def complete(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    ledger = BudgetLedger(max_wall_s=0.1)
    assert await asyncio.wait_for(
        request(SimpleNamespace(complete=complete), events, ledger, timeout_s=120),
        timeout=1) is None
    assert cancelled.is_set()
    [failure] = events.of_kind("root_call_failed")
    assert failure["phase"] == "provider"
    assert failure["timeout_scope"] == "overall"
    assert failure["timeout_s"] <= 0.09


async def test_no_retry_when_wall_budget_expires_during_failure(monkeypatch):
    delays = no_backoff(monkeypatch)
    ledger = BudgetLedger()
    calls = 0

    async def complete(*args, **kwargs):
        nonlocal calls
        calls += 1
        ledger._t0 -= ledger.max_wall_s
        raise TimeoutError("failed")

    assert await request(SimpleNamespace(complete=complete), Events(), ledger) is None
    assert calls == 1 and not delays


@pytest.mark.parametrize("error", [ValueError("bad model"), RuntimeError("bug"),
                                  SpendCeilingExceeded(2, 1)])
async def test_nonretryable_errors_propagate_once(error, monkeypatch):
    delays = no_backoff(monkeypatch)
    calls = 0

    async def complete(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise error

    events = Events()
    with pytest.raises(type(error)):
        await request(SimpleNamespace(complete=complete), events)
    assert calls == 1 and not delays
    assert events.of_kind("root_call_failed")[0]["retryable"] is False


@pytest.mark.parametrize("status,retries", [(400, False), (401, False), (403, False),
                                          (404, False), (429, True), (500, True),
                                          (502, True), (503, True)])
async def test_provider_status_controls_retries(status, retries, monkeypatch):
    no_backoff(monkeypatch)

    class ProviderError(Exception):
        status_code = status

    calls = 0

    async def complete(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise ProviderError("provider rejected request")

    if retries:
        assert await request(SimpleNamespace(complete=complete), Events()) is None
        assert calls == 3
    else:
        with pytest.raises(ProviderError):
            await request(SimpleNamespace(complete=complete), Events())
        assert calls == 1


async def test_governor_spend_refusal_is_not_retried():
    gov = Governor(spend_ceiling_usd=1, spent_usd=1)
    calls = []

    async def complete(*args, **kwargs):
        calls.append(kwargs)

    events = Events()
    with pytest.raises(SpendCeilingExceeded):
        await request(governed(SimpleNamespace(complete=complete), gov), events)
    assert not calls and gov.snapshot()["attempts"] == 0
    assert events.of_kind("root_call_failed")[0]["phase"] == "queue"


async def test_caller_cancellation_propagates_and_releases_permit():
    started = asyncio.Event()
    gov = Governor(max_in_flight=1)

    async def complete(*args, **kwargs):
        started.set()
        await asyncio.Event().wait()

    events = Events()
    task = asyncio.create_task(request(
        governed(SimpleNamespace(complete=complete), gov), events))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gov.snapshot()["in_flight"] == 0
    assert not events.of_kind("root_call_failed")


async def test_exhausted_spend_makes_no_request():
    async def complete(*args, **kwargs):
        raise AssertionError("must not dispatch")

    assert await request(SimpleNamespace(complete=complete), Events(),
                         BudgetLedger(max_spend_usd=0)) is None


@pytest.mark.parametrize("field,value", [("root_timeout_s", 0), ("root_timeout_s", -1),
                                        ("root_max_attempts", 0)])
def test_root_timeout_settings_reject_nonpositive_values(field, value):
    with pytest.raises(ValueError):
        Settings(**{field: value})


def test_root_timeout_settings_are_independent_of_cell_timeout(monkeypatch):
    monkeypatch.setenv("RNSR_ROOT_TIMEOUT_S", "45")
    monkeypatch.setenv("RNSR_ROOT_MAX_ATTEMPTS", "2")
    settings = Settings.from_env()
    assert settings.root_timeout_s == 45 and settings.root_max_attempts == 2
    assert settings.cell_timeout_s == 120
