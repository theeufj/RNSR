"""Run-level provider governance: in-flight cap, RPM pacing, spend ceiling."""

import asyncio
import time

import pytest

from rnsr.config import Settings
from rnsr.llm.governor import (
    Governor,
    SpendCeilingExceeded,
    governed,
    is_rate_limit,
    reset,
)
from rnsr.llm.mock import MockLLM


@pytest.fixture(autouse=True)
def fresh_governor():
    reset()
    yield
    reset()


class TestInFlightCap:
    async def test_cap_holds_across_independent_batches(self):
        # the bug this closes: each batch had its own semaphore, so N
        # concurrent loops meant N x sub_concurrency requests in flight
        from rnsr.llm.batch import map_prompts

        gov = Governor(max_in_flight=3)
        mock = MockLLM(delay_s=0.02, default="ok")
        client = governed(mock, gov)
        await asyncio.gather(*(
            map_prompts(client, ["p"] * 8, model="m", concurrency=8)
            for _ in range(4)))
        assert mock.max_in_flight <= 3
        assert len(mock.calls) == 32

    async def test_zero_disables_the_cap(self):
        gov = Governor(max_in_flight=0)
        mock = MockLLM(delay_s=0.02, default="ok")
        client = governed(mock, gov)
        await asyncio.gather(*(client.complete("p", model="m") for _ in range(6)))
        assert mock.max_in_flight > 1


class TestSpendCeiling:
    async def test_calls_refused_once_ceiling_reached(self):
        gov = Governor(spend_ceiling_usd=0.0025)   # 2 calls at $0.001 each
        client = governed(MockLLM(default="ok"), gov)
        for _ in range(3):
            await client.complete("p", model="m")
        with pytest.raises(SpendCeilingExceeded):
            await client.complete("p", model="m")

    async def test_spend_accumulates_across_roles(self):
        gov = Governor()
        client = governed(MockLLM(default="ok"), gov)
        await client.complete("p", model="m")
        await client.complete("p", model="m")
        assert gov.requests == 2
        assert gov.spent_usd == pytest.approx(0.002)

    async def test_embeddings_metered_without_usage(self):
        gov = Governor()
        client = governed(MockLLM(), gov)
        await client.embed(["a", "b"], model="e")
        assert gov.requests == 1
        assert gov.spent_usd == 0.0


class TestPacing:
    async def test_rpm_ceiling_delays_the_overflow_call(self):
        gov = Governor(max_rpm=2)
        client = governed(MockLLM(default="ok"), gov)
        await client.complete("p", model="m")
        await client.complete("p", model="m")
        # the third would have to wait ~60s; assert the gate computes a wait
        # rather than actually sleeping through it
        task = asyncio.ensure_future(client.complete("p", model="m"))
        await asyncio.sleep(0.05)
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_rate_limit_triggers_shared_cooldown(self):
        gov = Governor(cooldown_s=0.2)
        mock = MockLLM(default="ok", fail_times=1)   # first call raises
        client = governed(mock, gov)
        with pytest.raises(RuntimeError):
            await client.complete("p", model="m")
        assert gov.rate_limit_hits == 1
        t0 = time.monotonic()
        await client.complete("p", model="m")        # waits out the cooldown
        assert time.monotonic() - t0 >= 0.15

    def test_rate_limit_classifier(self):
        assert is_rate_limit(RuntimeError("rate limit (simulated 429)"))
        assert is_rate_limit(RuntimeError("Error code: 503 overloaded"))
        assert not is_rate_limit(ValueError("bad model name"))
        assert not is_rate_limit(SpendCeilingExceeded(1.0, 1.0))


class TestWiring:
    def test_router_wraps_clients(self, monkeypatch):
        from rnsr.llm.governor import GovernedClient
        from rnsr.llm.router import Router

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        r = Router(Settings(provider="openai", max_in_flight_requests=5,
                            run_spend_ceiling_usd=12.5))
        client = r.resolve("root").client
        assert isinstance(client, GovernedClient)
        assert client.provider == "openai"
        assert r.governor.max_in_flight == 5
        assert r.governor.spend_ceiling_usd == 12.5

    async def test_reused_across_event_loops(self):
        # the CLI calls asyncio.run more than once per process, and embedding
        # builds run their own loop in a worker thread
        gov = Governor(max_in_flight=2)
        client = governed(MockLLM(default="ok"), gov)

        def run_once():
            asyncio.run(client.complete("p", model="m"))

        await asyncio.to_thread(run_once)
        await asyncio.to_thread(run_once)
        assert gov.requests == 2
