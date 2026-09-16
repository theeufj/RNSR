"""Run-level provider governance: in-flight cap, RPM pacing, spend ceiling."""

import asyncio
import threading
import time

import pytest

from rnsr.config import Settings
from rnsr.llm.governor import (
    Governor,
    GovernorProtocol,
    SpendCeilingExceeded,
    configure,
    current,
    governed,
    install,
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
    async def test_cap_shared_by_simultaneous_event_loops(self):
        gov = Governor(max_in_flight=1)
        state = {"active": 0, "peak": 0}
        lock = threading.Lock()
        barrier = threading.Barrier(2)

        async def batch():
            for _ in range(5):
                await gov.acquire()
                try:
                    with lock:
                        state["active"] += 1
                        state["peak"] = max(state["peak"], state["active"])
                    await asyncio.sleep(0.005)
                    with lock:
                        state["active"] -= 1
                finally:
                    gov.release()

        def worker():
            barrier.wait(timeout=2)
            asyncio.run(batch())

        await asyncio.gather(asyncio.to_thread(worker), asyncio.to_thread(worker))
        assert state["peak"] == 1
        assert gov.snapshot()["attempts"] == 10
        assert gov.snapshot()["in_flight"] == 0

    async def test_cancelled_waiter_does_not_leak_permit(self):
        gov = Governor(max_in_flight=1)
        await gov.acquire()
        waiting = asyncio.create_task(gov.acquire())
        await asyncio.sleep(0.02)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        gov.release()
        await asyncio.wait_for(gov.acquire(), timeout=1)
        gov.release()
        assert gov.snapshot()["in_flight"] == 0

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
    async def test_waiting_request_rechecks_spend_before_admission(self):
        from rnsr.llm.base import Usage

        gov = Governor(max_in_flight=1, spend_ceiling_usd=1)
        await gov.acquire()
        waiting = asyncio.create_task(gov.acquire())
        await asyncio.sleep(0.02)
        gov.record(Usage(cost_usd=1))
        gov.release()
        with pytest.raises(SpendCeilingExceeded):
            await waiting

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
    async def test_embedding_failure_cancels_sibling_requests(self):
        class IndividualEmbedder:
            embeds_individually = True
            started = []
            completed = []

            async def embed(self, texts, *, model):
                text = texts[0]
                self.started.append(text)
                if text == "bad":
                    raise ValueError("bad embedding")
                await asyncio.sleep(0.05)
                self.completed.append(text)
                return [[1.0]]

        inner = IndividualEmbedder()
        gov = Governor(max_in_flight=1)
        with pytest.raises(ValueError, match="bad embedding"):
            await governed(inner, gov).embed(["bad", "later1", "later2"], model="e")
        attempts = gov.attempts
        await asyncio.sleep(0.1)
        assert inner.completed == []
        assert gov.attempts == attempts
        assert gov.snapshot()["in_flight"] == 0

    async def test_per_text_embedding_calls_are_individually_governed(self):
        class IndividualEmbedder(MockLLM):
            embeds_individually = True

        gov = Governor(max_in_flight=2)
        vectors = await governed(IndividualEmbedder(), gov).embed(["a", "b", "c"], model="e")
        assert len(vectors) == 3
        assert gov.requests == 3
        assert gov.attempts == 3
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


class RecordingGovernor:
    """A custom GovernorProtocol implementation (what a Redis-backed one
    looks like structurally): no inheritance from Governor required."""

    def __init__(self):
        self.events = []

    async def acquire(self):
        self.events.append("acquire")

    def release(self):
        self.events.append("release")

    def record(self, usage):
        self.events.append(("record", usage.cost_usd))

    def note_rate_limit(self):
        self.events.append("rate_limit")

    def snapshot(self):
        return {"requests": 1, "spend_usd": 0.0, "spend_ceiling_usd": 0.0,
                "rate_limit_hits": 0}


class TestPluggableGovernor:
    async def test_custom_governor_receives_the_call_lifecycle(self):
        gov = RecordingGovernor()
        client = governed(MockLLM(default="ok"), gov)
        await client.complete("p", model="m")
        assert gov.events[0] == "acquire"
        assert ("record", 0.001) in gov.events
        assert gov.events[-1] == "release"

    async def test_rate_limit_reaches_the_custom_governor(self):
        gov = RecordingGovernor()
        client = governed(MockLLM(default="ok", fail_times=1), gov)
        with pytest.raises(RuntimeError):
            await client.complete("p", model="m")
        assert "rate_limit" in gov.events

    def test_install_makes_it_process_wide(self):
        gov = RecordingGovernor()
        try:
            install(gov)
            assert current() is gov
            # clients wrapped without an explicit governor use it
            client = governed(MockLLM(default="ok"))
            assert client.governor is gov
        finally:
            reset()
        assert isinstance(current(), Governor)

    def test_configure_leaves_a_custom_governor_untouched(self):
        gov = RecordingGovernor()
        try:
            install(gov)
            got = configure(Settings(run_spend_ceiling_usd=5.0))
            assert got is gov                    # not replaced, not mutated
            assert not hasattr(gov, "spend_ceiling_usd")
        finally:
            reset()

    def test_install_rejects_incomplete_implementations(self):
        class NotAGovernor:
            async def acquire(self):
                pass

        with pytest.raises(TypeError, match="GovernorProtocol"):
            install(NotAGovernor())

    def test_default_governor_satisfies_the_protocol(self):
        assert isinstance(Governor(), GovernorProtocol)
