"""Process-wide provider governor: in-flight cap, RPM ceiling, spend ceiling.

The §7 BudgetLedger caps one query. Nothing capped the *run*: answer-csv
runs many loops concurrently, each with its own sub-call semaphore, so the
real request rate against a provider was (loops x sub_concurrency) and the
real spend was unbounded — a 999-question job could empty an account
before anyone read the console. The governor is the missing outer bound:

  - in-flight cap: one semaphore all provider traffic passes through, so
    adding concurrent loops stops multiplying the request rate;
  - RPM ceiling: a sliding-window gate, because providers bill per minute
    and 429 storms cost wall-clock in retries;
  - spend ceiling: aggregate USD across every role and loop. Once breached,
    further acquisitions raise rather than spend, so a runaway job stops
    at a number the operator chose;
  - shared cooldown: when one call sees a 429, every caller waits. Without
    it, sibling loops keep hammering a provider that just asked for quiet.

Wrapping happens in the Router (see governed()), so every path — root
calls, map_prompts fan-out, judges, ingest hooks — is covered by
construction rather than by remembering to call it.
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from collections import deque
from dataclasses import dataclass, field

from rnsr.errors import RNSRError
from rnsr.llm.base import LLMResponse, Usage
from rnsr.obs import get_logger, log, metrics

_LOG = get_logger("llm.governor")

_RATE_LIMIT_MARKERS = ("ratelimit", "rate limit", "429", "overloaded",
                       "quota", "insufficient_quota", "503")


class SpendCeilingExceeded(RNSRError):
    """The aggregate run spend ceiling was reached; no further calls are made."""

    def __init__(self, spent: float, ceiling: float):
        self.spent = spent
        self.ceiling = ceiling
        super().__init__(
            f"run spend ceiling reached: ${spent:.4f} >= ${ceiling:.2f}. "
            "Raise RNSR_RUN_SPEND_CEILING_USD or start a new run.")


def is_rate_limit(exc: BaseException) -> bool:
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(marker in blob for marker in _RATE_LIMIT_MARKERS)


@dataclass
class Governor:
    """Shared gate for all provider traffic. Zero on a limit disables it."""

    max_in_flight: int = 0
    max_rpm: int = 0
    spend_ceiling_usd: float = 0.0
    cooldown_s: float = 5.0

    spent_usd: float = 0.0
    requests: int = 0
    rate_limit_hits: int = 0
    waited_s: float = 0.0

    _window: deque[float] = field(default_factory=deque)
    _cooldown_until: float = 0.0
    # asyncio primitives bind to the running loop, and rnsr calls asyncio.run
    # more than once per process (CLI commands, embedding builds in worker
    # threads), so gates are created per loop. Keyed weakly by the loop
    # object: id() would be recycled after a loop is collected, handing a new
    # loop a semaphore bound to the dead one.
    _per_loop: weakref.WeakKeyDictionary = field(
        default_factory=weakref.WeakKeyDictionary)

    def _gates(self) -> tuple[asyncio.Semaphore | None, asyncio.Lock]:
        loop = asyncio.get_running_loop()
        gates = self._per_loop.get(loop)
        if gates is None:
            sem = asyncio.Semaphore(self.max_in_flight) if self.max_in_flight else None
            gates = (sem, asyncio.Lock())
            self._per_loop[loop] = gates
        return gates

    def check_spend(self) -> None:
        if self.spend_ceiling_usd and self.spent_usd >= self.spend_ceiling_usd:
            raise SpendCeilingExceeded(self.spent_usd, self.spend_ceiling_usd)

    def record(self, usage: Usage) -> None:
        self.spent_usd += usage.cost_usd
        self.requests += 1

    def note_rate_limit(self) -> None:
        self.rate_limit_hits += 1
        self._cooldown_until = max(self._cooldown_until,
                                   time.monotonic() + self.cooldown_s)
        log(_LOG, logging.WARNING, "provider.rate_limited",
            cooldown_s=self.cooldown_s, hits=self.rate_limit_hits)
        metrics().incr("provider_rate_limit_hits")

    async def _pace(self) -> None:
        """Serialize the window/cooldown decision, then sleep outside the lock."""
        _, lock = self._gates()
        while True:
            async with lock:
                now = time.monotonic()
                delay = max(0.0, self._cooldown_until - now)
                if not delay and self.max_rpm:
                    while self._window and now - self._window[0] >= 60.0:
                        self._window.popleft()
                    if len(self._window) >= self.max_rpm:
                        delay = 60.0 - (now - self._window[0])
                if delay <= 0:
                    if self.max_rpm:
                        self._window.append(now)
                    return
            self.waited_s += delay
            metrics().incr("provider_throttle_s", delay)
            await asyncio.sleep(delay)

    async def acquire(self) -> None:
        self.check_spend()
        await self._pace()
        sem, _ = self._gates()
        if sem is not None:
            await sem.acquire()

    def release(self) -> None:
        sem, _ = self._gates()
        if sem is not None:
            sem.release()

    def snapshot(self) -> dict:
        return {
            "requests": self.requests,
            "spend_usd": round(self.spent_usd, 6),
            "spend_ceiling_usd": self.spend_ceiling_usd,
            "rate_limit_hits": self.rate_limit_hits,
            "throttled_s": round(self.waited_s, 2),
        }


_GOVERNOR = Governor()


def configure(settings) -> Governor:
    """Point the process governor at the current Settings. Idempotent."""
    _GOVERNOR.max_in_flight = settings.max_in_flight_requests
    _GOVERNOR.max_rpm = settings.max_requests_per_minute
    _GOVERNOR.spend_ceiling_usd = settings.run_spend_ceiling_usd
    return _GOVERNOR


def current() -> Governor:
    return _GOVERNOR


def reset(**overrides) -> Governor:
    """Fresh counters and gates — for tests and for long-lived services that
    treat each job as its own spend envelope."""
    global _GOVERNOR
    _GOVERNOR = Governor(**overrides)
    return _GOVERNOR


class GovernedClient:
    """LLMClient wrapper that meters and paces every call."""

    def __init__(self, inner, governor: Governor | None = None):
        self._inner = inner
        self._governor = governor
        self.provider = getattr(inner, "provider", "")

    @property
    def governor(self) -> Governor:
        return self._governor or _GOVERNOR

    def __getattr__(self, name: str):     # passthrough for non-call helpers
        return getattr(self._inner, name)

    async def _guarded(self, coro_factory, *, usage_of=None):
        gov = self.governor
        await gov.acquire()
        try:
            result = await coro_factory()
        except BaseException as exc:
            if is_rate_limit(exc):
                # let the caller's retry policy decide what to do next, but
                # make every sibling loop wait too
                gov.note_rate_limit()
            raise
        else:
            gov.record(usage_of(result) if usage_of else Usage())
            return result
        finally:
            gov.release()

    async def complete(self, prompt, *, model, system=None, max_tokens=4096,
                       temperature=0.0, seed=None) -> LLMResponse:
        return await self._guarded(
            lambda: self._inner.complete(prompt, model=model, system=system,
                                         max_tokens=max_tokens,
                                         temperature=temperature, seed=seed),
            usage_of=lambda r: r.usage)

    async def embed(self, texts, *, model):
        return await self._guarded(lambda: self._inner.embed(texts, model=model))

    async def vision(self, prompt, image_png, *, model, max_tokens=4096):
        return await self._guarded(
            lambda: self._inner.vision(prompt, image_png, model=model,
                                       max_tokens=max_tokens),
            usage_of=lambda r: r.usage)


def governed(client, governor: Governor | None = None):
    """Wrap a client once; wrapping a wrapper is a no-op."""
    if isinstance(client, GovernedClient):
        return client
    return GovernedClient(client, governor)
