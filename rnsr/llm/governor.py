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

The governor is pluggable (GovernorProtocol): the in-memory Governor below
is the default and is all a single-process deployment needs. Multi-node
deployments that must share one rate limit or spend envelope across
workers implement the protocol over their own backend (e.g. Redis) and
install() it — RNSR deliberately ships no distributed infrastructure.
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

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


@runtime_checkable
class GovernorProtocol(Protocol):
    """What GovernedClient needs from a governor. Implement this over any
    backend (Redis, a broker, a sidecar) to share limits across processes.

    Contract, in call order around every provider request:
      - ``acquire()`` blocks until the call may proceed; it raises
        (SpendCeilingExceeded or an equivalent) to refuse the call outright.
      - the call runs; on an exception that looks like provider throttling,
        ``note_rate_limit()`` is invoked before the exception propagates.
      - on success, ``record(usage)`` is invoked with the call's Usage.
      - ``release()`` runs in all cases after a successful acquire.

    ``snapshot()`` returns the reporting dict; run reports and the CLI
    expect at least the keys requests, spend_usd, spend_ceiling_usd and
    rate_limit_hits.
    """

    async def acquire(self) -> None: ...
    def release(self) -> None: ...
    def record(self, usage: Usage) -> None: ...
    def note_rate_limit(self) -> None: ...
    def snapshot(self) -> dict: ...


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


_GOVERNOR: GovernorProtocol = Governor()


def configure(settings) -> GovernorProtocol:
    """Point the process governor at the current Settings. Idempotent.

    A custom governor installed via install() is returned untouched: its
    limits live in its own backend, not in this process's Settings.
    """
    if isinstance(_GOVERNOR, Governor):
        _GOVERNOR.max_in_flight = settings.max_in_flight_requests
        _GOVERNOR.max_rpm = settings.max_requests_per_minute
        _GOVERNOR.spend_ceiling_usd = settings.run_spend_ceiling_usd
    return _GOVERNOR


def current() -> GovernorProtocol:
    return _GOVERNOR


def install(governor: GovernorProtocol) -> GovernorProtocol:
    """Install a custom governor (e.g. Redis-backed) process-wide.

    Every client the Router wraps from this point on shares it. Call
    before building runners; reset() returns to the in-memory default.
    """
    global _GOVERNOR
    if not isinstance(governor, GovernorProtocol):
        raise TypeError(
            f"{type(governor).__name__} does not implement GovernorProtocol "
            "(needs acquire/release/record/note_rate_limit/snapshot)")
    _GOVERNOR = governor
    return _GOVERNOR


def reset(**overrides) -> Governor:
    """Fresh in-memory governor — for tests and for long-lived services that
    treat each job as its own spend envelope. Also uninstalls any custom
    governor."""
    global _GOVERNOR
    gov = Governor(**overrides)
    _GOVERNOR = gov
    return gov


class GovernedClient:
    """LLMClient wrapper that meters and paces every call."""

    def __init__(self, inner, governor: GovernorProtocol | None = None):
        self._inner = inner
        self._governor = governor
        self.provider = getattr(inner, "provider", "")

    @property
    def governor(self) -> GovernorProtocol:
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


def governed(client, governor: GovernorProtocol | None = None):
    """Wrap a client once; wrapping a wrapper is a no-op."""
    if isinstance(client, GovernedClient):
        return client
    return GovernedClient(client, governor)
