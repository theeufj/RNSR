"""Bounded root requests with separate queue and generation deadlines."""

from __future__ import annotations

import asyncio
import time

from rnsr.errors import BudgetExhausted
from rnsr.harness.budget import BudgetLedger
from rnsr.llm.base import LLMClient, LLMResponse
from rnsr.llm.governor import SpendCeilingExceeded, is_rate_limit


def _is_timeout(exc: Exception) -> bool:
    return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (BudgetExhausted, SpendCeilingExceeded)):
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 429) or 500 <= status < 600
    return (_is_timeout(exc) or isinstance(exc, ConnectionError)
            or "connect" in type(exc).__name__.lower() or is_rate_limit(exc))


async def complete_root(
    client: LLMClient, prompt: str, *, model: str, system: str,
    ledger: BudgetLedger, trajectory, seed: int | None = None,
    timeout_s: float = 120.0, attempts: int = 3,
) -> LLMResponse | None:
    """Retry transient failures without repeatedly cancelling queued requests.

    Admission waits count toward the overall deadline, not the active-call
    timeout. After a timeout, a retry may use twice the base allowance so a
    consistently slower response can finish. SDK limits are unchanged.
    All attempts/backoff share one deadline, reserving up to 30 seconds
    (10% of the remaining query time) for variable recovery. Cancellation
    propagates; terminal provider/configuration errors are never retried.
    """
    if timeout_s <= 0 or attempts < 1:
        raise ValueError("timeout_s and attempts must be positive")
    remaining = ledger.remaining_wall_s()
    if remaining <= 0 or ledger.spend_usd >= ledger.max_spend_usd:
        return None
    reserve = min(30.0, remaining * 0.1)
    deadline = time.monotonic() + remaining - reserve
    extended = False

    def available() -> float:
        return min(deadline - time.monotonic(), ledger.remaining_wall_s())

    for attempt in range(1, attempts + 1):
        remaining = available()
        if remaining <= 0 or ledger.spend_usd >= ledger.max_spend_usd:
            return None
        active_timeout = min(timeout_s * (2 if extended else 1), remaining)
        started = time.monotonic()
        admitted_at: float | None = None
        queue_wait_s: float | None = None

        def admitted(wait_s: float, *, _attempt=attempt, _timeout=active_timeout) -> None:
            nonlocal admitted_at, queue_wait_s
            admitted_at = time.monotonic()
            queue_wait_s = wait_s
            trajectory.event("root_call_started", attempt=_attempt,
                             queue_wait_s=round(wait_s, 3),
                             timeout_s=_timeout)

        overall_timeout = asyncio.timeout(remaining)
        try:
            async with overall_timeout:
                timed_complete = getattr(client, "complete_with_timeout", None)
                if callable(timed_complete):
                    response = await timed_complete(
                        prompt, model=model, system=system, max_tokens=8192,
                        seed=seed, timeout_s=active_timeout, on_admitted=admitted,
                    )
                else:
                    # Plain/custom clients have no separately exposed queue.
                    admitted(0.0)
                    async with asyncio.timeout(active_timeout):
                        response = await client.complete(
                            prompt, model=model, system=system,
                            max_tokens=8192, seed=seed,
                        )
            trajectory.event("root_call_completed", attempt=attempt,
                             elapsed_s=round(time.monotonic() - started, 3),
                             queue_wait_s=queue_wait_s)
            return response
        except Exception as exc:
            retryable = _retryable(exc)
            scope = ("overall" if overall_timeout.expired()
                     else "request" if _is_timeout(exc) else None)
            trajectory.event(
                "root_call_failed", attempt=attempt, timeout_s=active_timeout,
                error=f"{type(exc).__name__}: {exc}"[:200],
                phase="queue" if admitted_at is None else "provider",
                timeout_scope=scope, queue_wait_s=queue_wait_s,
                elapsed_s=round(time.monotonic() - started, 3),
                retryable=retryable,
            )
            if not retryable:
                raise
            if overall_timeout.expired() or attempt == attempts or available() <= 0:
                return None
            if _is_timeout(exc):
                extended = True
            # No delay after the last failure, and no sleep past the shared
            # deadline. The governor additionally enforces provider cooldown.
            backoff = min(5.0 * attempt, available() / 4)
            if backoff > 0.1:
                await asyncio.sleep(backoff)
    return None
