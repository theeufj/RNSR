"""Async sub-call fan-out with bounded concurrency (spec §7).

The paper's stated biggest inefficiency was sequential sub-calls; every
batched pathway in the system (semantic_annotate, rung-3 expansion, rung-5
sweeps, prose cross-checks) goes through map_prompts, which owns the
semaphore, the retry policy, and cost/count accounting via callbacks.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    stop_never,
    wait_exponential,
)

from rnsr.errors import BudgetExhausted, PermanentProviderError
from rnsr.llm.base import LLMClient, LLMResponse, Usage
from rnsr.llm.governor import SpendCeilingExceeded
from rnsr.llm.retry import is_terminal_provider_error

# Called after every completed sub-call; the harness BudgetLedger hooks in here.
UsageCallback = Callable[[Usage], None]


def _retryable(exc: BaseException) -> bool:
    if (isinstance(exc, (BudgetExhausted, SpendCeilingExceeded))
            or is_terminal_provider_error(exc)):
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in (408, 409, 429) or 500 <= status < 600
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return any(k in name or k in text for k in
               ("ratelimit", "rate limit", "429", "overloaded", "timeout",
                "connection", "500", "502", "503", "504"))


async def map_prompts(
    client: LLMClient,
    prompts: list[str],
    *,
    model: str,
    system: str | None = None,
    max_tokens: int = 4096,
    concurrency: int = 16,
    attempts: int | None = 4,
    on_usage: UsageCallback | None = None,
    on_attempt: Callable[[], None] | None = None,
    deadline: float | None = None,
    request_timeout_s: float | None = None,
    raise_permanent_errors: bool = False,
) -> list[LLMResponse | None]:
    """Run all prompts concurrently under a semaphore; order-preserving.

    A prompt whose retries exhaust resolves to None rather than failing the
    whole batch — callers decide whether partial coverage is acceptable.
    attempts=None retries transient failures until success or cancellation;
    each request can still have its own liveness timeout. Uncapped query
    callers surface permanent provider errors instead of empty coverage.
    """
    if concurrency < 1 or (attempts is not None and attempts < 1):
        raise ValueError("concurrency and attempts must be positive")
    if request_timeout_s is not None and request_timeout_s <= 0:
        raise ValueError("request_timeout_s must be positive")
    sem = asyncio.Semaphore(concurrency)

    async def one(prompt: str) -> LLMResponse | None:
        async with sem:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_never if attempts is None else stop_after_attempt(attempts),
                    wait=wait_exponential(multiplier=1, max=30),
                    retry=retry_if_exception(_retryable),
                    reraise=True,
                ):
                    with attempt:
                        if on_attempt:
                            on_attempt()
                        timed = getattr(client, "complete_with_timeout", None)
                        if request_timeout_s is not None and callable(timed):
                            resp = await timed(
                                prompt, model=model, system=system, max_tokens=max_tokens,
                                timeout_s=request_timeout_s)
                        else:
                            async with asyncio.timeout(request_timeout_s):
                                resp = await client.complete(
                                    prompt, model=model, system=system, max_tokens=max_tokens)
                        if on_usage:
                            on_usage(resp.usage)
                        return resp
            except (BudgetExhausted, SpendCeilingExceeded):
                raise
            except Exception as exc:
                if raise_permanent_errors and not _retryable(exc):
                    raise PermanentProviderError(
                        f"{type(exc).__name__}: {exc}") from exc
                return None
        return None

    tasks = [asyncio.create_task(one(p)) for p in prompts]
    try:
        async with asyncio.timeout(None if deadline is None else max(0, deadline - time.monotonic())):
            return list(await asyncio.gather(*tasks))
    finally:
        # A budget failure in one prompt must stop siblings and their retries.
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
