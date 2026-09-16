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
    wait_exponential,
)

from rnsr.errors import BudgetExhausted
from rnsr.llm.base import LLMClient, LLMResponse, Usage
from rnsr.llm.governor import SpendCeilingExceeded

# Called after every completed sub-call; the harness BudgetLedger hooks in here.
UsageCallback = Callable[[Usage], None]


def _retryable(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return any(k in name or k in text for k in
               ("ratelimit", "rate limit", "429", "overloaded", "timeout",
                "connection", "503", "502"))


async def map_prompts(
    client: LLMClient,
    prompts: list[str],
    *,
    model: str,
    system: str | None = None,
    max_tokens: int = 4096,
    concurrency: int = 16,
    attempts: int = 4,
    on_usage: UsageCallback | None = None,
    on_attempt: Callable[[], None] | None = None,
    deadline: float | None = None,
) -> list[LLMResponse | None]:
    """Run all prompts concurrently under a semaphore; order-preserving.

    A prompt whose retries exhaust resolves to None rather than failing the
    whole batch — callers decide whether partial coverage is acceptable
    (semantic_annotate reports it; verify paths treat None as failure).
    """
    if concurrency < 1 or attempts < 1:
        raise ValueError("concurrency and attempts must be positive")
    sem = asyncio.Semaphore(concurrency)

    async def one(prompt: str) -> LLMResponse | None:
        async with sem:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(attempts),
                    wait=wait_exponential(multiplier=1, max=30),
                    retry=retry_if_exception(_retryable),
                    reraise=True,
                ):
                    with attempt:
                        if on_attempt:
                            on_attempt()
                        resp = await client.complete(
                            prompt, model=model, system=system, max_tokens=max_tokens
                        )
                        if on_usage:
                            on_usage(resp.usage)
                        return resp
            except (BudgetExhausted, SpendCeilingExceeded):
                raise
            except Exception:
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
