"""Provider-agnostic LLM interface.

Roles (spec §4/§7): 'root' drives the REPL loop, 'sub' serves cheap batched
calls, 'embed' powers rung 4, 'vision' the table-extraction fallback.
Providers implement this protocol; the router maps roles to (provider,
model); batch.py owns concurrency and cost metering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Usage:
    """Metered tokens and list-price cost for one or more provider calls.

    ``input_tokens`` is the whole prompt (uncached + cache reads + cache
    writes), so token counts stay comparable whether or not a provider
    cached the prefix; the cache split is recorded alongside for pricing.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class PromptSegments:
    """A prompt split for provider-side prefix caching.

    ``stable`` blocks are an append-only prefix: across successive calls of
    one conversation, earlier blocks never change and new blocks are only
    appended. ``tail`` is the per-call text after the cache breakpoint. The
    flat rendering (``text()``) joins everything with newlines, so providers
    without prefix caching see exactly the same prompt as before.
    """

    stable: tuple[str, ...]
    tail: str = ""

    def text(self) -> str:
        parts = [*self.stable, self.tail] if self.tail else list(self.stable)
        return "\n".join(parts)


Prompt = str | PromptSegments


def prompt_text(prompt: Prompt) -> str:
    """Flat string form of a prompt, for providers without prefix caching."""
    return prompt.text() if isinstance(prompt, PromptSegments) else prompt


@runtime_checkable
class LLMClient(Protocol):
    provider: str

    async def complete(
        self,
        prompt: Prompt,
        *,
        model: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> LLMResponse: ...

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]: ...

    async def vision(
        self, prompt: str, image_png: bytes, *, model: str, max_tokens: int = 4096
    ) -> LLMResponse: ...
