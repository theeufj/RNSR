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
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
        )


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)


@runtime_checkable
class LLMClient(Protocol):
    provider: str

    async def complete(
        self,
        prompt: str,
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
