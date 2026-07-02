"""MockLLM for tests: scripted responses, call recording, concurrency probes."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

from rnsr.llm.base import LLMResponse, Usage


@dataclass
class Rule:
    pattern: str            # regex searched against the prompt
    response: str


@dataclass
class MockLLM:
    """Scripted client.

    Resolution order per call: next queued response if any, else the first
    matching rule, else `default`. Records every call; tracks peak
    concurrency for semaphore tests.
    """

    provider: str = "mock"
    queue: list[str] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    default: str = "UNCLEAR"
    delay_s: float = 0.0
    usage_per_call: Usage = field(default_factory=lambda: Usage(100, 10, 0.001))

    calls: list[dict] = field(default_factory=list)
    in_flight: int = 0
    max_in_flight: int = 0
    fail_times: int = 0          # raise on the first N calls (retry testing)

    def script(self, *responses: str) -> MockLLM:
        self.queue.extend(responses)
        return self

    def rule(self, pattern: str, response: str) -> MockLLM:
        self.rules.append(Rule(pattern, response))
        return self

    def _resolve(self, prompt: str) -> str:
        if self.queue:
            return self.queue.pop(0)
        for r in self.rules:
            if re.search(r.pattern, prompt, re.DOTALL):
                return r.response
        return self.default

    async def complete(self, prompt, *, model="mock-model", system=None,
                       max_tokens=4096, temperature=0.0, seed=None):
        self.calls.append({"kind": "complete", "prompt": prompt, "system": system,
                           "model": model})
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("rate limit (simulated 429)")
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            return LLMResponse(self._resolve(prompt), model, self.usage_per_call)
        finally:
            self.in_flight -= 1

    async def embed(self, texts, *, model="mock-embed"):
        self.calls.append({"kind": "embed", "n": len(texts), "model": model})
        # deterministic toy embeddings: char histogram over a tiny alphabet
        out = []
        for t in texts:
            v = [float(t.count(c)) for c in "aeiourstln"]
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / norm for x in v])
        return out

    async def vision(self, prompt, image_png, *, model="mock-vision", max_tokens=4096):
        self.calls.append({"kind": "vision", "prompt": prompt, "bytes": len(image_png),
                           "model": model})
        return LLMResponse(self._resolve(prompt), model, self.usage_per_call)
