"""Anthropic provider. No embedding models — router pairs it with another
provider for the embed role."""

from __future__ import annotations

import base64
import re

from rnsr.llm.base import LLMResponse, PromptSegments, Usage
from rnsr.llm.cost import make_usage

# Prompt caching (platform.claude.com/docs/en/build-with-claude/prompt-caching).
# A breakpoint caches the whole prefix up to and including its block; the
# next request is a hit when its prefix hash at some position within 20
# blocks of its own breakpoint matches a prior write. The root loop re-sends
# a growing transcript every iteration, so the system prompt gets one
# breakpoint (shared by every query on the same corpus) and the last
# *stable* transcript block gets the other. The per-call tail stays after
# the breakpoint: putting it on the varying block would write a new entry
# every call and never read one. Prompts under the model minimum (512
# tokens on 5.5) are silently uncached, so marking is always safe.
_CACHE = {"type": "ephemeral"}

# Models released after Claude Opus 4.6 reject sampling parameters: any
# temperature other than the default 1 fails with 400 "temperature is
# deprecated for this model" (top_p/top_k likewise). Only the 4.6-and-earlier
# generations still accept temperature. Verified against the API 2026-10.
_LEGACY_SAMPLING = re.compile(r"^claude-(opus-4-[0-6]|sonnet-4|haiku-4|3)")

# Sonnet 5.5 and Haiku 5.5 think adaptively by default, and thinking tokens
# count against max_tokens: a terse call such as a YES/NO judge with
# max_tokens=8 then returns an empty text block, and a strict-JSON review is
# truncated mid-object. The harness brokers short deterministic calls, so
# pre-response thinking is turned off where the API offers a switch. Each
# model accepts exactly one form and rejects the other with a 400; models
# before 5.5 do not think unless asked, and Opus 5.5 / Fable 5.1 cannot turn
# it off (effort is their only control). Verified against the API 2026-10.
_THINKING_OFF: tuple[tuple[re.Pattern[str], dict], ...] = (
    (re.compile(r"^claude-sonnet-5-5"), {"type": "between_tools"}),
    (re.compile(r"^claude-haiku-5-5"), {"type": "disabled"}),
)


def _sampling_kwargs(model: str, temperature: float) -> dict:
    if _LEGACY_SAMPLING.match(model):
        return {"temperature": temperature}
    for pattern, thinking in _THINKING_OFF:
        if pattern.match(model):
            return {"thinking": thinking}
    return {}


def _system_param(system: str | None) -> str | list[dict]:
    if not system:
        return ""
    return [{"type": "text", "text": system, "cache_control": _CACHE}]


def _content_param(prompt: str | PromptSegments) -> str | list[dict]:
    if not isinstance(prompt, PromptSegments):
        return prompt
    if not prompt.stable:
        return prompt.tail
    # Each block ends with the newline the flat rendering would place between
    # parts, so the model reads the same text either way.
    blocks = [{"type": "text", "text": text + "\n"} for text in prompt.stable]
    if prompt.tail:
        blocks.append({"type": "text", "text": prompt.tail})
    else:
        blocks[-1]["text"] = blocks[-1]["text"][:-1]
    last_stable = len(prompt.stable) - 1
    blocks[last_stable] = {**blocks[last_stable], "cache_control": _CACHE}
    return blocks


def _usage(model: str, usage) -> Usage:
    return make_usage(
        model, usage.input_tokens, usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
    )


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, api_key: str | None = None):
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)

    async def list_models(self) -> set[str]:
        page = await self._client.models.list()
        return {m.id async for m in page}

    async def complete(self, prompt, *, model, system=None, max_tokens=4096,
                       temperature=0.0, seed=None):
        # seed: not supported by the API; determinism comes from temperature=0
        # on models that still accept it (see _sampling_kwargs).
        msg = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            **_sampling_kwargs(model, temperature),
            system=_system_param(system),
            messages=[{"role": "user", "content": _content_param(prompt)}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(text, model, _usage(model, msg.usage))

    async def embed(self, texts, *, model):
        raise NotImplementedError("anthropic has no embedding API; configure openai/gemini for role 'embed'")

    async def vision(self, prompt, image_png, *, model, max_tokens=4096):
        msg = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            **_sampling_kwargs(model, 0.0),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": base64.b64encode(image_png).decode(),
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(text, model, _usage(model, msg.usage))
