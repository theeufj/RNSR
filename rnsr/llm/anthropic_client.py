"""Anthropic provider. No embedding models — router pairs it with another
provider for the embed role."""

from __future__ import annotations

import base64

from rnsr.llm.base import LLMResponse
from rnsr.llm.cost import make_usage


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, api_key: str | None = None):
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, prompt, *, model, system=None, max_tokens=4096,
                       temperature=0.0, seed=None):
        # seed: not supported by the API; determinism comes from temperature=0
        msg = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResponse(text, model,
                           make_usage(model, msg.usage.input_tokens, msg.usage.output_tokens))

    async def embed(self, texts, *, model):
        raise NotImplementedError("anthropic has no embedding API; configure openai/gemini for role 'embed'")

    async def vision(self, prompt, image_png, *, model, max_tokens=4096):
        msg = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
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
        return LLMResponse(text, model,
                           make_usage(model, msg.usage.input_tokens, msg.usage.output_tokens))
