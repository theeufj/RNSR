"""OpenAI provider."""

from __future__ import annotations

import base64

from rnsr.llm.base import LLMResponse
from rnsr.llm.cost import make_usage


class OpenAIClient:
    provider = "openai"

    def __init__(self, api_key: str | None = None):
        import openai

        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def complete(self, prompt, *, model, system=None, max_tokens=4096,
                       temperature=0.0, seed=None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        usage = resp.usage
        return LLMResponse(
            resp.choices[0].message.content or "", model,
            make_usage(model, usage.prompt_tokens if usage else 0,
                       usage.completion_tokens if usage else 0),
        )

    async def embed(self, texts, *, model):
        resp = await self._client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]

    async def vision(self, prompt, image_png, *, model, max_tokens=4096):
        b64 = base64.b64encode(image_png).decode()
        resp = await self._client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        usage = resp.usage
        return LLMResponse(
            resp.choices[0].message.content or "", model,
            make_usage(model, usage.prompt_tokens if usage else 0,
                       usage.completion_tokens if usage else 0),
        )
