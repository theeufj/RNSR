"""Google Gemini provider (google-genai SDK)."""

from __future__ import annotations

from rnsr.llm.base import LLMResponse
from rnsr.llm.cost import make_usage


def _usage_tokens(resp) -> tuple[int, int]:
    meta = getattr(resp, "usage_metadata", None)
    if meta is None:
        return 0, 0
    return (getattr(meta, "prompt_token_count", 0) or 0,
            getattr(meta, "candidates_token_count", 0) or 0)


class GeminiClient:
    provider = "gemini"

    def __init__(self, api_key: str | None = None):
        from google import genai

        self._client = genai.Client(api_key=api_key)

    async def complete(self, prompt, *, model, system=None, max_tokens=4096,
                       temperature=0.0, seed=None):
        from google.genai import types

        resp = await self._client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            ),
        )
        i, o = _usage_tokens(resp)
        return LLMResponse(resp.text or "", model, make_usage(model, i, o))

    async def embed(self, texts, *, model):
        resp = await self._client.aio.models.embed_content(model=model, contents=texts)
        return [e.values for e in resp.embeddings]

    async def vision(self, prompt, image_png, *, model, max_tokens=4096):
        from google.genai import types

        resp = await self._client.aio.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_png, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
        i, o = _usage_tokens(resp)
        return LLMResponse(resp.text or "", model, make_usage(model, i, o))
