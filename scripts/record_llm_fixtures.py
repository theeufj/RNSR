"""Opt-in live recorder for tests/llm_fixtures/<provider>/<op>.json.

Requires the matching provider key. Writes SDK model_dump() (or a
plain-dict fallback) so tests/test_llm_offline.py can rebuild stubs
without hitting the network.

  python scripts/record_llm_fixtures.py --provider openai
  python scripts/record_llm_fixtures.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "llm_fixtures"


def _dump(obj) -> dict:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return {"repr": repr(obj)}


def _write(provider: str, op: str, payload: dict) -> Path:
    dest = OUT / provider
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{op}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {path.relative_to(ROOT)}")
    return path


async def record_openai() -> None:
    import openai

    from rnsr.llm.router import DEFAULT_MODELS

    client = openai.AsyncOpenAI()
    models = DEFAULT_MODELS["openai"]
    complete = await client.chat.completions.create(
        model=models["sub"],
        messages=[{"role": "user", "content": "Reply with the single word pong."}],
        max_completion_tokens=16,
    )
    _write("openai", "complete", _dump(complete))
    embed = await client.embeddings.create(
        model=models["embed"], input=["revenue table"])
    _write("openai", "embed", _dump(embed))
    vision = await client.chat.completions.create(
        model=models["vision"],
        max_completion_tokens=16,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this 1x1 pixel."},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
                }},
            ],
        }],
    )
    _write("openai", "vision", _dump(vision))


async def record_anthropic() -> None:
    import anthropic

    from rnsr.llm.router import DEFAULT_MODELS

    client = anthropic.AsyncAnthropic()
    models = DEFAULT_MODELS["anthropic"]
    complete = await client.messages.create(
        model=models["sub"],
        max_tokens=16,
        messages=[{"role": "user", "content": "Reply with the single word pong."}],
    )
    _write("anthropic", "complete", _dump(complete))
    vision = await client.messages.create(
        model=models["vision"],
        max_tokens=16,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
                }},
                {"type": "text", "text": "Describe this 1x1 pixel."},
            ],
        }],
    )
    _write("anthropic", "vision", _dump(vision))


async def record_gemini() -> None:
    from google import genai
    from google.genai import types

    from rnsr.llm.router import DEFAULT_MODELS

    client = genai.Client()
    models = DEFAULT_MODELS["gemini"]
    complete = await client.aio.models.generate_content(
        model=models["sub"],
        contents="Reply with the single word pong.",
    )
    _write("gemini", "complete", _dump(complete))
    embed = await client.aio.models.embed_content(
        model=models["embed"], contents="revenue table")
    _write("gemini", "embed", _dump(embed))
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000d4944415478da6360000002000154a24f470000000049454e44ae426082"
    )
    vision = await client.aio.models.generate_content(
        model=models["vision"],
        contents=[types.Part.from_bytes(data=png, mime_type="image/png"),
                  "Describe this 1x1 pixel."],
    )
    _write("gemini", "vision", _dump(vision))


RECORDERS = {
    "openai": record_openai,
    "anthropic": record_anthropic,
    "gemini": record_gemini,
}

_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(RECORDERS))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    names = list(RECORDERS) if args.all else [args.provider]
    if not names or names == [None]:
        parser.error("pass --provider NAME or --all")
    for name in names:
        if not os.environ.get(_KEYS[name]):
            raise SystemExit(f"{_KEYS[name]} is not set; skip {name}")
        await RECORDERS[name]()


if __name__ == "__main__":
    asyncio.run(main())
