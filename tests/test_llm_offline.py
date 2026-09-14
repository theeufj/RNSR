"""Offline provider tests rebuilt from tests/llm_fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

FIXTURE = Path(__file__).parent / "llm_fixtures"


def _load(provider: str, op: str) -> dict:
    return json.loads((FIXTURE / provider / f"{op}.json").read_text())


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    return obj


PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d4944415478da6360000002000154a24f470000000049454e44ae426082"
)


class TestDefaultModelsPriced:
    def test_default_models_priced(self):
        from rnsr.llm.cost import PRICES_PER_MTOK
        from rnsr.llm.router import DEFAULT_MODELS

        missing = [
            (provider, role, model)
            for provider, roles in DEFAULT_MODELS.items()
            for role, model in roles.items()
            if model and model not in PRICES_PER_MTOK
        ]
        assert missing == [], f"unpriced DEFAULT_MODELS: {missing}"


class TestOpenAIOffline:
    async def test_complete_embed_vision(self, monkeypatch):
        from rnsr.llm.openai_client import OpenAIClient

        captured: dict = {}
        client = OpenAIClient(api_key="sk-test")

        async def create(**kwargs):
            captured["complete"] = kwargs
            return _ns(_load("openai", "complete"))

        async def embed_create(**kwargs):
            captured["embed"] = kwargs
            return _ns(_load("openai", "embed"))

        async def vision_create(**kwargs):
            captured["vision"] = kwargs
            return _ns(_load("openai", "vision"))

        monkeypatch.setattr(client._client.chat.completions, "create", create)
        resp = await client.complete("ping", model="gpt-5.6-luna", system="s")
        assert resp.text == "pong"
        assert resp.usage.input_tokens == 12
        assert resp.usage.output_tokens == 3
        assert resp.usage.cost_usd > 0
        assert captured["complete"]["model"] == "gpt-5.6-luna"
        assert captured["complete"]["messages"][-1]["content"] == "ping"
        assert "max_completion_tokens" in captured["complete"]

        monkeypatch.setattr(client._client.embeddings, "create", embed_create)
        vectors = await client.embed(["revenue table"], model="text-embedding-3-small")
        assert vectors == [[0.1, 0.2, 0.3]]
        assert captured["embed"]["input"] == ["revenue table"]

        monkeypatch.setattr(client._client.chat.completions, "create", vision_create)
        vis = await client.vision("what", PNG, model="gpt-5.6-luna")
        assert vis.text == "a single pixel"
        assert vis.usage.cost_usd > 0
        assert captured["vision"]["messages"][0]["content"][0]["type"] == "image_url"


class TestAnthropicOffline:
    async def test_complete_and_vision(self, monkeypatch):
        from rnsr.llm.anthropic_client import AnthropicClient

        captured: dict = {}
        client = AnthropicClient(api_key="sk-ant-test")

        async def create(**kwargs):
            captured.setdefault("calls", []).append(kwargs)
            op = "vision" if any(
                isinstance(p, dict) and p.get("type") == "image"
                for msg in kwargs.get("messages", [])
                for p in (msg.get("content") if isinstance(msg.get("content"), list) else [])
            ) else "complete"
            return _ns(_load("anthropic", op))

        monkeypatch.setattr(client._client.messages, "create", create)
        resp = await client.complete("ping", model="claude-haiku-4-5", system="s")
        assert resp.text == "pong"
        assert resp.usage.input_tokens == 10
        assert resp.usage.cost_usd > 0
        assert captured["calls"][0]["model"] == "claude-haiku-4-5"
        assert captured["calls"][0]["messages"][0]["content"] == "ping"

        vis = await client.vision("what", PNG, model="claude-haiku-4-5")
        assert vis.text == "a single pixel"
        assert vis.usage.cost_usd > 0
        assert captured["calls"][1]["messages"][0]["content"][0]["type"] == "image"

        with pytest.raises(NotImplementedError):
            await client.embed(["x"], model="none")


class TestGeminiOffline:
    async def test_complete_embed_vision(self, monkeypatch):
        from rnsr.llm.gemini_client import GeminiClient

        captured: dict = {}
        client = GeminiClient(api_key="g-test")

        async def generate_content(**kwargs):
            captured.setdefault("gen", []).append(kwargs)
            op = "vision" if isinstance(kwargs.get("contents"), list) else "complete"
            return _ns(_load("gemini", op))

        async def embed_content(**kwargs):
            captured["embed"] = kwargs
            return _ns(_load("gemini", "embed"))

        monkeypatch.setattr(client._client.aio.models, "generate_content",
                            generate_content)
        monkeypatch.setattr(client._client.aio.models, "embed_content",
                            embed_content)

        resp = await client.complete("ping", model="gemini-2.5-flash", system="s")
        assert resp.text == "pong"
        assert resp.usage.input_tokens == 8
        assert resp.usage.cost_usd > 0
        assert captured["gen"][0]["model"] == "gemini-2.5-flash"
        assert captured["gen"][0]["contents"] == "ping"

        vectors = await client.embed(["revenue table"], model="gemini-embedding-2")
        assert vectors == [[0.11, 0.22, 0.33]]
        assert captured["embed"]["contents"] == "revenue table"
        assert captured["embed"]["model"] == "gemini-embedding-2"

        vis = await client.vision("what", PNG, model="gemini-2.5-flash")
        assert vis.text == "a single pixel"
        assert vis.usage.cost_usd > 0
        assert isinstance(captured["gen"][1]["contents"], list)
