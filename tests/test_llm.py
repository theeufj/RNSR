"""llm layer: router role resolution, batch fan-out, cost, mock behavior."""

import pytest

from rnsr.config import Settings
from rnsr.llm.base import Usage
from rnsr.llm.batch import map_prompts
from rnsr.llm.cost import compute_cost
from rnsr.llm.mock import MockLLM
from rnsr.llm.router import DEFAULT_MODELS, Router, detect_provider


class TestRouter:
    def test_auto_detect_prefers_first_available(self, monkeypatch):
        for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert detect_provider(Settings(provider="auto")) == "openai"

    def test_no_keys_raises(self, monkeypatch):
        for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        with pytest.raises(RuntimeError, match="no LLM provider"):
            detect_provider(Settings(provider="auto"))

    def test_explicit_provider_wins(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert detect_provider(Settings(provider="gemini")) == "gemini"

    def test_role_defaults_and_overrides(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        r = Router(Settings(provider="openai", sub_model="gpt-5-mini-custom"))
        assert r.resolve("root").model == DEFAULT_MODELS["openai"]["root"]
        assert r.resolve("sub").model == "gpt-5-mini-custom"

    def test_anthropic_embed_falls_through(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
        r = Router(Settings(provider="anthropic"))
        resolved = r.resolve("embed")
        assert resolved.model == DEFAULT_MODELS["openai"]["embed"]
        assert resolved.client.provider == "openai"

    def test_anthropic_embed_without_fallback_raises(self, monkeypatch):
        for env in ("OPENAI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
        r = Router(Settings(provider="anthropic"))
        with pytest.raises(RuntimeError, match="no embeddings"):
            r.resolve("embed")

    def test_unknown_role_rejected(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        with pytest.raises(ValueError, match="unknown role"):
            Router(Settings(provider="openai")).resolve("oracle")


class TestBatch:
    async def test_order_preserved(self):
        mock = MockLLM().rule("alpha", "A").rule("beta", "B")
        out = await map_prompts(mock, ["say alpha", "say beta"], model="m")
        assert [r.text for r in out] == ["A", "B"]

    async def test_concurrency_bounded(self):
        mock = MockLLM(delay_s=0.01, default="ok")
        await map_prompts(mock, ["p"] * 40, model="m", concurrency=5)
        assert mock.max_in_flight <= 5
        assert len(mock.calls) == 40

    async def test_retry_on_rate_limit(self):
        mock = MockLLM(default="recovered", fail_times=2)
        out = await map_prompts(mock, ["p"], model="m", attempts=4)
        assert out[0] is not None and out[0].text == "recovered"
        assert len(mock.calls) == 3  # 2 failures + 1 success

    async def test_exhausted_retries_resolve_none(self):
        mock = MockLLM(default="never", fail_times=99)
        out = await map_prompts(mock, ["p"], model="m", attempts=2)
        assert out == [None]

    async def test_usage_callback_fires(self):
        seen: list[Usage] = []
        mock = MockLLM(default="ok")
        await map_prompts(mock, ["a", "b"], model="m", on_usage=seen.append)
        assert len(seen) == 2
        assert seen[0].input_tokens == 100


class TestCost:
    def test_known_model(self):
        # 1M in + 1M out at (1.0, 5.0) $/Mtok
        assert compute_cost("claude-haiku-4-5", 1_000_000, 1_000_000) == 6.0

    def test_unknown_model_zero_but_flagged(self):
        from rnsr.llm import cost

        assert compute_cost("mystery-lm", 1000, 1000) == 0.0
        assert "mystery-lm" in cost.unknown_models

    def test_usage_addition(self):
        total = Usage(10, 5, 0.1) + Usage(20, 10, 0.2)
        assert (total.input_tokens, total.output_tokens) == (30, 15)
        assert abs(total.cost_usd - 0.3) < 1e-9


class TestMock:
    async def test_queue_then_rules_then_default(self):
        mock = MockLLM(default="D").script("Q1").rule("hello", "R")
        r1 = await mock.complete("hello")     # queue wins
        r2 = await mock.complete("hello")     # rule
        r3 = await mock.complete("other")     # default
        assert (r1.text, r2.text, r3.text) == ("Q1", "R", "D")

    async def test_embed_deterministic(self):
        mock = MockLLM()
        a = await mock.embed(["revenue table"], model="e")
        b = await mock.embed(["revenue table"], model="e")
        assert a == b and len(a[0]) == 10
