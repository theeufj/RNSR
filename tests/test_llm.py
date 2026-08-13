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


class TestModelValidation:
    """An unpriced model makes every spend cap infinite; a retired model name
    fails every call. Both used to surface only mid-run."""

    def test_unpriced_model_warns_once(self, caplog):
        import logging

        from rnsr.llm import validate

        validate._warned.clear()
        with caplog.at_level(logging.WARNING, logger="rnsr.llm.validate"):
            assert validate.check_pricing("imaginary-model-9", "root") is False
            assert validate.check_pricing("imaginary-model-9", "root") is False
        warnings = [r for r in caplog.records if "pricing_missing" in r.message]
        assert len(warnings) == 1

    def test_priced_model_is_quiet(self, caplog):
        import logging

        from rnsr.llm import validate

        with caplog.at_level(logging.WARNING, logger="rnsr.llm.validate"):
            assert validate.check_pricing("claude-haiku-4-5", "root") is True
        assert caplog.records == []

    def test_every_default_model_has_a_price(self):
        # the check that would have caught the gpt-5.2 rot before a live run
        from rnsr.llm.cost import PRICES_PER_MTOK
        from rnsr.llm.router import DEFAULT_MODELS

        missing = [
            (provider, role, model)
            for provider, roles in DEFAULT_MODELS.items()
            for role, model in roles.items()
            if model and role not in ("embed",) and model not in PRICES_PER_MTOK
        ]
        assert missing == []

    async def test_live_check_reports_retired_names(self, monkeypatch):
        from rnsr.llm import validate

        class FakeResolved:
            def __init__(self, model):
                self.model = model

                class C:
                    provider = "openai"
                self.client = C()

        class FakeRouter:
            def resolve(self, role):
                return FakeResolved("gpt-5.2" if role == "root"
                                    else "text-embedding-3-small")

        async def fake_list(client, provider):
            return {"gpt-5.6-sol", "gpt-5.6-terra", "text-embedding-3-small"}

        monkeypatch.setattr(validate, "_list_models", fake_list)
        findings = await validate.check_models_live(FakeRouter(),
                                                   roles=("root",))
        assert any("not offered by the provider" in f["problem"] for f in findings)

    async def test_live_check_accepts_dated_snapshots(self, monkeypatch):
        from rnsr.llm import validate

        class FakeResolved:
            model = "claude-sonnet-4-5"

            class client:
                provider = "anthropic"

        class FakeRouter:
            def resolve(self, role):
                return FakeResolved()

        async def fake_list(client, provider):
            return {"claude-sonnet-4-5-20260514"}

        monkeypatch.setattr(validate, "_list_models", fake_list)
        findings = await validate.check_models_live(FakeRouter(), roles=("root",))
        assert findings == []


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


class TestOpenAIClient:
    @staticmethod
    def _fake_create(captured: dict):
        async def create(**kwargs):
            captured.clear()
            captured.update(kwargs)

            class Usage:
                prompt_tokens = 1
                completion_tokens = 1

            class Message:
                content = "ok"

            class Choice:
                message = Message()

            class Resp:
                usage = Usage()
                choices = [Choice()]
            return Resp()
        return create

    async def test_reasoning_models_omit_sampling_params(self, monkeypatch):
        # gpt-5.x rejects temperature != 1 with 400 unsupported_value; the
        # client must not send temperature/seed at all (seen live: every
        # OpenAI root call failed until these were stripped)
        from rnsr.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key="sk-test")
        captured: dict = {}
        monkeypatch.setattr(client._client.chat.completions, "create",
                            self._fake_create(captured))
        await client.complete("hi", model="gpt-5.6-terra",
                              temperature=0.0, seed=42)
        assert "temperature" not in captured
        assert "seed" not in captured
        assert captured["max_completion_tokens"] == 4096

    async def test_non_reasoning_models_keep_sampling_params(self, monkeypatch):
        from rnsr.llm.openai_client import OpenAIClient

        client = OpenAIClient(api_key="sk-test")
        captured: dict = {}
        monkeypatch.setattr(client._client.chat.completions, "create",
                            self._fake_create(captured))
        await client.complete("hi", model="gpt-4.1", temperature=0.0, seed=42)
        assert captured["temperature"] == 0.0
        assert captured["seed"] == 42


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

    def test_embed_override_still_falls_through_anthropic(self, monkeypatch):
        # an embed-model override must not route embeddings to a client
        # that has no embedding API (seen live: anthropic + gemini override)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
        monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        r = Router(Settings(provider="anthropic", embed_model="gemini-embedding-001"))
        resolved = r.resolve("embed")
        assert resolved.client.provider == "gemini"
        assert resolved.model == "gemini-embedding-001"
