"""Prefix caching for the root transcript, cache-aware pricing, and per-call
metering.

The root loop re-sends a growing transcript every iteration; in the October
2026 matter benchmark 93% of spend was Sonnet input tokens re-billed at full
price. These tests pin the three properties that make caching pay: the
rendered text is unchanged, earlier blocks never change once appended, and
the breakpoint sits on the last stable block rather than the varying tail.
"""
from types import SimpleNamespace

import pytest

from rnsr.harness.budget import BudgetLedger
from rnsr.harness.prompts.base import render_transcript, render_transcript_segments
from rnsr.harness.root_call import complete_root
from rnsr.llm.anthropic_client import AnthropicClient, _content_param, _system_param
from rnsr.llm.base import LLMResponse, PromptSegments, Usage, prompt_text
from rnsr.llm.cost import compute_cost, make_usage
from rnsr.llm.governor import Governor
from rnsr.llm.mock import MockLLM

TURNS = [("print(doc['a'][:10])", "Revenue was 42."),
         ("search('licence')", "0 hits\n[harness] note"),
         ("FINAL('42')", "[harness] FINAL is pending review")]


class TestTranscriptSegments:
    def test_flat_rendering_is_unchanged(self):
        seg = render_transcript_segments("Q?", TURNS, final_hint="FINAL_BATCH")
        assert seg.text() == render_transcript("Q?", TURNS, final_hint="FINAL_BATCH")
        assert prompt_text(seg) == seg.text()
        assert seg.text().startswith("TASK:\nQ?\n\n--- cell [1] ---")
        assert seg.text().endswith("Call FINAL_BATCH when done.")

    def test_prefix_is_append_only_across_iterations(self):
        previous = render_transcript_segments("Q?", TURNS[:1]).stable
        for n in range(2, len(TURNS) + 1):
            current = render_transcript_segments("Q?", TURNS[:n]).stable
            assert current[:len(previous)] == previous
            assert len(current) == len(previous) + 1
            previous = current

    def test_tail_holds_only_the_closing_instruction(self):
        seg = render_transcript_segments("Q?", TURNS)
        assert "Write the next python code cell" in seg.tail
        assert all("Write the next python code cell" not in block for block in seg.stable)


class TestAnthropicCacheControl:
    def test_breakpoint_on_last_stable_block_not_the_tail(self):
        seg = render_transcript_segments("Q?", TURNS)
        blocks = _content_param(seg)
        assert "".join(b["text"] for b in blocks) == seg.text()
        marked = [i for i, b in enumerate(blocks) if "cache_control" in b]
        assert marked == [len(seg.stable) - 1]
        assert "cache_control" not in blocks[-1]

    def test_plain_prompts_and_empty_system_are_untouched(self):
        assert _content_param("ping") == "ping"
        assert _system_param("") == "" and _system_param(None) == ""
        assert _system_param("sys") == [{"type": "text", "text": "sys",
                                         "cache_control": {"type": "ephemeral"}}]
        only_tail = PromptSegments((), "just this")
        assert _content_param(only_tail) == "just this"

    async def test_cache_tokens_are_read_from_the_response(self, monkeypatch):
        client = AnthropicClient(api_key="sk-ant-test")
        captured = {}

        async def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")],
                usage=SimpleNamespace(input_tokens=1_000, output_tokens=100,
                                      cache_read_input_tokens=30_000,
                                      cache_creation_input_tokens=5_000))

        monkeypatch.setattr(client._client.messages, "create", create)
        seg = render_transcript_segments("Q?", TURNS)
        resp = await client.complete(seg, model="claude-sonnet-5-5", system="rules")
        assert isinstance(captured["messages"][0]["content"], list)
        assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert resp.usage.input_tokens == 36_000
        assert (resp.usage.cache_read_tokens, resp.usage.cache_write_tokens) == (30_000, 5_000)
        # 1k uncached @ $2 + 30k reads @ $0.10 + 5k writes @ $2.50 + 100 out @ $10
        assert resp.usage.cost_usd == pytest.approx(0.0185)
        assert compute_cost("claude-sonnet-5-5", 36_000, 100) == pytest.approx(0.073)

    async def test_missing_cache_fields_price_as_uncached(self, monkeypatch):
        client = AnthropicClient(api_key="sk-ant-test")

        async def create(**kwargs):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")],
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=4))

        monkeypatch.setattr(client._client.messages, "create", create)
        resp = await client.complete("ping", model="claude-haiku-5-5")
        assert resp.usage == Usage(10, 4, compute_cost("claude-haiku-5-5", 10, 4))


class TestCachePricing:
    def test_multipliers_default_and_per_model(self):
        # Standard models: 0.1x reads, 1.25x writes on the base input price.
        assert compute_cost("claude-haiku-5-5", 0, 0, cache_read_tokens=1_000_000) == pytest.approx(0.01)
        assert compute_cost("claude-haiku-5-5", 0, 0, cache_write_tokens=1_000_000) == pytest.approx(0.125)
        # Sonnet 5.5 reads at 0.05x (pricing page 2026-10-07).
        assert compute_cost("claude-sonnet-5-5", 0, 0, cache_read_tokens=1_000_000) == pytest.approx(0.10)
        assert compute_cost("claude-sonnet-5-5", 0, 0, cache_write_tokens=1_000_000) == pytest.approx(2.50)

    def test_make_usage_reports_whole_prompt(self):
        u = make_usage("claude-sonnet-5-5", 100, 10, cache_read_tokens=900)
        assert u.input_tokens == 1_000 and u.cache_read_tokens == 900
        total = u + Usage(5, 1, 0.0, 2, 3)
        assert (total.cache_read_tokens, total.cache_write_tokens) == (902, 3)

    def test_other_providers_flatten_segments(self):
        mock = MockLLM(default="x")
        seg = PromptSegments(("a", "b"), "c")
        import asyncio

        asyncio.run(mock.complete(seg, model="m"))
        assert mock.calls[-1]["prompt"] == "a\nb\nc"


class _Events:
    def __init__(self):
        self.records = []

    def event(self, kind, **data):
        self.records.append({"kind": kind, **data})


class TestMetering:
    async def test_root_call_event_carries_per_call_usage(self):
        usage = make_usage("claude-sonnet-5-5", 1_000, 50, cache_read_tokens=20_000)

        async def complete(*_a, **_k):
            return LLMResponse("ok", "claude-sonnet-5-5", usage)

        events = _Events()
        await complete_root(SimpleNamespace(complete=complete), "p", model="m", system="s",
                            ledger=BudgetLedger(), trajectory=events)
        done = next(e for e in events.records if e["kind"] == "root_call_completed")
        assert done["input_tokens"] == 21_000 and done["output_tokens"] == 50
        assert done["cache_read_tokens"] == 20_000 and done["cache_write_tokens"] == 0
        assert done["cost_usd"] == pytest.approx(usage.cost_usd, abs=1e-6)

    def test_ledger_snapshot_and_governor_total_tokens(self):
        ledger = BudgetLedger()
        ledger.add_usage(Usage(100, 10, 0.5, 60, 20))
        snap = ledger.snapshot()
        assert (snap["cache_read_tokens"], snap["cache_write_tokens"]) == (60, 20)
        gov = Governor()
        gov.record(Usage(100, 10, 0.5, 60, 20))
        gov.record(Usage(50, 5, 0.1))
        snap = gov.snapshot()
        assert snap["requests"] == 2 and snap["spend_usd"] == pytest.approx(0.6)
        assert (snap["input_tokens"], snap["output_tokens"]) == (150, 15)
        assert (snap["cache_read_tokens"], snap["cache_write_tokens"]) == (60, 20)


class TestLenientCodeExtraction:
    """Replies Sonnet 5.5 produced live that the strict fence regex rejected;
    each one cost a wasted full-context call."""

    @pytest.fixture
    def extract(self):
        from rnsr.harness.loop import RootRunner

        return lambda text: RootRunner.__dict__["_extract_code"](None, text)

    def test_fenced_block_still_wins(self, extract):
        assert extract("```python\nx = 1\nprint(x)\n```") == "x = 1\nprint(x)"

    def test_lost_opening_fence_with_prose_either_side(self, extract):
        reply = ("Need X; search for it.\n\npython\nprint(doc['a'][:10])\n"
                 "for k in doc: print(k)\n```\n\nAlso read narrative docs next.")
        assert extract(reply) == "print(doc['a'][:10])\nfor k in doc: print(k)"

    def test_xml_tool_call_form_with_namespace_and_inner_fence(self, extract):
        ns = "antml"
        reply = (f'Let me look.\n\n<{ns}:invoke name="python">\n<{ns}:parameter name="code">'
                 f'```python\nfor d in ["a"]:\n    print(d)\n</{ns}:parameter>\n</{ns}:invoke>\nelements')
        assert extract(reply) == 'for d in ["a"]:\n    print(d)'
        twice = ('<parameter name="python">g(1)\ng(2)</parameter>\n\n'
                 '<parameter name="python">g(1)\ng(2)</parameter>')
        assert extract(twice) == "g(1)\ng(2)"

    def test_stray_tag_inside_valid_code_is_dropped(self, extract):
        reply = "\nimport re\ndef f():\n    pass\n</reasoning_effort>\n\nf()\n"
        assert extract(reply) == "import re\ndef f():\n    pass\n\nf()"

    def test_comparison_operators_survive(self, extract):
        code = "a = 1\nif a < 2 and a > 0:\n    print(a)"
        assert extract(code) == code

    @pytest.mark.parametrize("prose", [
        "I think the answer is probably in the context somewhere.",
        "Need the s56 due date.", "Yes", "print", "Done — see above.",
        '<invoke name="python">\n</invoke>', "",
    ])
    def test_prose_never_runs(self, extract, prose):
        assert extract(prose) is None
