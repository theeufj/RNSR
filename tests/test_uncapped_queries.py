"""An uncapped query reaches a final beyond the former limits and stays cancellable."""

import asyncio
import json
from pathlib import Path

import pytest

from rnsr.config import Settings
from rnsr.env.sandbox import SandboxedRepl
from rnsr.harness.budget import BudgetLedger
from rnsr.harness.loop import EnvSpec, RootRunner
from rnsr.llm.base import Usage
from rnsr.llm.mock import MockLLM

ENV = EnvSpec(mode="classic", context="North: 2\nSouth: 3")


def runner(root, sub=None, **settings):
    return RootRunner(root_client=root, root_model="root", sub_client=sub or MockLLM(),
                      sub_model="sub", settings=Settings(**settings))


async def test_reaches_final_past_all_four_previous_query_caps(tmp_path, monkeypatch):
    monkeypatch.setattr(BudgetLedger, "wall_s", property(lambda self: 7200.0))
    root = MockLLM(usage_per_call=Usage(100, 10, 0.2)).script(
        *(f"```python\nprint('inspection {i}')\n```" for i in range(21)),
        "```python\nlabels = llm_map(['classify row'] * 388)\nprint(len(labels))\n```",
        "```python\nFINAL(len(labels))\n```",
    )
    result = await runner(root, MockLLM(default="COMPLETE")).run(
        "How many rows were classified?", ENV, run_dir=tmp_path)
    assert result.status == "final" and result.answer == 388
    assert result.ledger["root_iters"] == 23
    assert result.ledger["sub_calls"] == 389  # includes completeness review
    assert result.ledger["spend_usd"] > 2 and result.ledger["wall_s"] > 600
    assert all("BUDGET LOW" not in call["prompt"] for call in root.calls)
    events = [json.loads(line) for line in Path(result.trajectory_path).read_text().splitlines()]
    assert not any(e["kind"] in ("budget_warning", "budget_breached") for e in events)
    json.dumps(events, allow_nan=False)


async def test_final_observation_review_has_no_infinite_number_conversion(tmp_path):
    root = MockLLM().script("```python\nprint(42)\nFINAL(42)\n```",
                            "```python\nFINAL(42)\n```")
    result = await runner(root).run("q", ENV, run_dir=tmp_path)
    assert result.status == "final" and result.answer == 42
    assert "No overall query time or iteration limit" in root.calls[1]["prompt"]


async def test_uncapped_nonfinal_loop_remains_cancellable_and_closes_child(tmp_path, monkeypatch):
    beyond_old_cap = asyncio.Event()

    class NonfinalRoot(MockLLM):
        async def complete(self, *args, **kwargs):
            response = await super().complete(*args, **kwargs)
            if len(self.calls) > 20:
                beyond_old_cap.set()
                await asyncio.Event().wait()
            return response

    sandboxes = []

    def sandbox_factory(**kwargs):
        sandbox = SandboxedRepl(**kwargs)
        sandboxes.append(sandbox)
        return sandbox

    monkeypatch.setattr("rnsr.harness.loop.SandboxedRepl", sandbox_factory)
    root = NonfinalRoot(default="```python\nx = 1\n```")
    task = asyncio.create_task(runner(root).run("q", ENV, run_dir=tmp_path))
    try:
        await asyncio.wait_for(beyond_old_cap.wait(), timeout=15)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert sandboxes[0]._proc is None or sandboxes[0]._proc.returncode is not None


async def test_permanent_worker_failure_is_visible_instead_of_repeated_empty_labels(tmp_path):
    class Unauthenticated(MockLLM):
        async def complete(self, *args, **kwargs):
            raise ValueError("HTTP 401: invalid API key")

    root = MockLLM().script("```python\nlabels = llm_map(['classify'])\n```")
    result = await runner(root, Unauthenticated()).run("q", ENV, run_dir=tmp_path)
    assert result.status == "error" and result.answer is None
    assert len(root.calls) == 1
    events = [json.loads(line) for line in Path(result.trajectory_path).read_text().splitlines()]
    assert any(e["kind"] == "error" and "PermanentProviderError" in e["error"] for e in events)


def test_verbose_output_cannot_hide_error_feedback():
    from rnsr.env.sandbox import CellResult

    observation = runner(MockLLM())._observe(
        CellResult(ok=False, stdout="source excerpt " * 1000, error="Quote 2 did not match the source"))
    assert "Quote 2 did not match the source" in observation
    assert "truncated" in observation
