import pytest

from rnsr.answer_semantics import (
    AnswerKind,
    classify_answer,
    is_negative,
    publish_answer,
    requires_quote,
)
from rnsr.eval.metrics import EvalResult, score_answer, summarize
from rnsr.eval.regression import score_run
from rnsr.forms.fanout import is_negative as form_negative
from rnsr.harness.budget import BudgetLedger
from rnsr.harness.loop import ConsensusAnswer, RootRunner
from rnsr.llm.mock import MockLLM


@pytest.mark.parametrize("value", ["blank", "leave blank", "not reached", "na", "nil",
                                   "not specified", "Not found", "NOT_FOUND",
                                   "Not found in matter corpus", "not_applicable", "No!"])
def test_absent_vocabulary_agrees_across_product_and_scoring(value):
    assert is_negative(value) and form_negative(value)
    report = score_run({"missing": []}, {"missing": value}, max_false_positive_rate=0)
    assert report.passed and report.false_positive_rate == 0
    result = EvalResult("missing", "absent", value, "", True, "final", 0, 0, 0, 1)
    assert summarize([result])["false_positive_rate"] == 0


def test_classification_preserves_domain_distinctions():
    assert classify_answer("No") == AnswerKind.NEGATIVE
    assert classify_answer("unknown") == AnswerKind.ABSENT
    assert classify_answer("not applicable") == AnswerKind.NOT_APPLICABLE
    assert requires_quote("Mallory Smith") and requires_quote(42)
    assert not requires_quote("yes")
    assert not score_answer("", "a real value")


@pytest.mark.parametrize(("tier", "expected"), [("low", "NEEDS REVIEW"),
    ("medium", "NEEDS REVIEW"), ("high", "answer"), (None, "NEEDS REVIEW")])
def test_publication_threshold_is_inclusive_and_unknown_fails_closed(tier, expected):
    assert publish_answer("answer", tier, "medium") == expected
    assert publish_answer(None, tier, "medium") is None


def test_split_consensus_is_contested_without_tiebreak():
    assert ConsensusAnswer("a", "split", .5).contested
    assert ConsensusAnswer("a", "tiebreak", .5).contested
    assert not ConsensusAnswer("a", "unanimous", 1).contested


def test_extract_code_accepts_bare_final_rejects_prose():
    runner = RootRunner(MockLLM(), "mock", MockLLM(), "mock")
    assert runner._extract_code("FINAL('yes')") == "FINAL('yes')"
    assert runner._extract_code("the answer is probably yes") is None
    assert runner._extract_code("answer") is None


async def test_completeness_check_never_starts_after_cap():
    sub = MockLLM(default="COMPLETE")
    runner = RootRunner(MockLLM(), "mock", sub, "mock")
    for ledger in (BudgetLedger(max_sub_calls=0), BudgetLedger(max_wall_s=0)):
        assert await runner._completeness_gap("q", {"value": "a"}, ledger) is None
    assert sub.calls == []


async def test_completeness_attempt_counted_on_failure():
    sub = MockLLM(default="COMPLETE")
    sub.fail_times = 1
    runner = RootRunner(MockLLM(), "mock", sub, "mock")
    ledger = BudgetLedger()
    assert await runner._completeness_gap("q", {"value": "a"}, ledger) is None
    assert ledger.sub_calls == 1


@pytest.mark.parametrize(("count", "enabled"), [(199, False), (200, True)])
async def test_embedding_threshold_reaches_sandbox(monkeypatch, tmp_path, count, enabled):
    from rnsr.harness.loop import EnvSpec
    observed = []

    class Sandbox:
        def __init__(self, **kwargs):
            pass
        async def start(self, **kwargs):
            observed.append(kwargs["init_extra"]["enable_embeddings"])
            raise RuntimeError("stop before running provider")
        async def close(self):
            pass

    monkeypatch.setattr("rnsr.harness.loop.SandboxedRepl", Sandbox)
    runner = RootRunner(MockLLM(), "mock", MockLLM(), "mock", embed_client=MockLLM())
    await runner.run("q", EnvSpec("docdb", manifest={"documents": [{}] * count}), run_dir=tmp_path)
    assert observed == [enabled]


async def test_recovery_namespace_read_obeys_remaining_wall_budget():
    import asyncio
    from types import SimpleNamespace

    from rnsr.harness.recovery import recover_variable

    class SlowSandbox:
        async def vars(self):
            await asyncio.Event().wait()

    root = MockLLM()
    ledger = BudgetLedger(max_wall_s=0.02)
    result = await asyncio.wait_for(recover_variable(
        SlowSandbox(), SimpleNamespace(root_client=root), "question", [], None,
        ledger), timeout=1.0)
    assert result is None
    assert root.calls == [] and ledger.root_iters == 0
