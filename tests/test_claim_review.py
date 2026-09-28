"""Advisory semantic review must not become a new source of answer authority."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from rnsr.config import Settings
from rnsr.env.sandbox import CellResult
from rnsr.harness import claim_review
from rnsr.harness.budget import BudgetLedger
from rnsr.harness.claim_review import review_final
from rnsr.harness.evidence import from_final
from rnsr.harness.loop import EnvSpec, RootRunner
from rnsr.harness.trajectory import redact
from rnsr.llm.mock import MockLLM


def verified(quote="Revenue was 42.", heading="Fiscal 2020"):
    match = {"doc_id": "report", "char_start": 11, "char_end": 11 + len(quote),
             "source_context": {"page": 2, "heading_paths": [heading],
                                "text": f"{heading}\n{quote} Excludes discontinued operations.",
                                "source_path": "/private/source/should-not-be-forwarded.pdf"}}
    return {"passed": True, "check": "lexical_source_match",
            "quotes": [{"quote": quote, "matched": True, **match, "matches": [match]}]}


def final(answer="42", **kwargs):
    return {"value": answer, "verification": verified(**kwargs)}


def reply(verdict="supported", fid="f0", refs=None, reason="The source supports the claim."):
    return {"field_id": fid, "verdict": verdict,
            "evidence_ids": [f"{fid}q0m0"] if refs is None else refs, "reason": reason}


def client(*rows):
    return MockLLM(default=json.dumps({"reviews": list(rows or [reply()])}))


async def run_review(value=None, *, question="What was revenue in 2020?", sub=None,
                     ledger=None, **kwargs):
    return await review_final(value or final(), [("question-1", question)], batch=False,
                              client=sub or client(), model="review-model",
                              ledger=ledger or BudgetLedger(), **kwargs)


@pytest.mark.parametrize("verdict", ["supported", "contradicted", "insufficient"])
async def test_records_verdict_without_mutating_accepted_final(verdict):
    submitted = final()
    original = copy.deepcopy(submitted)
    sub, ledger = client(reply(verdict)), BudgetLedger()
    result = await run_review(submitted, sub=sub, ledger=ledger)
    assert submitted == original
    assert result["advisory"] and result["scope"] == "provided_verified_evidence_only"
    assert result["fields"][0]["verdict"] == verdict
    assert ledger.sub_calls == 1 and ledger.spend_usd == 0.001
    assert result["usage"]["cost_usd"] == 0.001
    assert result["model"] == result["response_model"] == "review-model"
    assert result["provider"] == "mock"


async def test_uses_actual_question_definitions_and_verified_scope_only():
    sub = client(reply("insufficient"))
    submitted = final("Yes", quote="A royalty-free licence is granted.")
    submitted["gold"] = "GOLD_SENTINEL"
    submitted["verification"]["untrusted_injected_metadata"] = "UNTRUSTED_SENTINEL"
    question = "Does this clause restrict resale prices?"
    definitions = "Price restriction means limits on resale pricing, not licence royalties."
    result = await run_review(submitted, question=question, sub=sub,
                              category_definitions=definitions)
    prompt = json.loads(sub.calls[0]["prompt"])
    assert prompt["category_definitions"] == definitions
    field = prompt["fields"][0]
    assert field["question"] == question and field["answer"] == "Yes"
    assert field["evidence"][0]["heading_paths"] == ["Fiscal 2020"]
    assert "Excludes discontinued operations" in field["evidence"][0]["source_context"]
    for secret in ("GOLD_SENTINEL", "UNTRUSTED_SENTINEL", "/private/source/"):
        assert secret not in sub.calls[0]["prompt"]
        assert secret not in json.dumps(result)
    assert "task DATA" in sub.calls[0]["system"]
    assert "negative/absence" in sub.calls[0]["system"]


@pytest.mark.parametrize("answer", ["Yes", "No", "NOT_FOUND", "unknown"])
async def test_no_quote_claims_are_insufficient_without_paid_call(answer):
    sub = client()
    result = await run_review({"value": answer, "verification": {
        "passed": False, "quotes": [], "zero_quotes": True}}, sub=sub)
    assert not sub.calls and not result["attempted"]
    assert result["fields"][0]["verdict"] == "insufficient"
    assert result["fields"][0]["reason_code"] == "no_verified_evidence"


@pytest.mark.parametrize("answer", ["Yes", "No"])
async def test_quoted_boolean_claims_do_reach_reviewer(answer):
    sub = client(reply("contradicted"))
    result = await run_review(final(answer), sub=sub)
    assert len(sub.calls) == 1 and result["fields"][0]["verdict"] == "contradicted"


async def test_batch_fields_are_independent_and_use_one_call():
    sub = client(reply("contradicted", "f1"), reply("supported", "f0"))
    submitted = {"value": {"a": "42", "b": "Yes", "c": "No"}, "verification": {
        "a": verified(), "b": verified(), "c": {"passed": False, "quotes": []}}}
    result = await review_final(submitted, [("a", "Revenue?"), ("b", "Exclusive?"),
                                          ("c", "Perpetual?")], batch=True,
                                client=sub, model="sub", ledger=BudgetLedger())
    assert len(sub.calls) == 1
    assert [f["verdict"] for f in result["fields"]] == ["supported", "contradicted", "insufficient"]
    assert [f["field_id"] for f in json.loads(sub.calls[0]["prompt"])["fields"]] == ["f0", "f1"]


@pytest.mark.parametrize("text", [
    "supported", '{"reviews": []}', '{"reviews": null}',
    json.dumps({"reviews": [reply("yes")]}),
    json.dumps({"reviews": [reply(refs=[])]}),
    json.dumps({"reviews": [reply(refs=["f1q0m0"])]}),
    json.dumps({"reviews": [reply(fid="other")]}),
    json.dumps({"reviews": [reply(reason="")]}),
    "x" * (claim_review.MAX_RESPONSE_CHARS + 1),
])
async def test_invalid_reviews_cannot_become_support(text):
    ledger = BudgetLedger()
    result = await run_review(sub=MockLLM(default=text), ledger=ledger)
    assert result["status"] == "error"
    assert result["fields"][0]["verdict"] is None
    assert ledger.sub_calls == 1 and ledger.spend_usd == 0.001  # response still billable
    assert "reply" not in result  # no unvalidated provider body in logs


_SAVED_REVIEWS = json.loads((Path(__file__).parent / "fixtures" / "claim_review" /
                             "haiku_fenced_reviews.json").read_text())["cases"]


@pytest.mark.parametrize("saved", _SAVED_REVIEWS, ids=lambda saved: saved["case"])
def test_saved_fenced_provider_replies_preserve_strict_verdicts(saved):
    text = saved["response_text"]
    assert hashlib.sha256(text.encode()).hexdigest() == saved["response_sha256"]
    fields = [{"field_id": saved["field_id"], "evidence": [
        {"evidence_id": ref} for ref in saved["available_evidence_ids"]]}]
    outcomes, reasons = claim_review._parse_reply(text, fields)
    assert outcomes["f0"]["verdict"] == saved["expected_verdict"]
    assert outcomes["f0"]["evidence_ids"] == saved["expected_evidence_ids"]
    assert reasons["f0"]


@pytest.mark.parametrize("wrapper", [
    "```json\n{}\n```", " \n```json\r\n{}\r\n```\n ",
])
async def test_whole_json_fence_accepted_once_through_review(wrapper):
    sub = MockLLM(default=wrapper.format(json.dumps({"reviews": [reply()]})))
    result = await run_review(sub=sub)
    assert result["status"] == "completed" and len(sub.calls) == 1
    assert result["fields"][0]["verdict"] == "supported"


@pytest.mark.parametrize("wrapper", [
    "Here is the result:\n```json\n{}\n```", "```json\n{}\n```\nExtra text",
    "```json\n{}\n```\n```json\n{{}}\n```", "```python\n{}\n```",
    "```\n{}\n```", "```json\n{}\n{{}}\n```",
])
async def test_json_fence_does_not_allow_prose_multiple_objects_or_other_languages(wrapper):
    result = await run_review(sub=MockLLM(
        default=wrapper.format(json.dumps({"reviews": [reply()]}))))
    assert result["status"] == "error" and result["fields"][0]["verdict"] is None


@pytest.mark.parametrize("bad_reply", [reply("yes"), reply(refs=[]), reply(refs=["other"]),
                                       reply(fid="other"), reply(reason="")])
async def test_fenced_object_keeps_existing_schema_validation(bad_reply):
    text = "```json\n" + json.dumps({"reviews": [bad_reply]}) + "\n```"
    result = await run_review(sub=MockLLM(default=text))
    assert result["status"] == "error" and result["fields"][0]["verdict"] is None


async def test_duplicate_batch_field_rejected():
    sub = client(reply(), reply())
    result = await review_final({"value": {"a": "1", "b": "2"},
                                 "verification": {"a": verified(), "b": verified()}},
                                [("a", "A?"), ("b", "B?")], batch=True,
                                client=sub, model="sub", ledger=BudgetLedger())
    assert result["status"] == "error"
    assert all(f["verdict"] is None for f in result["fields"])


@pytest.mark.parametrize("ledger", [
    BudgetLedger(max_sub_calls=1, sub_calls=1),
    BudgetLedger(max_spend_usd=1, spend_usd=1),
    BudgetLedger(max_root_iters=1, root_iters=1),
    BudgetLedger(max_wall_s=0.001, _t0=0),
])
async def test_existing_caps_skip_advisory(ledger):
    sub = client()
    result = await run_review(sub=sub, ledger=ledger)
    assert not sub.calls and result["status"] == "skipped"
    assert result["fields"][0]["verdict"] is None


async def test_uncapped_query_can_review_after_former_limits():
    sub = client()
    ledger = BudgetLedger(max_root_iters=0, max_sub_calls=0,
                          max_wall_s=0, max_spend_usd=0,
                          root_iters=100, sub_calls=1000, spend_usd=100, _t0=0)
    result = await run_review(sub=sub, ledger=ledger)
    assert len(sub.calls) == 1 and result["status"] == "completed"
    assert result["fields"][0]["verdict"] == "supported"
    assert ledger.sub_calls == 1001


async def test_remaining_wall_budget_skips_call():
    sub = client()
    result = await run_review(sub=sub, ledger=BudgetLedger(max_wall_s=0.5))
    assert not sub.calls and result["status"] == "skipped"
    assert result["fields"][0]["reason_code"] == "remaining_wall_budget"


@pytest.mark.parametrize("bound", ["review", "query"])
async def test_timeout_cancels_call_without_retry(monkeypatch, bound):
    if bound == "review":
        monkeypatch.setattr(claim_review, "MAX_REVIEW_S", 0.03)
    monkeypatch.setattr(claim_review, "MIN_REMAINING_S", 0)
    sub = MockLLM(delay_s=20)
    ledger = BudgetLedger(max_wall_s=0.03 if bound == "query" else 600)
    result = await run_review(sub=sub, ledger=ledger)
    assert result["status"] == "error" and result["error_type"] == "TimeoutError"
    assert len(sub.calls) == ledger.sub_calls == 1 and sub.in_flight == 0


async def test_caller_cancellation_propagates_and_leaves_no_worker():
    started, stopped = asyncio.Event(), asyncio.Event()

    async def hang(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    task = asyncio.create_task(run_review(sub=SimpleNamespace(provider="mock", complete=hang)))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()


async def test_provider_failure_does_not_retry_or_log_response_body():
    async def fail(*args, **kwargs):
        raise RuntimeError("SECRET_PROVIDER_BODY")

    ledger = BudgetLedger()
    result = await run_review(sub=SimpleNamespace(provider="mock", complete=fail), ledger=ledger)
    assert result["status"] == "error" and ledger.sub_calls == 1
    assert "SECRET" not in json.dumps(result)


@pytest.mark.parametrize("part,limit", [
    ("question", claim_review.MAX_QUESTION_CHARS),
    ("answer", claim_review.MAX_ANSWER_CHARS),
    ("definition", claim_review.MAX_DEFINITION_CHARS),
    ("quote", claim_review.MAX_QUOTE_CHARS),
])
@pytest.mark.parametrize("extra", [0, 1])
async def test_core_input_limit_never_reviews_silently_truncated_claims(part, limit, extra):
    text, sub = "x" * (limit + extra), client()
    kwargs = {"sub": sub}
    if part == "question":
        kwargs["question"] = text
    elif part == "answer":
        kwargs["value"] = final(text)
    elif part == "quote":
        kwargs["value"] = final(quote=text)
    else:
        kwargs["category_definitions"] = text
    result = await run_review(**kwargs)
    assert bool(sub.calls) is (extra == 0)
    if extra:
        assert result["fields"][0]["verdict"] is None


async def test_oversized_batch_and_prompt_are_bounded_and_report_omissions():
    questions = [(str(i), "q" * claim_review.MAX_QUESTION_CHARS)
                 for i in range(claim_review.MAX_FIELDS + 2)]
    submitted = {"value": {q: "42" for q, _ in questions},
                 "verification": {q: verified() for q, _ in questions}}
    sub = client()
    result = await review_final(submitted, questions, batch=True, client=sub,
                                model="sub", ledger=BudgetLedger())
    assert len(sub.calls) == 1
    assert len(sub.calls[0]["prompt"]) <= claim_review.MAX_PROMPT_CHARS
    assert len(result["fields"]) == len(questions)
    assert any(f["reason_code"] == "prompt_limit" for f in result["fields"])
    assert result["fields"][-1]["reason_code"] == "field_limit"


async def test_alternative_locations_and_context_truncation_are_explicit():
    submitted = final()
    quote = submitted["verification"]["quotes"][0]
    quote["matches"] *= 3
    quote["matches"][0]["source_context"]["text"] = "c" * 2100
    sub = client()
    result = await run_review(submitted, sub=sub)
    field = json.loads(sub.calls[0]["prompt"])["fields"][0]
    assert len(field["evidence"]) == 2 and field["evidence_truncated"]
    assert len(field["evidence"][0]["source_context"]) == claim_review.MAX_CONTEXT_CHARS
    assert result["fields"][0]["evidence_truncated"]


async def test_content_redaction_retains_outcomes_and_safe_identity():
    result = await run_review(sub=client(reply(reason="CLIENT_SENSITIVE_REASON")))
    raw = json.dumps(result)
    assert "CLIENT_SENSITIVE_REASON" in raw and "Revenue" not in raw
    for mode in ("redacted", "metadata"):
        protected = redact({"kind": "claim_review", **result}, mode)
        assert "CLIENT_SENSITIVE_REASON" not in json.dumps(protected)
        assert protected["fields"][0]["verdict"] == "supported"
        assert len(protected["fields"][0]["answer_sha256"]) == 64
    changed = await run_review(final("43"))
    assert result["fields"][0]["answer_sha256"] != changed["fields"][0]["answer_sha256"]


def stub_sandbox(monkeypatch, *finals):
    cells = [CellResult(ok=True, stdout="", final=f) for f in finals]
    sandbox = SimpleNamespace(start=AsyncMock(), close=AsyncMock(),
                              exec_cell=AsyncMock(side_effect=cells))
    monkeypatch.setattr("rnsr.harness.loop.SandboxedRepl", lambda **kwargs: sandbox)
    return sandbox


def runner(sub, **settings):
    return RootRunner(root_client=MockLLM(default="```python\nFINAL('draft')\n```"),
                      root_model="root", sub_client=sub, sub_model="sub",
                      settings=Settings(**settings))


@pytest.mark.parametrize("enabled", [False, True])
async def test_integration_shadow_cannot_block_or_upgrade_final(tmp_path, monkeypatch, enabled):
    submitted = final()
    stub_sandbox(monkeypatch, submitted)
    sub = MockLLM(default="COMPLETE").rule(r'"fields"', json.dumps({
        "reviews": [reply("contradicted")]}))
    result = await runner(sub, claim_review_enabled=enabled).run(
        "What was revenue?", EnvSpec(mode="docdb"), run_dir=tmp_path)
    assert result.answer == "42" and result.status == "final" and result.final == submitted
    events = [json.loads(line) for line in Path(result.trajectory_path).read_text().splitlines()]
    reviews = [e for e in events if e["kind"] == "claim_review"]
    assert len(reviews) == int(enabled)
    assert result.ledger["sub_calls"] == 1 + int(enabled)
    if enabled:
        assert reviews[0]["fields"][0]["verdict"] == "contradicted"
    assert result.evidence.to_dict() == from_final(submitted).to_dict()


async def test_integration_reviews_only_final_after_completeness_pushback(tmp_path, monkeypatch):
    stub_sandbox(monkeypatch, final("42"), final("42 million"))
    sub = MockLLM().script("MISSING: units", json.dumps({"reviews": [reply()]}))
    result = await runner(sub, claim_review_enabled=True).run(
        "Revenue and units?", EnvSpec(mode="docdb"), run_dir=tmp_path)
    assert result.answer == "42 million"
    assert len(sub.calls) == 2  # completeness plus one support review
    assert json.loads(sub.calls[-1]["prompt"])["fields"][0]["answer"] == "42 million"


async def test_integration_batch_keeps_question_and_verdict_mapping(tmp_path, monkeypatch):
    submitted = {"value": {"a": "42", "b": "Yes"},
                 "verification": {"a": verified(), "b": verified()}}
    stub_sandbox(monkeypatch, submitted)
    sub = client(reply("supported", "f0"), reply("contradicted", "f1"))
    result = await runner(sub, claim_review_enabled=True).run_batch(
        [("a", "What was revenue?"), ("b", "Is the licence exclusive?")],
        EnvSpec(mode="docdb", category_definitions="Exclusive excludes other licensees."),
        run_dir=tmp_path)
    assert result.answers == submitted["value"] and result.result.status == "final"
    assert len(sub.calls) == 1
    fields = json.loads(sub.calls[0]["prompt"])["fields"]
    assert [f["question"] for f in fields] == ["What was revenue?", "Is the licence exclusive?"]
    assert result.evidence["b"].to_dict() == from_final(submitted, qid="b").to_dict()


async def test_integration_internal_reviewer_fault_cannot_discard_answer(tmp_path, monkeypatch):
    stub_sandbox(monkeypatch, final())
    monkeypatch.setattr("rnsr.harness.loop.review_final",
                        AsyncMock(side_effect=ValueError("SENSITIVE_CONTEXT")))
    result = await runner(MockLLM(default="COMPLETE"), claim_review_enabled=True).run(
        "Revenue?", EnvSpec(mode="docdb"), run_dir=tmp_path)
    assert result.answer == "42" and result.status == "final"
    events = [json.loads(line) for line in Path(result.trajectory_path).read_text().splitlines()]
    event = next(e for e in events if e["kind"] == "claim_review")
    assert event["status"] == "error" and "SENSITIVE" not in json.dumps(event)


async def test_classic_mode_is_not_given_unverified_claim_review(tmp_path, monkeypatch):
    stub_sandbox(monkeypatch, {"value": "42"})
    sub = MockLLM(default="COMPLETE")
    result = await runner(sub, claim_review_enabled=True).run(
        "Revenue?", EnvSpec(mode="classic", context="Revenue 42"), run_dir=tmp_path)
    assert result.answer == "42" and len(sub.calls) == 1
    assert "claim_review" not in Path(result.trajectory_path).read_text()


async def test_real_parent_verified_context_reaches_advisory_without_source_writes(tmp_path):
    from rnsr.db.artifact import CorpusDB
    from rnsr.ingest.model import Element, ParsedDocument
    from rnsr.ingest.pipeline import ingest

    def parse(path):
        return ParsedDocument(doc_id="report", source_path=str(path), sha256="a" * 64,
                              n_pages=1, parser="test", elements=[
                                  Element("heading", "Fiscal 2020", 1, heading_level=1),
                                  Element("text", "Revenue was 42.", 1)], tables=[])

    corpus = tmp_path / "corpus.db"
    ingest([tmp_path / "input.pdf"], corpus, parse=parse)
    with CorpusDB(corpus) as db:
        manifest = db.manifest_dict()
    before = hashlib.sha256(corpus.read_bytes()).hexdigest()
    root = MockLLM(default="```python\nFINAL('42', quotes=['Revenue was 42.'])\n```")
    sub = MockLLM(default="COMPLETE").rule(r'"fields"', json.dumps({
        "reviews": [reply("insufficient", reason="Evidence is from fiscal 2020, not 2024.")]}))
    r = runner(sub, claim_review_enabled=True)
    r.root_client = root
    result = await r.run("What was revenue in 2024?", EnvSpec(
        mode="docdb", corpus_db=str(corpus), manifest=manifest), run_dir=tmp_path / "runs")
    assert result.status == "final" and result.answer == "42"  # shadow only
    call = next(c for c in sub.calls if '"fields"' in c["prompt"])
    evidence = json.loads(call["prompt"])["fields"][0]["evidence"][0]
    assert evidence["heading_paths"] == ["Fiscal 2020"] and evidence["page"] == 1
    assert hashlib.sha256(corpus.read_bytes()).hexdigest() == before


def test_claim_review_config_is_opt_in_and_env_parsed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert not Settings().claim_review_enabled
    monkeypatch.setenv("RNSR_CLAIM_REVIEW_ENABLED", "true")
    assert Settings.from_env().claim_review_enabled
    monkeypatch.setenv("RNSR_CLAIM_REVIEW_ENABLED", "false")
    assert not Settings.from_env().claim_review_enabled
    monkeypatch.setenv("RNSR_CLAIM_REVIEW_ENABLED", "invalid")
    with pytest.raises(ValueError):
        Settings.from_env()
