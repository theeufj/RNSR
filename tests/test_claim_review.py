"""Evidence review gates acceptance without inventing source authority."""
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
    assert not result["advisory"] and result["scope"] == "provided_verified_evidence_only"
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


async def test_legacy_lexical_mode_is_explicitly_available(tmp_path, monkeypatch):
    submitted = final()
    stub_sandbox(monkeypatch, submitted)
    result = await runner(MockLLM(default="COMPLETE"), claim_review_enabled=False).run(
        "What was revenue?", EnvSpec(mode="docdb"), run_dir=tmp_path)
    assert result.answer == "42" and result.status == "final" and result.final == submitted
    assert result.claim_review is None
    assert "claim_review" not in Path(result.trajectory_path).read_text()


async def test_integration_rejects_wrong_period_then_accepts_supported_repair(tmp_path, monkeypatch):
    wrong = final("42", heading="Fiscal 2020")
    repaired = final("48", quote="Revenue was 48.", heading="Fiscal 2024")
    sandbox = stub_sandbox(monkeypatch, wrong, repaired)
    sub = MockLLM().script("COMPLETE", json.dumps({"reviews": [reply(
        "insufficient", reason="This evidence is from fiscal 2020, not requested 2024.")]}),
        json.dumps({"reviews": [reply()]}))
    r = runner(sub)
    result = await r.run("What was revenue in 2024?", EnvSpec(mode="docdb"), run_dir=tmp_path)
    assert result.answer == "48" and result.status == "final"
    assert result.claim_review["status"] == "supported"
    assert sandbox.exec_cell.await_count == 2
    assert "fiscal 2020" in r.root_client.calls[1]["prompt"]
    events = [json.loads(line) for line in Path(result.trajectory_path).read_text().splitlines()]
    assert [e["supported"] for e in events if e["kind"] == "claim_review_decision"] == [False, True]
    assert len([e for e in events if e["kind"] == "final"]) == 1


async def test_integration_reviews_only_final_after_completeness_pushback(tmp_path, monkeypatch):
    stub_sandbox(monkeypatch, final("42"), final("42 million"))
    sub = MockLLM().script("MISSING: units", json.dumps({"reviews": [reply()]}))
    result = await runner(sub, claim_review_enabled=True).run(
        "Revenue and units?", EnvSpec(mode="docdb"), run_dir=tmp_path)
    assert result.answer == "42 million"
    assert len(sub.calls) == 2  # completeness plus one support review
    assert json.loads(sub.calls[-1]["prompt"])["fields"][0]["answer"] == "42 million"


async def test_integration_batch_keeps_supported_fields_and_repairs_related_concept(tmp_path, monkeypatch):
    submitted = {"value": {"a": "42", "b": "Yes"},
                 "verification": {"a": verified(), "b": verified("A licence is granted.")}}
    repaired = copy.deepcopy(submitted)
    repaired["value"]["b"] = "No ownership transfer; only a licence is granted."
    stub_sandbox(monkeypatch, submitted, repaired)
    sub = MockLLM().script(json.dumps({"reviews": [reply("supported", "f0"), reply(
        "insufficient", "f1", reason="A licence does not establish ownership transfer.")]}),
        json.dumps({"reviews": [reply()]}))
    result = await runner(sub).run_batch(
        [("a", "What was revenue?"), ("b", "Is ownership transferred?")],
        EnvSpec(mode="docdb", category_definitions="Ownership transfer excludes a mere licence."),
        run_dir=tmp_path)
    assert result.answers == repaired["value"] and result.result.status == "final"
    assert len(sub.calls) == 2
    assert [f["question"] for f in json.loads(sub.calls[1]["prompt"])["fields"]] == [
        "Is ownership transferred?"]  # unchanged supported field uses its evidence-bound cache


async def test_integration_reviewer_fault_never_accepts_draft(tmp_path, monkeypatch):
    stub_sandbox(monkeypatch, final())
    monkeypatch.setattr(claim_review, "review_final", AsyncMock(side_effect=ValueError("fault")))
    result = await runner(MockLLM(default="COMPLETE")).run(
        "Revenue?", EnvSpec(mode="docdb"), run_dir=tmp_path)
    assert result.answer is None and result.status == "error"
    assert not any(json.loads(line)["kind"] == "final" for line in
                   Path(result.trajectory_path).read_text().splitlines())


async def test_classic_mode_is_not_given_unverified_claim_review(tmp_path, monkeypatch):
    stub_sandbox(monkeypatch, {"value": "42"})
    sub = MockLLM(default="COMPLETE")
    result = await runner(sub, claim_review_enabled=True).run(
        "Revenue?", EnvSpec(mode="classic", context="Revenue 42"), run_dir=tmp_path)
    assert result.answer == "42" and len(sub.calls) == 1
    assert "claim_review" not in Path(result.trajectory_path).read_text()


async def test_real_wrong_year_source_is_not_accepted_and_source_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr("rnsr.harness.loop.recover_variable", AsyncMock(return_value=None))
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
    r = runner(sub, claim_review_enabled=True, max_root_iters=2)
    r.root_client = root
    result = await r.run("What was revenue in 2024?", EnvSpec(
        mode="docdb", corpus_db=str(corpus), manifest=manifest), run_dir=tmp_path / "runs")
    assert result.status == "budget_exhausted" and result.answer is None
    call = next(c for c in sub.calls if '"fields"' in c["prompt"])
    evidence = json.loads(call["prompt"])["fields"][0]["evidence"][0]
    assert evidence["heading_paths"] == ["Fiscal 2020"] and evidence["page"] == 1
    assert hashlib.sha256(corpus.read_bytes()).hexdigest() == before


def test_claim_review_config_is_default_on_and_env_parsed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert Settings().claim_review_enabled
    monkeypatch.setenv("RNSR_CLAIM_REVIEW_ENABLED", "true")
    assert Settings.from_env().claim_review_enabled
    monkeypatch.setenv("RNSR_CLAIM_REVIEW_ENABLED", "false")
    assert not Settings.from_env().claim_review_enabled
    monkeypatch.setenv("RNSR_CLAIM_REVIEW_ENABLED", "invalid")
    with pytest.raises(ValueError):
        Settings.from_env()


async def gate_check(gate, submitted=None, *, sub=None, adjudicator=None,
                     question="What was revenue in 2024?", **kwargs):
    return await gate.check(submitted or final(), [("q", question)], batch=False,
                            client=sub or client(), model="sub",
                            adjudicator=adjudicator or client(), adjudicator_model="root",
                            ledger=BudgetLedger(), **kwargs)


async def test_confirmed_wrong_year_veto_is_cached_without_ending_uncapped_query():
    gate = claim_review.ReviewGate()
    sub = client(reply("insufficient", reason="Fiscal 2020 does not establish fiscal 2024."))
    adjudicator = client(reply("insufficient", reason="The requested period is missing."))
    first = await gate_check(gate, sub=sub, adjudicator=adjudicator)
    second = await gate_check(gate, sub=sub, adjudicator=adjudicator)
    for _ in range(20):
        repeated = await gate_check(gate, sub=sub, adjudicator=adjudicator)
        assert not repeated["supported"] and repeated["fields"][0]["cached"]
        assert not repeated["events"]
    assert not first["supported"] and not second["supported"]
    assert len(sub.calls) == len(adjudicator.calls) == 1
    assert "requested period" in claim_review.repair_instruction(repeated, [("q", "Revenue?")])
    # New source evidence is eligible even after arbitrarily many unchanged drafts.
    repaired = await gate_check(gate, final(heading="Fiscal 2024"), sub=client())
    assert repaired["supported"]


async def test_independent_adjudication_can_clear_false_veto_without_reference_answer():
    gate = claim_review.ReviewGate()
    sub = client(reply("contradicted", reason="INITIAL_REVIEW_REASON_SENTINEL"))

    class Adjudicator(MockLLM):
        def _resolve(self, prompt):
            field = json.loads(prompt)["fields"][0]
            row = reply()
            row["objection_resolution"] = {
                "objection_id": field["prior_objection"]["objection_id"], "resolved": True,
                "evidence_ids": ["f0q0m0"],
                "source_quotes": [{"evidence_id": "f0q0m0", "quote": "Revenue was 42."}],
                "explanation": "The direct source statement establishes the requested revenue value."}
            return json.dumps({"reviews": [row]})

    adjudicator = Adjudicator()
    submitted = final()
    submitted["gold"] = "EVALUATION_GOLD_SENTINEL"
    assert not (await gate_check(gate, submitted, sub=sub))["supported"]
    decision = await gate_check(gate, submitted, sub=sub, adjudicator=adjudicator)
    assert decision["supported"] and decision["fields"][0]["phase"] == "adjudication"
    assert len(adjudicator.calls) == 1 and adjudicator.calls[0]["model"] == "root"
    assert "INITIAL_REVIEW_REASON_SENTINEL" in adjudicator.calls[0]["prompt"]
    assert "EVALUATION_GOLD_SENTINEL" not in adjudicator.calls[0]["prompt"]


async def test_review_transport_failure_is_not_cached_as_support_or_contradiction():
    gate = claim_review.ReviewGate()
    failed = MockLLM(fail_times=1)
    result = await gate_check(gate, sub=failed)
    assert not result["supported"] and result["fields"][0]["verdict"] is None
    assert result["fields"][0]["status"] == "error"
    recovered = await gate_check(gate, sub=client())
    assert recovered["supported"] and recovered["fields"][0]["phase"] == "review"


async def test_terminal_reviewer_provider_error_is_loud_without_provider_body():
    from rnsr.errors import PermanentProviderError

    class ProviderError(Exception):
        status_code = 400

    async def denied(*args, **kwargs):
        raise ProviderError("PRIVATE_PROVIDER_REQUEST")

    with pytest.raises(PermanentProviderError, match="ProviderError") as exc:
        await gate_check(claim_review.ReviewGate(), sub=SimpleNamespace(provider="mock", complete=denied))
    assert "PRIVATE" not in str(exc.value)


async def test_governor_queue_wait_is_not_the_active_review_timeout(monkeypatch):
    from rnsr.llm.base import LLMResponse, Usage

    monkeypatch.setattr(claim_review, "MAX_REVIEW_S", 0.01)
    monkeypatch.setattr(claim_review, "MIN_REMAINING_S", 0)
    seen = []

    async def queued(*args, **kwargs):
        seen.append(kwargs["timeout_s"])
        await asyncio.sleep(0.03)  # provider admission queue precedes active timeout
        return LLMResponse(json.dumps({"reviews": [reply()]}), "sub", Usage(1, 1, 0))

    decision = await gate_check(claim_review.ReviewGate(), sub=SimpleNamespace(
        provider="mock", complete_with_timeout=queued))
    assert decision["supported"] and 0 < seen[0] <= 0.01


async def test_negative_source_claims_are_reviewed_but_empty_quotes_do_not_prove_absence():
    gate = claim_review.ReviewGate()
    sub = client()
    explicit = final("No", quote="The parties shall not transfer ownership of the licensed software.")
    supported = await gate_check(gate, explicit, question="Does ownership transfer?", sub=sub)
    assert supported["supported"]
    absent = await gate_check(gate, {"value": "NOT_FOUND", "verification": {
        "passed": True, "quotes": [], "zero_quotes": True}}, question="Does ownership transfer?", sub=sub)
    assert not absent["supported"] and absent["fields"][0]["verdict"] == "insufficient"
    assert len(sub.calls) == 1


async def test_parent_arithmetic_graph_reaches_review_without_arbitrary_metadata():
    submitted = final("-4", quote="Current assets 10, current liabilities 14.")
    report = submitted["verification"]
    report.update(check="source_bound_calculations", answer="-4", records=[
        {"id": "src_a", "kind": "source", "value": "10", "unit": "USD millions", "period": "2024",
         "source": {"doc_id": "report", "table_title": "Balance sheet", "source_path": "/private/secret"}},
        {"id": "src_b", "kind": "source", "value": "14", "unit": "USD millions", "period": "2024"},
        {"id": "calc_x", "kind": "calculation", "value": "-4", "operation": "subtract",
         "operand_ids": ["src_a", "src_b"], "financial_metric": {
             "metric": "working_capital", "convention": "total", "input_roles": {
                 "current_assets": "src_a", "current_liabilities": "src_b"}},
         "untrusted": "GOLD_SENTINEL"}], calculation_answers={"result": {
             "value": "-4", "rendered_value": "-4", "calculation_id": "calc_x"}})
    report["quotes"] *= 4  # trusted graph may legitimately cite more than three source values
    sub = client(reply(refs=["f0p0"]))
    result = await run_review(submitted, question="Calculate working capital in 2024.", sub=sub)
    assert result["fields"][0]["verdict"] == "supported"
    data = json.loads(sub.calls[0]["prompt"])["fields"][0]
    proof = next(e for e in data["evidence"] if e["evidence_id"] == "f0p0")
    assert len(proof["records"]) == 3
    assert proof["records"][2]["financial_metric"]["convention"] == "total"
    assert "GOLD_SENTINEL" not in sub.calls[0]["prompt"] and "/private" not in sub.calls[0]["prompt"]
    assert "computed numeric claims require" in sub.calls[0]["system"]
    assert "Directional/narrative questions" in sub.calls[0]["system"]


async def test_classification_count_proof_retains_vocabulary_scope_and_rubric():
    submitted = {"value": "3", "verification": {"passed": True, "check": "classification_aggregate",
        "answer": "3", "quotes": [], "classification": {
            "counts": {"human": 3, "animal": 2}, "total": 5, "allowed_labels": ["human", "animal"],
            "annotation_id": "ann1", "annotation_version": "v2", "selection_sha256": "a" * 64,
            "table": "instances", "column": "classification", "where": "id > 0",
            "instruction": "Classify the referent in each instance.", "operation": "count",
            "labels": ["human"], "gold": "GOLD_SENTINEL"}}}
    sub = client(reply(refs=["f0p0"]))
    result = await run_review(submitted, question="How many instances are human?", sub=sub)
    assert result["fields"][0]["verdict"] == "supported"
    classification = json.loads(sub.calls[0]["prompt"])["fields"][0]["evidence"][0]["classification"]
    assert classification["instruction"] == "Classify the referent in each instance."
    assert classification["allowed_labels"] == ["human", "animal"]
    assert classification["selection_sha256"] == "a" * 64
    assert "GOLD_SENTINEL" not in sub.calls[0]["prompt"]
    assert "Classified counts require a parent" in sub.calls[0]["system"]


async def test_large_batch_reviews_every_field_instead_of_silently_omitting_them():
    from rnsr.llm.base import LLMResponse, Usage

    calls = []

    async def respond(prompt, **kwargs):
        fields = json.loads(prompt)["fields"]
        calls.append(fields)
        rows = [reply(fid=f["field_id"], refs=[f["evidence"][0]["evidence_id"]]) for f in fields]
        return LLMResponse(json.dumps({"reviews": rows}), "sub", Usage(1, 1, 0))

    questions = [(str(i), "Revenue?") for i in range(27)]
    submitted = {"value": {q: "42" for q, _ in questions},
                 "verification": {q: verified() for q, _ in questions}}
    sub = SimpleNamespace(provider="mock", complete=respond)
    decision = await claim_review.ReviewGate().check(
        submitted, questions, batch=True, client=sub, model="sub",
        adjudicator=sub, adjudicator_model="root", ledger=BudgetLedger())
    assert decision["supported"] and len(decision["fields"]) == 27
    assert len(calls) == 3 and max(map(len, calls)) == claim_review.MAX_FIELDS
    assert [f["field_id"] for f in decision["fields"]] == [f"f{i}" for i in range(27)]


async def test_trusted_model_identities_are_passed_to_sandbox(tmp_path, monkeypatch):
    sandbox = stub_sandbox(monkeypatch, final())
    sub = MockLLM().script("COMPLETE", json.dumps({"reviews": [reply()]}))
    r = runner(sub)
    await r.run("Revenue?", EnvSpec(mode="docdb"), run_dir=tmp_path)
    assert sandbox.start.call_args.kwargs["init_extra"]["model_identities"] == {
        "root": "mock:root", "sub": "mock:sub"}


@pytest.mark.parametrize("has_contract", [False, True])
async def test_nonpositive_result_clipping_requires_satisfied_caller_contract(has_contract):
    report = verified()
    report.update(check="source_bound_calculation", answer="0", calculation_id="calc0", records=[{
        "id": "calc0", "kind": "calculation", "value": "0",
        "operation": "caller_nonpositive_numerator_zero", "operand_ids": ["calc_negative"]}])
    if has_contract:
        report.update(metric_contract_satisfied=True, metric_contract={
            "authority": "caller_declared", "formula": "ebitdar_to_ebit_coverage",
            "nonpositive_numerator": "zero", "basis": "Caller requests a zero floor for coverage",
            "period": "2024", "unit": "millions", "sha256": "c" * 64})
    sub = client(reply(refs=["f0p0"]))
    result = await run_review({"value": "0", "verification": report}, sub=sub)
    if has_contract:
        assert result["fields"][0]["verdict"] == "supported"
        proof = json.loads(sub.calls[0]["prompt"])["fields"][0]["evidence"][-1]
        assert proof["metric_contract"]["nonpositive_numerator"] == "zero"
        assert proof["metric_contract_satisfied"] is True
    else:
        assert result["fields"][0]["verdict"] is None and not sub.calls
        assert result["fields"][0]["reason_code"] == "missing_caller_metric_contract"


async def test_long_verified_quote_is_segmented_without_omitting_source_text():
    text = "Relevant source text. " * 160
    sub = client(reply(refs=["f0q0m0s0"]))
    result = await run_review(final(quote=text), sub=sub)
    assert result["fields"][0]["verdict"] == "supported"
    evidence = json.loads(sub.calls[0]["prompt"])["fields"][0]["evidence"]
    assert len(evidence) == 2
    assert "".join(e["quote"] for e in evidence) == text
    assert all(e["quote_segment_count"] == 2 and len(e["quote"]) <= claim_review.MAX_QUOTE_CHARS
               for e in evidence)
    assert evidence[0]["char_start"] == evidence[1]["char_start"]  # whole verified span, not invented offsets


async def test_scoped_not_found_can_use_complete_quoted_source_section():
    quote = "Revenue table (all reporting periods)\nYear | Revenue\nFY2023 | 42\nEnd of table."
    submitted = final("NOT_FOUND", quote=quote, heading="Revenue table")
    match = submitted["verification"]["quotes"][0]["matches"][0]
    match["source_context"]["sections"] = [{"heading_path": "Revenue table",
        "char_start": match["char_start"], "char_end": match["char_end"]}]
    sub = client(reply(reason="The complete requested table contains FY2023 only, not FY2021."))
    result = await gate_check(claim_review.ReviewGate(), submitted, sub=sub,
                             question="What is FY2021 revenue in the Revenue table? Return NOT_FOUND if that table lacks it.")
    assert result["supported"]
    source = json.loads(sub.calls[0]["prompt"])["fields"][0]["evidence"][0]
    assert source["fully_quoted_sections"] == match["source_context"]["sections"]
    assert "task-scoped NOT_FOUND can be supported" in sub.calls[0]["system"]


async def test_directional_financial_prose_does_not_require_invented_precise_values():
    submitted = final("Payroll costs increased and reduced operating margin.",
        quote="Operating margin decreased primarily due to deleverage in store payroll and benefits.",
        heading="Operating results 2024")
    sub = client()
    result = await gate_check(claim_review.ReviewGate(), submitted, sub=sub,
        question="How did payroll costs affect operating margin in 2024?")
    assert result["supported"]
    sources = json.loads(sub.calls[0]["prompt"])["fields"][0]["evidence"]
    assert all(e.get("kind") not in {"source_bound_calculation", "source_bound_calculations"} for e in sources)


@pytest.mark.parametrize("trigger", ["budget", "root_timeout"])
async def test_recovery_cannot_publish_unreviewed_docdb_candidate(tmp_path, monkeypatch, trigger):
    sandbox = stub_sandbox(monkeypatch)
    candidate = final("UNSUPPORTED_CANDIDATE")
    monkeypatch.setattr("rnsr.harness.loop.recover_variable", AsyncMock(return_value=candidate))
    r = runner(MockLLM(default="COMPLETE"), max_wall_s=0.000001 if trigger == "budget" else 0)
    if trigger == "root_timeout":
        r._root_complete = AsyncMock(return_value=None)
    result = await r.run("Revenue?", EnvSpec(mode="docdb"), run_dir=tmp_path)
    assert result.status == "budget_exhausted" and result.answer is None
    assert result.unverified_answer == "UNSUPPORTED_CANDIDATE"
    assert result.claim_review["status"] == "unverified"
    assert result.evidence.tier == "low"
    assert sandbox.exec_cell.await_count == 0
    assert not any(json.loads(line)["kind"] == "final" for line in
                   Path(result.trajectory_path).read_text().splitlines())


def test_reissued_calculation_handles_do_not_trigger_identical_evidence_reviews():
    def draft(suffix, value="10"):
        report = verified()
        report.update(check="source_bound_calculation", answer="2", calculation_id="calc" + suffix,
            records=[{"id": "calc" + suffix, "kind": "calculation", "operation": "divide",
                      "value": "2", "operand_ids": ["a" + suffix, "b" + suffix]},
                     {"id": "a" + suffix, "kind": "source", "value": value},
                     {"id": "b" + suffix, "kind": "source", "value": "5"}])
        return {"value": "2", "verification": report}
    def key(value):
        return claim_review.candidate_key(value, ("q", "Ratio?"), batch=False)
    assert key(draft("old")) == key(draft("fresh"))
    assert key(draft("old")) != key(draft("fresh", "11"))


def classification_draft(annotation_id=1, version="first", **changes):
    classification = {
        "certified": True, "counts": {"human": 3, "animal": 2}, "total": 5,
        "allowed_labels": ["human", "animal"], "annotation_id": annotation_id,
        "annotation_version": version, "selection_sha256": "a" * 64,
        "labels_sha256": "b" * 64, "model_identity": "anthropic:claude-sonnet",
        "table": "instances", "column": "classification", "where": "id > 0",
        "instruction": "Classify each referent.", "operation": "count", "labels": ["human"],
    }
    classification.update(changes)
    return {"value": "3", "verification": {"passed": True, "check": "classification_aggregate",
        "answer": "3", "quotes": [], "classification": classification}}


async def test_identical_labels_with_new_publication_id_reuse_rejected_review():
    gate = claim_review.ReviewGate()
    sub = client(reply("insufficient", refs=["f0p0"], reason="The rubric omits requested categories."))
    adjudicator = client(reply("insufficient", refs=["f0p0"], reason="The requested vocabulary is incomplete."))
    original = classification_draft()
    before = copy.deepcopy(original)
    first = await gate_check(gate, original, sub=sub, adjudicator=adjudicator)
    second = await gate_check(gate, classification_draft(2, "second"), sub=sub, adjudicator=adjudicator)
    third = await gate_check(gate, classification_draft(3, "third"), sub=sub, adjudicator=adjudicator)
    assert not first["supported"] and not second["supported"] and not third["supported"]
    assert len(sub.calls) == len(adjudicator.calls) == 1
    assert third["fields"][0]["cached"] and third["events"] == []
    assert original == before  # parent freshness identity is never removed from the actual proof
    proof = json.loads(sub.calls[0]["prompt"])["fields"][0]["evidence"][0]["classification"]
    assert proof["annotation_id"] == 1 and proof["annotation_version"] == "first"
    assert proof["labels_sha256"] == "b" * 64 and proof["model_identity"] == "anthropic:claude-sonnet"

    # A human/animal swap keeps aggregate counts identical but changes the exact
    # row-label assignment; it must receive a fresh primary review.
    changed = await gate_check(gate, classification_draft(4, "fourth", labels_sha256="c" * 64),
                               sub=sub, adjudicator=adjudicator)
    assert not changed["fields"][0]["cached"] and changed["fields"][0]["phase"] == "review"
    assert len(sub.calls) == 2 and len(adjudicator.calls) == 1


@pytest.mark.parametrize("change", [
    {"selection_sha256": "c" * 64}, {"model_identity": "anthropic:claude-haiku"},
    {"where": "id >= 0"}, {"instruction": "A different classification rubric."},
    {"allowed_labels": ["human", "animal", "place"]},
])
def test_classification_review_identity_binds_scope_model_and_rubric(change):
    def key(value):
        return claim_review.candidate_key(value, ("q", "How many humans?"), batch=False)
    assert key(classification_draft()) != key(classification_draft(2, "new", **change))


@pytest.mark.parametrize("missing", ["labels_sha256", "selection_sha256", "model_identity"])
def test_legacy_classification_proofs_without_stable_fingerprints_keep_freshness_ids(missing):
    first, second = classification_draft(), classification_draft(2, "new")
    first["verification"]["classification"].pop(missing)
    second["verification"]["classification"].pop(missing)
    key1 = claim_review.candidate_key(first, ("q", "Count?"), batch=False)
    key2 = claim_review.candidate_key(second, ("q", "Count?"), batch=False)
    assert key1 != key2


@pytest.mark.parametrize("input_part", ["question", "definition", "batch_question"])
async def test_immutable_oversized_review_inputs_fail_before_sandbox_or_provider(tmp_path, monkeypatch, input_part):
    sandbox = stub_sandbox(monkeypatch)
    root, sub = MockLLM(), MockLLM()
    r = runner(sub)
    r.root_client = root
    env = EnvSpec(mode="docdb")
    question, batch_questions = "Revenue?", None
    if input_part == "question":
        question = "q" * (claim_review.MAX_QUESTION_CHARS + 1)
    elif input_part == "definition":
        env.category_definitions = "d" * (claim_review.MAX_DEFINITION_CHARS + 1)
    else:
        batch_questions = [("a", "Short task"), ("b", "q" * (claim_review.MAX_QUESTION_CHARS + 1))]
    with pytest.raises(ValueError, match="exceed.*supported"):
        await r.run(question, env, batch_questions=batch_questions, run_dir=tmp_path)
    assert root.calls == sub.calls == []
    assert sandbox.start.await_count == 0 and sandbox.exec_cell.await_count == 0
    assert list(tmp_path.iterdir()) == []  # fail before creating a trajectory too


def test_review_preflight_applies_per_question_not_combined_batch_length():
    claim_review.validate_review_task([("a", "x" * 3000), ("b", "y" * 3000)], "Allowed categories")


_CLASSIFICATION_QUESTION = 'Classify each original instance as Human or Place. How many instances are Human?'


@pytest.mark.parametrize('answer', ['3', '0', 'No', 'Answer: 3', 'NOT_FOUND: the count is 3'])
@pytest.mark.parametrize('question', [_CLASSIFICATION_QUESTION,
    'In the above data, how many data points should be classified as label Person?'])
async def test_lexical_quotes_cannot_replace_recorded_classification_aggregate(answer, question):
    sub = client()
    submitted = final(answer, quote='Complete dataset: Who is Mira? Who is Owen? Who is Tara?')
    result = await run_review(submitted, question=question, sub=sub)
    assert not sub.calls and not result['attempted']
    assert result['fields'][0]['verdict'] == 'insufficient'
    assert result['fields'][0]['reason_code'] == 'missing_parent_classification_proof'
    assert 'semantic_classify' in result['reply']['f0']
    assert 'FINAL_CLASSIFICATION' in result['reply']['f0']
    assert 'expected_count' in result['reply']['f0']


async def test_reviewer_and_adjudicator_cannot_override_missing_parent_proof_then_repair_can_proceed():
    gate = claim_review.ReviewGate()
    sub, adjudicator = client(reply(refs=['f0p0'])), client()
    submitted = final('3', quote='The complete source consists of three person questions.')
    for _ in range(3):
        rejected = await gate_check(gate, submitted, question=_CLASSIFICATION_QUESTION,
                                    sub=sub, adjudicator=adjudicator)
        assert not rejected['supported']
        assert rejected['fields'][0]['reason_code'] == 'missing_parent_classification_proof'
    assert not sub.calls and not adjudicator.calls
    repaired = await gate_check(gate, classification_draft(), question=_CLASSIFICATION_QUESTION,
                                sub=sub, adjudicator=adjudicator)
    assert repaired['supported'] and len(sub.calls) == 1 and not adjudicator.calls
    reviewed_proof = json.loads(sub.calls[0]['prompt'])['fields'][0]['evidence'][0]['classification']
    assert reviewed_proof['certified'] is True
    assert reviewed_proof['selection_sha256'] == 'a' * 64


@pytest.mark.parametrize('change', [
    lambda report: report.update(passed=False),
    lambda report: report.update(check='lexical_source_match'),
    lambda report: report['classification'].update(certified=False),
    lambda report: report['classification'].pop('certified'),
])
async def test_uncertified_classification_proof_cannot_reach_model(change):
    submitted, sub = classification_draft(), client(reply(refs=['f0p0']))
    change(submitted['verification'])
    result = await run_review(submitted, question=_CLASSIFICATION_QUESTION, sub=sub)
    assert not sub.calls
    assert result['fields'][0]['reason_code'] == 'missing_parent_classification_proof'


async def test_mixed_batch_reviews_factual_count_and_preserves_classification_repair():
    sub = client(reply(fid='f1'))
    submitted = {'value': {'aggregate': '3', 'factual': '42'},
                 'verification': {'aggregate': verified(), 'factual': verified('The company has 42 employees.')}}
    result = await review_final(submitted, [('aggregate', _CLASSIFICATION_QUESTION),
                                ('factual', 'How many employees does the company have?')], batch=True,
                                client=sub, model='sub', ledger=BudgetLedger())
    assert [f['verdict'] for f in result['fields']] == ['insufficient', 'supported']
    assert [f['field_id'] for f in json.loads(sub.calls[0]['prompt'])['fields']] == ['f1']
    assert 'FINAL_CLASSIFICATION' in result['reply']['f0'] and result['reply']['f1']


@pytest.mark.parametrize('answer', ['NOT_FOUND', 'unknown', 'Not found in matter corpus'])
async def test_absent_classification_dataset_can_be_reviewed_for_scoped_coverage(answer):
    quote = 'Complete attachment index: there are no question records in this supplied collection.'
    sub = client(reply(reason='The complete requested attachment index establishes the dataset is absent.'))
    result = await run_review(final(answer, quote=quote), question=_CLASSIFICATION_QUESTION, sub=sub)
    assert len(sub.calls) == 1 and result['fields'][0]['verdict'] == 'supported'
    assert json.loads(sub.calls[0]['prompt'])['fields'][0]['answer'] == answer


async def test_quote_free_classification_absence_cannot_gain_support():
    sub = client()
    result = await run_review({'value': 'NOT_FOUND', 'verification': {'passed': False, 'quotes': []}},
                              question=_CLASSIFICATION_QUESTION, sub=sub)
    assert not sub.calls and result['fields'][0]['reason_code'] == 'no_verified_evidence'


async def test_classification_absence_still_respects_reviewer_source_coverage_rejection():
    sub = client(reply('insufficient', reason='One omitted page does not establish absence of the dataset.'))
    result = await run_review(final('NOT_FOUND', quote='This page lists only administrative notices.'),
                              question=_CLASSIFICATION_QUESTION, sub=sub)
    assert len(sub.calls) == 1 and result['fields'][0]['verdict'] == 'insufficient'
