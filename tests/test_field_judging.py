"""Field judges see question contracts and retain unsuccessful decisions."""

import csv
import json

import pytest

from rnsr.eval.regression import (
    FieldContext,
    judge_disagreements,
    load_field_contexts,
    merge_field_contexts,
    score_run,
)
from rnsr.llm.base import LLMResponse


class Judge:
    def __init__(self, text):
        self.text = text
        self.calls = []

    async def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if isinstance(self.text, Exception):
            raise self.text
        return LLMResponse(text=self.text, model="test-judge")


@pytest.mark.asyncio
@pytest.mark.parametrize("decision,expected,status", [
    ("equivalent", True, "accepted"), ("different", False, "rejected"),
    ("unavailable", False, "unavailable"),
])
async def test_question_context_and_every_explained_verdict_are_saved(tmp_path, decision,
                                                                  expected, status):
    report = score_run({"field123": ["Alex Smith"]}, {"field123": "Applicant 1"},
                       contexts={"field123": FieldContext(
                           question="With whom does the child live?",
                           field_context="Applicant 1 is Alex Smith.",
                           answer_type="role", output_requirements="An established role is allowed.")})
    judge = Judge(json.dumps({"verdict": decision, "reason": "Specific contextual reason."}))
    await judge_disagreements(report, judge, "test-judge")
    result = report.results[0]
    assert result.agrees is expected and result.judge_status == status
    prompt = judge.calls[0][0]
    assert "With whom does the child live?" in prompt
    assert "Applicant 1 is Alex Smith." in prompt
    assert "An established role is allowed." in prompt
    report.write(tmp_path)
    saved = json.loads((tmp_path / "judge_decisions.jsonl").read_text())
    assert saved["reason"] == "Specific contextual reason."
    assert saved["attempts"][0]["response"] == judge.text
    with (tmp_path / "comparison.csv").open() as f:
        row = next(csv.DictReader(f))
    assert row["judge_status"] == status
    assert row["question"] == result.context.question


@pytest.mark.asyncio
@pytest.mark.parametrize("reply,status", [
    ("YES", "unavailable"), ('{"verdict":"different"}', "unavailable"),
    (TimeoutError("private provider details"), "error"),
])
async def test_unusable_judge_retains_deterministic_failure_and_reason(reply, status):
    report = score_run({"x": ["spouse"]}, {"x": "married partner"},
                       contexts={"x": "What is the person's relationship?"})
    await judge_disagreements(report, Judge(reply), "judge")
    result = report.results[0]
    assert not result.agrees
    assert result.judge_status == status and result.judge_verdict is None
    assert result.judge_reason and len(result.judge_history) == 1
    if status == "error":
        assert result.judge_history[0]["error_type"] == "TimeoutError"
        assert "private provider details" not in result.judge_reason


@pytest.mark.asyncio
async def test_missing_question_does_not_get_replaced_by_an_id():
    report = score_run({"x": ["spouse"]}, {"x": "married partner"})
    judge = Judge(AssertionError("must not call provider without question"))
    await judge_disagreements(report, judge, "judge")
    assert not judge.calls
    assert report.results[0].judge_status == "unavailable"
    assert "Missing question context" in report.results[0].judge_reason


@pytest.mark.asyncio
async def test_missing_answer_is_recorded_without_provider_call():
    report = score_run({"x": ["spouse"]}, {}, contexts={"x": "Relationship?"})
    judge = Judge(AssertionError("must not call provider without candidate"))
    await judge_disagreements(report, judge, "judge")
    assert not judge.calls
    assert report.results[0].judge_reason == "No candidate answer to judge."


@pytest.mark.asyncio
async def test_repeated_judge_attempts_keep_previous_unavailable_reason():
    report = score_run({"x": ["spouse"]}, {"x": "married partner"},
                       contexts={"x": "Relationship?"})
    await judge_disagreements(report, Judge("invalid"), "judge")
    await judge_disagreements(report, Judge(json.dumps(
        {"verdict": "different", "reason": "Not equivalent."})), "judge")
    assert [d["status"] for d in report.results[0].judge_history] == ["unavailable", "rejected"]


@pytest.mark.parametrize("answer_type,gold,answer,agrees", [
    ("boolean", "Yes — three children", "Yes", True),
    ("boolean", "Yes — three children", "No", False),
    ("boolean", "No", "Yes; no further details", False),
    ("boolean", "", "No", True),
    ("boolean_with_details", "Yes — three children", "Yes", False),
    ("person_name", "Alex Smith", "Alex", False),
    ("person_name", "Alex Smith", "Not Alex Smith", False),
    ("person_name", "Alex Smith", "Applicant 1", False),
])
def test_explicit_contract_does_not_use_substring_equivalence(answer_type, gold, answer, agrees):
    report = score_run({"x": [gold]}, {"x": answer}, contexts={"x": FieldContext(
        question="Field question", answer_type=answer_type)})
    assert report.results[0].agrees is agrees


def test_actual_question_prevents_unsafe_legacy_name_containment():
    report = score_run({"x": ["Alex Smith"]}, {"x": "Not Alex Smith"},
                       contexts={"x": "Who is the applicant?"})
    assert not report.results[0].agrees


@pytest.mark.asyncio
@pytest.mark.parametrize("answer_type,gold,answer", [
    ("boolean", "Yes — three children", "No"),
    ("boolean_with_details", "Yes — three children", "Yes"),
    ("person_name", "Alex Smith", "Applicant 1"),
])
async def test_clear_contract_failures_cannot_be_overridden_by_judge(answer_type, gold, answer):
    report = score_run({"x": [gold]}, {"x": answer}, contexts={"x": FieldContext(
        question="Field question", answer_type=answer_type)})
    judge = Judge(AssertionError("do not spend a judge call on an explicit contract failure"))
    await judge_disagreements(report, judge, "judge")
    assert not judge.calls
    assert report.results[0].judge_status == "rejected"
    assert report.results[0].judge_history[0]["source"] == "output_requirement"


def test_context_loader_excludes_gold_and_trap_notes_from_requirements(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps({"context": "Applicant is Alex Smith", "fields": [{
        "id": "x", "title": "Is there a child?", "field_type": "checkbox",
        "golden": ["Yes — three children"], "notes": "TRAP: actually there are three",
    }]}))
    context = load_field_contexts(path)["x"]
    assert context.question == "Is there a child?" and context.answer_type == "boolean"
    assert "TRAP" not in str(context) and "three" not in str(context)
    questions = tmp_path / "questions.csv"
    questions.write_text('qid,ground_truth_question\nx,Is there a child of the relationship?\n')
    merged = merge_field_contexts({"x": context}, load_field_contexts(questions))["x"]
    assert merged.answer_type == "boolean"
    assert merged.question == "Is there a child of the relationship?"
    assert "Applicant is Alex Smith" in merged.field_context


def test_group_map_supplies_question_and_per_field_requirement(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"items": [{"item_id": "g1", "question": "What is the role?",
        "members": [{"id": "parent", "option_label": "Parent", "field_type": "checkbox"},
                    {"id": "other", "option_label": "Other, specify", "answer_type": "value"}]}]}))
    contexts = load_field_contexts(path)
    assert set(contexts) == {"parent", "other"}
    assert contexts["parent"].answer_type == "boolean"
    assert contexts["other"].answer_type == "value"
    assert "Field option: Parent" in contexts["parent"].field_context


def test_unknown_contract_type_is_refused():
    with pytest.raises(ValueError, match="answer_type"):
        FieldContext(answer_type="accept-anything")


def test_cli_questions_reach_field_judge_and_saved_audit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from rnsr.cli import app
    from rnsr.llm.router import Router

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"fields": [{"id": "person", "title": "Who?",
                                              "golden": ["Alex Smith"]}]}))
    answers = tmp_path / "answers.csv"
    answers.write_text("field_id,answer\nperson,Applicant 1\n")
    questions = tmp_path / "questions.csv"
    questions.write_text("qid,ground_truth_question\nperson,With whom does the child live? Applicant 1 is Alex Smith.\n")
    judge = Judge('{"verdict":"equivalent","reason":"The supplied role identifies the same person."}')
    monkeypatch.setattr(Router, "__init__", lambda *args: None)
    monkeypatch.setattr(Router, "resolve", lambda *args: SimpleNamespace(client=judge, model="judge"))
    result = CliRunner().invoke(app, ["regress", "--answers", str(answers),
                                     "--golden", str(golden), "--questions", str(questions),
                                     "--judge", "--out", str(tmp_path / "scored")])
    assert result.exit_code == 0, result.output
    assert "With whom does the child live?" in judge.calls[0][0]
    saved = json.loads((tmp_path / "scored/judge_decisions.jsonl").read_text())
    assert saved["status"] == "accepted" and "Applicant 1 is Alex Smith" in saved["question"]


@pytest.mark.asyncio
async def test_sdk_preserves_golden_question_and_explicit_caller_contract(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rnsr.llm.router import Router
    from rnsr.sdk import score_answers

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"items": [{"qid": "x", "question": "Who is the applicant?",
                                             "golden": ["Alex Smith"]}]}))
    judge = Judge('{"verdict":"different","reason":"The names identify different people."}')
    monkeypatch.setattr(Router, "__init__", lambda *args: None)
    monkeypatch.setattr(Router, "resolve", lambda *args: SimpleNamespace(client=judge, model="judge"))
    report = await score_answers(golden, {"x": "Avery Smith"}, judge=True,
                                 contexts={"x": FieldContext(answer_type="person_name")})
    assert report.results[0].context.question == "Who is the applicant?"
    assert report.results[0].context.answer_type == "person_name"
    assert report.results[0].judge_status == "rejected" and not report.results[0].agrees


@pytest.mark.asyncio
async def test_unconfigured_sdk_judge_still_returns_auditable_report():
    from rnsr.sdk import score_answers

    # The offline fixture removes credentials. Setup fails before any API call.
    report = await score_answers({"x": ["spouse"]}, {"x": "partner"},
                                 judge=True, contexts={"x": "Relationship?"})
    result = report.results[0]
    assert not result.agrees and result.judge_status == "error"
    assert result.judge_history[0]["error_type"] == "RuntimeError"
    assert "setup failed" in result.judge_reason
