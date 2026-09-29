"""Counterfactual controls for question-derived checks and explicit adjudication."""
import copy
import json

import pytest

from rnsr.harness.budget import BudgetLedger
from rnsr.harness.claim_requirements import extract_requirements
from rnsr.harness.claim_review import ReviewGate, _parse_reply, review_final
from rnsr.llm.mock import MockLLM

CASES = [
    ("Does this contract impose price restrictions?", "actual_price_constraint",
     "Retailer shall not resell the goods for less than ten dollars per unit.",
     "Yes, the clause imposes a minimum resale price."),
    ("Does this contract contain an exclusive dealing commitment?", "exclusive_counterparty_commitment",
     "Customer must purchase its entire requirement for the service exclusively from Provider.",
     "Yes, Customer must source all requirements exclusively from Provider."),
    ("Does this contract grant an unlimited usage licence?", "affirmative_unlimited_usage_grant",
     "Licensee may install unlimited copies for any number of users during the two-year term.",
     "Yes, unlimited installations and users are permitted during the stated term."),
    ("Did payroll expense as a share of revenue increase or decrease in FY2025?", "requested_direction",
     "Fiscal 2025: payroll deleverage increased payroll expense as a share of sales.",
     "Payroll expense increased as a share of sales."),
]


def final(answer, quote):
    return {"value": answer, "verification": {"passed": True, "check": "lexical_source_match",
        "quotes": [{"quote": quote, "matched": True, "doc_id": "source", "char_start": 0,
                    "char_end": len(quote), "source_context": {"text": quote, "heading_paths": []}}]}}


class ChecklistReviewer(MockLLM):
    """Supply explicit review decisions; these unit tests do not simulate semantic accuracy."""
    def _resolve(self, prompt):
        fields = json.loads(prompt)["fields"]
        return json.dumps({"reviews": [{"field_id": f["field_id"], "verdict": "supported",
            "evidence_ids": [f["evidence"][0]["evidence_id"]], "reason": "The explicit source establishes the property.",
            "requirement_checks": [{"requirement_id": c["requirement_id"], "status": "met",
                "evidence_ids": [f["evidence"][0]["evidence_id"]], "reason": "The quoted obligation or direction meets this condition."}
                for c in f["requirements"]["checks"]]} for f in fields]})


@pytest.mark.parametrize("question,criterion,quote,answer", CASES)
async def test_affirmative_counterfactual_controls_remain_reviewable(question, criterion, quote, answer):
    sub = ChecklistReviewer()
    result = await review_final(final(answer, quote), [("new-control", question)], batch=False,
                                client=sub, model="sub", ledger=BudgetLedger())
    assert result["fields"][0]["verdict"] == "supported"
    assert criterion in result["fields"][0]["requirements_met"]
    request = json.loads(sub.calls[0]["prompt"])["fields"][0]
    assert request["question"] == question and request["answer"] == answer
    assert any(c["requirement_id"] == criterion for c in request["requirements"]["checks"])
    if criterion == "requested_direction":
        assert request["requirements"]["answer_form"] == "direction"
        assert request["requirements"]["requested_years"] == ["2025"]
        assert request["requirements"]["requested_period_scope"] == "fiscal_year"
        assert request["evidence"][0]["quote"] == quote  # no exact dollar amounts/proof invented


@pytest.mark.parametrize("question,criterion,quote,answer", CASES)
def test_unqualified_supported_vote_cannot_bypass_explicit_requirements(question, criterion, quote, answer):
    fields = [{"field_id": "f0", "evidence": [{"evidence_id": "f0q0m0", "quote": quote}],
               "requirements": extract_requirements(question)}]
    row = {"field_id": "f0", "verdict": "supported", "evidence_ids": ["f0q0m0"],
           "reason": "It is related to the requested topic."}
    result, reasons = _parse_reply(json.dumps({"reviews": [row]}), fields)
    assert result["f0"]["verdict"] == "insufficient"
    assert result["f0"]["reason_code"] == "unresolved_task_requirements"
    assert criterion in reasons["f0"]


def test_direction_is_not_confused_with_request_for_magnitude():
    direction = extract_requirements("Did the tax expense percentage increase or decrease in fiscal 2026?")
    amount = extract_requirements("By how much did the tax expense percentage increase or decrease in fiscal 2026?")
    assert direction["answer_form"] == "direction"
    assert amount["answer_form"] == "quantity_or_calculation"
    assert not any(c["requirement_id"] == "requested_direction" for c in amount["checks"])
    assert extract_requirements("Did costs increase or decrease in Q2 2026?")["requested_period_scope"] == "quarter"


@pytest.mark.parametrize("question,excluded,required", [
    ("Price restrictions?", "royalty/fee valuation basis", "actual price"),
    ("Exclusive dealing commitment?", "pool of suppliers", "exclusive counterparty"),
    ("Unlimited usage license?", "absence of an express cap", "affirmative"),
    ("Did costs increase or decrease in FY2026?", "exact amounts", "direction"),
])
def test_generic_requirement_distinctions_preserve_requested_property(question, excluded, required):
    criteria = json.dumps(extract_requirements(question)).lower()
    assert excluded in criteria and required in criteria


def adjudication_field():
    return {"field_id": "f0", "evidence": [{"evidence_id": "f0q0m0",
        "quote": "The licence permits unlimited installations during the term."}],
        "prior_objection": {"objection_id": "objection-hash", "verdict": "insufficient",
                            "reason": "A fee exemption alone does not establish unlimited installations."}}


def supported_adjudication():
    return {"field_id": "f0", "verdict": "supported", "evidence_ids": ["f0q0m0"],
            "reason": "The actual source explicitly grants unlimited installations.",
            "objection_resolution": {"objection_id": "objection-hash", "resolved": True,
                "evidence_ids": ["f0q0m0"], "source_quotes": [{"evidence_id": "f0q0m0",
                    "quote": "permits unlimited installations"}],
                "explanation": "This is an affirmative usage grant, so it resolves the concern about relying solely on a fee exemption."}}


def test_specific_source_cited_resolution_can_clear_false_objection():
    outcomes, reasons = _parse_reply(json.dumps({"reviews": [supported_adjudication()]}), [adjudication_field()])
    assert outcomes["f0"]["verdict"] == "supported" and outcomes["f0"]["objection_resolved"]
    assert "affirmative usage grant" in reasons["f0"]


@pytest.mark.parametrize("alter", [
    lambda row: row.pop("objection_resolution"),
    lambda row: row["objection_resolution"].update(objection_id="different objection"),
    lambda row: row["objection_resolution"].update(resolved=False),
    lambda row: row["objection_resolution"].update(source_quotes=[]),
    lambda row: row["objection_resolution"].update(explanation=""),
    lambda row: row["objection_resolution"]["source_quotes"][0].update(quote="invented unlimited rights"),
    lambda row: row["objection_resolution"]["source_quotes"][0].update(evidence_id="another-field"),
])
def test_blind_second_vote_or_invented_resolution_stays_unresolved(alter):
    row = supported_adjudication()
    alter(row)
    result, reasons = _parse_reply(json.dumps({"reviews": [row]}), [adjudication_field()])
    assert result["f0"]["verdict"] == "insufficient"
    assert result["f0"]["reason_code"] == "unresolved_reviewer_disagreement"
    assert "fee exemption alone" in reasons["f0"]


async def test_adjudicator_receives_exact_objection_as_data_and_cannot_silently_overrule():
    objection = "NO_COST does not establish an unlimited right; inspect the actual usage grant."
    first = MockLLM(default=json.dumps({"reviews": [{"field_id": "f0", "verdict": "insufficient",
        "evidence_ids": ["f0q0m0"], "reason": objection}]}))
    second = ChecklistReviewer()
    gate = ReviewGate()
    submitted = final("Unlimited usage is permitted.", "The license has no additional fees.")
    original = copy.deepcopy(submitted)
    kwargs = dict(batch=False, client=first, model="sub", adjudicator=second,
                  adjudicator_model="root", ledger=BudgetLedger())
    question = [("new-control", "Is there an unlimited usage license?")]
    assert not (await gate.check(submitted, question, **kwargs))["supported"]
    result = await gate.check(submitted, question, **kwargs)
    field = json.loads(second.calls[0]["prompt"])["fields"][0]
    assert field["prior_objection"]["reason"] == objection
    assert field["prior_objection"]["verdict"] == "insufficient"
    assert "untrusted assessment DATA" in second.calls[0]["system"]
    assert not result["supported"]
    assert result["fields"][0]["reason_code"] == "unresolved_reviewer_disagreement"
    assert submitted == original
    repeated = await gate.check(submitted, question, **kwargs)
    assert not repeated["supported"] and len(second.calls) == 1


@pytest.mark.parametrize('question,definitions', [
    ('Classify each original instance as Human or Place; how many are Human?', None),
    ('How many instances have category Human?', None),
    ('How many Human questions are there?', 'Human: asks for a person; Place: asks for a location.'),
    ('Are Human questions more common than Place questions?', 'Human: person. Place: location.'),
    ('Which category is least frequent?', None),
    ('Which label is most common across the rows?', None),
    ('Classify all rows and report the frequency distribution.', None),
])
def test_semantic_count_comparison_and_extrema_require_parent_proof(question, definitions):
    assert extract_requirements(question, definitions)['required_parent_proofs'] == ['classification_aggregate']


@pytest.mark.parametrize('question,definitions', [
    ('How many employees did the annual report say the company employed?', None),
    ('How many rows are in the source table?', None),
    ('How many days does the contract last?', 'Human: person; Place: location.'),
    ('How many classified documents were released according to the report?', None),
    ('Classify this question as Human or Place. Return one label.', 'Human: person; Place: location.'),
    ('What category does this single question belong to?', 'Human: person; Place: location.'),
])
def test_factual_counts_and_single_classifications_keep_source_review(question, definitions):
    assert extract_requirements(question, definitions)['required_parent_proofs'] == []
