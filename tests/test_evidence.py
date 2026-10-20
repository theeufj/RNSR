"""Trust-tier rules and evidence reconstruction."""

import hashlib
import json

from rnsr.harness.evidence import AnswerEvidence, assign_tier, from_final, from_records


def test_recovered_and_third_strike_are_low():
    assert assign_tier(AnswerEvidence(source="recovered")) == "low"
    assert assign_tier(AnswerEvidence(status="budget_exhausted")) == "low"
    assert assign_tier(AnswerEvidence(third_strike=True)) == "low"
    assert assign_tier(AnswerEvidence(negative_audit="flagged")) == "low"
    assert assign_tier(AnswerEvidence(resolved_by="unresolved")) == "low"


def test_zero_quote_final_is_medium_final_var_is_high():
    assert assign_tier(AnswerEvidence(source="final", zero_quotes=True)) == "medium"
    assert assign_tier(AnswerEvidence(source="final_var", zero_quotes=True)) == "high"
    assert assign_tier(AnswerEvidence(
        source="final", quotes_total=2, quotes_verified=2)) == "high"


def test_survived_audit_and_pushback_are_medium():
    assert assign_tier(AnswerEvidence(negative_audit="survived")) == "medium"
    assert assign_tier(AnswerEvidence(pushbacks=1)) == "medium"
    assert assign_tier(AnswerEvidence(agreement=0.5, resolved_by="tiebreak")) == "medium"


def test_from_final_marks_empty_batch_zero_quotes():
    ev = from_final({"value": {"q000": "yes"}, "is_var": True}, status="final")
    assert ev.zero_quotes
    assert ev.source == "final"
    assert ev.tier == "medium"


def test_from_final_third_strike():
    ev = from_final({
        "value": "42",
        "verification": {"passed": False, "quotes": [], "third_strike": True},
    })
    assert ev.third_strike
    assert ev.tier == "low"


def test_from_records_survived_negative_audit():
    records = [
        {"kind": "negative_audit", "flagged": ["q000"]},
        {"kind": "final", "value": "NOT_FOUND", "verification": {
            "passed": True, "quotes": []}},
        {"kind": "end", "status": "final"},
    ]
    ev = from_records(records)
    assert ev.negative_audit == "survived"
    assert ev.tier == "medium"


def test_source_table_quality_is_scoped_to_each_batch_field():
    def report(table, quality):
        return {"passed": True, "quotes": [{"matched": True, "doc_id": "report"}],
                "source_tables": [{"table": table, "status": quality}]}
    final = {"value": {"a": "42", "b": "43"}, "verification": {
        "a": report("trusted_table", "trusted"), "b": report("unchecked_table", "unchecked")}}
    first, second = from_final(final, qid="a"), from_final(final, qid="b")
    assert first.tier == "high" and first.cited_tables == ["trusted_table"]
    assert second.tier == "medium" and second.cited_table_statuses == ["unchecked"]


def test_empty_question_id_keeps_its_own_document_scope():
    final = {"value": {"": "42", "b": "43"}, "verification": {
        qid: {"passed": True, "quotes": [{"matched": True, "doc_id": doc}]}
        for qid, doc in [("", "first"), ("b", "second")]}}
    assert from_final(final, qid="").docs_cited == ["first"]


def test_review_must_support_the_selected_field_for_high_tier():
    qhash = hashlib.sha256(json.dumps("a").encode()).hexdigest()
    final = {"value": {"a": "42", "b": "43"}, "verification": {
        q: {"passed": True, "quotes": [{"matched": True}]} for q in ("a", "b")},
        "claim_review": {"status": "supported", "fields": [
            {"qid_sha256": qhash, "verdict": "supported"}]}}
    assert from_final(final, qid="a", claim_review_required=True).tier == "high"
    absent = from_final(final, qid="b", claim_review_required=True)
    assert absent.claim_review_status == "unverified" and absent.tier == "low"
    assert assign_tier(AnswerEvidence(claim_review_status="disabled")) == "medium"
    assert assign_tier(AnswerEvidence(claim_review_required=True)) == "medium"


def test_replay_retains_review_and_table_checks():
    records = [{"kind": "start", "mode": "docdb", "claim_review_enabled": True},
               {"kind": "final", "value": "42", "verification": {
                   "passed": True, "quotes": [{"matched": True}],
                   "source_tables": [{"table": "table", "status": "unchecked"}]},
                "claim_review": {"status": "supported"}},
               {"kind": "end", "status": "final"}]
    evidence = from_records(records)
    assert evidence.claim_review_status == "supported"
    assert evidence.claim_review_required and evidence.tier == "medium"
    assert evidence.cited_tables == ["table"]
