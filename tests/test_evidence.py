"""Trust-tier rules and evidence reconstruction."""

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
