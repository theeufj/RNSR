"""Per-answer evidence and trust tier (no LLM in the loop).

The answering loop already computes quote verification, negative-answer
audits, completeness pushbacks, and recovery. Those signals used to die
in the trajectory. AnswerEvidence is the caller-visible record; ``tier``
is a deterministic function of that record, documented in
``docs/trust-tiers.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from rnsr.answer_semantics import QueryStatus, Resolution, TrustTier

TIERS = ("high", "medium", "low")
SOURCES = ("final", "final_var", "recovered")
NEG_AUDIT = ("none", "probed", "flagged", "survived")
REVIEW_STATUSES = ("supported", "disabled", "not_applicable", "unverified", "unknown")


def _reports(verification: object) -> list[dict]:
    """Normalise solo-report vs per-qid dict vs FINAL_BATCH wrapper."""
    if not verification or not isinstance(verification, dict):
        return []
    if (isinstance(verification.get("quotes"), list)
            or type(verification.get("passed")) is bool
            or type(verification.get("zero_quotes")) is bool
            or isinstance(verification.get("check"), str)):
        return [verification]
    out: list[dict] = []
    for block in verification.values():
        if isinstance(block, dict):
            out.append(block)
    return out


def _docs_from_verification(verification: object) -> list[str]:
    docs: list[str] = []
    for report in _reports(verification):
        for q in report.get("quotes") or []:
            if isinstance(q, dict) and q.get("doc_id"):
                docs.append(str(q["doc_id"]))
    return list(dict.fromkeys(docs))


@dataclass
class AnswerEvidence:
    """Mechanical signals that produced one answer (or one batch field)."""

    quotes_total: int = 0
    quotes_verified: int = 0
    third_strike: bool = False
    zero_quotes: bool = False
    negative_audit: Literal["none", "probed", "flagged", "survived"] = "none"          # none | probed | flagged | survived
    pushbacks: int = 0
    source: Literal["final", "final_var", "recovered"] = "final"                 # final | final_var | recovered
    rungs_used: list[int] = field(default_factory=list)
    docs_cited: list[str] = field(default_factory=list)
    cited_table_statuses: list[str] = field(default_factory=list)
    cited_tables: list[str] = field(default_factory=list)
    claim_review_status: Literal["supported", "disabled", "not_applicable", "unverified", "unknown"] = "unknown"
    claim_review_required: bool = False
    agreement: float | None = None
    resolved_by: Resolution | None = None
    votes: list[str | None] = field(default_factory=list)
    health_grade: str | None = None
    budget_warned: bool = False
    status: QueryStatus = "final"

    @property
    def tier(self) -> TrustTier:
        return assign_tier(self)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tier"] = self.tier
        return d


def assign_tier(ev: AnswerEvidence) -> TrustTier:
    """Transparent rules. Order is load-bearing: low checks first.

    See docs/trust-tiers.md. Never calls a model.
    """
    if ev.source == "recovered" or ev.status not in ("final",):
        return "low"
    if ev.third_strike:
        return "low"
    if ev.health_grade == "blocked":
        return "low"
    if ev.negative_audit == "flagged":
        return "low"
    if ev.resolved_by == "unresolved":
        return "low"
    if ev.claim_review_required and ev.claim_review_status == "unverified":
        return "low"

    medium = False
    if ev.zero_quotes and ev.source == "final":
        medium = True
    if ev.quotes_total and ev.quotes_verified < ev.quotes_total:
        medium = True
    if ev.negative_audit == "survived":
        medium = True
    if ev.pushbacks:
        medium = True
    if ev.health_grade == "degraded":
        medium = True
    if ev.budget_warned:
        medium = True
    if ev.resolved_by in ("tiebreak", "split"):
        medium = True
    if ev.agreement is not None and ev.agreement < 1.0:
        medium = True
    if any(s in ("untrusted", "unchecked", "unknown") for s in ev.cited_table_statuses):
        medium = True
    if ev.claim_review_status == "disabled" or (
            ev.claim_review_required and ev.claim_review_status != "supported"):
        medium = True
    if medium:
        return "medium"
    return "high"


def from_final(
    final: dict | None,
    *,
    status: str = "final",
    pushbacks: int = 0,
    negative_audit: str = "none",
    budget_warned: bool = False,
    health_grade: str | None = None,
    rungs_used: list[int] | None = None,
    cited_table_statuses: list[str] | None = None,
    claim_review_status: str | None = None,
    claim_review_required: bool = False,
    agreement: float | None = None,
    resolved_by: str | None = None,
    votes: list[str | None] | None = None,
    qid: str | None = None,
) -> AnswerEvidence:
    """Build evidence from a FINAL-shaped dict plus loop signals."""
    verification = (final or {}).get("verification")
    value = (final or {}).get("value")
    is_var = bool(final and final.get("is_var"))

    report = None
    if qid is not None and isinstance(verification, dict) and qid in verification:
        report = verification[qid]
        reports = _reports(report)
    else:
        reports = _reports(verification)

    quotes = [q for r in reports for q in (r.get("quotes") or [])]
    quotes_total = len(quotes)
    quotes_verified = sum(1 for q in quotes if isinstance(q, dict) and q.get("matched"))
    third_strike = any(r.get("third_strike") for r in reports)
    flagged_zero = any(r.get("zero_quotes") for r in reports)
    source_tables = [t for r in reports for t in (r.get("source_tables") or [])
                     if isinstance(t, dict)]
    review = (final or {}).get("claim_review") or {}
    review_status = claim_review_status or review.get("status") or "unknown"
    # 'partial': the loop stopped a stalled review cycle and finalised with
    # some fields unresolved. Per field, a supported verdict stands; an
    # unresolved one is unverified (low tier). Without a field to look at,
    # the whole answer is treated as unverified.
    per_field = review.get("status") in ("supported", "partial") and bool(review.get("fields"))
    if review_status == "partial":
        review_status = "unverified"
    if qid is not None and per_field and review_status not in ("disabled", "not_applicable"):
        qhash = hashlib.sha256(json.dumps(qid, ensure_ascii=False, sort_keys=True)
                              .encode("utf-8", "surrogatepass")).hexdigest()
        reviewed = [f for f in review["fields"] if f.get("qid_sha256") == qhash]
        if not reviewed or any(f.get("verdict") != "supported" for f in reviewed):
            review_status = "unverified"
        else:
            review_status = "supported"

    if status == "recovered":
        source = "recovered"
    elif isinstance(value, dict) and not reports:
        source = "final"
        flagged_zero = True
    elif reports:
        source = "final"
    elif is_var:
        source = "final_var"
    else:
        source = "final"
        flagged_zero = flagged_zero or quotes_total == 0

    zero_quotes = flagged_zero or (source == "final" and quotes_total == 0)
    return AnswerEvidence(
        quotes_total=quotes_total,
        quotes_verified=quotes_verified,
        third_strike=third_strike,
        zero_quotes=zero_quotes,
        negative_audit=negative_audit if negative_audit in NEG_AUDIT else "none",
        pushbacks=pushbacks,
        source=source,
        rungs_used=list(rungs_used or []),
        docs_cited=_docs_from_verification(report if qid is not None else verification),
        cited_table_statuses=list(dict.fromkeys(
            [*list(cited_table_statuses or []), *[t.get("status", "unknown") for t in source_tables]])),
        cited_tables=list(dict.fromkeys(t["table"] for t in source_tables if t.get("table"))),
        claim_review_status=review_status if review_status in REVIEW_STATUSES else "unknown",
        claim_review_required=claim_review_required,
        agreement=agreement,
        resolved_by=resolved_by,
        votes=list(votes or []),
        health_grade=health_grade,
        budget_warned=budget_warned,
        status=status,
    )


def from_records(records: list[dict], *,
                 status: str = "final",
                 health_grade: str | None = None,
                 qid: str | None = None) -> AnswerEvidence:
    """Rebuild evidence from a trajectory when the live object is gone."""
    final = next((r for r in reversed(records) if r.get("kind") == "final"), {})
    end = next((r for r in reversed(records) if r.get("kind") == "end"), {})
    status = end.get("status") or status
    pushbacks = sum(1 for r in records if r.get("kind") in
                    ("completeness_pushback",))
    neg = "none"
    if any(r.get("kind") == "negative_audit" for r in records):
        flagged = any((r.get("flagged") or [])
                      for r in records if r.get("kind") == "negative_audit")
        neg = "flagged" if flagged else "probed"
        if flagged and status == "final":
            neg = "survived"
    rungs = []
    for r in records:
        if r.get("kind") == "search_rung" and r.get("rung") is not None:
            rungs.append(int(r["rung"]))
    recovered = any(r.get("kind") == "recovery" for r in records)
    start = next((r for r in records if r.get("kind") == "start"), {})
    return from_final(
        {"value": final.get("value"),
         "is_var": final.get("is_var"),
         "verification": final.get("verification"), "claim_review": final.get("claim_review")},
        status="recovered" if recovered and status != "final" else status,
        pushbacks=pushbacks,
        negative_audit=neg,
        budget_warned=any(r.get("kind") == "budget_warning" for r in records),
        health_grade=health_grade,
        rungs_used=list(dict.fromkeys(rungs)),
        claim_review_status=("not_applicable" if start.get("mode") == "classic" else
                             "disabled" if start.get("claim_review_enabled") is False else None),
        claim_review_required=start.get("mode") == "docdb",
        qid=qid,
    )
