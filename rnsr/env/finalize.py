"""Final-answer quote contract, also enforced outside the untrusted child."""
from __future__ import annotations

from rnsr.answer_semantics import requires_quote

MAX_FINAL_FIELDS = 1000
MAX_QUOTES_PER_FIELD = 3
MAX_QUOTE_CHARS = 20_000
MAX_TOTAL_QUOTE_CHARS = 200_000


def validate_final(value, quotes, verifier, *, batch: bool = False) -> dict:
    if batch:
        if not isinstance(value, dict) or not isinstance(quotes, (dict, type(None))):
            raise ValueError("FINAL_BATCH requires an answers dict and per-question quotes dict")
        if len(value) > MAX_FINAL_FIELDS:
            raise ValueError(f"FINAL_BATCH permits at most {MAX_FINAL_FIELDS} fields")
        submitted = []
        for qid in value:
            field_quotes = (quotes or {}).get(qid) or []
            if isinstance(field_quotes, str):
                field_quotes = [field_quotes]
            if not isinstance(field_quotes, list) or any(not isinstance(q, str) for q in field_quotes):
                raise ValueError("quotes must be a list of verbatim source strings")
            submitted.extend(field_quotes)
        if sum(map(len, submitted)) > MAX_TOTAL_QUOTE_CHARS:
            raise ValueError(f"FINAL_BATCH permits at most {MAX_TOTAL_QUOTE_CHARS} quote characters")
        return {qid: validate_final(answer, (quotes or {}).get(qid), verifier)
                for qid, answer in value.items()}
    if quotes is None:
        quotes = []
    if isinstance(quotes, str):
        quotes = [quotes]
    if not isinstance(quotes, list) or any(not isinstance(q, str) for q in quotes):
        raise ValueError("quotes must be a list of verbatim source strings")
    if not quotes:
        if requires_quote(value):
            raise ValueError("FINAL rejected: a value-bearing answer requires supporting source quotes")
        return {"passed": False, "answer": str(value), "quotes": [], "zero_quotes": True}
    if len(quotes) > MAX_QUOTES_PER_FIELD or any(len(q) > MAX_QUOTE_CHARS for q in quotes):
        raise ValueError("FINAL rejected: supply 1-3 short source quotes")
    report = verifier.verify(str(value), quotes)
    if not report["passed"]:
        raise ValueError("FINAL rejected: quotes do not match retained source text")
    return report


def submitted_quotes(report, *, batch: bool = False):
    """Extract only submitted strings; never trust the child's match/offset claims."""
    if batch:
        return {qid: submitted_quotes(item) for qid, item in (report or {}).items()}
    if not isinstance(report, dict):
        return []
    return [q.get("quote") for q in report.get("quotes", []) if isinstance(q, dict)]
