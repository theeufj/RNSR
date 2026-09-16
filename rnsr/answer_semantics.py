"""Shared answer meaning and publication policy.

Classification is intentionally separate from voting/numeric comparison: an
explicit No, missing evidence and not-applicable remain distinct domain states.
"""
from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Literal

NOT_FOUND = "Not found in matter corpus"
DEFAULT_NOT_FOUND = NOT_FOUND
TrustTier = Literal["low", "medium", "high"]
AbstainBelow = Literal["off", "low", "medium", "high"]
QueryStatus = Literal["final", "recovered", "budget_exhausted", "error", "unanswered"]
Resolution = Literal["unanimous", "majority", "split", "tiebreak", "unresolved"]
TIER_RANK = {"low": 0, "medium": 1, "high": 2}


class AnswerKind(StrEnum):
    VALUE = "value"
    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"
    ABSENT = "absent"
    NOT_APPLICABLE = "not_applicable"


def normalize_label(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"[☐☑☒✓✗]", " ", text)
    return re.sub(r"\s+", " ", text.replace("_", " ").casefold()).strip(" .!`*\"'\t\n")



def comparison_key(value: str) -> str:
    """Equality for consensus/form regression, retaining value punctuation."""
    text = re.sub(r"[☐☑☒✓✗]", " ", value or "")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s/@.:$%-]", " ", text.lower())).strip(" .")


def classify_answer(value: object, not_found: str = NOT_FOUND) -> AnswerKind:
    text = normalize_label(value)
    if text == "yes":
        return AnswerKind.AFFIRMATIVE
    if text == "no":
        return AnswerKind.NEGATIVE
    if text in {"n/a", "na", "not applicable"}:
        return AnswerKind.NOT_APPLICABLE
    if text in {"", "none", "nil", "unknown", "blank", "leave blank", "not reached",
                "not specified", "needs review"}:
        return AnswerKind.ABSENT
    for prefix in {"not found", normalize_label(not_found)} - {""}:
        if text == prefix or text.startswith(prefix + " ") or text.startswith(prefix + ":"):
            return AnswerKind.ABSENT
    return AnswerKind.VALUE


def is_negative(value: object, not_found: str = NOT_FOUND) -> bool:
    """Whether an answer needs absence/negative auditing (including explicit No)."""
    return classify_answer(value, not_found) in {
        AnswerKind.NEGATIVE, AnswerKind.ABSENT, AnswerKind.NOT_APPLICABLE,
    }


def requires_quote(value: object) -> bool:
    return classify_answer(value) == AnswerKind.VALUE


def needs_review(tier: str | None) -> bool:
    return tier != "high"


def publish_answer(value: object, tier: str | None,
                   abstain_below: AbstainBelow = "off") -> object:
    """Inclusive threshold; missing evidence is never treated as high trust."""
    if abstain_below != "off" and abstain_below not in TIER_RANK:
        raise ValueError("abstain_below must be off, low, medium, or high")
    if value is None:
        return None
    if abstain_below != "off" and TIER_RANK.get(tier, -1) <= TIER_RANK[abstain_below]:
        return "NEEDS REVIEW"
    return value


def coerce_batch(value: object) -> dict | None:
    """Lenient parse of a batch-final value into a qid -> answer dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        start, end = value.find("{"), value.rfind("}")
        if start != -1 and end > start:
            try:
                out = json.loads(value[start:end + 1])
            except json.JSONDecodeError:
                return None
            if isinstance(out, dict):
                return out
    return None
