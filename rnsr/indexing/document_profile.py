"""
Document Profile Extraction

Extracts structured metadata from KG entities and document content
at ingestion time. Provides deterministic answers to common factual
questions (judge name, citation, court, date) without requiring
LLM navigation at query time.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger(__name__)

# Neutral citation pattern: [YYYY] COURT NNN  (e.g. [2025] HCA 7)
_CITATION_RE = re.compile(
    r"\[(\d{4})\]\s+([A-Z][A-Za-z]{1,15}(?:\s+[A-Z][A-Za-z]{0,10})?)\s+(\d{1,6})"
)

_JUDGE_TITLE_RE = re.compile(
    r"\b(?:Justice|Judge|Honour|Magistrate|"
    r"Chief\s+Justice|Deputy\s+President|President)\b",
    re.IGNORECASE,
)

_JUDGE_SUFFIX_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
    r"(?:CJ|JJ?|DCJ|SC\s+DCJ|AJ|JA|AM|SC)\b"
)

_COURT_KEYWORDS = re.compile(
    r"\b(?:court|tribunal|commission|magistrat|house\s+of\s+lords|privy\s+council)\b",
    re.IGNORECASE,
)

# Patterns to extract judgment/hearing dates from document header text.
# These are ordered by specificity — most specific first.
# Date capture groups handle multiple formats: DD Month YYYY, DD-Mon-YY, DD/MM/YYYY.
_DATE_VALUE = (
    r"(\d{1,2}[\s./-]+\w+[\s./-]+\d{2,4})"
)
_JUDGMENT_DATE_PATTERNS: list[re.Pattern[str]] = [
    # "JUDGMENT GIVEN ON 29 October 2009" or "[Date of Judgment]: 29-Oct-09"
    re.compile(
        r"\[?(?:JUDGMENT\s+GIVEN\s+ON|Date\s+of\s+Judgment|Judgment\s+date|"
        r"Date\s+of\s+(?:Decision|Order|Reasons))\]?[:\s]*\s*"
        + _DATE_VALUE,
        re.IGNORECASE | re.DOTALL,
    ),
    # "Date of Hearing: 13 July 2009" or "Hearing date: 13-Jul-09"
    re.compile(
        r"\[?Date\s+of\s+Hearing\]?[:\s]*\s*" + _DATE_VALUE,
        re.IGNORECASE | re.DOTALL,
    ),
    # "Decided: October 29, 2009" or "Decided 29/10/2009"
    re.compile(
        r"\b(?:Decided|Delivered|Handed\s+down)[:\s]+\s*"
        + _DATE_VALUE,
        re.IGNORECASE | re.DOTALL,
    ),
]

# Non-date strings sometimes extracted as DATE entities from legal docs.
_NON_DATE_TERMS = re.compile(
    r"(?:Michaelmas|Trinity|Hilary|Easter)\s+Term"
    r"|(?:Spring|Summer|Autumn|Winter|Fall)\s+\d{4}"
    r"|(?:First|Second|Third|Fourth)\s+Quarter"
    r"|\b(?:Term|Quarter|Session|Semester)\b",
    re.IGNORECASE,
)

_DOC_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bjudgm?ent\b", re.I), "judgment"),
    (re.compile(r"\border\b", re.I), "order"),
    (re.compile(r"\bagreement\b", re.I), "agreement"),
    (re.compile(r"\bletter\b", re.I), "letter"),
    (re.compile(r"\binvoice\b", re.I), "invoice"),
    (re.compile(r"\bcosts?\s+agreement\b", re.I), "costs_agreement"),
    (re.compile(r"\baffidavit\b", re.I), "affidavit"),
    (re.compile(r"\bstatement\b", re.I), "statement"),
    (re.compile(r"\bcontract\b", re.I), "contract"),
    (re.compile(r"\bwill\b", re.I), "will"),
    (re.compile(r"\blease\b", re.I), "lease"),
    (re.compile(r"\bretainer\b", re.I), "retainer"),
    (re.compile(r"\breport\b", re.I), "report"),
    (re.compile(r"\bapplication\b", re.I), "application"),
]


class DocumentProfile(BaseModel):
    """Structured metadata extracted from a document at ingestion time."""

    document_type: str | None = None
    citation: str | None = None
    court: str | None = None
    judge: str | None = None
    primary_date: str | None = None
    parties: list[str] = Field(default_factory=list)
    jurisdiction: str | None = None
    page_count: int | None = None


def extract_profile(
    kg: Any,
    doc_id: str,
    title: str,
    root_content: str | None = None,
    tail_content: str | None = None,
    page_count: int | None = None,
) -> DocumentProfile:
    """Build a DocumentProfile from KG entities and raw text.

    This avoids extra LLM calls — it reuses the entities and
    relationships already extracted during KG build.

    Args:
        kg: KnowledgeGraph instance.
        doc_id: Document identifier.
        title: Document title.
        root_content: First ~2000 chars of the document (root + first child).
        tail_content: Last ~1000 chars of the document (last child).
        page_count: Authoritative page count if known.
    """
    from rnsr.extraction.models import EntityType, RelationType

    profile = DocumentProfile(page_count=page_count)

    # ---- document type (deterministic, from title + root text) ----
    probe = (title or "") + " " + (root_content or "")[:500]
    for pat, dtype in _DOC_TYPE_PATTERNS:
        if pat.search(probe):
            profile.document_type = dtype
            break

    entities = kg.find_entities_in_document(doc_id)
    if not entities:
        # No KG data yet — fall back to regex on raw text
        profile.citation = _extract_citation_regex(root_content)
        profile.court = _extract_court_regex(root_content)
        profile.judge = _extract_judge_regex(root_content, tail_content)
        return profile

    # ---- citation (REFERENCE entities + regex) ----
    for ent in entities:
        if ent.type == EntityType.REFERENCE:
            m = _CITATION_RE.search(ent.canonical_name)
            if m:
                profile.citation = m.group(0)
                break
    if not profile.citation:
        profile.citation = _extract_citation_regex(root_content)

    # ---- court (ORGANIZATION entities containing court keywords) ----
    # Prefer courts that appear in the document header (first ~500 chars)
    # over courts mentioned later in body text.
    header_text = (root_content or "")[:500].lower()
    court_candidates: list[tuple[str, bool]] = []
    for ent in entities:
        if ent.type == EntityType.ORGANIZATION and _COURT_KEYWORDS.search(
            ent.canonical_name
        ):
            in_header = ent.canonical_name.lower() in header_text
            court_candidates.append((ent.canonical_name, in_header))

    if court_candidates:
        header_courts = [c for c, h in court_candidates if h]
        if header_courts:
            profile.court = max(header_courts, key=len)
        else:
            profile.court = court_candidates[0][0]
    if not profile.court:
        profile.court = _extract_court_regex(root_content)

    # ---- judge (PERSON entities with judicial context) ----
    profile.judge = _find_judge_entity(entities, kg)
    if not profile.judge:
        profile.judge = _extract_judge_regex(root_content, tail_content)

    # ---- primary date ----
    # For judgments, prefer the judgment/hearing date from the header text
    # over KG entity dates (which are often event dates from the case facts).
    if profile.document_type == "judgment":
        profile.primary_date = _extract_judgment_date(root_content)

    if not profile.primary_date:
        date_entities = [
            e for e in entities
            if e.type == EntityType.DATE
            and not _NON_DATE_TERMS.search(e.canonical_name)
        ]
        if date_entities:
            for de in date_entities:
                rels = kg.get_entity_relationships(de.id)
                for rel in rels:
                    if rel.type == RelationType.HAS_DATE:
                        profile.primary_date = de.canonical_name
                        break
                if profile.primary_date:
                    break
            if not profile.primary_date:
                profile.primary_date = date_entities[0].canonical_name

    # ---- parties (PERSON entities that are parties) ----
    for ent in entities:
        if ent.type == EntityType.PERSON:
            rels = kg.get_entity_relationships(ent.id)
            for rel in rels:
                if rel.type == RelationType.PARTY_TO:
                    profile.parties.append(ent.canonical_name)
                    break

    # ---- jurisdiction (from court entity or root text) ----
    if profile.court:
        lower = profile.court.lower()
        if "criminal" in lower:
            profile.jurisdiction = "Criminal"
        elif "family" in lower:
            profile.jurisdiction = "Family"
        elif "federal" in lower:
            profile.jurisdiction = "Federal"

    logger.info(
        "document_profile_extracted",
        doc_id=doc_id,
        has_citation=profile.citation is not None,
        has_judge=profile.judge is not None,
        has_court=profile.court is not None,
        has_date=profile.primary_date is not None,
        parties=len(profile.parties),
    )
    return profile


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_judge_entity(entities: list, kg: Any) -> str | None:
    """Find a judge among PERSON entities using context and relationships.

    Returns a name only when exactly one candidate is found. When multiple
    judicial figures appear, returns None so the resolver falls back to
    navigation rather than guessing.
    """
    from rnsr.extraction.models import EntityType

    candidates: list[str] = []
    for ent in entities:
        if ent.type != EntityType.PERSON:
            continue
        name = ent.canonical_name

        if _JUDGE_SUFFIX_RE.search(name):
            candidates.append(name)
            continue

        for mention in ent.mentions:
            if _JUDGE_TITLE_RE.search(mention.context):
                candidates.append(name)
                break
        else:
            meta_role = (ent.metadata.get("role") or "").lower()
            if any(t in meta_role for t in ("judge", "justice", "magistrate")):
                candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]
    return None


def _extract_judgment_date(text: str | None) -> str | None:
    """Extract a judgment/hearing date from document header text."""
    if not text:
        return None
    for pat in _JUDGMENT_DATE_PATTERNS:
        m = pat.search(text[:3000])
        if m:
            return m.group(1).strip()
    return None


def _extract_citation_regex(text: str | None) -> str | None:
    if not text:
        return None
    m = _CITATION_RE.search(text[:1500])
    return m.group(0) if m else None


_COURT_REGEX = re.compile(
    r"(?:"
    # Multi-word court names that must be matched first to avoid partial matches
    r"COURT\s+OF\s+(?:APPEAL|CRIMINAL\s+APPEAL|FIRST\s+INSTANCE)"
    r"(?:\s+OF\s+(?:THE\s+)?[A-Z][A-Z ]*?)?"
    r"|UNITED\s+STATES\s+(?:DISTRICT|CIRCUIT|BANKRUPTCY)\s+COURT"
    r"(?:[,\s]+(?:FOR\s+(?:THE\s+)?)?(?:(?:NORTHERN|SOUTHERN|EASTERN|WESTERN|CENTRAL|MIDDLE)\s+)?"
    r"(?:DISTRICT|CIRCUIT)\s+OF\s+[A-Z][A-Z ]*?)?"
    r"|(?:HIGH|SUPREME|DISTRICT|FAMILY|FEDERAL|CROWN|COUNTY|CIRCUIT"
    r"|EMPLOYMENT|LAND|LAND\s+AND\s+ENVIRONMENT|CHILDREN(?:'?S)?|CORONER(?:'?S)?)"
    r"\s+COURT"
    r"(?:\s+OF\s+(?:THE\s+)?[A-Z][A-Z ]*?)?"
    r"|FEDERAL\s+CIRCUIT(?:\s+AND\s+FAMILY)?\s+COURT"
    r"(?:\s+OF\s+[A-Z][A-Z ]*?)?"
    r"|MAGISTRATES?(?:'?S?)?\s+COURT"
    r"(?:\s+OF\s+[A-Z][A-Z ]*?)?"
    r"|HOUSE\s+OF\s+LORDS"
    r"|PRIVY\s+COUNCIL"
    r"|(?:FAIR\s+WORK|ADMINISTRATIVE\s+APPEALS|MENTAL\s+HEALTH)\s+(?:COMMISSION|TRIBUNAL)"
    r"(?:\s+OF\s+[A-Z][A-Z ]*?)?"
    r")",
    re.IGNORECASE,
)


def _extract_court_regex(text: str | None) -> str | None:
    if not text:
        return None
    m = _COURT_REGEX.search(text[:2000])
    return m.group(0).strip() if m else None


def _extract_judge_regex(
    root_text: str | None, tail_text: str | None
) -> str | None:
    combined = ((root_text or "")[:1500] + "\n" + (tail_text or "")[:1000])
    m = _JUDGE_SUFFIX_RE.search(combined)
    return m.group(0).strip() if m else None
