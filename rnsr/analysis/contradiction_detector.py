"""
Contradiction Detector -- find conflicting claims within or across documents.

Combines three detection strategies:
1. KG-based: look for CONTRADICTS / SUPPORTS relationships already extracted.
2. Heuristic: pairwise negation and numeric conflict checks (via ProvenanceTracker).
3. LLM-based: semantic contradiction detection for ambiguous cases.

Usage:
    from rnsr.analysis import detect_document_contradictions

    contradictions = detect_document_contradictions(kg, skeleton, kv_store)
    for c in contradictions:
        print(f"[{c.type}] {c.explanation}")
        print(f"  Claim 1: {c.claim_1}  ({c.source_1})")
        print(f"  Claim 2: {c.claim_2}  ({c.source_2})")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FactContradiction:
    """A detected contradiction between two claims."""

    claim_1: str
    """Text of the first claim."""
    claim_2: str
    """Text of the contradicting claim."""
    source_1: str
    """Section / document reference for claim 1."""
    source_2: str
    """Section / document reference for claim 2."""
    type: str = "semantic"
    """One of: direct, numeric, temporal, semantic."""
    confidence: float = 0.5
    """Confidence that this is a genuine contradiction (0.0 -- 1.0)."""
    explanation: str = ""
    """Human-readable explanation of the conflict."""


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

_NEGATION_PAIRS: list[tuple[str, str]] = [
    ("is not", "is"),
    ("was not", "was"),
    ("did not", "did"),
    ("cannot", "can"),
    ("never", "always"),
    ("false", "true"),
    ("incorrect", "correct"),
    ("denied", "granted"),
    ("rejected", "accepted"),
    ("excluded", "included"),
]


def _check_negation(text_a: str, text_b: str) -> str | None:
    """Return a description if the two texts contain opposing negation."""
    a, b = text_a.lower(), text_b.lower()
    for neg, pos in _NEGATION_PAIRS:
        if (neg in a and pos in b and neg not in b) or (
            neg in b and pos in a and neg not in a
        ):
            return f"Negation detected: '{neg}' vs '{pos}'"
    return None


def _check_numeric_conflict(text_a: str, text_b: str) -> str | None:
    """Return a description if the two texts cite different numbers."""
    nums_a = set(re.findall(r"\$?[\d,]+\.?\d*", text_a))
    nums_b = set(re.findall(r"\$?[\d,]+\.?\d*", text_b))
    if nums_a and nums_b and nums_a != nums_b:
        return f"Numeric conflict: {nums_a} vs {nums_b}"
    return None


# ---------------------------------------------------------------------------
# Main detection
# ---------------------------------------------------------------------------


def detect_document_contradictions(
    kg: Any,  # KnowledgeGraph
    skeleton: dict[str, Any] | None = None,
    kv_store: Any | None = None,
    doc_id: str | None = None,
    llm_fn: Any | None = None,
    max_pairs: int = 200,
) -> list[FactContradiction]:
    """Detect contradictions within a document or across documents.

    Args:
        kg: A ``KnowledgeGraph`` instance.
        skeleton: Optional skeleton index for section label lookup.
        kv_store: Optional KV store for retrieving full text.
        doc_id: Restrict to a single document (default: all).
        llm_fn: Optional ``Callable[[str], str]`` for semantic detection.
        max_pairs: Maximum claim pairs to evaluate (to bound cost).

    Returns:
        List of :class:`FactContradiction` sorted by confidence (highest first).
    """
    from rnsr.extraction.models import RelationType

    contradictions: list[FactContradiction] = []

    # ------------------------------------------------------------------
    # Strategy 1: KG CONTRADICTS relationships
    # ------------------------------------------------------------------
    try:
        stats = kg.get_stats()
        if stats.get("relationship_count", 0) > 0:
            _find_kg_contradictions(kg, doc_id, skeleton, contradictions)
    except Exception as exc:
        logger.debug("kg_contradiction_scan_error", error=str(exc))

    # ------------------------------------------------------------------
    # Strategy 2: Heuristic pairwise checks on key claims
    # ------------------------------------------------------------------
    if kv_store and skeleton:
        claims = _extract_key_claims(skeleton, kv_store, doc_id)
        _heuristic_pairwise(claims, contradictions, max_pairs)

    # ------------------------------------------------------------------
    # Strategy 3: LLM-based semantic detection (optional)
    # ------------------------------------------------------------------
    if llm_fn and kv_store and skeleton:
        claims = _extract_key_claims(skeleton, kv_store, doc_id)
        _llm_semantic_check(claims, llm_fn, contradictions, max_pairs)

    # Deduplicate and sort
    contradictions = _deduplicate(contradictions)
    contradictions.sort(key=lambda c: c.confidence, reverse=True)

    logger.info(
        "contradictions_detected",
        total=len(contradictions),
        by_type={
            t: sum(1 for c in contradictions if c.type == t)
            for t in {"direct", "numeric", "temporal", "semantic"}
        },
    )
    return contradictions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _Claim:
    """Internal: a claim extracted from a section."""
    text: str
    section: str
    node_id: str
    doc_id: str = ""


def _find_kg_contradictions(
    kg: Any,
    doc_id: str | None,
    skeleton: dict | None,
    out: list[FactContradiction],
) -> None:
    """Scan KG for CONTRADICTS relationships."""
    from rnsr.extraction.models import RelationType

    # Walk all entities and look for CONTRADICTS rels
    for entity_type_name in ("person", "organization", "legal_concept", "event", "other"):
        try:
            from rnsr.extraction.models import EntityType

            etype = EntityType(entity_type_name)
            entities = kg.find_entities_by_type(etype, doc_id=doc_id)
        except (ValueError, Exception):
            continue

        for entity in entities:
            rels = kg.get_entity_relationships(entity.id)
            for rel in rels:
                if rel.type in (
                    RelationType.CONTRADICTS,
                ):
                    source_label = entity.canonical_name
                    target_entity = (
                        kg.get_entity(rel.target_id) if hasattr(kg, "get_entity") else None
                    )
                    target_label = (
                        target_entity.canonical_name if target_entity else rel.target_id
                    )

                    out.append(
                        FactContradiction(
                            claim_1=source_label,
                            claim_2=target_label,
                            source_1=rel.doc_id or "",
                            source_2=rel.doc_id or "",
                            type="direct",
                            confidence=rel.confidence,
                            explanation=rel.evidence or f"{source_label} contradicts {target_label}",
                        )
                    )


def _extract_key_claims(
    skeleton: dict[str, Any],
    kv_store: Any,
    doc_id: str | None,
    max_claims: int = 50,
) -> list[_Claim]:
    """Extract the first sentence from each section as a 'claim'."""
    claims: list[_Claim] = []
    for node_id, node in skeleton.items():
        content = kv_store.get(node_id) or ""
        if len(content.strip()) < 30:
            continue
        # Take the first meaningful sentence
        for sent in re.split(r"[.!?]\s+", content[:500]):
            sent = sent.strip()
            if len(sent) > 20:
                claims.append(
                    _Claim(
                        text=sent,
                        section=getattr(node, "header", node_id),
                        node_id=node_id,
                        doc_id=doc_id or "",
                    )
                )
                break
        if len(claims) >= max_claims:
            break
    return claims


def _heuristic_pairwise(
    claims: list[_Claim],
    out: list[FactContradiction],
    max_pairs: int,
) -> None:
    """Run negation and numeric checks on claim pairs."""
    checked = 0
    for i, c1 in enumerate(claims):
        for c2 in claims[i + 1:]:
            if checked >= max_pairs:
                return
            checked += 1

            neg = _check_negation(c1.text, c2.text)
            if neg:
                out.append(
                    FactContradiction(
                        claim_1=c1.text,
                        claim_2=c2.text,
                        source_1=c1.section,
                        source_2=c2.section,
                        type="direct",
                        confidence=0.6,
                        explanation=neg,
                    )
                )
                continue

            num = _check_numeric_conflict(c1.text, c2.text)
            if num:
                out.append(
                    FactContradiction(
                        claim_1=c1.text,
                        claim_2=c2.text,
                        source_1=c1.section,
                        source_2=c2.section,
                        type="numeric",
                        confidence=0.4,
                        explanation=num,
                    )
                )


def _llm_semantic_check(
    claims: list[_Claim],
    llm_fn: Any,
    out: list[FactContradiction],
    max_pairs: int,
) -> None:
    """Use an LLM to detect semantic contradictions."""
    import json as _json

    # Take the most important claims (first N)
    top_claims = claims[:20]
    if len(top_claims) < 2:
        return

    claims_text = "\n".join(
        f"{i+1}. [{c.section}] {c.text[:200]}"
        for i, c in enumerate(top_claims)
    )

    prompt = f"""Analyze the following claims from a document and identify any contradictions.
Return a JSON array of contradictions. If none exist, return an empty array.

Claims:
{claims_text}

Return ONLY valid JSON (no markdown, no extra text):
[{{"claim_1_index": 1, "claim_2_index": 3, "type": "direct|numeric|temporal|semantic", "explanation": "brief reason"}}]"""

    try:
        raw = llm_fn(prompt)
        # Parse JSON from response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return
        results = _json.loads(match.group(0))
        for item in results:
            idx1 = item.get("claim_1_index", 0) - 1
            idx2 = item.get("claim_2_index", 0) - 1
            if 0 <= idx1 < len(top_claims) and 0 <= idx2 < len(top_claims):
                out.append(
                    FactContradiction(
                        claim_1=top_claims[idx1].text,
                        claim_2=top_claims[idx2].text,
                        source_1=top_claims[idx1].section,
                        source_2=top_claims[idx2].section,
                        type=item.get("type", "semantic"),
                        confidence=0.7,
                        explanation=item.get("explanation", ""),
                    )
                )
    except Exception as exc:
        logger.debug("llm_contradiction_check_failed", error=str(exc))


def _deduplicate(items: list[FactContradiction]) -> list[FactContradiction]:
    """Remove duplicate contradictions (same pair of claims)."""
    seen: set[str] = set()
    unique: list[FactContradiction] = []
    for c in items:
        key = f"{c.claim_1[:50]}|{c.claim_2[:50]}"
        rev_key = f"{c.claim_2[:50]}|{c.claim_1[:50]}"
        if key not in seen and rev_key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Cross-document contradiction detection
# ---------------------------------------------------------------------------


def _heuristic_pairwise_cross_doc(
    claims: list[_Claim],
    out: list[FactContradiction],
    max_pairs: int,
) -> None:
    """Run negation and numeric checks on claim pairs from *different* documents only."""
    checked = 0
    for i, c1 in enumerate(claims):
        for c2 in claims[i + 1:]:
            # Only compare claims from different documents
            if c1.doc_id == c2.doc_id:
                continue
            if checked >= max_pairs:
                return
            checked += 1

            neg = _check_negation(c1.text, c2.text)
            if neg:
                out.append(
                    FactContradiction(
                        claim_1=c1.text,
                        claim_2=c2.text,
                        source_1=f"[{c1.doc_id}] {c1.section}",
                        source_2=f"[{c2.doc_id}] {c2.section}",
                        type="direct",
                        confidence=0.6,
                        explanation=neg,
                    )
                )
                continue

            num = _check_numeric_conflict(c1.text, c2.text)
            if num:
                out.append(
                    FactContradiction(
                        claim_1=c1.text,
                        claim_2=c2.text,
                        source_1=f"[{c1.doc_id}] {c1.section}",
                        source_2=f"[{c2.doc_id}] {c2.section}",
                        type="numeric",
                        confidence=0.4,
                        explanation=num,
                    )
                )


def _llm_semantic_check_cross_doc(
    claims: list[_Claim],
    llm_fn: Any,
    out: list[FactContradiction],
    max_pairs: int,
) -> None:
    """Use an LLM to detect semantic contradictions across documents."""
    import json as _json

    # Take claims from different documents (interleave)
    top_claims = claims[:30]
    if len(top_claims) < 2:
        return

    doc_ids_present = {c.doc_id for c in top_claims}
    if len(doc_ids_present) < 2:
        return

    claims_text = "\n".join(
        f"{i+1}. [Doc: {c.doc_id} | {c.section}] {c.text[:200]}"
        for i, c in enumerate(top_claims)
    )

    prompt = f"""Analyze the following claims from MULTIPLE documents and identify contradictions BETWEEN documents.
Only flag contradictions where two DIFFERENT documents make conflicting statements.
Return a JSON array of contradictions. If none exist, return an empty array.

Claims:
{claims_text}

Return ONLY valid JSON (no markdown, no extra text):
[{{"claim_1_index": 1, "claim_2_index": 3, "type": "direct|numeric|temporal|semantic", "explanation": "brief reason"}}]"""

    try:
        raw = llm_fn(prompt)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return
        results = _json.loads(match.group(0))
        for item in results:
            idx1 = item.get("claim_1_index", 0) - 1
            idx2 = item.get("claim_2_index", 0) - 1
            if 0 <= idx1 < len(top_claims) and 0 <= idx2 < len(top_claims):
                c1, c2 = top_claims[idx1], top_claims[idx2]
                out.append(
                    FactContradiction(
                        claim_1=c1.text,
                        claim_2=c2.text,
                        source_1=f"[{c1.doc_id}] {c1.section}",
                        source_2=f"[{c2.doc_id}] {c2.section}",
                        type=item.get("type", "semantic"),
                        confidence=0.7,
                        explanation=item.get("explanation", ""),
                    )
                )
    except Exception as exc:
        logger.debug("llm_cross_doc_contradiction_check_failed", error=str(exc))


def detect_cross_document_contradictions(
    kg: Any,  # KnowledgeGraph
    documents: list[tuple[str, dict[str, Any], Any]],
    llm_fn: Any | None = None,
    max_pairs: int = 300,
) -> list[FactContradiction]:
    """Detect contradictions *across* multiple documents.

    Unlike :func:`detect_document_contradictions` which analyses a single
    document, this function extracts claims from every document provided
    and then runs pairwise comparisons only between claims originating
    from *different* documents.

    Args:
        kg: A workspace-wide ``KnowledgeGraph`` that contains entities
            from all documents (e.g. from ``DocumentStore.get_workspace_kg()``).
        documents: A list of ``(doc_id, skeleton, kv_store)`` tuples —
            one per document to include in the analysis.
        llm_fn: Optional ``Callable[[str], str]`` for semantic detection.
        max_pairs: Maximum claim pairs to evaluate (to bound cost).

    Returns:
        List of :class:`FactContradiction` sorted by confidence (highest first).

    Example::

        store = DocumentStore("./docs")
        kg = store.get_workspace_kg()
        docs = [
            (doc_id, *store.get_document(doc_id))
            for doc_id in store
        ]
        contradictions = detect_cross_document_contradictions(kg, docs)
    """
    contradictions: list[FactContradiction] = []

    # ------------------------------------------------------------------
    # Strategy 1: KG CONTRADICTS relationships (workspace-wide)
    # ------------------------------------------------------------------
    try:
        stats = kg.get_stats()
        if stats.get("relationship_count", 0) > 0:
            _find_kg_contradictions(kg, doc_id=None, skeleton=None, out=contradictions)
    except Exception as exc:
        logger.debug("cross_doc_kg_contradiction_scan_error", error=str(exc))

    # ------------------------------------------------------------------
    # Strategy 2: Heuristic pairwise checks across documents
    # ------------------------------------------------------------------
    all_claims: list[_Claim] = []
    for doc_id, skeleton, kv_store in documents:
        doc_claims = _extract_key_claims(skeleton, kv_store, doc_id, max_claims=30)
        all_claims.extend(doc_claims)

    if len(all_claims) >= 2:
        _heuristic_pairwise_cross_doc(all_claims, contradictions, max_pairs)

    # ------------------------------------------------------------------
    # Strategy 3: LLM-based semantic detection (optional)
    # ------------------------------------------------------------------
    if llm_fn and len(all_claims) >= 2:
        _llm_semantic_check_cross_doc(all_claims, llm_fn, contradictions, max_pairs)

    # Deduplicate and sort
    contradictions = _deduplicate(contradictions)
    contradictions.sort(key=lambda c: c.confidence, reverse=True)

    logger.info(
        "cross_doc_contradictions_detected",
        total=len(contradictions),
        documents=len(documents),
        claims_analysed=len(all_claims),
        by_type={
            t: sum(1 for c in contradictions if c.type == t)
            for t in {"direct", "numeric", "temporal", "semantic"}
        },
    )
    return contradictions
