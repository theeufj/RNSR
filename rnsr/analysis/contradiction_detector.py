"""
Contradiction Detector -- find conflicting claims within or across documents.

Combines six detection strategies:
1. KG-based: look for CONTRADICTS / SUPPORTS relationships already extracted.
2. Heuristic: pairwise negation and numeric conflict checks (subject-gated).
3. LLM-based: semantic contradiction detection for ambiguous cases.
4. Structure-Parallel: match parallel sections across documents by header
   similarity (e.g. "Diagnosis" in two expert reports) and compare them.
5. Entity-Centric: group claims by the KG entities they reference, then
   compare only claims that discuss the same real-world entity.
6. Relationship Divergence: compare KG relationships for the same linked
   entity across documents (e.g. "has PTSD" vs "does not have PTSD").

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
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
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
    """Return a description if the two texts contain opposing negation **about
    the same subject**.

    We require that the two texts share at least one meaningful content word
    (longer than 4 chars, not a stopword) so we only flag negation between
    statements that are actually about the same thing.
    """
    if not _texts_share_subject(text_a, text_b):
        return None

    a, b = text_a.lower(), text_b.lower()
    for neg, pos in _NEGATION_PAIRS:
        if (neg in a and pos in b and neg not in b) or (
            neg in b and pos in a and neg not in a
        ):
            return f"Negation detected: '{neg}' vs '{pos}'"
    return None


# Numbers that are part of dates, reference codes, or section numbers are
# typically *not* the kind of numeric facts that contradict each other.
_DATE_PATTERN = re.compile(
    r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE,
)
_REF_CODE_PATTERN = re.compile(
    r"(?:#|Incident|WDC|WHS|SOP|Ref|Case|DOB)[:\-\s]*[\w\-]+",
    re.IGNORECASE,
)
_SECTION_NUM_PATTERN = re.compile(r"^\s*\d+\.\s")

# Stopwords / noise words to ignore when checking subject overlap
_STOPWORDS = frozenset(
    "the a an is was were are been be have has had do does did will would "
    "shall should may might can could of in on at to for with by from and "
    "or but not that this these those it its he she they his her their "
    "him them who which what when where how than more most very also as".split()
)


def _content_words(text: str) -> set[str]:
    """Extract meaningful content words (>4 chars, not stopwords)."""
    words = set(re.findall(r"[a-zA-Z]{4,}", text.lower()))
    return words - _STOPWORDS


def _texts_share_subject(text_a: str, text_b: str, min_shared: int = 2) -> bool:
    """Check whether two texts share enough content words to be about the
    same subject.  Returns True if they share at least *min_shared* meaningful
    content words.
    """
    words_a = _content_words(text_a)
    words_b = _content_words(text_b)
    shared = words_a & words_b
    return len(shared) >= min_shared


def _strip_noise_numbers(text: str) -> str:
    """Remove dates, reference codes, and section numbers before extracting
    numeric values so they don't pollute the comparison."""
    text = _DATE_PATTERN.sub("", text)
    text = _REF_CODE_PATTERN.sub("", text)
    text = _SECTION_NUM_PATTERN.sub("", text)
    return text


def _check_numeric_conflict(text_a: str, text_b: str) -> str | None:
    """Return a description if the two texts cite genuinely conflicting numbers.

    Improvements over the naive version:
    1. The two texts must share a subject (overlapping content words).
    2. Dates, reference codes, and section numbers are stripped before
       extracting numeric values.
    3. Single-digit numbers and very short matches are ignored.
    """
    if not _texts_share_subject(text_a, text_b):
        return None

    cleaned_a = _strip_noise_numbers(text_a)
    cleaned_b = _strip_noise_numbers(text_b)

    # Only extract numbers that look like quantities / values (at least 2 digits
    # or a dollar/percentage sign)
    nums_a = set(re.findall(r"\$[\d,]+\.?\d*|\b\d{2,}(?:\.\d+)?%?\b", cleaned_a))
    nums_b = set(re.findall(r"\$[\d,]+\.?\d*|\b\d{2,}(?:\.\d+)?%?\b", cleaned_b))

    if not nums_a or not nums_b:
        return None

    # Only flag when there are numbers that appear in one but not the other
    # AND the sets actually overlap on at least one number context (so they're
    # talking about the same kind of quantity).
    only_a = nums_a - nums_b
    only_b = nums_b - nums_a
    if only_a and only_b:
        return f"Numeric conflict: {only_a} vs {only_b}"
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
    """Extract meaningful claim sentences from each section.

    Improved over the original: takes up to the first two substantive
    sentences per section (instead of just one) and prepends the section
    header so subject-gating works better.
    """
    claims: list[_Claim] = []
    for node_id, node in skeleton.items():
        content = kv_store.get(node_id) or ""
        if len(content.strip()) < 30:
            continue
        header = getattr(node, "header", node_id)
        sentences = re.split(r"[.!?]\s+", content[:600])
        added = 0
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20:
                # Prefix with header for better subject matching
                claim_text = f"{header}\n{sent}"
                claims.append(
                    _Claim(
                        text=claim_text,
                        section=header,
                        node_id=node_id,
                        doc_id=doc_id or "",
                    )
                )
                added += 1
                if added >= 2:
                    break
        if len(claims) >= max_claims:
            break
    return claims


def _heuristic_pairwise(
    claims: list[_Claim],
    out: list[FactContradiction],
    max_pairs: int,
) -> None:
    """Run negation and numeric checks on claim pairs.

    Only compares claims that share a subject (overlapping content words) to
    avoid the combinatorial explosion of comparing every date/number across
    unrelated sections.
    """
    checked = 0
    for i, c1 in enumerate(claims):
        for c2 in claims[i + 1:]:
            if checked >= max_pairs:
                return

            # Pre-filter: skip pairs that clearly aren't about the same thing
            if not _texts_share_subject(c1.text, c2.text):
                continue

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
                        confidence=0.5,
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
# Strategy 4: Structure-Parallel Section Matching
# ---------------------------------------------------------------------------


def _header_similarity(h1: str, h2: str) -> float:
    """Score header similarity using SequenceMatcher on normalised text."""
    n1 = re.sub(r"^\d+[\.\)]\s*", "", h1).strip().lower()
    n2 = re.sub(r"^\d+[\.\)]\s*", "", h2).strip().lower()
    if n1 == n2:
        return 1.0
    return SequenceMatcher(None, n1, n2).ratio()


def _structure_parallel_cross_doc(
    documents: list[tuple[str, dict[str, Any], Any]],
    llm_fn: Any | None,
    out: list[FactContradiction],
    header_threshold: float = 0.70,
    max_section_chars: int = 800,
) -> None:
    """Find sections with similar headers across documents and compare them.

    For example, if two expert reports both have a section called "Diagnosis",
    we extract the content from each and ask the LLM (or use heuristics) to
    find conflicts.  This is vastly more targeted than comparing random claims.
    """
    import json as _json

    if len(documents) < 2:
        return

    # Build a list of (doc_id, header, content) for every non-trivial section
    _SectionInfo = tuple  # (doc_id, node_id, header, content)

    sections: list[_SectionInfo] = []
    for doc_id, skeleton, kv_store in documents:
        for node_id, node in skeleton.items():
            content = kv_store.get(node_id) or ""
            if len(content.strip()) < 40:
                continue
            header = getattr(node, "header", node_id)
            sections.append((doc_id, node_id, header, content[:max_section_chars]))

    # Match sections with similar headers across different documents
    matched_pairs: list[tuple[_SectionInfo, _SectionInfo]] = []
    for i, s1 in enumerate(sections):
        for s2 in sections[i + 1:]:
            if s1[0] == s2[0]:  # same document
                continue
            if _header_similarity(s1[2], s2[2]) >= header_threshold:
                matched_pairs.append((s1, s2))

    if not matched_pairs:
        return

    logger.info(
        "structure_parallel_matches",
        matched_section_pairs=len(matched_pairs),
    )

    # Use LLM if available, else heuristic
    if llm_fn:
        _compare_parallel_sections_llm(matched_pairs, llm_fn, out)
    else:
        _compare_parallel_sections_heuristic(matched_pairs, out)


def _compare_parallel_sections_heuristic(
    pairs: list[tuple[tuple, tuple]],
    out: list[FactContradiction],
) -> None:
    """Heuristic comparison of parallel sections (negation + numeric)."""
    for s1, s2 in pairs:
        doc_id_1, _, header_1, content_1 = s1
        doc_id_2, _, header_2, content_2 = s2

        neg = _check_negation(content_1, content_2)
        if neg:
            out.append(FactContradiction(
                claim_1=content_1[:300],
                claim_2=content_2[:300],
                source_1=f"[{doc_id_1}] {header_1}",
                source_2=f"[{doc_id_2}] {header_2}",
                type="direct",
                confidence=0.70,
                explanation=f"Parallel sections '{header_1}' — {neg}",
            ))

        num = _check_numeric_conflict(content_1, content_2)
        if num:
            out.append(FactContradiction(
                claim_1=content_1[:300],
                claim_2=content_2[:300],
                source_1=f"[{doc_id_1}] {header_1}",
                source_2=f"[{doc_id_2}] {header_2}",
                type="numeric",
                confidence=0.60,
                explanation=f"Parallel sections '{header_1}' — {num}",
            ))


def _compare_parallel_sections_llm(
    pairs: list[tuple[tuple, tuple]],
    llm_fn: Any,
    out: list[FactContradiction],
) -> None:
    """LLM comparison of parallel sections (much higher quality)."""
    import json as _json

    # Batch up to 5 pairs per LLM call to bound cost
    for batch_start in range(0, len(pairs), 5):
        batch = pairs[batch_start:batch_start + 5]

        comparisons = []
        for idx, (s1, s2) in enumerate(batch, 1):
            doc_id_1, _, header_1, content_1 = s1
            doc_id_2, _, header_2, content_2 = s2
            comparisons.append(
                f"--- Pair {idx} ---\n"
                f"SECTION A [Doc: {doc_id_1[:12]}, Header: {header_1}]:\n"
                f"{content_1[:500]}\n\n"
                f"SECTION B [Doc: {doc_id_2[:12]}, Header: {header_2}]:\n"
                f"{content_2[:500]}\n"
            )

        prompt = (
            "You are a contradiction detector for legal / professional documents.\n"
            "Below are pairs of parallel sections from DIFFERENT documents that "
            "have similar headings. For each pair, determine if there is a genuine "
            "factual contradiction, disagreement, or conflicting opinion.\n\n"
            "IMPORTANT: Different dates, different authors, or different patients "
            "are NOT contradictions. Only flag genuine conflicts where the two "
            "sections make incompatible statements about the same fact or issue.\n\n"
            + "\n".join(comparisons) + "\n\n"
            "Return ONLY valid JSON (no markdown, no extra text):\n"
            '[{"pair": 1, "contradiction": true/false, "type": "direct|numeric|temporal|semantic", '
            '"confidence": 0.0-1.0, "explanation": "brief reason", '
            '"claim_a_summary": "...", "claim_b_summary": "..."}]'
        )

        try:
            raw = llm_fn(prompt)
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                continue
            results = _json.loads(match.group(0))
            for item in results:
                if not item.get("contradiction"):
                    continue
                pair_idx = item.get("pair", 1) - 1
                if pair_idx < 0 or pair_idx >= len(batch):
                    continue
                s1, s2 = batch[pair_idx]
                doc_id_1, _, header_1, _ = s1
                doc_id_2, _, header_2, _ = s2
                out.append(FactContradiction(
                    claim_1=item.get("claim_a_summary", s1[3][:200]),
                    claim_2=item.get("claim_b_summary", s2[3][:200]),
                    source_1=f"[{doc_id_1}] {header_1}",
                    source_2=f"[{doc_id_2}] {header_2}",
                    type=item.get("type", "semantic"),
                    confidence=item.get("confidence", 0.75),
                    explanation=item.get("explanation", ""),
                ))
        except Exception as exc:
            logger.debug("structure_parallel_llm_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Strategy 5: Entity-Centric Claim Comparison
# ---------------------------------------------------------------------------


def _entity_centric_cross_doc(
    kg: Any,
    documents: list[tuple[str, dict[str, Any], Any]],
    llm_fn: Any | None,
    out: list[FactContradiction],
) -> None:
    """For each entity that appears across multiple documents, gather the
    statements made about that entity in each document and compare them.

    This is the highest-signal strategy: it only compares passages that
    actually discuss the same real-world entity.
    """
    import json as _json
    from rnsr.extraction.models import EntityType

    # Build a map: entity_id -> list of (doc_id, node_id, header, content)
    entity_doc_passages: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)

    # Index doc content by node_id for quick lookup
    doc_content: dict[str, dict[str, tuple[str, str]]] = {}  # doc_id -> {node_id -> (header, content)}
    for doc_id, skeleton, kv_store in documents:
        node_map: dict[str, tuple[str, str]] = {}
        for node_id, node in skeleton.items():
            content = kv_store.get(node_id) or ""
            if len(content.strip()) < 30:
                continue
            header = getattr(node, "header", node_id)
            node_map[node_id] = (header, content[:600])
        doc_content[doc_id] = node_map

    # For each entity in the KG, find which documents mention it
    all_entity_types = ["person", "organization", "legal_concept", "event", "other",
                        "date", "location", "reference", "monetary", "document"]
    all_entities = []
    for etype_name in all_entity_types:
        try:
            etype = EntityType(etype_name)
            all_entities.extend(kg.find_entities_by_type(etype))
        except (ValueError, Exception):
            continue

    # Group by cross-doc presence
    for entity in all_entities:
        doc_ids_for_entity: set[str] = set()
        for mention in entity.mentions:
            if mention.doc_id and mention.doc_id in doc_content:
                node_info = doc_content[mention.doc_id].get(mention.node_id)
                if node_info:
                    header, content = node_info
                    entity_doc_passages[entity.id].append(
                        (mention.doc_id, mention.node_id, header, content)
                    )
                    doc_ids_for_entity.add(mention.doc_id)

        # Also check linked entities (cross-doc resolution)
        try:
            linked = kg.find_entity_across_documents(entity.id, min_confidence=0.5)
            for linked_ent in linked:
                if linked_ent.id == entity.id:
                    continue
                for mention in linked_ent.mentions:
                    if mention.doc_id and mention.doc_id in doc_content:
                        node_info = doc_content[mention.doc_id].get(mention.node_id)
                        if node_info:
                            header, content = node_info
                            entity_doc_passages[entity.id].append(
                                (mention.doc_id, mention.node_id, header, content)
                            )
        except Exception:
            pass

    # Now for each entity with cross-doc passages, compare them
    cross_doc_entities = {
        eid: passages
        for eid, passages in entity_doc_passages.items()
        if len({p[0] for p in passages}) >= 2  # must span 2+ docs
    }

    if not cross_doc_entities:
        return

    logger.info(
        "entity_centric_cross_doc_candidates",
        entities_spanning_docs=len(cross_doc_entities),
    )

    for entity_id, passages in cross_doc_entities.items():
        entity = kg.get_entity(entity_id)
        entity_name = entity.canonical_name if entity else entity_id

        # Deduplicate passages by (doc_id, node_id)
        seen_keys: set[str] = set()
        unique_passages: list[tuple[str, str, str, str]] = []
        for p in passages:
            key = f"{p[0]}|{p[1]}"
            if key not in seen_keys:
                seen_keys.add(key)
                unique_passages.append(p)

        # Group by document
        by_doc: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
        for p in unique_passages:
            by_doc[p[0]].append(p)

        doc_ids = list(by_doc.keys())
        if len(doc_ids) < 2:
            continue

        if llm_fn:
            _compare_entity_passages_llm(
                entity_name, by_doc, doc_ids, llm_fn, out
            )
        else:
            _compare_entity_passages_heuristic(
                entity_name, by_doc, doc_ids, out
            )


def _compare_entity_passages_heuristic(
    entity_name: str,
    by_doc: dict[str, list[tuple[str, str, str, str]]],
    doc_ids: list[str],
    out: list[FactContradiction],
) -> None:
    """Heuristic entity-centric comparison."""
    for i, d1 in enumerate(doc_ids):
        for d2 in doc_ids[i + 1:]:
            for p1 in by_doc[d1]:
                for p2 in by_doc[d2]:
                    neg = _check_negation(p1[3], p2[3])
                    if neg:
                        out.append(FactContradiction(
                            claim_1=p1[3][:300],
                            claim_2=p2[3][:300],
                            source_1=f"[{d1}] {p1[2]}",
                            source_2=f"[{d2}] {p2[2]}",
                            type="direct",
                            confidence=0.70,
                            explanation=f"Entity '{entity_name}' — {neg}",
                        ))


def _compare_entity_passages_llm(
    entity_name: str,
    by_doc: dict[str, list[tuple[str, str, str, str]]],
    doc_ids: list[str],
    llm_fn: Any,
    out: list[FactContradiction],
) -> None:
    """LLM entity-centric comparison -- highest quality."""
    import json as _json

    # Build a summary of what each document says about this entity
    doc_summaries: list[str] = []
    doc_id_map: list[str] = []
    header_map: list[str] = []
    for doc_id in doc_ids:
        passages = by_doc[doc_id]
        combined = "\n".join(
            f"[{p[2]}]: {p[3][:300]}"
            for p in passages[:4]  # max 4 sections per doc
        )
        doc_summaries.append(
            f"DOCUMENT {len(doc_summaries)+1} [{doc_id[:12]}]:\n{combined}"
        )
        doc_id_map.append(doc_id)
        header_map.append(passages[0][2] if passages else "")

    if len(doc_summaries) < 2:
        return

    prompt = (
        f"You are analysing statements about the entity \"{entity_name}\" "
        f"across {len(doc_summaries)} different documents.\n\n"
        "Identify any genuine contradictions, disagreements, or conflicting "
        "claims about this entity. Different dates or different contexts are "
        "NOT contradictions. Only flag where two documents make incompatible "
        "claims about the SAME aspect of this entity.\n\n"
        + "\n\n".join(doc_summaries) + "\n\n"
        "Return ONLY valid JSON (no markdown):\n"
        '[{"doc_a": 1, "doc_b": 2, "type": "direct|numeric|temporal|semantic", '
        '"confidence": 0.0-1.0, "explanation": "brief reason", '
        '"claim_a": "what doc A says", "claim_b": "what doc B says"}]'
    )

    try:
        raw = llm_fn(prompt)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return
        results = _json.loads(match.group(0))
        for item in results:
            idx_a = item.get("doc_a", 1) - 1
            idx_b = item.get("doc_b", 2) - 1
            if idx_a < 0 or idx_a >= len(doc_id_map):
                continue
            if idx_b < 0 or idx_b >= len(doc_id_map):
                continue
            out.append(FactContradiction(
                claim_1=item.get("claim_a", ""),
                claim_2=item.get("claim_b", ""),
                source_1=f"[{doc_id_map[idx_a]}] re: {entity_name}",
                source_2=f"[{doc_id_map[idx_b]}] re: {entity_name}",
                type=item.get("type", "semantic"),
                confidence=item.get("confidence", 0.80),
                explanation=f"Entity '{entity_name}': {item.get('explanation', '')}",
            ))
    except Exception as exc:
        logger.debug("entity_centric_llm_failed", entity=entity_name, error=str(exc))


# ---------------------------------------------------------------------------
# Strategy 6: Relationship Divergence Detection
# ---------------------------------------------------------------------------


def _relationship_divergence_cross_doc(
    kg: Any,
    out: list[FactContradiction],
) -> None:
    """Compare KG relationships for the same entity across documents.

    If entity A has relationship R1 in Doc1 and relationship R2 in Doc2,
    and R1 and R2 are contradictory (e.g. SUPPORTS vs CONTRADICTS, or
    the same relationship type with conflicting targets/evidence),
    surface them.
    """
    from rnsr.extraction.models import EntityType, RelationType

    # Gather all entities
    all_entities = []
    for etype_name in ("person", "organization", "legal_concept", "event", "other"):
        try:
            etype = EntityType(etype_name)
            all_entities.extend(kg.find_entities_by_type(etype))
        except (ValueError, Exception):
            continue

    # For each entity, check if it has cross-doc mentions and contradictory relationships
    for entity in all_entities:
        # Check for linked versions across docs
        try:
            linked_entities = kg.find_entity_across_documents(entity.id, min_confidence=0.5)
        except Exception:
            linked_entities = [entity]

        # Collect all relationships for this entity cluster
        all_rels: list[tuple[str, Any]] = []  # (doc_id, relationship)
        seen_rel_ids: set[str] = set()
        for ent in linked_entities:
            try:
                rels = kg.get_entity_relationships(ent.id)
                for rel in rels:
                    if rel.id not in seen_rel_ids:
                        seen_rel_ids.add(rel.id)
                        all_rels.append((rel.doc_id or ent.source_doc_id or "", rel))
            except Exception:
                continue

        if len(all_rels) < 2:
            continue

        # Look for contradictory relationship patterns
        # Pattern 1: SUPPORTS in one doc, CONTRADICTS in another
        supports = [(d, r) for d, r in all_rels if r.type == RelationType.SUPPORTS]
        contradicts = [(d, r) for d, r in all_rels if r.type == RelationType.CONTRADICTS]

        for d1, r1 in supports:
            for d2, r2 in contradicts:
                if d1 == d2:
                    continue
                target1 = kg.get_entity(r1.target_id) if hasattr(kg, "get_entity") else None
                target2 = kg.get_entity(r2.target_id) if hasattr(kg, "get_entity") else None
                t1_name = target1.canonical_name if target1 else r1.target_id
                t2_name = target2.canonical_name if target2 else r2.target_id

                out.append(FactContradiction(
                    claim_1=f"{entity.canonical_name} supports: {t1_name}",
                    claim_2=f"{entity.canonical_name} contradicts: {t2_name}",
                    source_1=f"[{d1}] KG relationship",
                    source_2=f"[{d2}] KG relationship",
                    type="direct",
                    confidence=0.80,
                    explanation=(
                        f"Entity '{entity.canonical_name}' has a SUPPORTS relationship "
                        f"in doc {d1[:12]} but a CONTRADICTS relationship in doc {d2[:12]}"
                    ),
                ))

        # Pattern 2: Same relationship type but conflicting evidence text
        rels_by_type: dict[str, list[tuple[str, Any]]] = defaultdict(list)
        for d, r in all_rels:
            rels_by_type[r.type.value].append((d, r))

        for rel_type, type_rels in rels_by_type.items():
            if len(type_rels) < 2:
                continue
            for i, (d1, r1) in enumerate(type_rels):
                for d2, r2 in type_rels[i + 1:]:
                    if d1 == d2:
                        continue
                    # Check if the evidence texts are contradictory
                    if r1.evidence and r2.evidence and len(r1.evidence) > 20 and len(r2.evidence) > 20:
                        neg = _check_negation(r1.evidence, r2.evidence)
                        if neg:
                            out.append(FactContradiction(
                                claim_1=r1.evidence[:300],
                                claim_2=r2.evidence[:300],
                                source_1=f"[{d1}] KG: {entity.canonical_name} → {rel_type}",
                                source_2=f"[{d2}] KG: {entity.canonical_name} → {rel_type}",
                                type="direct",
                                confidence=0.75,
                                explanation=(
                                    f"Entity '{entity.canonical_name}' has contradictory "
                                    f"'{rel_type}' relationship evidence across documents"
                                ),
                            ))


# ---------------------------------------------------------------------------
# Cross-document contradiction detection (heuristic helpers)
# ---------------------------------------------------------------------------


def _heuristic_pairwise_cross_doc(
    claims: list[_Claim],
    out: list[FactContradiction],
    max_pairs: int,
) -> None:
    """Run negation and numeric checks on claim pairs from *different* documents only.

    Only compares claims that share a subject (overlapping content words) to
    avoid the combinatorial explosion of flagging every unrelated pair that
    happens to contain different numbers.
    """
    checked = 0
    for i, c1 in enumerate(claims):
        for c2 in claims[i + 1:]:
            # Only compare claims from different documents
            if c1.doc_id == c2.doc_id:
                continue

            # Pre-filter: skip pairs that clearly aren't about the same thing
            if not _texts_share_subject(c1.text, c2.text):
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
                        confidence=0.65,
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
                        confidence=0.5,
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

    Uses six complementary strategies:

    1. **KG CONTRADICTS** -- Explicit contradiction relationships in the KG.
    2. **Heuristic pairwise** -- Subject-gated negation/numeric checks.
    3. **LLM semantic** -- Broad semantic scan of all claims (optional).
    4. **Structure-Parallel** -- Match sections with similar headers across
       documents (e.g. two "Diagnosis" sections) and compare them.
    5. **Entity-Centric** -- Group passages by the KG entities they mention,
       then compare only passages about the same real-world entity.
    6. **Relationship Divergence** -- Compare KG relationships for the same
       linked entity across documents.

    Args:
        kg: A workspace-wide ``KnowledgeGraph`` that contains entities
            from all documents (e.g. from ``DocumentStore.get_workspace_kg()``).
        documents: A list of ``(doc_id, skeleton, kv_store)`` tuples —
            one per document to include in the analysis.
        llm_fn: Optional ``Callable[[str], str]`` for semantic detection.
            When provided, strategies 3-5 use the LLM for higher-quality
            results.  When ``None``, heuristic fallbacks are used.
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

    # ------------------------------------------------------------------
    # Strategy 4: Structure-Parallel section matching
    # ------------------------------------------------------------------
    try:
        _structure_parallel_cross_doc(documents, llm_fn, contradictions)
    except Exception as exc:
        logger.debug("structure_parallel_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Strategy 5: Entity-Centric claim comparison
    # ------------------------------------------------------------------
    try:
        _entity_centric_cross_doc(kg, documents, llm_fn, contradictions)
    except Exception as exc:
        logger.debug("entity_centric_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Strategy 6: Relationship Divergence detection
    # ------------------------------------------------------------------
    try:
        _relationship_divergence_cross_doc(kg, contradictions)
    except Exception as exc:
        logger.debug("relationship_divergence_failed", error=str(exc))

    # Deduplicate and sort
    contradictions = _deduplicate(contradictions)
    contradictions.sort(key=lambda c: c.confidence, reverse=True)

    logger.info(
        "cross_doc_contradictions_detected",
        total=len(contradictions),
        documents=len(documents),
        claims_analysed=len(all_claims),
        strategies_used="1-6" if llm_fn else "1-2,4-6 (heuristic)",
        by_type={
            t: sum(1 for c in contradictions if c.type == t)
            for t in {"direct", "numeric", "temporal", "semantic"}
        },
    )
    return contradictions
