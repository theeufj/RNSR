"""
Timeline Extractor -- build chronological timelines from knowledge graphs.

Extracts temporal events via two complementary strategies:

Strategy 1 -- KG-Derived (existing):
    Find DATE entities and HAS_DATE / TEMPORAL_* relationships in the KG.

Strategy 2 -- Direct Text Scan (new):
    A targeted LLM pass over raw section text that specifically asks
    "what events happened and when?"  This catches events that the
    general-purpose entity extractor missed because its prompt was
    not timeline-focused.

Usage:
    from rnsr.extraction.timeline_extractor import extract_timeline

    # KG-only (fast, relies on what the extractor found)
    events = extract_timeline(kg)

    # KG + direct text scan (higher recall)
    events = extract_timeline(kg, skeleton=skeleton, kv_store=kv_store)
"""

from __future__ import annotations

import json as _json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TimelineEvent:
    """A single event on a timeline."""

    date_str: str
    """Raw date string as it appeared in the document."""
    date_parsed: datetime | None
    """Parsed datetime (may be None if parsing failed)."""
    description: str
    """Human-readable description of the event."""
    entities_involved: list[str] = field(default_factory=list)
    """Canonical names of entities involved."""
    doc_id: str = ""
    """Source document ID."""
    node_id: str = ""
    """Source skeleton node ID."""
    confidence: float = 1.0
    """Confidence score (0.0 -- 1.0)."""
    relationship_type: str = ""
    """KG relationship type that produced this event."""

    @property
    def sort_key(self) -> tuple:
        """Key for chronological sorting.  Events without a parsed date
        are placed at the end."""
        if self.date_parsed:
            return (0, self.date_parsed)
        return (1, datetime.max)


# ---------------------------------------------------------------------------
# Date parsing helpers
# ---------------------------------------------------------------------------

# Common date patterns found in legal / financial documents.
_DATE_PATTERNS: list[tuple[str, str]] = [
    # ISO: 2024-03-15
    (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),
    # US: March 15, 2024 / Mar 15, 2024
    (r"[A-Z][a-z]+ \d{1,2},? \d{4}", None),  # type: ignore[arg-type]
    # UK/AU: 15 March 2024
    (r"\d{1,2} [A-Z][a-z]+ \d{4}", None),  # type: ignore[arg-type]
    # Compact: 15/03/2024, 03/15/2024
    (r"\d{1,2}/\d{1,2}/\d{4}", None),  # type: ignore[arg-type]
]


def _parse_date(text: str) -> datetime | None:
    """Best-effort date parsing.  Returns ``None`` on failure."""
    if not text or not text.strip():
        return None

    text = text.strip()

    # Try dateutil first (handles most formats)
    try:
        from dateutil import parser as dateutil_parser

        return dateutil_parser.parse(text, dayfirst=False, fuzzy=True)
    except Exception:
        pass

    # Fallback: manual patterns
    for pattern, fmt in _DATE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            matched = m.group(0)
            if fmt:
                try:
                    return datetime.strptime(matched, fmt)
                except ValueError:
                    continue
            else:
                try:
                    from dateutil import parser as dateutil_parser

                    return dateutil_parser.parse(matched, dayfirst=False)
                except Exception:
                    continue

    return None


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Regex date pre-scan  (Layer 4: Source Grounding)
# ---------------------------------------------------------------------------

# Broad date-like patterns to pre-extract from source text.
_GROUNDING_DATE_PATTERNS: list[str] = [
    # ISO: 2024-03-15
    r"\d{4}-\d{2}-\d{2}",
    # UK/AU: 15 March 2024, 3 July 2022
    r"\d{1,2}\s+[A-Z][a-z]+\s+\d{4}",
    # US: March 15, 2024
    r"[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}",
    # Short month: 15 Mar 2024, Mar 15, 2024
    r"\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4}",
    r"[A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4}",
    # Compact: 15/03/2024, 03/15/2024, 2024/03/15
    r"\d{1,2}/\d{1,2}/\d{4}",
    r"\d{4}/\d{1,2}/\d{1,2}",
    # Quarter: Q1 2024, Q2 2025
    r"Q[1-4]\s+\d{4}",
    # Month-Year: January 2024, Feb 2024
    r"[A-Z][a-z]+\s+\d{4}",
    # Year-only (as last resort, captured but lower priority)
    r"\b(?:19|20)\d{2}\b",
]


def _regex_date_scan(content: str) -> list[str]:
    """Pre-extract all date-like strings from *content* using regex.

    Returns a deduplicated list sorted by position in the text.
    """
    found: dict[str, int] = {}  # date_str -> first_position
    for pattern in _GROUNDING_DATE_PATTERNS:
        for m in re.finditer(pattern, content):
            date_str = m.group(0).strip()
            if date_str not in found:
                found[date_str] = m.start()
    # Sort by first occurrence in the text
    return sorted(found, key=lambda d: found[d])


def _is_date_grounded(date_str: str, source_content: str) -> bool:
    """Check whether *date_str* (or a close variant) appears in *source_content*.

    This prevents the LLM from hallucinating dates that don't exist in the text.
    """
    if not date_str or not source_content:
        return False
    # Exact substring match (case-insensitive)
    if date_str.lower() in source_content.lower():
        return True
    # Try matching just the numeric components (handles "15 Jan 2024" vs "15 January 2024")
    nums = re.findall(r"\d+", date_str)
    if len(nums) >= 2 and all(n in source_content for n in nums):
        return True
    return False


_TIMELINE_SCAN_PROMPT = """You are a precise timeline extractor.  Given a document section, extract EVERY event that has a date or temporal marker.

Section Header: {header}

Section Content:
---
{content}
---

Dates found in this section (use ONLY these dates, do NOT invent new ones):
{grounded_dates}

For EACH event, return:
- "date": the date string exactly as written (e.g. "15 January 2019", "Q2 2025")
- "description": a concise one-sentence description of what happened
- "entities": list of people, organisations, or things involved

Return ONLY valid JSON -- no markdown, no code fences:
{{
  "events": [
    {{"date": "...", "description": "...", "entities": ["..."]}}
  ]
}}

RULES:
1. Extract ALL events, even minor ones.  Do not summarise or skip.
2. One event per date.  If multiple things happened on the same date, create separate entries.
3. Use the exact date string from the text, do not infer dates.
4. If a date range is given (e.g. "between Nov 2023 and Feb 2024"), use the start date.
5. ONLY use dates from the "Dates found" list above.  Do NOT hallucinate dates.

JSON only:"""


def _direct_text_scan(
    skeleton: dict[str, Any],
    kv_store: Any,
    doc_id: str = "",
    llm_fn: Any | None = None,
) -> list[TimelineEvent]:
    """Strategy 2: scan raw section text with a dedicated timeline prompt.

    This catches events the general-purpose extractor missed because its
    prompt was person-centric and didn't include event/temporal types.

    Args:
        skeleton: The skeleton index (node_id -> SkeletonNode).
        kv_store: The KV store for looking up section content.
        doc_id: Document ID for provenance.
        llm_fn: A ``Callable[[str], str]``.  If None, will try ``get_llm()``.

    Returns:
        List of TimelineEvent (unsorted -- caller sorts).
    """
    if llm_fn is None:
        try:
            from rnsr.llm import get_llm
            _llm = get_llm()
            # Prefer JSON-mode completion for structured timeline output
            _json_fn = getattr(_llm, "complete_json", None) or _llm.complete
            llm_fn = lambda prompt: str(_json_fn(prompt))
        except Exception as exc:
            logger.warning("timeline_scan_no_llm", error=str(exc))
            return []

    events: list[TimelineEvent] = []

    for node_id, node in skeleton.items():
        content = ""
        if hasattr(kv_store, "get"):
            content = kv_store.get(node_id) or ""
        elif isinstance(kv_store, dict):
            content = kv_store.get(node_id, "")

        content = content.strip()
        if len(content) < 30:
            continue

        # Skip root nodes that are just document titles with no content
        header = getattr(node, "header", node_id) or node_id

        # --- Grounding: pre-extract dates from source text ---
        grounded_dates = _regex_date_scan(content)
        grounded_dates_str = ", ".join(grounded_dates) if grounded_dates else "(none found)"

        prompt = _TIMELINE_SCAN_PROMPT.format(
            header=header,
            content=content,
            grounded_dates=grounded_dates_str,
        )

        try:
            raw = llm_fn(prompt).strip()
            # Strip markdown fences
            if raw.startswith("```"):
                lines = raw.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw = "\n".join(lines)

            data = _json.loads(raw)
            for item in data.get("events", []):
                date_str = item.get("date", "")
                description = item.get("description", "")
                entities_involved = item.get("entities", [])

                if not date_str and not description:
                    continue

                # --- Grounding check: reject hallucinated dates ---
                if date_str and not _is_date_grounded(date_str, content):
                    logger.debug(
                        "timeline_event_date_not_grounded",
                        date_str=date_str,
                        node_id=node_id,
                    )
                    continue  # Skip events with hallucinated dates

                date_parsed = _parse_date(date_str) if date_str else None

                events.append(
                    TimelineEvent(
                        date_str=date_str,
                        date_parsed=date_parsed,
                        description=description,
                        entities_involved=entities_involved,
                        doc_id=doc_id,
                        node_id=node_id,
                        confidence=0.9,
                        relationship_type="direct_scan",
                    )
                )

        except Exception as exc:
            logger.debug(
                "timeline_scan_section_failed",
                node_id=node_id,
                error=str(exc),
            )

    logger.info(
        "timeline_direct_scan_complete",
        events_found=len(events),
        sections_scanned=len(skeleton),
    )
    return events


def _event_quality_score(ev: TimelineEvent) -> tuple:
    """Score an event for deduplication preference.

    Higher is better.  Prefers:
    1. direct_scan events (richer, timeline-focused descriptions)
    2. longer descriptions (more informative)
    3. higher confidence
    """
    is_scan = 1 if ev.relationship_type == "direct_scan" else 0
    desc_len = len(ev.description)
    return (is_scan, desc_len, ev.confidence)


def _deduplicate_events(events: list[TimelineEvent]) -> list[TimelineEvent]:
    """Deduplicate events by (date_parsed, description-keywords).

    When the same event is found by both the KG strategy and the direct
    text scan, keep the one with the best quality (direct_scan preferred,
    then longer description, then higher confidence).
    """
    # Group events by dedup key, keeping the best in each group
    groups: dict[str, TimelineEvent] = {}

    for ev in events:
        # Build a dedup key from the parsed date + first few content words
        date_key = ev.date_parsed.isoformat() if ev.date_parsed else ev.date_str
        # Normalise description to a few keywords
        desc_words = sorted(
            set(w.lower() for w in re.findall(r"[a-zA-Z]{3,}", ev.description[:200]))
        )[:5]
        dedup_key = f"{date_key}|{'|'.join(desc_words)}"

        if dedup_key not in groups:
            groups[dedup_key] = ev
        else:
            # Keep the higher-quality event
            if _event_quality_score(ev) > _event_quality_score(groups[dedup_key]):
                groups[dedup_key] = ev

    return list(groups.values())


def extract_timeline(
    kg: Any,  # KnowledgeGraph
    doc_ids: list[str] | None = None,
    skeleton: dict[str, Any] | None = None,
    kv_store: Any | None = None,
    llm_fn: Any | None = None,
) -> list[TimelineEvent]:
    """Extract a chronological timeline from a knowledge graph.

    Uses two complementary strategies:

    1. **KG-Derived**: DATE/EVENT entities + HAS_DATE/TEMPORAL_* relationships.
    2. **Direct Text Scan** (optional): A targeted LLM pass over raw section
       text that specifically asks for events and dates.  Activated when
       ``skeleton`` and ``kv_store`` are provided.

    Args:
        kg: A ``KnowledgeGraph`` (or ``InMemoryKnowledgeGraph``) instance.
        doc_ids: Restrict to specific document IDs (default: all).
        skeleton: Optional skeleton index for direct text scan (Strategy 2).
        kv_store: Optional KV store for direct text scan (Strategy 2).
        llm_fn: Optional ``Callable[[str], str]`` for Strategy 2.

    Returns:
        List of :class:`TimelineEvent` sorted chronologically.
    """
    from rnsr.extraction.models import EntityType, RelationType

    events: list[TimelineEvent] = []
    seen_keys: set[str] = set()  # (entity_id, date_str) to deduplicate

    # ── Strategy 1: KG-Derived events ──────────────────────────────────

    # 1a. Collect all DATE entities
    date_entities = kg.find_entities_by_type(EntityType.DATE)
    if doc_ids:
        date_entities = [
            e for e in date_entities if e.source_doc_id in doc_ids
        ]

    for date_entity in date_entities:
        date_str = date_entity.canonical_name
        date_parsed = _parse_date(date_str)

        # Find all relationships pointing to this date
        rels = kg.get_entity_relationships(date_entity.id, direction="incoming")
        for rel in rels:
            if rel.type not in (
                RelationType.HAS_DATE,
                RelationType.TEMPORAL_BEFORE,
                RelationType.TEMPORAL_AFTER,
            ):
                continue

            dedup_key = f"{rel.source_id}|{date_str}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # Resolve the source entity name
            source_entity = kg.get_entity(rel.source_id) if hasattr(kg, "get_entity") else None
            source_name = source_entity.canonical_name if source_entity else rel.source_id

            description = f"{source_name} -- {date_str}"
            if rel.evidence:
                description = rel.evidence

            events.append(
                TimelineEvent(
                    date_str=date_str,
                    date_parsed=date_parsed,
                    description=description,
                    entities_involved=[source_name],
                    doc_id=rel.doc_id or "",
                    confidence=rel.confidence,
                    relationship_type=rel.type.value,
                )
            )

    # 1b. Also scan EVENT entities with embedded dates
    event_entities = kg.find_entities_by_type(EntityType.EVENT)
    if doc_ids:
        event_entities = [
            e for e in event_entities if e.source_doc_id in doc_ids
        ]

    for event_entity in event_entities:
        date_parsed = _parse_date(event_entity.canonical_name)
        if date_parsed is None:
            # Check metadata for date
            meta_date = event_entity.metadata.get("date", "")
            if meta_date:
                date_parsed = _parse_date(meta_date)

        dedup_key = f"event|{event_entity.id}"
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)

        events.append(
            TimelineEvent(
                date_str=event_entity.canonical_name,
                date_parsed=date_parsed,
                description=event_entity.canonical_name,
                entities_involved=[event_entity.canonical_name],
                doc_id=event_entity.source_doc_id or "",
                confidence=1.0,
                relationship_type="event",
            ),
        )

    kg_count = len(events)

    # ── Strategy 2: Direct Text Scan ───────────────────────────────────

    if skeleton is not None and kv_store is not None:
        doc_id = doc_ids[0] if doc_ids else ""
        scan_events = _direct_text_scan(
            skeleton=skeleton,
            kv_store=kv_store,
            doc_id=doc_id,
            llm_fn=llm_fn,
        )
        events.extend(scan_events)

    # ── Deduplicate and sort ───────────────────────────────────────────

    events = _deduplicate_events(events)
    events.sort(key=lambda e: e.sort_key)

    logger.info(
        "timeline_extracted",
        total_events=len(events),
        kg_events=kg_count,
        scan_events=len(events) - kg_count if skeleton else 0,
        dated_events=sum(1 for e in events if e.date_parsed is not None),
    )
    return events


def format_timeline(events: list[TimelineEvent]) -> str:
    """Format a timeline as a human-readable string."""
    if not events:
        return "No timeline events found."

    lines: list[str] = []
    for i, ev in enumerate(events, 1):
        date_label = ev.date_str if ev.date_str else "Unknown date"
        lines.append(f"{i}. [{date_label}] {ev.description}")
        if ev.entities_involved:
            lines.append(f"   Entities: {', '.join(ev.entities_involved)}")
        if ev.doc_id:
            lines.append(f"   Source: {ev.doc_id}")

    return "\n".join(lines)
