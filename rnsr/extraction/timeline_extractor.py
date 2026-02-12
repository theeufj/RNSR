"""
Timeline Extractor -- build chronological timelines from knowledge graphs.

Extracts temporal events by:
1. Finding DATE entities and HAS_DATE / TEMPORAL_* relationships in the KG
2. Parsing date strings into sortable values
3. Building a list of TimelineEvent objects sorted chronologically

Usage:
    from rnsr.extraction.timeline_extractor import extract_timeline

    events = extract_timeline(kg)
    for ev in events:
        print(f"{ev.date_str}  {ev.description}")
"""

from __future__ import annotations

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


def extract_timeline(
    kg: Any,  # KnowledgeGraph
    doc_ids: list[str] | None = None,
) -> list[TimelineEvent]:
    """Extract a chronological timeline from a knowledge graph.

    The function looks for:
    - Entities of type DATE
    - Relationships of type HAS_DATE, TEMPORAL_BEFORE, TEMPORAL_AFTER

    Args:
        kg: A ``KnowledgeGraph`` (or ``InMemoryKnowledgeGraph``) instance.
        doc_ids: Restrict to specific document IDs (default: all).

    Returns:
        List of :class:`TimelineEvent` sorted chronologically.
    """
    from rnsr.extraction.models import EntityType, RelationType

    events: list[TimelineEvent] = []
    seen_keys: set[str] = set()  # (entity_id, date_str) to deduplicate

    # 1. Collect all DATE entities
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

    # 2. Also scan EVENT entities with embedded dates
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

    # 3. Sort chronologically
    events.sort(key=lambda e: e.sort_key)

    logger.info(
        "timeline_extracted",
        total_events=len(events),
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
