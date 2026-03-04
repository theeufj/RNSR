"""
KG-First Query Resolver

Attempts to answer factual questions directly from the Knowledge Graph
and Document Profiles before falling back to expensive tree navigation.

When a direct answer isn't possible, returns entity mention node_ids
so the navigator can jump straight to the most relevant sections.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Intent classification — deterministic keyword matching
# ---------------------------------------------------------------------------

@dataclass
class _IntentPattern:
    intent: str
    patterns: list[re.Pattern[str]]


_INTENT_PATTERNS: list[_IntentPattern] = [
    _IntentPattern(
        "judge",
        [
            re.compile(r"\b(?:who\s+(?:is|was)\s+the\s+judge|before\s+whom|presid(?:ed|ing)\s+judge|which\s+judge)", re.I),
            re.compile(r"\bjudge\b.*\b(?:name|who)\b", re.I),
        ],
    ),
    _IntentPattern(
        "citation",
        [
            re.compile(r"\b(?:neutral\s+citation|case\s+citation|citation\s+number|medium\s+neutral\s+citation)\b", re.I),
            re.compile(r"\bwhat\s+is\s+the\s+(?:neutral\s+)?citation\b", re.I),
        ],
    ),
    _IntentPattern(
        "court",
        [
            # Only match metadata-style court questions ("which court heard this"),
            # NOT document-content questions ("which courts are identified/listed in section X")
            re.compile(r"\b(?:which\s+court|what\s+court|name\s+of\s+(?:the\s+)?court)(?!\s+order)(?!s?\s+(?:are|is|were)\s+(?:identified|listed|mentioned|named|set out))", re.I),
            re.compile(r"\bcourt\s+(?:was|that|where)\b", re.I),
        ],
    ),
    _IntentPattern(
        "jurisdiction",
        [
            re.compile(r"\b(?:what\s+jurisdiction|type\s+of\s+jurisdiction|jurisdiction\s+(?:is|was))\b", re.I),
        ],
    ),
    _IntentPattern(
        "date",
        [
            # Only match document-level date questions (judgment, hearing),
            # NOT event-specific dates ("when was X assessed", "last updated")
            re.compile(r"\bwhen\s+was\s+(?:the\s+)?(?:judgment|order|hearing)\s+(?:given|made|delivered|handed\s+down)\b", re.I),
            re.compile(r"\bdate\s+(?:of|was)\s+(?:the\s+)?(?:judgment|order|hearing)\b", re.I),
            re.compile(r"\bwhen\s+was\s+judgment\s+given\b", re.I),
        ],
    ),
    _IntentPattern(
        "parties",
        [
            re.compile(r"\bwho\s+(?:is|are|was|were)\s+the\s+(?:applicant|respondent|plaintiff|defendant|appellant|claimant|parties)\b", re.I),
        ],
    ),
    _IntentPattern(
        "page_count",
        [
            re.compile(r"\bhow\s+many\s+pages\b", re.I),
        ],
    ),
]

# Regex to extract which party role is being asked about
_PARTY_ROLE_RE = re.compile(
    r"\bwho\s+(?:is|are|was|were)\s+the\s+(applicant|respondent|plaintiff|defendant|appellant|claimant)",
    re.I,
)


@dataclass
class KGResolution:
    """Result of a KG-first resolution attempt."""

    answer: str | None = None
    confidence: float = 0.0
    resolved: bool = False
    intent: str | None = None
    entity_node_ids: list[str] = field(default_factory=list)
    source_entity: str | None = None


class KGResolver:
    """Resolves factual questions from KG entities and document profiles."""

    def __init__(self, kg: Any, profiles: dict[str, Any] | None = None):
        """
        Args:
            kg: KnowledgeGraph instance.
            profiles: Map of doc_id -> DocumentProfile (or dict representation).
        """
        self.kg = kg
        self._profiles = profiles or {}

    def try_resolve(
        self,
        question: str,
        doc_ids: list[str] | None = None,
    ) -> KGResolution:
        """Attempt to answer *question* from KG entities and profiles.

        Returns a KGResolution with ``resolved=True`` if a high-confidence
        answer was found. Otherwise ``entity_node_ids`` contains mention
        node_ids that the navigator should visit first.
        """
        intent = self._classify_intent(question)
        if not intent:
            return self._entity_guidance(question, doc_ids)

        logger.info("kg_resolver_intent", intent=intent, question=question[:80])

        # For parties intent, extract the requested role and resolve
        # only if we can match the specific role.
        if intent == "parties":
            result = self._resolve_party_by_role(question, doc_ids)
            if result is not None:
                return result
            # If role-based resolution returned None (couldn't determine
            # role or no match), fall through to entity guidance.
            guidance = self._entity_guidance(question, doc_ids)
            guidance.intent = intent
            return guidance

        # Try profile-based resolution first
        answer = self._resolve_from_profiles(intent, doc_ids)
        if answer:
            logger.info("kg_resolved_from_profile", intent=intent, answer=answer[:80])
            return KGResolution(
                answer=answer,
                confidence=0.95,
                resolved=True,
                intent=intent,
            )

        # Try KG entity lookup
        result = self._resolve_from_kg(intent, doc_ids)
        if result.resolved:
            logger.info("kg_resolved_from_entities", intent=intent, answer=(result.answer or "")[:80])
            return result

        # Even if we can't resolve, return entity guidance
        guidance = self._entity_guidance(question, doc_ids)
        guidance.intent = intent
        return guidance

    # ------------------------------------------------------------------
    # Role-specific party resolution
    # ------------------------------------------------------------------

    def _resolve_party_by_role(
        self, question: str, doc_ids: list[str] | None
    ) -> KGResolution | None:
        """Resolve a party question by matching the specific role asked about.

        Returns None when no role can be extracted or no match is found,
        signalling that we should fall through to navigation.
        """
        m = _PARTY_ROLE_RE.search(question)
        if not m:
            return None

        asked_role = m.group(1).lower()
        from rnsr.extraction.models import EntityType, RelationType

        entities = self._get_entities(EntityType.PERSON, doc_ids)
        for ent in entities:
            rels = self.kg.get_entity_relationships(ent.id)
            for rel in rels:
                if rel.type != RelationType.PARTY_TO:
                    continue
                # Check entity metadata or mention context for the role
                meta_role = (ent.metadata.get("role") or "").lower()
                if asked_role in meta_role:
                    node_ids = self._collect_node_ids([ent])
                    return KGResolution(
                        answer=ent.canonical_name,
                        confidence=0.85,
                        resolved=True,
                        intent="parties",
                        entity_node_ids=node_ids,
                        source_entity=ent.id,
                    )
                # Also check mention context for role labels
                for mention in ent.mentions:
                    if asked_role in mention.context.lower():
                        node_ids = self._collect_node_ids([ent])
                        return KGResolution(
                            answer=ent.canonical_name,
                            confidence=0.85,
                            resolved=True,
                            intent="parties",
                            entity_node_ids=node_ids,
                            source_entity=ent.id,
                        )

        # No entity matched the specific role — don't guess
        return None

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_intent(question: str) -> str | None:
        for ip in _INTENT_PATTERNS:
            for pat in ip.patterns:
                if pat.search(question):
                    return ip.intent
        return None

    # ------------------------------------------------------------------
    # Profile-based resolution
    # ------------------------------------------------------------------

    def _resolve_from_profiles(
        self, intent: str, doc_ids: list[str] | None
    ) -> str | None:
        targets = doc_ids or list(self._profiles.keys())

        for did in targets:
            profile = self._profiles.get(did)
            if profile is None:
                continue
            if isinstance(profile, dict):
                value = profile.get(intent) or profile.get(f"primary_{intent}")
            else:
                value = getattr(profile, intent, None) or getattr(
                    profile, f"primary_{intent}", None
                )

            if value:
                if isinstance(value, list):
                    return ", ".join(str(v) for v in value)
                return str(value)

        return None

    # ------------------------------------------------------------------
    # KG entity-based resolution
    # ------------------------------------------------------------------

    def _resolve_from_kg(
        self, intent: str, doc_ids: list[str] | None
    ) -> KGResolution:
        from rnsr.extraction.models import EntityType

        dispatch = {
            "judge": (EntityType.PERSON, self._filter_judge),
            "court": (EntityType.ORGANIZATION, self._filter_court),
            "citation": (EntityType.REFERENCE, self._filter_citation),
            "date": (EntityType.DATE, self._filter_date),
            "parties": (EntityType.PERSON, self._filter_parties),
        }

        entry = dispatch.get(intent)
        if not entry:
            return KGResolution(intent=intent)

        entity_type, filter_fn = entry
        all_entities = self._get_entities(entity_type, doc_ids)
        matched = filter_fn(all_entities)

        if not matched:
            node_ids = self._collect_node_ids(all_entities)
            return KGResolution(intent=intent, entity_node_ids=node_ids)

        node_ids = self._collect_node_ids(matched)

        # For single-value intents, only resolve directly when there is
        # exactly one match.  Multiple matches mean ambiguity — fall back
        # to entity-guided navigation so the navigator can pick the right one.
        _SINGLE_VALUE_INTENTS = {"judge", "court", "citation", "date"}
        if intent in _SINGLE_VALUE_INTENTS and len(matched) > 1:
            logger.info(
                "kg_ambiguous_entities",
                intent=intent,
                count=len(matched),
                entities=[e.canonical_name for e in matched[:5]],
            )
            return KGResolution(
                intent=intent,
                entity_node_ids=node_ids,
            )

        if intent == "parties":
            answer = ", ".join(e.canonical_name for e in matched)
        else:
            answer = matched[0].canonical_name

        return KGResolution(
            answer=answer,
            confidence=0.85,
            resolved=True,
            intent=intent,
            entity_node_ids=node_ids,
            source_entity=matched[0].id,
        )

    # ------------------------------------------------------------------
    # Entity guidance (when intent is unknown or unresolvable)
    # ------------------------------------------------------------------

    def _entity_guidance(
        self, question: str, doc_ids: list[str] | None
    ) -> KGResolution:
        """Find entities whose names appear in the question and return
        their mention node_ids so the navigator can jump there."""
        from rnsr.extraction.models import EntityType

        q_lower = question.lower()
        node_ids: list[str] = []

        for etype in EntityType:
            entities = self._get_entities(etype, doc_ids)
            for ent in entities:
                for name in ent.all_names:
                    if len(name) >= 4 and name.lower() in q_lower:
                        node_ids.extend(
                            m.node_id for m in ent.mentions
                            if m.node_id not in node_ids
                        )
                        break

        return KGResolution(entity_node_ids=node_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_entities(self, entity_type: Any, doc_ids: list[str] | None) -> list:
        if doc_ids and len(doc_ids) == 1:
            return self.kg.find_entities_by_type(entity_type, doc_id=doc_ids[0])
        entities = self.kg.find_entities_by_type(entity_type)
        if doc_ids:
            did_set = set(doc_ids)
            entities = [
                e for e in entities
                if e.source_doc_id in did_set
                or any(m.doc_id in did_set for m in e.mentions)
            ]
        return entities

    @staticmethod
    def _collect_node_ids(entities: list) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for ent in entities:
            for m in ent.mentions:
                if m.node_id not in seen:
                    seen.add(m.node_id)
                    out.append(m.node_id)
        return out

    # ---- per-intent filters ----

    _JUDGE_TITLE_RE = re.compile(
        r"\b(?:Justice|Judge|Honour|Magistrate|CJ|JJ?|DCJ|AJ|JA|SC\s+DCJ|"
        r"Chief\s+Justice|Deputy\s+President|President)\b",
        re.IGNORECASE,
    )

    def _filter_judge(self, entities: list) -> list:
        out = []
        for ent in entities:
            if self._JUDGE_TITLE_RE.search(ent.canonical_name):
                out.append(ent)
                continue
            for m in ent.mentions:
                if self._JUDGE_TITLE_RE.search(m.context):
                    out.append(ent)
                    break
            else:
                meta_role = (ent.metadata.get("role") or "").lower()
                if any(
                    t in meta_role
                    for t in ("judge", "justice", "magistrate", "honour")
                ):
                    out.append(ent)
        return out

    _COURT_RE = re.compile(
        r"\b(?:court|tribunal|commission)\b", re.IGNORECASE
    )

    def _filter_court(self, entities: list) -> list:
        return [e for e in entities if self._COURT_RE.search(e.canonical_name)]

    _CITATION_RE = re.compile(
        r"\[\d{4}\]\s+[A-Z][A-Za-z]{1,15}(?:\s+[A-Z][A-Za-z]{0,10})?\s+\d{1,6}"
    )

    def _filter_citation(self, entities: list) -> list:
        return [
            e for e in entities if self._CITATION_RE.search(e.canonical_name)
        ]

    def _filter_date(self, entities: list) -> list:
        from rnsr.extraction.models import RelationType

        primary: list = []
        for ent in entities:
            rels = self.kg.get_entity_relationships(ent.id)
            for rel in rels:
                if rel.type == RelationType.HAS_DATE:
                    primary.append(ent)
                    break
        return primary or entities[:3]

    def _filter_parties(self, entities: list) -> list:
        from rnsr.extraction.models import RelationType

        out = []
        for ent in entities:
            rels = self.kg.get_entity_relationships(ent.id)
            for rel in rels:
                if rel.type == RelationType.PARTY_TO:
                    out.append(ent)
                    break
        return out
