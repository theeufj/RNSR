"""
RNSR Relationship Extractor

DEPRECATED: This extractor uses LLM-first approach which can hallucinate.
Use RLMUnifiedExtractor instead for grounded, accurate extraction.

LLM-based relationship extraction between entities and sections.
Extracts temporal, causal, semantic, and reference relationships.
"""

from __future__ import annotations

import json
import re
import time
import warnings
from typing import Any

import structlog

from rnsr.extraction.models import (
    Entity,
    EntityType,
    Relationship,
    RelationType,
)
from rnsr.llm import get_llm

logger = structlog.get_logger(__name__)

# Deprecation warning
_DEPRECATION_WARNING = """
RelationshipExtractor is deprecated and may hallucinate relationships.
Use RLMUnifiedExtractor instead for grounded, accurate extraction:

    from rnsr.extraction import RLMUnifiedExtractor
    extractor = RLMUnifiedExtractor()
    result = extractor.extract(node_id, doc_id, header, content)
    # result.relationships contains validated relationships
"""


# Relationship extraction prompt template
RELATIONSHIP_EXTRACTION_PROMPT = """You are an expert relationship extractor for legal and business documents.

Analyze the following document section and extract relationships between:
1. Entities (people, organizations, dates, events, legal concepts)
2. This section and other referenced sections/documents

Document Section:
---
{content}
---

Section ID: {node_id}
Document ID: {doc_id}
Section Header: {header}

Known entities in this section:
{entities_json}

Extract relationships of the following types:

ENTITY-TO-ENTITY RELATIONSHIPS:
- TEMPORAL_BEFORE: Event/date X occurred before Event/date Y
- TEMPORAL_AFTER: Event/date X occurred after Event/date Y
- CAUSAL: Action X caused/led to Outcome Y (e.g., breach led to damages)
- AFFILIATED_WITH: Person X is affiliated with Organization Y
- PARTY_TO: Entity X is party to Document/Event Y (e.g., signatory, defendant)

SECTION/DOCUMENT RELATIONSHIPS:
- SUPPORTS: This section supports a claim or finding in another section
- CONTRADICTS: This section contradicts another section
- REFERENCES: This section references another document or section (exhibit, citation)
- SUPERSEDES: This section supersedes/overrides another
- AMENDS: This section amends another section/document

For each relationship, provide:
1. type: One of the types above
2. source_id: ID of the source entity (from the known entities list) or "{node_id}" for this section
3. source_type: "entity" or "node"
4. target_id: ID of the target entity or node ID of the target section (use descriptive placeholder if referencing external doc, e.g., "exhibit_a")
5. target_type: "entity" or "node"
6. confidence: 0.0-1.0 based on how explicit the relationship is
7. evidence: The exact quote that establishes this relationship

Return your response as a JSON array:
```json
[
  {{
    "type": "CAUSAL",
    "source_id": "ent_abc123",
    "source_type": "entity",
    "target_id": "ent_def456",
    "target_type": "entity",
    "confidence": 0.9,
    "evidence": "The breach of contract by Defendant led to significant damages..."
  }},
  {{
    "type": "REFERENCES",
    "source_id": "{node_id}",
    "source_type": "node",
    "target_id": "exhibit_a",
    "target_type": "node",
    "confidence": 1.0,
    "evidence": "As shown in Exhibit A..."
  }}
]
```

If no relationships are found, return an empty array: []

Important:
- Only extract relationships that are explicitly stated or strongly implied
- Include the exact quote as evidence
- Use entity IDs from the provided list when possible
- For temporal relationships, be precise about the direction (BEFORE vs AFTER)
- For causal relationships, the source is the cause and target is the effect
"""


class RelationshipExtractor:
    """
    DEPRECATED: Extracts relationships from document sections using LLM-first approach.
    
    This extractor can hallucinate relationships. Use RLMUnifiedExtractor instead.
    
    Identifies connections between:
    - Entities (temporal, causal, affiliation)
    - Sections (supports, contradicts, references)
    - Documents (cross-references, amendments)
    """
    
    def __init__(
        self,
        llm: Any | None = None,
        min_content_length: int = 50,
        max_content_length: int | None = None,
        suppress_deprecation_warning: bool = False,
    ):
        # Emit deprecation warning
        if not suppress_deprecation_warning:
            warnings.warn(
                _DEPRECATION_WARNING,
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning("deprecated_extractor_used", extractor="RelationshipExtractor")
        """
        Initialize the relationship extractor.
        
        Args:
            llm: LLM instance to use. If None, uses get_llm().
            min_content_length: Minimum content length to process.
            max_content_length: Optional max content length per extraction call.
                Defaults to ``None`` (no truncation).
        """
        self.llm = llm or get_llm()
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        
        # Cache for extracted relationships (node_id -> relationships)
        self._cache: dict[str, list[Relationship]] = {}
    
    def extract_from_node(
        self,
        node_id: str,
        doc_id: str,
        header: str,
        content: str,
        entities: list[Entity],
    ) -> list[Relationship]:
        """
        Extract relationships from a single document node.
        
        Args:
            node_id: Skeleton node ID.
            doc_id: Document ID.
            header: Section header text.
            content: Full section content.
            entities: Entities already extracted from this node.
            
        Returns:
            List of extracted Relationship objects.
        """
        start_time = time.time()
        
        # Skip very short content
        if len(content.strip()) < self.min_content_length:
            logger.debug(
                "skipping_short_content",
                node_id=node_id,
                content_length=len(content),
            )
            return []
        
        # Check cache
        cache_key = f"{doc_id}:{node_id}"
        if cache_key in self._cache:
            logger.debug("using_cached_relationships", node_id=node_id)
            return self._cache[cache_key]
        
        # Truncate content only if an explicit cap was set
        if self.max_content_length and len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."
        
        try:
            relationships = self._extract_with_llm(
                node_id=node_id,
                doc_id=doc_id,
                header=header,
                content=content,
                entities=entities,
            )
            
            # Cache results
            self._cache[cache_key] = relationships
            
        except Exception as e:
            logger.error(
                "relationship_extraction_failed",
                node_id=node_id,
                error=str(e),
            )
            relationships = []
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "relationships_extracted",
            node_id=node_id,
            relationship_count=len(relationships),
            processing_time_ms=processing_time_ms,
        )
        
        return relationships
    
    def _extract_with_llm(
        self,
        node_id: str,
        doc_id: str,
        header: str,
        content: str,
        entities: list[Entity],
    ) -> list[Relationship]:
        """
        Use LLM to extract relationships from content.
        
        Args:
            node_id: Skeleton node ID.
            doc_id: Document ID.
            header: Section header.
            content: Section content.
            entities: Known entities in this section.
            
        Returns:
            List of extracted Relationship objects.
        """
        # Format entities for prompt
        entities_json = json.dumps([
            {
                "id": e.id,
                "type": e.type.value,
                "name": e.canonical_name,
                "aliases": e.aliases,
            }
            for e in entities
        ], indent=2)
        
        prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
            content=content,
            node_id=node_id,
            doc_id=doc_id,
            header=header,
            entities_json=entities_json,
        )
        
        # Call LLM (prefer JSON mode for structured output)
        _complete_fn = getattr(self.llm, "complete_json", None) or self.llm.complete
        response = _complete_fn(prompt)
        response_text = str(response) if not isinstance(response, str) else response
        
        # Parse JSON from response
        relationships = self._parse_llm_response(
            response_text=response_text,
            node_id=node_id,
            doc_id=doc_id,
            entities=entities,
        )
        
        return relationships
    
    def _parse_llm_response(
        self,
        response_text: str,
        node_id: str,
        doc_id: str,
        entities: list[Entity],
    ) -> list[Relationship]:
        """
        Parse LLM response into Relationship objects.
        
        Args:
            response_text: Raw LLM response.
            node_id: Source node ID.
            doc_id: Source document ID.
            entities: Known entities.
            
        Returns:
            List of Relationship objects.
        """
        # Extract JSON from response
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                logger.warning(
                    "no_json_found_in_response",
                    response_preview=response_text[:200],
                )
                return []
        
        try:
            raw_relationships = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(
                "json_parse_error",
                error=str(e),
                json_preview=json_str[:200],
            )
            return []
        
        if not isinstance(raw_relationships, list):
            logger.warning("expected_list_of_relationships", got=type(raw_relationships).__name__)
            return []
        
        # Create a set of valid entity IDs
        valid_entity_ids = {e.id for e in entities}
        
        relationships = []
        for raw in raw_relationships:
            try:
                relationship = self._create_relationship_from_raw(
                    raw=raw,
                    node_id=node_id,
                    doc_id=doc_id,
                    valid_entity_ids=valid_entity_ids,
                )
                if relationship:
                    relationships.append(relationship)
            except Exception as e:
                logger.debug(
                    "failed_to_create_relationship",
                    raw=raw,
                    error=str(e),
                )
        
        return relationships
    
    def _create_relationship_from_raw(
        self,
        raw: dict[str, Any],
        node_id: str,
        doc_id: str,
        valid_entity_ids: set[str],
    ) -> Relationship | None:
        """
        Create a Relationship object from raw LLM output.
        
        Args:
            raw: Raw relationship dict from LLM.
            node_id: Source node ID.
            doc_id: Source document ID.
            valid_entity_ids: Set of valid entity IDs.
            
        Returns:
            Relationship object or None if invalid.
        """
        # Parse relationship type
        type_str = raw.get("type", "").upper()
        
        type_mapping = {
            "TEMPORAL_BEFORE": RelationType.TEMPORAL_BEFORE,
            "TEMPORAL_AFTER": RelationType.TEMPORAL_AFTER,
            "BEFORE": RelationType.TEMPORAL_BEFORE,
            "AFTER": RelationType.TEMPORAL_AFTER,
            "CAUSAL": RelationType.CAUSAL,
            "CAUSED": RelationType.CAUSAL,
            "AFFILIATED_WITH": RelationType.AFFILIATED_WITH,
            "AFFILIATION": RelationType.AFFILIATED_WITH,
            "PARTY_TO": RelationType.PARTY_TO,
            "PARTY": RelationType.PARTY_TO,
            "SUPPORTS": RelationType.SUPPORTS,
            "SUPPORT": RelationType.SUPPORTS,
            "CONTRADICTS": RelationType.CONTRADICTS,
            "CONTRADICT": RelationType.CONTRADICTS,
            "REFERENCES": RelationType.REFERENCES,
            "REFERENCE": RelationType.REFERENCES,
            "CITES": RelationType.REFERENCES,
            "SUPERSEDES": RelationType.SUPERSEDES,
            "SUPERSEDE": RelationType.SUPERSEDES,
            "AMENDS": RelationType.AMENDS,
            "AMEND": RelationType.AMENDS,
            "MENTIONS": RelationType.MENTIONS,
            "DEFINED_IN": RelationType.DEFINED_IN,
        }
        
        rel_type = type_mapping.get(type_str)
        if not rel_type:
            logger.debug("unknown_relationship_type", type=type_str)
            return None
        
        # Get source and target
        source_id = raw.get("source_id", "")
        target_id = raw.get("target_id", "")
        source_type = raw.get("source_type", "entity").lower()
        target_type = raw.get("target_type", "entity").lower()
        
        if not source_id or not target_id:
            return None
        
        # Validate source_type and target_type
        if source_type not in ("entity", "node"):
            source_type = "entity"
        if target_type not in ("entity", "node"):
            target_type = "entity"
        
        # Get confidence and evidence
        confidence = raw.get("confidence", 0.8)
        if not isinstance(confidence, (int, float)):
            confidence = 0.8
        confidence = max(0.0, min(1.0, float(confidence)))
        
        evidence = raw.get("evidence", "").strip()
        
        # Create relationship
        relationship = Relationship(
            type=rel_type,
            source_id=source_id,
            target_id=target_id,
            source_type=source_type,
            target_type=target_type,
            doc_id=doc_id,
            confidence=confidence,
            evidence=evidence,
        )
        
        return relationship
    
    def extract_entity_to_section_relationships(
        self,
        entities: list[Entity],
    ) -> list[Relationship]:
        """
        Create MENTIONS relationships linking entities to their sections.
        
        Args:
            entities: List of entities with mentions.
            
        Returns:
            List of MENTIONS relationships.
        """
        relationships = []
        
        for entity in entities:
            for mention in entity.mentions:
                rel = Relationship(
                    type=RelationType.MENTIONS,
                    source_id=mention.node_id,
                    target_id=entity.id,
                    source_type="node",
                    target_type="entity",
                    doc_id=mention.doc_id,
                    confidence=mention.confidence,
                    evidence=mention.context,
                )
                relationships.append(rel)
        
        return relationships
    
    def extract_batch(
        self,
        nodes: list[dict[str, Any]],
        entities_by_node: dict[str, list[Entity]],
    ) -> list[Relationship]:
        """
        Extract relationships from multiple nodes.
        
        Args:
            nodes: List of node dicts with keys: node_id, doc_id, header, content
            entities_by_node: Mapping of node_id to entities in that node.
            
        Returns:
            List of all extracted Relationship objects.
        """
        all_relationships = []
        
        for node in nodes:
            node_id = node.get("node_id", "")
            entities = entities_by_node.get(node_id, [])
            
            relationships = self.extract_from_node(
                node_id=node_id,
                doc_id=node.get("doc_id", ""),
                header=node.get("header", ""),
                content=node.get("content", ""),
                entities=entities,
            )
            all_relationships.extend(relationships)
        
        return all_relationships
    
    def clear_cache(self) -> None:
        """Clear the relationship cache."""
        self._cache.clear()


def extract_implicit_relationships(
    entities: list[Entity],
    doc_id: str,
) -> list[Relationship]:
    """
    Extract implicit relationships based on entity metadata.
    
    For example:
    - PERSON with role "defendant" -> PARTY_TO -> any legal proceeding
    - PERSON affiliated with ORGANIZATION
    
    Args:
        entities: List of entities.
        doc_id: Document ID.
        
    Returns:
        List of implicit relationships.
    """
    relationships = []
    
    # Group entities by type
    persons = [e for e in entities if e.type == EntityType.PERSON]
    orgs = [e for e in entities if e.type == EntityType.ORGANIZATION]
    events = [e for e in entities if e.type == EntityType.EVENT]
    documents = [e for e in entities if e.type == EntityType.DOCUMENT]
    
    # Extract affiliations from person metadata
    for person in persons:
        # Check for organization affiliation in metadata
        affiliated_org = person.metadata.get("organization") or person.metadata.get("employer")
        if affiliated_org:
            # Find matching org
            for org in orgs:
                if affiliated_org.lower() in org.canonical_name.lower() or any(
                    affiliated_org.lower() in alias.lower() for alias in org.aliases
                ):
                    rel = Relationship(
                        type=RelationType.AFFILIATED_WITH,
                        source_id=person.id,
                        target_id=org.id,
                        source_type="entity",
                        target_type="entity",
                        doc_id=doc_id,
                        confidence=0.9,
                        evidence=f"{person.canonical_name} affiliated with {org.canonical_name}",
                    )
                    relationships.append(rel)
                    break
        
        # Check for role-based party relationships
        role = person.metadata.get("role", "").lower()
        if role in ("defendant", "plaintiff", "respondent", "petitioner", "applicant"):
            for event in events:
                if any(kw in event.canonical_name.lower() for kw in ["case", "proceeding", "trial", "hearing"]):
                    rel = Relationship(
                        type=RelationType.PARTY_TO,
                        source_id=person.id,
                        target_id=event.id,
                        source_type="entity",
                        target_type="entity",
                        doc_id=doc_id,
                        confidence=0.85,
                        evidence=f"{person.canonical_name} is {role} in {event.canonical_name}",
                    )
                    relationships.append(rel)
            
            for doc in documents:
                if any(kw in doc.canonical_name.lower() for kw in ["complaint", "motion", "order", "judgment"]):
                    rel = Relationship(
                        type=RelationType.PARTY_TO,
                        source_id=person.id,
                        target_id=doc.id,
                        source_type="entity",
                        target_type="entity",
                        doc_id=doc_id,
                        confidence=0.85,
                        evidence=f"{person.canonical_name} is {role} in {doc.canonical_name}",
                    )
                    relationships.append(rel)
    
    return relationships
