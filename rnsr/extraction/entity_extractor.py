"""
RNSR Entity Extractor

DEPRECATED: This extractor uses LLM-first approach which can hallucinate.
Use RLMUnifiedExtractor instead for grounded, accurate extraction.

LLM-based entity extraction from document sections.
Extracts people, organizations, dates, legal concepts, and other entities.

Features adaptive learning: when the LLM discovers new entity types, they are
stored in a learned types registry and used in future extraction prompts.
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
    ExtractionResult,
    Mention,
)
from rnsr.extraction.learned_types import (
    get_learned_type_registry,
    record_learned_type,
)
from rnsr.llm import get_llm

logger = structlog.get_logger(__name__)

# Deprecation warning
_DEPRECATION_WARNING = """
EntityExtractor is deprecated and may hallucinate entities.
Use RLMUnifiedExtractor instead for grounded, accurate extraction:

    from rnsr.extraction import RLMUnifiedExtractor
    extractor = RLMUnifiedExtractor()
    result = extractor.extract(node_id, doc_id, header, content)
"""


# Entity extraction prompt template
ENTITY_EXTRACTION_PROMPT = """You are an expert entity extractor for legal and business documents.

Analyze the following document section and extract all significant entities.

Document Section:
---
{content}
---

Section ID: {node_id}
Document ID: {doc_id}
Section Header: {header}

Extract entities of the following types:
- PERSON: Names of individuals, including their roles if mentioned (e.g., "plaintiff", "defendant", "witness", "CEO")
- ORGANIZATION: Companies, agencies, courts, government bodies
- LEGAL_CONCEPT: Legal claims, breaches, obligations, remedies, causes of action
- DATE: Specific dates, time periods, deadlines
- EVENT: Significant occurrences (hearings, signings, breaches, filings)
- LOCATION: Places, addresses, jurisdictions
- REFERENCE: Section references, exhibit numbers, document citations
- MONETARY: Dollar amounts, financial figures
- DOCUMENT: Referenced documents (contracts, exhibits, agreements)
{learned_types_section}
For each entity, provide:
1. type: One of the types above (or your own descriptive type if none fit)
2. canonical_name: The standardized/normalized name
3. aliases: Any alternative names or spellings found
4. context: The surrounding sentence or phrase where the entity appears
5. metadata: Any additional relevant information (roles, dates, amounts)

Return your response as a JSON array of entities:
```json
[
  {{
    "type": "PERSON",
    "canonical_name": "John Smith",
    "aliases": ["Mr. Smith", "J. Smith"],
    "context": "John Smith, the defendant, filed a motion...",
    "metadata": {{"role": "defendant"}}
  }},
  ...
]
```

If no entities are found, return an empty array: []

Important:
- Be thorough but precise - only extract clearly identifiable entities
- Normalize names (e.g., "Mr. John Smith" -> "John Smith")
- Include context that helps understand the entity's role
- For legal concepts, use standardized legal terminology
- If an entity doesn't fit the predefined types, use your own descriptive type name
"""


class EntityExtractor:
    """
    DEPRECATED: Extracts entities from document sections using LLM-first approach.
    
    This extractor can hallucinate entities. Use RLMUnifiedExtractor instead.
    
    Supports batch processing, caching, and adaptive learning of entity types.
    When new entity types are discovered, they are stored and used in future prompts.
    """
    
    def __init__(
        self,
        llm: Any | None = None,
        min_content_length: int = 50,
        max_content_length: int | None = None,
        enable_type_learning: bool = True,
        learned_type_min_count: int = 2,
        suppress_deprecation_warning: bool = False,
    ):
        # Emit deprecation warning
        if not suppress_deprecation_warning:
            warnings.warn(
                _DEPRECATION_WARNING,
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning("deprecated_extractor_used", extractor="EntityExtractor")
        """
        Initialize the entity extractor.
        
        Args:
            llm: LLM instance to use. If None, uses get_llm().
            min_content_length: Minimum content length to process.
            max_content_length: Optional max content length per extraction call.
                Defaults to ``None`` (no truncation — modern LLMs handle full sections).
            enable_type_learning: Whether to learn new entity types.
            learned_type_min_count: Minimum occurrences before a learned type
                                    is included in extraction prompts.
        """
        self.llm = llm or get_llm()
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.enable_type_learning = enable_type_learning
        self.learned_type_min_count = learned_type_min_count
        
        # Cache for extracted entities (node_id -> entities)
        self._cache: dict[str, list[Entity]] = {}
        
        # Get learned type registry
        self._type_registry = get_learned_type_registry() if enable_type_learning else None
    
    def extract_from_node(
        self,
        node_id: str,
        doc_id: str,
        header: str,
        content: str,
        page_num: int | None = None,
    ) -> ExtractionResult:
        """
        Extract entities from a single document node.
        
        Args:
            node_id: Skeleton node ID.
            doc_id: Document ID.
            header: Section header text.
            content: Full section content.
            page_num: Page number if available.
            
        Returns:
            ExtractionResult with extracted entities.
        """
        start_time = time.time()
        result = ExtractionResult(
            node_id=node_id,
            doc_id=doc_id,
            extraction_method="llm",
        )
        
        # Skip very short content
        if len(content.strip()) < self.min_content_length:
            logger.debug(
                "skipping_short_content",
                node_id=node_id,
                content_length=len(content),
            )
            return result
        
        # Check cache
        cache_key = f"{doc_id}:{node_id}"
        if cache_key in self._cache:
            result.entities = self._cache[cache_key]
            logger.debug("using_cached_entities", node_id=node_id)
            return result
        
        # Truncate content only if an explicit cap was set
        if self.max_content_length and len(content) > self.max_content_length:
            original_len = len(content)
            content = content[:self.max_content_length] + "..."
            result.warnings.append(f"Content truncated from {original_len} chars")
        
        try:
            entities = self._extract_with_llm(
                node_id=node_id,
                doc_id=doc_id,
                header=header,
                content=content,
                page_num=page_num,
            )
            result.entities = entities
            
            # Cache results
            self._cache[cache_key] = entities
            
        except Exception as e:
            logger.error(
                "entity_extraction_failed",
                node_id=node_id,
                error=str(e),
            )
            result.warnings.append(f"Extraction failed: {str(e)}")
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "entities_extracted",
            node_id=node_id,
            entity_count=len(result.entities),
            processing_time_ms=result.processing_time_ms,
        )
        
        return result
    
    def _extract_with_llm(
        self,
        node_id: str,
        doc_id: str,
        header: str,
        content: str,
        page_num: int | None = None,
    ) -> list[Entity]:
        """
        Use LLM to extract entities from content.
        
        Args:
            node_id: Skeleton node ID.
            doc_id: Document ID.
            header: Section header.
            content: Section content.
            page_num: Page number.
            
        Returns:
            List of extracted Entity objects.
        """
        # Build learned types section for prompt
        learned_types_section = ""
        if self._type_registry:
            learned_types = self._type_registry.get_types_for_prompt(
                min_count=self.learned_type_min_count,
                limit=15,
            )
            if learned_types:
                types_list = ", ".join(learned_types).upper()
                learned_types_section = f"\nAdditionally, these domain-specific types have been learned from previous documents:\n- {types_list}\n"
        
        prompt = ENTITY_EXTRACTION_PROMPT.format(
            content=content,
            node_id=node_id,
            doc_id=doc_id,
            header=header,
            learned_types_section=learned_types_section,
        )
        
        # Call LLM (prefer JSON mode for structured output)
        _complete_fn = getattr(self.llm, "complete_json", None) or self.llm.complete
        response = _complete_fn(prompt)
        response_text = str(response) if not isinstance(response, str) else response
        
        # Parse JSON from response
        entities = self._parse_llm_response(
            response_text=response_text,
            node_id=node_id,
            doc_id=doc_id,
            page_num=page_num,
        )
        
        return entities
    
    def _parse_llm_response(
        self,
        response_text: str,
        node_id: str,
        doc_id: str,
        page_num: int | None = None,
    ) -> list[Entity]:
        """
        Parse LLM response into Entity objects.
        
        Args:
            response_text: Raw LLM response.
            node_id: Source node ID.
            doc_id: Source document ID.
            page_num: Page number.
            
        Returns:
            List of Entity objects.
        """
        # Extract JSON from response (may be wrapped in markdown code block)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find a JSON array directly
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
            raw_entities = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(
                "json_parse_error",
                error=str(e),
                json_preview=json_str[:200],
            )
            return []
        
        if not isinstance(raw_entities, list):
            logger.warning("expected_list_of_entities", got=type(raw_entities).__name__)
            return []
        
        entities = []
        for raw in raw_entities:
            try:
                entity = self._create_entity_from_raw(
                    raw=raw,
                    node_id=node_id,
                    doc_id=doc_id,
                    page_num=page_num,
                )
                if entity:
                    entities.append(entity)
            except Exception as e:
                logger.debug(
                    "failed_to_create_entity",
                    raw=raw,
                    error=str(e),
                )
        
        return entities
    
    def _create_entity_from_raw(
        self,
        raw: dict[str, Any],
        node_id: str,
        doc_id: str,
        page_num: int | None = None,
    ) -> Entity | None:
        """
        Create an Entity object from raw LLM output.
        
        Args:
            raw: Raw entity dict from LLM.
            node_id: Source node ID.
            doc_id: Source document ID.
            page_num: Page number.
            
        Returns:
            Entity object or None if invalid.
        """
        # Parse entity type
        type_str = raw.get("type", "").upper()
        original_type = type_str  # Preserve for metadata
        
        try:
            entity_type = EntityType(type_str.lower())
        except ValueError:
            # Try mapping common variations
            type_mapping = {
                "PERSON": EntityType.PERSON,
                "PEOPLE": EntityType.PERSON,
                "INDIVIDUAL": EntityType.PERSON,
                "NAME": EntityType.PERSON,
                "ORGANIZATION": EntityType.ORGANIZATION,
                "ORG": EntityType.ORGANIZATION,
                "COMPANY": EntityType.ORGANIZATION,
                "AGENCY": EntityType.ORGANIZATION,
                "COURT": EntityType.ORGANIZATION,
                "LEGAL_CONCEPT": EntityType.LEGAL_CONCEPT,
                "LEGAL": EntityType.LEGAL_CONCEPT,
                "CONCEPT": EntityType.LEGAL_CONCEPT,
                "CLAIM": EntityType.LEGAL_CONCEPT,
                "OBLIGATION": EntityType.LEGAL_CONCEPT,
                "DATE": EntityType.DATE,
                "TIME": EntityType.DATE,
                "DATETIME": EntityType.DATE,
                "PERIOD": EntityType.DATE,
                "EVENT": EntityType.EVENT,
                "OCCURRENCE": EntityType.EVENT,
                "INCIDENT": EntityType.EVENT,
                "LOCATION": EntityType.LOCATION,
                "PLACE": EntityType.LOCATION,
                "ADDRESS": EntityType.LOCATION,
                "JURISDICTION": EntityType.LOCATION,
                "REFERENCE": EntityType.REFERENCE,
                "REF": EntityType.REFERENCE,
                "CITATION": EntityType.REFERENCE,
                "SECTION": EntityType.REFERENCE,
                "MONETARY": EntityType.MONETARY,
                "MONEY": EntityType.MONETARY,
                "AMOUNT": EntityType.MONETARY,
                "CURRENCY": EntityType.MONETARY,
                "FINANCIAL": EntityType.MONETARY,
                "DOCUMENT": EntityType.DOCUMENT,
                "DOC": EntityType.DOCUMENT,
                "CONTRACT": EntityType.DOCUMENT,
                "AGREEMENT": EntityType.DOCUMENT,
                "EXHIBIT": EntityType.DOCUMENT,
            }
            entity_type = type_mapping.get(type_str)
            
            if not entity_type:
                # Check if we have a learned mapping for this type
                if self._type_registry:
                    mappings = self._type_registry.get_mappings()
                    if type_str.lower() in mappings:
                        mapped_type = mappings[type_str.lower()]
                        try:
                            entity_type = EntityType(mapped_type.lower())
                            logger.debug(
                                "using_learned_mapping",
                                original=type_str,
                                mapped_to=mapped_type,
                            )
                        except ValueError:
                            pass
                
                if not entity_type:
                    # Use OTHER as fallback - never drop entities
                    logger.debug("unmapped_entity_type_using_other", type=type_str)
                    entity_type = EntityType.OTHER
        
        # Get canonical name
        canonical_name = raw.get("canonical_name", "").strip()
        if not canonical_name:
            canonical_name = raw.get("name", "").strip()
        if not canonical_name:
            return None
        
        # Get aliases
        aliases = raw.get("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [a.strip() for a in aliases if a and a.strip()]
        
        # Get context
        context = raw.get("context", "").strip()
        
        # Get metadata
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        
        # Preserve original type if we used the OTHER fallback
        if entity_type == EntityType.OTHER and original_type:
            metadata["original_type"] = original_type.lower()
            
            # Record this type for adaptive learning
            if self._type_registry and self.enable_type_learning:
                self._type_registry.record_type(
                    type_name=original_type.lower(),
                    context=context,
                    entity_name=canonical_name,
                )
        
        # Create mention
        mention = Mention(
            node_id=node_id,
            doc_id=doc_id,
            context=context,
            page_num=page_num,
            confidence=1.0,
        )
        
        # Create entity
        entity = Entity(
            type=entity_type,
            canonical_name=canonical_name,
            aliases=aliases,
            mentions=[mention],
            metadata=metadata,
            source_doc_id=doc_id,
        )
        
        return entity
    
    def extract_batch(
        self,
        nodes: list[dict[str, Any]],
    ) -> list[ExtractionResult]:
        """
        Extract entities from multiple nodes.
        
        Args:
            nodes: List of node dicts with keys: node_id, doc_id, header, content, page_num
            
        Returns:
            List of ExtractionResult objects.
        """
        results = []
        
        for node in nodes:
            result = self.extract_from_node(
                node_id=node.get("node_id", ""),
                doc_id=node.get("doc_id", ""),
                header=node.get("header", ""),
                content=node.get("content", ""),
                page_num=node.get("page_num"),
            )
            results.append(result)
        
        return results
    
    def clear_cache(self) -> None:
        """Clear the entity cache."""
        self._cache.clear()


def merge_entities(entities: list[Entity]) -> list[Entity]:
    """
    Merge duplicate entities based on canonical name and type.
    
    Combines mentions and aliases from duplicates.
    
    Args:
        entities: List of entities to merge.
        
    Returns:
        Deduplicated list of entities.
    """
    # Group by (type, normalized canonical_name)
    grouped: dict[tuple[EntityType, str], list[Entity]] = {}
    
    for entity in entities:
        key = (entity.type, entity.canonical_name.lower().strip())
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(entity)
    
    # Merge each group
    merged = []
    for entities_group in grouped.values():
        if len(entities_group) == 1:
            merged.append(entities_group[0])
        else:
            # Merge into first entity
            primary = entities_group[0]
            for other in entities_group[1:]:
                # Merge mentions
                primary.mentions.extend(other.mentions)
                
                # Merge aliases
                for alias in other.aliases:
                    primary.add_alias(alias)
                
                # Merge metadata (prefer primary's values on conflict)
                for k, v in other.metadata.items():
                    if k not in primary.metadata:
                        primary.metadata[k] = v
            
            merged.append(primary)
    
    return merged
