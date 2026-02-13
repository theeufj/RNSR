"""
RNSR Grounded Entity Extractor

Implements the RLM pattern for entity extraction:
1. Pre-extract candidates using regex/patterns (CODE FIRST)
2. LLM classifies and validates candidates (LLM SECOND)
3. Recursive refinement if needed

This approach PREVENTS HALLUCINATION because:
- Every entity is tied to an exact text span
- LLM classifies existing text, doesn't generate entities
- Pattern matching provides grounded candidates
- LLM's job is validation, not invention

Validation Modes:
- SIMPLE: Basic LLM classification (faster, single call)
- TOT: Tree of Thoughts validation with probabilities and navigation (more accurate)

Inspired by the RLM paper's insight: use code to filter/extract
before sending to LLM for reasoning.
"""

from __future__ import annotations

import json
import re
import time
from enum import Enum
from typing import Any, TYPE_CHECKING

import structlog

from rnsr.extraction.models import (
    Entity,
    EntityType,
    ExtractionResult,
    Mention,
)
from rnsr.extraction.candidate_extractor import (
    CandidateExtractor,
    EntityCandidate,
)
from rnsr.extraction.learned_types import (
    get_learned_type_registry,
)
from rnsr.llm import get_llm

if TYPE_CHECKING:
    from rnsr.models import DocumentTree


class ValidationMode(str, Enum):
    """Validation mode for candidate entities."""
    
    SIMPLE = "simple"   # Basic LLM classification (faster)
    TOT = "tot"         # Tree of Thoughts with navigation (more accurate)

logger = structlog.get_logger(__name__)


# LLM prompt for CLASSIFYING pre-extracted candidates (not generating)
CLASSIFICATION_PROMPT = """You are classifying entity candidates that have already been extracted from a document.

Your job is to VALIDATE and CLASSIFY each candidate - NOT to generate new entities.
These candidates were extracted by pattern matching and are grounded in the actual text.

Document Section:
---
{content}
---

Pre-extracted candidates to classify:
{candidates_json}

For each candidate, provide:
1. valid: true if this is a real entity worth tracking, false if it's noise
2. type: The entity type (PERSON, ORGANIZATION, DATE, EVENT, LEGAL_CONCEPT, LOCATION, REFERENCE, MONETARY, DOCUMENT, or other descriptive type)
3. canonical_name: Normalized/cleaned name (e.g., "Mr. John Smith" → "John Smith")
4. role: Any role or relationship mentioned (e.g., "defendant", "CEO")

Return JSON array:
```json
[
  {{
    "candidate_id": 0,
    "valid": true,
    "type": "PERSON",
    "canonical_name": "John Smith",
    "role": "defendant"
  }},
  {{
    "candidate_id": 1,
    "valid": false,
    "reason": "Generic reference, not a specific entity"
  }}
]
```

Rules:
- ONLY classify the candidates provided - do not add new entities
- Set valid=false for generic terms, partial matches, or noise
- Use the exact text span provided - don't modify the match boundaries
- Be conservative - when uncertain, set valid=false
"""

# Prompt for finding entities the patterns might have missed
SUPPLEMENTARY_PROMPT = """Review this text for important entities that might have been missed by pattern matching.

Text:
---
{content}
---

Already extracted: {existing_entities}

Are there any CLEARLY IDENTIFIABLE entities that were missed?
Only list entities that:
1. Are explicitly named in the text (not implied)
2. Are significant (not passing mentions)
3. Have a clear type

If there are missed entities, return:
```json
[
  {{
    "text": "exact text as it appears",
    "type": "ENTITY_TYPE",
    "canonical_name": "Normalized Name",
    "reason": "Why this is important"
  }}
]
```

If nothing important was missed, return: []
"""


class GroundedEntityExtractor:
    """
    Entity extractor following the RLM pattern:
    CODE FIRST (pattern extraction) → LLM SECOND (classification).
    
    This prevents hallucination by grounding entities in actual text.
    """
    
    def __init__(
        self,
        llm: Any | None = None,
        candidate_extractor: CandidateExtractor | None = None,
        min_content_length: int = 50,
        max_candidates_per_batch: int = 30,
        enable_supplementary_extraction: bool = True,
        enable_type_learning: bool = True,
        validation_mode: ValidationMode | str = ValidationMode.SIMPLE,
        tot_selection_threshold: float = 0.6,
        tot_enable_navigation: bool = True,
    ):
        """
        Initialize the grounded extractor.
        
        Args:
            llm: LLM instance. If None, uses get_llm().
            candidate_extractor: Pre-extraction engine.
            min_content_length: Minimum content length to process.
            max_candidates_per_batch: Max candidates per LLM call.
            enable_supplementary_extraction: Check for missed entities.
            enable_type_learning: Learn new entity types.
            validation_mode: SIMPLE (faster) or TOT (more accurate with navigation).
            tot_selection_threshold: Probability threshold for ToT mode.
            tot_enable_navigation: Navigate tree for uncertain candidates in ToT mode.
        """
        self.llm = llm
        self.candidate_extractor = candidate_extractor or CandidateExtractor()
        self.min_content_length = min_content_length
        self.max_candidates_per_batch = max_candidates_per_batch
        self.enable_supplementary_extraction = enable_supplementary_extraction
        self.enable_type_learning = enable_type_learning
        
        # Validation mode
        if isinstance(validation_mode, str):
            validation_mode = ValidationMode(validation_mode.lower())
        self.validation_mode = validation_mode
        self.tot_selection_threshold = tot_selection_threshold
        self.tot_enable_navigation = tot_enable_navigation
        
        # Lazy init for ToT validator
        self._tot_validator = None
        
        # Lazy LLM init
        self._llm_initialized = False
        
        # Type registry for learning
        self._type_registry = get_learned_type_registry() if enable_type_learning else None
        
        # Cache
        self._cache: dict[str, list[Entity]] = {}
    
    def _get_llm(self) -> Any:
        """Get or initialize LLM."""
        if self.llm is None and not self._llm_initialized:
            self.llm = get_llm()
            self._llm_initialized = True
        return self.llm
    
    def _get_tot_validator(self) -> "TotEntityValidator":
        """Get or initialize ToT validator."""
        if self._tot_validator is None:
            from rnsr.extraction.tot_validator import TotEntityValidator
            self._tot_validator = TotEntityValidator(
                llm=self._get_llm(),
                selection_threshold=self.tot_selection_threshold,
                enable_navigation=self.tot_enable_navigation,
                max_candidates_per_batch=self.max_candidates_per_batch,
            )
        return self._tot_validator
    
    def extract_from_node(
        self,
        node_id: str,
        doc_id: str,
        header: str,
        content: str,
        page_num: int | None = None,
        document_tree: "DocumentTree | None" = None,
    ) -> ExtractionResult:
        """
        Extract entities using the grounded approach.
        
        Flow:
        1. Pattern-extract candidates (grounded)
        2. Validate with LLM (SIMPLE mode) or ToT (TOT mode)
        3. Optionally check for missed entities
        4. Return validated entities
        
        Args:
            node_id: Section node ID.
            doc_id: Document ID.
            header: Section header.
            content: Section content.
            page_num: Page number.
            document_tree: Optional tree for ToT navigation.
            
        Returns:
            ExtractionResult with grounded entities.
        """
        start_time = time.time()
        result = ExtractionResult(
            node_id=node_id,
            doc_id=doc_id,
            extraction_method=f"grounded_{self.validation_mode.value}",
        )
        
        # Skip short content
        if len(content.strip()) < self.min_content_length:
            return result
        
        # Check cache
        cache_key = f"{doc_id}:{node_id}"
        if cache_key in self._cache:
            result.entities = self._cache[cache_key]
            return result
        
        # STEP 1: Extract candidates using patterns (CODE FIRST)
        candidates = self.candidate_extractor.extract_candidates(content)
        
        result.warnings.append(f"Pattern extraction found {len(candidates)} candidates")
        
        if not candidates:
            # No pattern matches - try supplementary if enabled
            if self.enable_supplementary_extraction:
                entities = self._supplementary_extraction(
                    content, node_id, doc_id, page_num, []
                )
                result.entities = entities
            
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
        
        # STEP 2: Validate candidates (LLM SECOND)
        if self.validation_mode == ValidationMode.TOT:
            # Use Tree of Thoughts validation (more accurate, can navigate)
            entities = self._validate_with_tot(
                candidates=candidates,
                header=header,
                content=content,
                node_id=node_id,
                doc_id=doc_id,
                page_num=page_num,
                document_tree=document_tree,
            )
        else:
            # Use simple classification (faster)
            entities = self._classify_candidates(
                candidates=candidates,
                content=content,
                node_id=node_id,
                doc_id=doc_id,
                page_num=page_num,
            )
        
        # STEP 3: Check for missed entities (optional)
        if self.enable_supplementary_extraction:
            existing_names = [e.canonical_name for e in entities]
            supplementary = self._supplementary_extraction(
                content, node_id, doc_id, page_num, existing_names
            )
            entities.extend(supplementary)
        
        result.entities = entities
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        # Cache
        self._cache[cache_key] = entities
        
        logger.info(
            "grounded_extraction_complete",
            node_id=node_id,
            candidates=len(candidates),
            entities=len(entities),
            validation_mode=self.validation_mode.value,
            time_ms=result.processing_time_ms,
        )
        
        return result
    
    def _validate_with_tot(
        self,
        candidates: list[EntityCandidate],
        header: str,
        content: str,
        node_id: str,
        doc_id: str,
        page_num: int | None,
        document_tree: "DocumentTree | None",
    ) -> list[Entity]:
        """
        Validate candidates using Tree of Thoughts pattern.
        
        This uses the same ToT approach as document navigation:
        - Evaluate each candidate with probability + reasoning
        - Navigate to related sections for uncertain candidates
        - More accurate than simple classification
        """
        tot_validator = self._get_tot_validator()
        
        # Run ToT validation
        validation_result = tot_validator.validate_candidates(
            candidates=candidates,
            section_header=header,
            section_content=content,
            document_tree=document_tree,
            node_id=node_id,
        )
        
        # Convert to entities
        entities = tot_validator.candidates_to_entities(
            candidates=candidates,
            validation_result=validation_result,
            node_id=node_id,
            doc_id=doc_id,
            page_num=page_num,
        )
        
        # Learn from OTHER types
        if self._type_registry:
            for entity in entities:
                if entity.type == EntityType.OTHER:
                    original_type = entity.metadata.get("original_type", "unknown")
                    context = entity.mentions[0].context if entity.mentions else ""
                    self._type_registry.record_type(
                        type_name=original_type,
                        context=context,
                        entity_name=entity.canonical_name,
                    )
        
        logger.info(
            "tot_validation_complete",
            candidates=len(candidates),
            validated=len(entities),
            high_confidence=validation_result.high_confidence_count,
            low_confidence=validation_result.low_confidence_count,
        )
        
        return entities
    
    def _classify_candidates(
        self,
        candidates: list[EntityCandidate],
        content: str,
        node_id: str,
        doc_id: str,
        page_num: int | None,
    ) -> list[Entity]:
        """
        Use LLM to classify pre-extracted candidates.
        
        The LLM's job is VALIDATION and CLASSIFICATION,
        not generation of new entities.
        """
        llm = self._get_llm()
        if llm is None:
            # No LLM - return candidates as-is with pattern-based types
            return self._candidates_to_entities(
                candidates, node_id, doc_id, page_num
            )
        
        entities = []
        
        # Process in batches
        for i in range(0, len(candidates), self.max_candidates_per_batch):
            batch = candidates[i:i + self.max_candidates_per_batch]
            batch_entities = self._classify_batch(
                batch, content, node_id, doc_id, page_num
            )
            entities.extend(batch_entities)
        
        return entities
    
    def _classify_batch(
        self,
        candidates: list[EntityCandidate],
        content: str,
        node_id: str,
        doc_id: str,
        page_num: int | None,
    ) -> list[Entity]:
        """Classify a batch of candidates with LLM."""
        # Format candidates for prompt
        candidates_json = json.dumps([
            {
                "id": idx,
                "text": c.text,
                "type_hint": c.candidate_type,
                "context": c.context,
            }
            for idx, c in enumerate(candidates)
        ], indent=2)
        
        prompt = CLASSIFICATION_PROMPT.format(
            content=content,
            candidates_json=candidates_json,
        )
        
        try:
            _complete_fn = getattr(self.llm, "complete_json", None) or self.llm.complete
            response = _complete_fn(prompt)
            response_text = str(response) if not isinstance(response, str) else response
            
            # Parse classifications
            classifications = self._parse_classification_response(response_text)
            
            # Convert to entities
            entities = []
            for classification in classifications:
                candidate_id = classification.get("candidate_id")
                
                if candidate_id is None or candidate_id >= len(candidates):
                    continue
                
                if not classification.get("valid", False):
                    continue
                
                candidate = candidates[candidate_id]
                entity = self._create_entity_from_classification(
                    candidate=candidate,
                    classification=classification,
                    node_id=node_id,
                    doc_id=doc_id,
                    page_num=page_num,
                )
                
                if entity:
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.warning("classification_failed", error=str(e))
            # Fallback: return candidates as-is
            return self._candidates_to_entities(
                candidates, node_id, doc_id, page_num
            )
    
    def _parse_classification_response(
        self,
        response_text: str,
    ) -> list[dict[str, Any]]:
        """Parse LLM classification response."""
        # Extract JSON
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            json_str = json_match.group(0) if json_match else "[]"
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return []
    
    def _create_entity_from_classification(
        self,
        candidate: EntityCandidate,
        classification: dict[str, Any],
        node_id: str,
        doc_id: str,
        page_num: int | None,
    ) -> Entity | None:
        """Create Entity from candidate + LLM classification."""
        entity_type_str = classification.get("type", candidate.candidate_type).upper()
        
        # Map to EntityType
        entity_type = self._map_entity_type(entity_type_str)
        
        # Record learned type if OTHER
        if entity_type == EntityType.OTHER and self._type_registry:
            self._type_registry.record_type(
                type_name=entity_type_str.lower(),
                context=candidate.context,
                entity_name=candidate.text,
            )
        
        # Get canonical name (LLM-cleaned or original)
        canonical_name = classification.get("canonical_name", candidate.text).strip()
        if not canonical_name:
            canonical_name = candidate.text
        
        # Build metadata
        metadata = {
            "grounded": True,  # Flag that this is grounded in text
            "pattern": candidate.pattern_name,
            "span_start": candidate.start,
            "span_end": candidate.end,
        }
        
        if classification.get("role"):
            metadata["role"] = classification["role"]
        
        if entity_type == EntityType.OTHER:
            metadata["original_type"] = entity_type_str.lower()
        
        # Create mention
        mention = Mention(
            node_id=node_id,
            doc_id=doc_id,
            span_start=candidate.start,
            span_end=candidate.end,
            context=candidate.context,
            page_num=page_num,
            confidence=candidate.confidence,
        )
        
        return Entity(
            type=entity_type,
            canonical_name=canonical_name,
            aliases=[candidate.text] if candidate.text != canonical_name else [],
            mentions=[mention],
            metadata=metadata,
            source_doc_id=doc_id,
        )
    
    def _map_entity_type(self, type_str: str) -> EntityType:
        """Map type string to EntityType enum."""
        type_str = type_str.upper()
        
        mapping = {
            "PERSON": EntityType.PERSON,
            "PEOPLE": EntityType.PERSON,
            "INDIVIDUAL": EntityType.PERSON,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "ORG": EntityType.ORGANIZATION,
            "COMPANY": EntityType.ORGANIZATION,
            "COURT": EntityType.ORGANIZATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
            "EVENT": EntityType.EVENT,
            "LEGAL_CONCEPT": EntityType.LEGAL_CONCEPT,
            "LEGAL": EntityType.LEGAL_CONCEPT,
            "LOCATION": EntityType.LOCATION,
            "PLACE": EntityType.LOCATION,
            "ADDRESS": EntityType.LOCATION,
            "REFERENCE": EntityType.REFERENCE,
            "CITATION": EntityType.REFERENCE,
            "MONETARY": EntityType.MONETARY,
            "MONEY": EntityType.MONETARY,
            "DOCUMENT": EntityType.DOCUMENT,
        }
        
        try:
            return EntityType(type_str.lower())
        except ValueError:
            return mapping.get(type_str, EntityType.OTHER)
    
    def _candidates_to_entities(
        self,
        candidates: list[EntityCandidate],
        node_id: str,
        doc_id: str,
        page_num: int | None,
    ) -> list[Entity]:
        """Convert candidates directly to entities (no LLM)."""
        entities = []
        
        for candidate in candidates:
            entity_type = self._map_entity_type(candidate.candidate_type)
            
            mention = Mention(
                node_id=node_id,
                doc_id=doc_id,
                span_start=candidate.start,
                span_end=candidate.end,
                context=candidate.context,
                page_num=page_num,
                confidence=candidate.confidence,
            )
            
            entity = Entity(
                type=entity_type,
                canonical_name=candidate.text,
                mentions=[mention],
                metadata={
                    "grounded": True,
                    "pattern": candidate.pattern_name,
                    "llm_validated": False,
                },
                source_doc_id=doc_id,
            )
            entities.append(entity)
        
        return entities
    
    def _supplementary_extraction(
        self,
        content: str,
        node_id: str,
        doc_id: str,
        page_num: int | None,
        existing_names: list[str],
    ) -> list[Entity]:
        """
        Check for entities that patterns might have missed.
        
        This is a safety net, but the LLM is instructed to be
        conservative and only add clearly identifiable entities.
        """
        llm = self._get_llm()
        if llm is None:
            return []
        
        prompt = SUPPLEMENTARY_PROMPT.format(
            content=content,
            existing_entities=", ".join(existing_names) if existing_names else "None",
        )
        
        try:
            _complete_fn = getattr(llm, "complete_json", None) or llm.complete
            response = _complete_fn(prompt)
            response_text = str(response) if not isinstance(response, str) else response
            
            # Parse response
            json_match = re.search(r'\[[\s\S]*?\]', response_text)
            if not json_match:
                return []
            
            missed = json.loads(json_match.group())
            
            if not isinstance(missed, list):
                return []
            
            entities = []
            for item in missed:
                text = item.get("text", "").strip()
                if not text or text in existing_names:
                    continue
                
                entity_type = self._map_entity_type(item.get("type", "OTHER"))
                canonical = item.get("canonical_name", text)
                
                # Find the text in content to get position
                match = re.search(re.escape(text), content)
                span_start = match.start() if match else None
                span_end = match.end() if match else None
                
                mention = Mention(
                    node_id=node_id,
                    doc_id=doc_id,
                    span_start=span_start,
                    span_end=span_end,
                    context=content[max(0, (span_start or 0) - 50):(span_end or 0) + 50] if span_start else "",
                    page_num=page_num,
                    confidence=0.6,  # Lower confidence for supplementary
                )
                
                entity = Entity(
                    type=entity_type,
                    canonical_name=canonical,
                    aliases=[text] if text != canonical else [],
                    mentions=[mention],
                    metadata={
                        "grounded": span_start is not None,
                        "supplementary": True,
                        "reason": item.get("reason", ""),
                    },
                    source_doc_id=doc_id,
                )
                entities.append(entity)
            
            if entities:
                logger.debug(
                    "supplementary_entities_found",
                    count=len(entities),
                    names=[e.canonical_name for e in entities],
                )
            
            return entities
            
        except Exception as e:
            logger.debug("supplementary_extraction_failed", error=str(e))
            return []
    
    def clear_cache(self) -> None:
        """Clear the extraction cache."""
        self._cache.clear()
