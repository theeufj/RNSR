"""
RNSR Relationship Validator (ToT Pattern)

Validates relationship candidates using Tree of Thoughts reasoning.
Same pattern as entity validation:

1. Pattern extraction provides grounded candidates
2. ToT evaluates each with probability + reasoning
3. Navigate for context if uncertain
4. Prevents hallucinated relationships
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import structlog

from rnsr.extraction.models import Entity, Relationship, RelationType
from rnsr.extraction.relationship_patterns import RelationshipCandidate
from rnsr.llm import get_llm

if TYPE_CHECKING:
    from rnsr.models import DocumentTree

logger = structlog.get_logger(__name__)


# ToT prompt for relationship validation
TOT_RELATIONSHIP_VALIDATION_PROMPT = """You are validating relationship candidates extracted from a document.

Current Section: {section_header}
Section Content:
---
{section_content}
---

Known Entities:
{entities_formatted}

Relationship Candidates to Validate:
{candidates_formatted}

EVALUATION TASK:
For each relationship candidate, determine if it represents a real, meaningful relationship.

RELATIONSHIP TYPES:
- MENTIONS: Section/document mentions an entity
- TEMPORAL_BEFORE: X occurred before Y
- TEMPORAL_AFTER: X occurred after Y
- CAUSAL: X caused/led to Y
- SUPPORTS: X supports claim in Y
- CONTRADICTS: X contradicts Y
- AFFILIATED_WITH: Person affiliated with organization
- PARTY_TO: Entity is party to document/case
- REFERENCES: References another document/section
- SUPERSEDES: X supersedes/replaces Y
- AMENDS: X amends/modifies Y

For each candidate, provide:
1. valid: true if this is a real, meaningful relationship
2. probability: 0.0-1.0 confidence score
3. relationship_type: Corrected type if pattern was wrong
4. reasoning: Brief explanation

OUTPUT FORMAT (JSON):
{{
    "evaluations": [
        {{
            "candidate_id": 0,
            "valid": true,
            "probability": 0.85,
            "relationship_type": "AFFILIATED_WITH",
            "reasoning": "Evidence clearly shows John Smith is CEO of Acme Corp"
        }},
        {{
            "candidate_id": 1,
            "valid": false,
            "probability": 0.2,
            "reasoning": "Co-occurrence does not indicate actual relationship"
        }}
    ],
    "selected_relationships": [0],
    "needs_more_context": []
}}

Rules:
- Only validate relationships with clear evidence in the text
- Co-occurrence alone is NOT sufficient - need explicit connection
- Set valid=false for weak or ambiguous connections
- Be conservative - uncertain relationships should be rejected

Respond ONLY with the JSON, no other text."""


@dataclass
class RelationshipValidationResult:
    """Result of validating a relationship candidate."""
    
    candidate_id: int
    probability: float
    is_valid: bool
    relationship_type: str | None = None
    reasoning: str = ""
    used_navigation: bool = False


@dataclass
class RelationshipBatchResult:
    """Result of validating a batch of relationship candidates."""
    
    evaluations: list[RelationshipValidationResult] = field(default_factory=list)
    selected_relationships: list[int] = field(default_factory=list)
    needs_more_context: list[int] = field(default_factory=list)


class RelationshipValidator:
    """
    Tree of Thoughts relationship validator.
    
    Validates grounded relationship candidates with:
    - Probability scores for each candidate
    - Explicit reasoning
    - Optional navigation for context
    """
    
    def __init__(
        self,
        llm: Any | None = None,
        selection_threshold: float = 0.6,
        rejection_threshold: float = 0.3,
        max_candidates_per_batch: int = 15,
    ):
        """
        Initialize the relationship validator.
        
        Args:
            llm: LLM instance.
            selection_threshold: Probability threshold for accepting.
            rejection_threshold: Probability threshold for rejecting.
            max_candidates_per_batch: Max candidates per LLM call.
        """
        self.llm = llm
        self.selection_threshold = selection_threshold
        self.rejection_threshold = rejection_threshold
        self.max_candidates_per_batch = max_candidates_per_batch
        
        self._llm_initialized = False
    
    def _get_llm(self) -> Any:
        """Get or initialize LLM."""
        if self.llm is None and not self._llm_initialized:
            self.llm = get_llm()
            self._llm_initialized = True
        return self.llm
    
    def validate_candidates(
        self,
        candidates: list[RelationshipCandidate],
        entities: list[Entity],
        section_header: str,
        section_content: str,
    ) -> RelationshipBatchResult:
        """
        Validate relationship candidates using ToT reasoning.
        
        Args:
            candidates: Pre-extracted relationship candidates.
            entities: Known entities in the section.
            section_header: Section header for context.
            section_content: Section content.
            
        Returns:
            RelationshipBatchResult with validated relationships.
        """
        if not candidates:
            return RelationshipBatchResult()
        
        llm = self._get_llm()
        if llm is None:
            return self._accept_high_confidence(candidates)
        
        # Process in batches
        all_results = RelationshipBatchResult()
        
        for i in range(0, len(candidates), self.max_candidates_per_batch):
            batch = candidates[i:i + self.max_candidates_per_batch]
            batch_offset = i
            
            batch_result = self._validate_batch(
                candidates=batch,
                batch_offset=batch_offset,
                entities=entities,
                section_header=section_header,
                section_content=section_content,
            )
            
            all_results.evaluations.extend(batch_result.evaluations)
            all_results.selected_relationships.extend(batch_result.selected_relationships)
            all_results.needs_more_context.extend(batch_result.needs_more_context)
        
        return all_results
    
    def _validate_batch(
        self,
        candidates: list[RelationshipCandidate],
        batch_offset: int,
        entities: list[Entity],
        section_header: str,
        section_content: str,
    ) -> RelationshipBatchResult:
        """Validate a batch with ToT."""
        # Format entities
        entities_formatted = "\n".join([
            f"- [{e.id}] {e.canonical_name} ({e.type.value})"
            for e in entities
        ]) if entities else "(no entities)"
        
        # Format candidates
        candidates_formatted = "\n".join([
            f"[{i + batch_offset}] {c.source_text} --[{c.relationship_type}]--> {c.target_text}\n"
            f"    Evidence: \"{c.evidence}\"\n"
            f"    Pattern: {c.pattern_name}, Confidence: {c.confidence:.2f}"
            for i, c in enumerate(candidates)
        ])
        
        prompt = TOT_RELATIONSHIP_VALIDATION_PROMPT.format(
            section_header=section_header,
            section_content=section_content,
            entities_formatted=entities_formatted,
            candidates_formatted=candidates_formatted,
        )
        
        try:
            _complete_fn = getattr(self.llm, "complete_json", None) or self.llm.complete
            response = _complete_fn(prompt)
            response_text = str(response) if not isinstance(response, str) else response
            
            return self._parse_validation_response(response_text, len(candidates), batch_offset)
            
        except Exception as e:
            logger.warning("relationship_validation_failed", error=str(e))
            return self._accept_high_confidence(candidates, offset=batch_offset)
    
    def _parse_validation_response(
        self,
        response_text: str,
        candidate_count: int,
        batch_offset: int,
    ) -> RelationshipBatchResult:
        """Parse ToT validation response."""
        result = RelationshipBatchResult()
        
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            return result
        
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return result
        
        for eval_data in data.get("evaluations", []):
            try:
                validation = RelationshipValidationResult(
                    candidate_id=eval_data.get("candidate_id", 0),
                    probability=float(eval_data.get("probability", 0.5)),
                    is_valid=eval_data.get("valid", False),
                    relationship_type=eval_data.get("relationship_type"),
                    reasoning=eval_data.get("reasoning", ""),
                )
                result.evaluations.append(validation)
            except (KeyError, TypeError, ValueError):
                continue
        
        result.selected_relationships = [
            idx for idx in data.get("selected_relationships", [])
            if isinstance(idx, int)
        ]
        
        result.needs_more_context = [
            idx for idx in data.get("needs_more_context", [])
            if isinstance(idx, int)
        ]
        
        return result
    
    def _accept_high_confidence(
        self,
        candidates: list[RelationshipCandidate],
        offset: int = 0,
    ) -> RelationshipBatchResult:
        """Accept only high confidence candidates (fallback)."""
        result = RelationshipBatchResult()
        
        for i, candidate in enumerate(candidates):
            idx = i + offset
            # Only accept high confidence pattern matches
            is_valid = candidate.confidence >= 0.7
            
            result.evaluations.append(RelationshipValidationResult(
                candidate_id=idx,
                probability=candidate.confidence,
                is_valid=is_valid,
                relationship_type=candidate.relationship_type,
                reasoning=f"Pattern: {candidate.pattern_name}" if is_valid else "Low confidence",
            ))
            
            if is_valid:
                result.selected_relationships.append(idx)
        
        return result
    
    def candidates_to_relationships(
        self,
        candidates: list[RelationshipCandidate],
        validation_result: RelationshipBatchResult,
        node_id: str,
        doc_id: str,
    ) -> list[Relationship]:
        """
        Convert validated candidates to Relationship objects.
        """
        relationships = []
        
        eval_by_id = {e.candidate_id: e for e in validation_result.evaluations}
        
        for idx in validation_result.selected_relationships:
            if idx >= len(candidates):
                continue
            
            candidate = candidates[idx]
            evaluation = eval_by_id.get(idx)
            
            if not evaluation or not evaluation.is_valid:
                continue
            
            # Get relationship type
            rel_type_str = evaluation.relationship_type or candidate.relationship_type
            rel_type = self._map_relationship_type(rel_type_str)
            
            # Determine source and target
            source_id = candidate.source_entity_id or node_id
            source_type = "entity" if candidate.source_entity_id else "node"
            target_id = candidate.target_entity_id or f"{doc_id}:{candidate.target_text}"
            target_type = "entity" if candidate.target_entity_id else "node"
            
            relationship = Relationship(
                type=rel_type,
                source_id=source_id,
                source_type=source_type,
                target_id=target_id,
                target_type=target_type,
                confidence=evaluation.probability,
                evidence=candidate.evidence,
                doc_id=doc_id,
                node_id=node_id,
                metadata={
                    "grounded": True,
                    "tot_validated": True,
                    "tot_probability": evaluation.probability,
                    "tot_reasoning": evaluation.reasoning,
                    "pattern": candidate.pattern_name,
                },
            )
            relationships.append(relationship)
        
        return relationships
    
    def _map_relationship_type(self, type_str: str) -> RelationType:
        """Map type string to RelationType enum."""
        type_str = type_str.upper()
        
        try:
            return RelationType(type_str.lower())
        except ValueError:
            mapping = {
                "TEMPORAL_BEFORE": RelationType.TEMPORAL_BEFORE,
                "TEMPORAL_AFTER": RelationType.TEMPORAL_AFTER,
                "TEMPORAL": RelationType.TEMPORAL_BEFORE,
                "CAUSAL": RelationType.CAUSAL,
                "SUPPORTS": RelationType.SUPPORTS,
                "CONTRADICTS": RelationType.CONTRADICTS,
                "AFFILIATED_WITH": RelationType.AFFILIATED_WITH,
                "PARTY_TO": RelationType.PARTY_TO,
                "REFERENCES": RelationType.REFERENCES,
                "SUPERSEDES": RelationType.SUPERSEDES,
                "AMENDS": RelationType.AMENDS,
            }
            return mapping.get(type_str, RelationType.MENTIONS)
