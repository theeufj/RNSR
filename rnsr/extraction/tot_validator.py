"""
RNSR Tree of Thoughts Entity Validator

Applies the ToT pattern from the RLM Navigator to entity validation:

1. Given pre-extracted candidates, evaluate each with probability + reasoning
2. Navigate the document tree for additional context when uncertain
3. Make multi-step decisions (like backtracking in document navigation)

This prevents hallucination because:
- Candidates are already grounded in text (from pattern extraction)
- ToT provides structured evaluation with explicit probabilities
- Navigation provides additional context for ambiguous cases
- Same battle-tested pattern used for document Q&A
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from rnsr.extraction.candidate_extractor import EntityCandidate
from rnsr.extraction.models import Entity, EntityType, Mention
from rnsr.llm import get_llm
from rnsr.models import DocumentTree

logger = structlog.get_logger(__name__)


# ToT-style prompt for entity validation (mirrors graph.py ToT_SYSTEM_PROMPT pattern)
TOT_ENTITY_VALIDATION_PROMPT = """You are validating entity candidates extracted from a document.

Current Section: {section_header}
Section Content:
---
{section_content}
---

Entity Candidates to Evaluate:
{candidates_formatted}

EVALUATION TASK:
For each candidate, estimate the probability (0.0 to 1.0) that it is a valid, 
significant entity worth tracking, AND classify its type.

INSTRUCTIONS:
1. Evaluate: For each candidate, analyze its context and estimate validity probability.
2. Valid entities have: clear identity, specific name, significance to document.
3. Invalid entities are: generic terms, partial matches, noise, common words.
4. If probability >= {selection_threshold}, include in selected_entities.
5. If probability < {rejection_threshold}, mark as rejected.
6. Provide brief reasoning for each decision.
7. Classify type: PERSON, ORGANIZATION, DATE, LOCATION, MONETARY, REFERENCE, 
   DOCUMENT, EVENT, LEGAL_CONCEPT, or describe a custom type.

OUTPUT FORMAT (JSON):
{{
    "evaluations": [
        {{
            "candidate_id": 0,
            "probability": 0.85,
            "is_valid": true,
            "entity_type": "PERSON",
            "canonical_name": "John Smith",
            "role": "defendant",
            "reasoning": "Clear person name with title, mentioned as party to case"
        }},
        {{
            "candidate_id": 1,
            "probability": 0.30,
            "is_valid": false,
            "entity_type": null,
            "canonical_name": null,
            "reasoning": "Generic reference to 'the agreement', not a specific entity"
        }}
    ],
    "selected_entities": [0],
    "needs_more_context": [],
    "high_confidence_count": 1,
    "low_confidence_count": 1
}}

If uncertain about a candidate (probability 0.4-0.6), add its id to "needs_more_context".
We may navigate to related sections to gather more information.

Respond ONLY with the JSON, no other text."""


# Prompt for gathering context from related sections
TOT_CONTEXT_GATHERING_PROMPT = """You need more context to validate an entity candidate.

Entity candidate: "{candidate_text}" (type hint: {type_hint})
Original section: {original_section}

Related sections found:
{related_sections}

Based on this additional context, provide your evaluation:
{{
    "candidate_id": {candidate_id},
    "probability": 0.XX,
    "is_valid": true/false,
    "entity_type": "TYPE",
    "canonical_name": "Full Name",
    "reasoning": "With additional context from section X, this is clearly a..."
}}

Respond ONLY with the JSON, no other text."""


@dataclass
class TotValidationResult:
    """Result of ToT entity validation."""
    
    candidate_id: int
    probability: float
    is_valid: bool
    entity_type: str | None = None
    canonical_name: str | None = None
    role: str | None = None
    reasoning: str = ""
    used_navigation: bool = False
    

@dataclass
class TotBatchResult:
    """Result of validating a batch of candidates."""
    
    evaluations: list[TotValidationResult] = field(default_factory=list)
    selected_entities: list[int] = field(default_factory=list)
    needs_more_context: list[int] = field(default_factory=list)
    high_confidence_count: int = 0
    low_confidence_count: int = 0


class TotEntityValidator:
    """
    Tree of Thoughts entity validator.
    
    Uses the same ToT pattern as document navigation:
    - Evaluate candidates with explicit probabilities
    - Navigate for context when uncertain
    - Structured JSON output for reliable parsing
    """
    
    def __init__(
        self,
        llm: Any | None = None,
        selection_threshold: float = 0.6,
        rejection_threshold: float = 0.3,
        enable_navigation: bool = True,
        max_navigation_depth: int = 2,
        max_candidates_per_batch: int = 20,
    ):
        """
        Initialize the ToT validator.
        
        Args:
            llm: LLM instance.
            selection_threshold: Probability threshold for accepting entity.
            rejection_threshold: Probability threshold for rejecting entity.
            enable_navigation: Navigate tree for uncertain candidates.
            max_navigation_depth: Max depth to navigate for context.
            max_candidates_per_batch: Max candidates per LLM call.
        """
        self.llm = llm
        self.selection_threshold = selection_threshold
        self.rejection_threshold = rejection_threshold
        self.enable_navigation = enable_navigation
        self.max_navigation_depth = max_navigation_depth
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
        candidates: list[EntityCandidate],
        section_header: str,
        section_content: str,
        document_tree: DocumentTree | None = None,
        node_id: str | None = None,
    ) -> TotBatchResult:
        """
        Validate entity candidates using ToT reasoning.
        
        Args:
            candidates: Pre-extracted candidates to validate.
            section_header: Current section header.
            section_content: Current section content.
            document_tree: Optional tree for navigation.
            node_id: Current node ID for navigation.
            
        Returns:
            TotBatchResult with validated entities.
        """
        if not candidates:
            return TotBatchResult()
        
        llm = self._get_llm()
        if llm is None:
            # No LLM - accept all candidates with pattern-based types
            return self._accept_all_candidates(candidates)
        
        # Process in batches
        all_results = TotBatchResult()
        
        for i in range(0, len(candidates), self.max_candidates_per_batch):
            batch = candidates[i:i + self.max_candidates_per_batch]
            batch_offset = i
            
            batch_result = self._validate_batch(
                candidates=batch,
                batch_offset=batch_offset,
                section_header=section_header,
                section_content=section_content,
            )
            
            # Merge results
            all_results.evaluations.extend(batch_result.evaluations)
            all_results.selected_entities.extend(batch_result.selected_entities)
            all_results.needs_more_context.extend(batch_result.needs_more_context)
            all_results.high_confidence_count += batch_result.high_confidence_count
            all_results.low_confidence_count += batch_result.low_confidence_count
        
        # Handle uncertain candidates if navigation is enabled
        if self.enable_navigation and document_tree and all_results.needs_more_context:
            all_results = self._resolve_uncertain_candidates(
                candidates=candidates,
                batch_result=all_results,
                document_tree=document_tree,
                current_node_id=node_id,
            )
        
        return all_results
    
    def _validate_batch(
        self,
        candidates: list[EntityCandidate],
        batch_offset: int,
        section_header: str,
        section_content: str,
    ) -> TotBatchResult:
        """Validate a batch of candidates with ToT."""
        # Format candidates for prompt
        candidates_formatted = "\n".join([
            f"[{i + batch_offset}] Text: \"{c.text}\" | Type Hint: {c.candidate_type} | "
            f"Context: \"...{c.context}...\""
            for i, c in enumerate(candidates)
        ])
        
        prompt = TOT_ENTITY_VALIDATION_PROMPT.format(
            section_header=section_header,
            section_content=section_content,
            candidates_formatted=candidates_formatted,
            selection_threshold=self.selection_threshold,
            rejection_threshold=self.rejection_threshold,
        )
        
        try:
            _complete_fn = getattr(self.llm, "complete_json", None) or self.llm.complete
            response = _complete_fn(prompt)
            response_text = str(response) if not isinstance(response, str) else response
            
            return self._parse_validation_response(response_text, len(candidates), batch_offset)
            
        except Exception as e:
            logger.warning("tot_validation_failed", error=str(e))
            return self._accept_all_candidates(candidates, offset=batch_offset)
    
    def _parse_validation_response(
        self,
        response_text: str,
        candidate_count: int,
        batch_offset: int,
    ) -> TotBatchResult:
        """Parse ToT validation response."""
        result = TotBatchResult()
        
        # Extract JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            logger.warning("tot_no_json_found")
            return result
        
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning("tot_json_parse_failed", error=str(e))
            return result
        
        # Parse evaluations
        for eval_data in data.get("evaluations", []):
            try:
                validation = TotValidationResult(
                    candidate_id=eval_data.get("candidate_id", 0),
                    probability=float(eval_data.get("probability", 0.5)),
                    is_valid=eval_data.get("is_valid", False),
                    entity_type=eval_data.get("entity_type"),
                    canonical_name=eval_data.get("canonical_name"),
                    role=eval_data.get("role"),
                    reasoning=eval_data.get("reasoning", ""),
                )
                result.evaluations.append(validation)
                
                if validation.is_valid:
                    result.high_confidence_count += 1
                else:
                    result.low_confidence_count += 1
                    
            except (KeyError, TypeError, ValueError) as e:
                logger.debug("tot_eval_parse_error", error=str(e))
                continue
        
        # Parse selected entities (adjust for batch offset)
        result.selected_entities = [
            idx for idx in data.get("selected_entities", [])
            if isinstance(idx, int)
        ]
        
        # Parse needs_more_context
        result.needs_more_context = [
            idx for idx in data.get("needs_more_context", [])
            if isinstance(idx, int)
        ]
        
        return result
    
    def _resolve_uncertain_candidates(
        self,
        candidates: list[EntityCandidate],
        batch_result: TotBatchResult,
        document_tree: DocumentTree,
        current_node_id: str | None,
    ) -> TotBatchResult:
        """
        Navigate document tree to resolve uncertain candidates.
        
        This is like backtracking in document Q&A - gather more context
        to make a better decision.
        """
        if not batch_result.needs_more_context:
            return batch_result
        
        logger.info(
            "tot_navigating_for_context",
            uncertain_count=len(batch_result.needs_more_context),
        )
        
        # Find related sections
        related_sections = self._find_related_sections(
            document_tree=document_tree,
            current_node_id=current_node_id,
            depth=self.max_navigation_depth,
        )
        
        if not related_sections:
            # No related sections - accept uncertain candidates with lower confidence
            for idx in batch_result.needs_more_context:
                if idx < len(candidates):
                    # Add as selected with moderate confidence
                    batch_result.selected_entities.append(idx)
            return batch_result
        
        # Re-evaluate uncertain candidates with additional context
        for idx in batch_result.needs_more_context:
            if idx >= len(candidates):
                continue
                
            candidate = candidates[idx]
            
            resolved = self._resolve_single_candidate(
                candidate=candidate,
                candidate_id=idx,
                related_sections=related_sections,
            )
            
            if resolved and resolved.is_valid:
                # Update the evaluation
                for i, eval_item in enumerate(batch_result.evaluations):
                    if eval_item.candidate_id == idx:
                        batch_result.evaluations[i] = resolved
                        break
                else:
                    batch_result.evaluations.append(resolved)
                
                batch_result.selected_entities.append(idx)
                batch_result.high_confidence_count += 1
        
        # Clear needs_more_context since we've processed them
        batch_result.needs_more_context = []
        
        return batch_result
    
    def _find_related_sections(
        self,
        document_tree: DocumentTree,
        current_node_id: str | None,
        depth: int,
    ) -> list[dict[str, str]]:
        """Find related sections for context gathering."""
        sections = []
        
        if not document_tree or not document_tree.root:
            return sections
        
        # Collect sections from tree (siblings and nearby nodes)
        def collect_sections(node: Any, current_depth: int) -> None:
            if current_depth > depth:
                return
            
            if hasattr(node, 'header') and hasattr(node, 'content'):
                node_id = getattr(node, 'id', str(id(node)))
                if node_id != current_node_id:
                    sections.append({
                        "header": node.header or "(no header)",
                        "content": node.content or "",
                    })
            
            if hasattr(node, 'children'):
                for child in node.children:
                    collect_sections(child, current_depth + 1)
        
        collect_sections(document_tree.root, 0)
        
        return sections
    
    def _resolve_single_candidate(
        self,
        candidate: EntityCandidate,
        candidate_id: int,
        related_sections: list[dict[str, str]],
    ) -> TotValidationResult | None:
        """Resolve a single uncertain candidate with additional context."""
        llm = self._get_llm()
        if llm is None:
            return None
        
        # Format related sections
        sections_text = "\n\n".join([
            f"### {s['header']}\n{s['content']}"
            for s in related_sections[:5]
        ])
        
        prompt = TOT_CONTEXT_GATHERING_PROMPT.format(
            candidate_text=candidate.text,
            type_hint=candidate.candidate_type,
            original_section=candidate.context,
            related_sections=sections_text,
            candidate_id=candidate_id,
        )
        
        try:
            _complete_fn = getattr(llm, "complete_json", None) or llm.complete
            response = _complete_fn(prompt)
            response_text = str(response) if not isinstance(response, str) else response
            
            # Parse response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                return None
            
            data = json.loads(json_match.group())
            
            return TotValidationResult(
                candidate_id=candidate_id,
                probability=float(data.get("probability", 0.5)),
                is_valid=data.get("is_valid", False),
                entity_type=data.get("entity_type"),
                canonical_name=data.get("canonical_name"),
                role=data.get("role"),
                reasoning=data.get("reasoning", ""),
                used_navigation=True,
            )
            
        except Exception as e:
            logger.debug("tot_context_resolution_failed", error=str(e))
            return None
    
    def _accept_all_candidates(
        self,
        candidates: list[EntityCandidate],
        offset: int = 0,
    ) -> TotBatchResult:
        """Accept all candidates without validation (fallback)."""
        result = TotBatchResult()
        
        for i, candidate in enumerate(candidates):
            idx = i + offset
            result.evaluations.append(TotValidationResult(
                candidate_id=idx,
                probability=candidate.confidence,
                is_valid=True,
                entity_type=candidate.candidate_type.upper(),
                canonical_name=candidate.text,
                reasoning="Accepted without LLM validation",
            ))
            result.selected_entities.append(idx)
        
        result.high_confidence_count = len(candidates)
        return result
    
    def candidates_to_entities(
        self,
        candidates: list[EntityCandidate],
        validation_result: TotBatchResult,
        node_id: str,
        doc_id: str,
        page_num: int | None = None,
    ) -> list[Entity]:
        """
        Convert validated candidates to Entity objects.
        
        Only includes candidates that passed ToT validation.
        """
        entities = []
        
        # Build lookup for evaluations
        eval_by_id = {e.candidate_id: e for e in validation_result.evaluations}
        
        for idx in validation_result.selected_entities:
            if idx >= len(candidates):
                continue
            
            candidate = candidates[idx]
            evaluation = eval_by_id.get(idx)
            
            if not evaluation or not evaluation.is_valid:
                continue
            
            # Map entity type
            entity_type = self._map_entity_type(
                evaluation.entity_type or candidate.candidate_type
            )
            
            # Get canonical name
            canonical_name = evaluation.canonical_name or candidate.text
            
            # Create mention
            mention = Mention(
                node_id=node_id,
                doc_id=doc_id,
                span_start=candidate.start,
                span_end=candidate.end,
                context=candidate.context,
                page_num=page_num,
                confidence=evaluation.probability,
            )
            
            # Build metadata
            metadata = {
                "grounded": True,
                "tot_validated": True,
                "tot_probability": evaluation.probability,
                "tot_reasoning": evaluation.reasoning,
                "pattern": candidate.pattern_name,
            }
            
            if evaluation.role:
                metadata["role"] = evaluation.role
            
            if evaluation.used_navigation:
                metadata["used_context_navigation"] = True
            
            if entity_type == EntityType.OTHER:
                metadata["original_type"] = (evaluation.entity_type or "").lower()
            
            entity = Entity(
                type=entity_type,
                canonical_name=canonical_name,
                aliases=[candidate.text] if candidate.text != canonical_name else [],
                mentions=[mention],
                metadata=metadata,
                source_doc_id=doc_id,
            )
            entities.append(entity)
        
        return entities
    
    def _map_entity_type(self, type_str: str) -> EntityType:
        """Map type string to EntityType enum."""
        type_str = type_str.upper()
        
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "ORG": EntityType.ORGANIZATION,
            "DATE": EntityType.DATE,
            "LOCATION": EntityType.LOCATION,
            "MONETARY": EntityType.MONETARY,
            "MONEY": EntityType.MONETARY,
            "REFERENCE": EntityType.REFERENCE,
            "DOCUMENT": EntityType.DOCUMENT,
            "EVENT": EntityType.EVENT,
            "LEGAL_CONCEPT": EntityType.LEGAL_CONCEPT,
            "LEGAL": EntityType.LEGAL_CONCEPT,
        }
        
        try:
            return EntityType(type_str.lower())
        except ValueError:
            return mapping.get(type_str, EntityType.OTHER)
