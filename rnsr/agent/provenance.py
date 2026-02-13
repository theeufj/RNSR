"""
RNSR Provenance and Citation System

Every answer should trace back to exact document evidence.
Provides structured citations with:
- Exact document location (doc_id, node_id, page_num)
- Exact quote with character spans
- Confidence score per citation
- Contradiction detection when sources disagree

Critical for legal, academic, and enterprise use cases where
answers must be verifiable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Citation Models
# =============================================================================


class CitationStrength(str, Enum):
    """How strongly a citation supports a claim."""
    
    DIRECT = "direct"          # Explicitly states the claim
    SUPPORTING = "supporting"  # Implies or supports the claim
    CONTEXTUAL = "contextual"  # Provides background context
    WEAK = "weak"              # Tangentially related


class ContradictionType(str, Enum):
    """Types of contradictions between sources."""
    
    DIRECT = "direct"            # Sources directly contradict
    TEMPORAL = "temporal"        # Different time periods
    PARTIAL = "partial"          # Partially contradictory
    INTERPRETATION = "interpretation"  # Different interpretations


@dataclass
class Citation:
    """
    A structured citation linking an answer to document evidence.
    
    Provides complete traceability from answer to source.
    """
    
    id: str = field(default_factory=lambda: f"cite_{str(uuid4())[:8]}")
    
    # Document location
    doc_id: str = ""
    node_id: str = ""
    page_num: int | None = None
    
    # Exact quote
    quote: str = ""
    span_start: int | None = None
    span_end: int | None = None
    
    # Context around the quote
    context_before: str = ""
    context_after: str = ""
    
    # Relevance and confidence
    strength: CitationStrength = CitationStrength.SUPPORTING
    confidence: float = 0.5
    relevance_score: float = 0.5
    
    # Claim this citation supports
    claim: str = ""
    
    # Metadata
    extracted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    section_header: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "node_id": self.node_id,
            "page_num": self.page_num,
            "quote": self.quote,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "claim": self.claim,
            "section_header": self.section_header,
            "extracted_at": self.extracted_at,
        }
    
    def to_formatted_string(self, include_context: bool = False) -> str:
        """Format citation for display."""
        parts = []
        
        if self.doc_id:
            parts.append(f"[{self.doc_id}]")
        
        if self.section_header:
            parts.append(f"Section: {self.section_header}")
        
        if self.page_num:
            parts.append(f"Page {self.page_num}")
        
        location = ", ".join(parts) if parts else "Unknown location"
        
        quote_display = self.quote
        if len(quote_display) > 200:
            quote_display = quote_display[:200] + "..."
        
        result = f'{location}: "{quote_display}"'
        
        if include_context and (self.context_before or self.context_after):
            result += f"\n  Context: ...{self.context_before[-50:]}" if self.context_before else ""
            result += f"[QUOTE]{self.context_after[:50]}..." if self.context_after else ""
        
        return result


@dataclass
class Contradiction:
    """A detected contradiction between citations."""
    
    id: str = field(default_factory=lambda: f"contra_{str(uuid4())[:8]}")
    
    citation_1_id: str = ""
    citation_2_id: str = ""
    
    type: ContradictionType = ContradictionType.PARTIAL
    
    description: str = ""
    resolution_suggestion: str = ""
    
    confidence: float = 0.5
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "citation_1_id": self.citation_1_id,
            "citation_2_id": self.citation_2_id,
            "type": self.type.value,
            "description": self.description,
            "resolution_suggestion": self.resolution_suggestion,
            "confidence": self.confidence,
        }


@dataclass
class ProvenanceRecord:
    """
    Complete provenance for an answer.
    
    Links an answer to all supporting citations and any contradictions.
    """
    
    id: str = field(default_factory=lambda: f"prov_{str(uuid4())[:8]}")
    
    # The answer being traced
    answer: str = ""
    question: str = ""
    
    # All citations supporting this answer
    citations: list[Citation] = field(default_factory=list)
    
    # Detected contradictions
    contradictions: list[Contradiction] = field(default_factory=list)
    
    # Overall confidence based on citations
    aggregate_confidence: float = 0.0
    
    # Summary
    evidence_summary: str = ""
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "answer": self.answer,
            "question": self.question,
            "citations": [c.to_dict() for c in self.citations],
            "contradictions": [c.to_dict() for c in self.contradictions],
            "aggregate_confidence": self.aggregate_confidence,
            "evidence_summary": self.evidence_summary,
            "created_at": self.created_at,
        }
    
    def to_markdown(self) -> str:
        """Export as markdown for documentation."""
        lines = [
            f"# Provenance Record",
            f"",
            f"**Question:** {self.question}",
            f"",
            f"**Answer:** {self.answer}",
            f"",
            f"**Confidence:** {self.aggregate_confidence:.2%}",
            f"",
            f"## Citations ({len(self.citations)})",
            f"",
        ]
        
        for i, citation in enumerate(self.citations, 1):
            lines.append(f"### Citation {i}")
            lines.append(f"- **Document:** {citation.doc_id}")
            lines.append(f"- **Section:** {citation.section_header or citation.node_id}")
            if citation.page_num:
                lines.append(f"- **Page:** {citation.page_num}")
            lines.append(f"- **Strength:** {citation.strength.value}")
            lines.append(f"- **Confidence:** {citation.confidence:.2%}")
            lines.append(f"")
            lines.append(f"> {citation.quote}")
            lines.append(f"")
        
        if self.contradictions:
            lines.append(f"## Contradictions ({len(self.contradictions)})")
            lines.append(f"")
            for contra in self.contradictions:
                lines.append(f"- **Type:** {contra.type.value}")
                lines.append(f"- **Description:** {contra.description}")
                if contra.resolution_suggestion:
                    lines.append(f"- **Resolution:** {contra.resolution_suggestion}")
                lines.append(f"")
        
        return "\n".join(lines)


# =============================================================================
# Provenance Tracker
# =============================================================================


class ProvenanceTracker:
    """
    Tracks provenance for answers.
    
    Extracts citations from navigation results and detects contradictions.
    """
    
    def __init__(
        self,
        kv_store: Any | None = None,
        skeleton: dict | None = None,
        min_quote_length: int = 20,
        context_window: int = 100,
    ):
        """
        Initialize the provenance tracker.
        
        Args:
            kv_store: KV store for retrieving full content.
            skeleton: Skeleton index for node metadata.
            min_quote_length: Minimum quote length to consider.
            context_window: Characters of context around quotes.
        """
        self.kv_store = kv_store
        self.skeleton = skeleton or {}
        self.min_quote_length = min_quote_length
        self.context_window = context_window
    
    def extract_citations(
        self,
        answer: str,
        question: str,
        variables: dict[str, Any],
        trace: list[dict] | None = None,
    ) -> list[Citation]:
        """
        Extract citations from navigation results.
        
        Args:
            answer: The generated answer.
            question: The original question.
            variables: Variable store contents (contains retrieved content).
            trace: Navigation trace entries.
            
        Returns:
            List of Citation objects.
        """
        citations = []
        
        # Extract from variables (most reliable source)
        for var_name, var_data in variables.items():
            if isinstance(var_data, dict):
                citation = self._extract_citation_from_variable(
                    var_name, var_data, answer
                )
                if citation:
                    citations.append(citation)
        
        # Extract from trace if available
        if trace:
            trace_citations = self._extract_citations_from_trace(trace, answer)
            citations.extend(trace_citations)
        
        # Deduplicate
        citations = self._deduplicate_citations(citations)
        
        # Score relevance to answer
        for citation in citations:
            citation.relevance_score = self._score_relevance(
                citation.quote, answer
            )
        
        # Sort by relevance
        citations.sort(key=lambda c: -c.relevance_score)
        
        logger.info(
            "citations_extracted",
            count=len(citations),
            question=question[:50],
        )
        
        return citations
    
    def _extract_citation_from_variable(
        self,
        var_name: str,
        var_data: dict,
        answer: str,
    ) -> Citation | None:
        """Extract citation from a stored variable."""
        content = var_data.get("content", "")
        
        if not content or len(content) < self.min_quote_length:
            return None
        
        # Find the most relevant quote from this content
        quote, span_start, span_end = self._find_best_quote(content, answer)
        
        if not quote:
            return None
        
        # Get context around quote
        context_before = content[max(0, span_start - self.context_window):span_start]
        context_after = content[span_end:span_end + self.context_window]
        
        # Determine citation strength
        strength = self._determine_strength(quote, answer)
        
        # Get node metadata
        node_id = var_data.get("node_id", var_name)
        doc_id = var_data.get("doc_id", "")
        page_num = var_data.get("page_num")
        
        # Get section header from skeleton
        section_header = ""
        if node_id in self.skeleton:
            section_header = self.skeleton[node_id].get("header", "")
        
        return Citation(
            doc_id=doc_id,
            node_id=node_id,
            page_num=page_num,
            quote=quote,
            span_start=span_start,
            span_end=span_end,
            context_before=context_before,
            context_after=context_after,
            strength=strength,
            confidence=0.8 if strength == CitationStrength.DIRECT else 0.6,
            section_header=section_header,
        )
    
    def _extract_citations_from_trace(
        self,
        trace: list[dict],
        answer: str,
    ) -> list[Citation]:
        """Extract citations from navigation trace."""
        citations = []
        
        for entry in trace:
            if entry.get("action") == "read_content":
                node_id = entry.get("node_id", "")
                content = entry.get("content", "")
                
                if content and len(content) >= self.min_quote_length:
                    quote, start, end = self._find_best_quote(content, answer)
                    
                    if quote:
                        citations.append(Citation(
                            node_id=node_id,
                            quote=quote,
                            span_start=start,
                            span_end=end,
                            strength=self._determine_strength(quote, answer),
                        ))
        
        return citations
    
    def _find_best_quote(
        self,
        content: str,
        answer: str,
    ) -> tuple[str, int, int]:
        """Find the most relevant quote from content."""
        # Split answer into key phrases
        answer_words = set(answer.lower().split())
        
        # Find sentences in content
        sentences = re.split(r'[.!?]\s+', content)
        
        best_quote = ""
        best_score = 0
        best_start = 0
        best_end = 0
        
        current_pos = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < self.min_quote_length:
                current_pos += len(sentence) + 2
                continue
            
            # Score by word overlap
            sentence_words = set(sentence.lower().split())
            overlap = len(answer_words & sentence_words)
            score = overlap / max(len(answer_words), 1)
            
            if score > best_score:
                best_score = score
                best_quote = sentence
                best_start = content.find(sentence, current_pos)
                best_end = best_start + len(sentence)
            
            current_pos += len(sentence) + 2
        
        return best_quote, best_start, best_end
    
    def _determine_strength(self, quote: str, answer: str) -> CitationStrength:
        """Determine how strongly a quote supports the answer."""
        quote_lower = quote.lower()
        answer_lower = answer.lower()
        
        # Check for direct overlap
        answer_words = set(answer_lower.split())
        quote_words = set(quote_lower.split())
        overlap = len(answer_words & quote_words) / max(len(answer_words), 1)
        
        if overlap > 0.5:
            return CitationStrength.DIRECT
        elif overlap > 0.3:
            return CitationStrength.SUPPORTING
        elif overlap > 0.1:
            return CitationStrength.CONTEXTUAL
        else:
            return CitationStrength.WEAK
    
    def _score_relevance(self, quote: str, answer: str) -> float:
        """Score relevance of quote to answer."""
        if not quote or not answer:
            return 0.0
        
        quote_words = set(quote.lower().split())
        answer_words = set(answer.lower().split())
        
        if not answer_words:
            return 0.0
        
        overlap = len(quote_words & answer_words)
        return overlap / len(answer_words)
    
    def _deduplicate_citations(
        self,
        citations: list[Citation],
    ) -> list[Citation]:
        """Remove duplicate citations."""
        seen_quotes = set()
        unique = []
        
        for citation in citations:
            # Hash the quote
            quote_hash = hashlib.md5(citation.quote.encode()).hexdigest()[:16]
            
            if quote_hash not in seen_quotes:
                seen_quotes.add(quote_hash)
                unique.append(citation)
        
        return unique
    
    def detect_contradictions(
        self,
        citations: list[Citation],
        llm_fn: Any | None = None,
    ) -> list[Contradiction]:
        """
        Detect contradictions between citations.
        
        Args:
            citations: List of citations to check.
            llm_fn: Optional LLM function for semantic contradiction detection.
            
        Returns:
            List of Contradiction objects.
        """
        contradictions = []
        
        if len(citations) < 2:
            return contradictions
        
        # Simple heuristic-based detection
        for i, c1 in enumerate(citations):
            for c2 in citations[i + 1:]:
                contradiction = self._check_contradiction(c1, c2)
                if contradiction:
                    contradictions.append(contradiction)
        
        # Optional: LLM-based semantic detection
        if llm_fn and len(citations) >= 2:
            semantic_contradictions = self._detect_semantic_contradictions(
                citations, llm_fn
            )
            contradictions.extend(semantic_contradictions)
        
        return contradictions
    
    def _check_contradiction(
        self,
        c1: Citation,
        c2: Citation,
    ) -> Contradiction | None:
        """Check for contradiction between two citations using heuristics."""
        q1 = c1.quote.lower()
        q2 = c2.quote.lower()
        
        # Look for negation patterns
        negation_pairs = [
            ("is not", "is"),
            ("was not", "was"),
            ("did not", "did"),
            ("cannot", "can"),
            ("never", "always"),
            ("false", "true"),
            ("incorrect", "correct"),
        ]
        
        for neg, pos in negation_pairs:
            if (neg in q1 and pos in q2 and neg not in q2) or \
               (neg in q2 and pos in q1 and neg not in q1):
                return Contradiction(
                    citation_1_id=c1.id,
                    citation_2_id=c2.id,
                    type=ContradictionType.DIRECT,
                    description=f"Potential negation contradiction detected",
                    confidence=0.6,
                )
        
        # Look for number contradictions
        nums_1 = re.findall(r'\$?[\d,]+\.?\d*', q1)
        nums_2 = re.findall(r'\$?[\d,]+\.?\d*', q2)
        
        if nums_1 and nums_2 and nums_1 != nums_2:
            # Could be contradictory numbers
            return Contradiction(
                citation_1_id=c1.id,
                citation_2_id=c2.id,
                type=ContradictionType.PARTIAL,
                description=f"Different numbers mentioned: {nums_1} vs {nums_2}",
                confidence=0.4,
            )
        
        return None
    
    def _detect_semantic_contradictions(
        self,
        citations: list[Citation],
        llm_fn: Any,
    ) -> list[Contradiction]:
        """Use LLM to detect semantic contradictions."""
        if len(citations) < 2:
            return []
        
        # Build prompt with top citations
        quotes_text = "\n".join([
            f"[{i+1}] {c.quote}"
            for i, c in enumerate(citations)
        ])
        
        prompt = f"""Analyze these quotes for contradictions:

{quotes_text}

Do any of these quotes contradict each other? If yes, specify which quotes (by number) and explain the contradiction.

Respond in JSON:
{{
  "contradictions": [
    {{"quote_1": 1, "quote_2": 2, "type": "direct|temporal|partial", "explanation": "..."}}
  ]
}}

If no contradictions, respond: {{"contradictions": []}}"""

        try:
            response = llm_fn(prompt)
            
            # Parse response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return []
            
            data = json.loads(json_match.group())
            
            contradictions = []
            for c in data.get("contradictions", []):
                idx1 = c.get("quote_1", 1) - 1
                idx2 = c.get("quote_2", 2) - 1
                
                if 0 <= idx1 < len(citations) and 0 <= idx2 < len(citations):
                    contradictions.append(Contradiction(
                        citation_1_id=citations[idx1].id,
                        citation_2_id=citations[idx2].id,
                        type=ContradictionType(c.get("type", "partial")),
                        description=c.get("explanation", ""),
                        confidence=0.7,
                    ))
            
            return contradictions
            
        except Exception as e:
            logger.warning("semantic_contradiction_detection_failed", error=str(e))
            return []
    
    def create_provenance_record(
        self,
        answer: str,
        question: str,
        variables: dict[str, Any],
        trace: list[dict] | None = None,
        llm_fn: Any | None = None,
    ) -> ProvenanceRecord:
        """
        Create a complete provenance record for an answer.
        
        Args:
            answer: The generated answer.
            question: The original question.
            variables: Variable store contents.
            trace: Navigation trace.
            llm_fn: Optional LLM for contradiction detection.
            
        Returns:
            ProvenanceRecord with citations and contradictions.
        """
        # Extract citations
        citations = self.extract_citations(answer, question, variables, trace)
        
        # Detect contradictions
        contradictions = self.detect_contradictions(citations, llm_fn)
        
        # Calculate aggregate confidence
        if citations:
            # Weighted average by relevance
            total_weight = sum(c.relevance_score for c in citations)
            if total_weight > 0:
                aggregate = sum(
                    c.confidence * c.relevance_score for c in citations
                ) / total_weight
            else:
                aggregate = sum(c.confidence for c in citations) / len(citations)
            
            # Reduce confidence if contradictions found
            if contradictions:
                aggregate *= 0.8
        else:
            aggregate = 0.0
        
        # Generate evidence summary
        summary = self._generate_evidence_summary(citations, contradictions)
        
        record = ProvenanceRecord(
            answer=answer,
            question=question,
            citations=citations,
            contradictions=contradictions,
            aggregate_confidence=aggregate,
            evidence_summary=summary,
        )
        
        logger.info(
            "provenance_record_created",
            citations=len(citations),
            contradictions=len(contradictions),
            confidence=aggregate,
        )
        
        return record
    
    def _generate_evidence_summary(
        self,
        citations: list[Citation],
        contradictions: list[Contradiction],
    ) -> str:
        """Generate a summary of evidence quality."""
        if not citations:
            return "No supporting evidence found."
        
        direct = sum(1 for c in citations if c.strength == CitationStrength.DIRECT)
        supporting = sum(1 for c in citations if c.strength == CitationStrength.SUPPORTING)
        
        parts = []
        parts.append(f"Found {len(citations)} citation(s)")
        
        if direct:
            parts.append(f"{direct} directly supporting")
        if supporting:
            parts.append(f"{supporting} supporting")
        
        if contradictions:
            parts.append(f"WARNING: {len(contradictions)} contradiction(s) detected")
        
        return ". ".join(parts) + "."


# =============================================================================
# Convenience Functions
# =============================================================================


def create_citation(
    doc_id: str,
    node_id: str,
    quote: str,
    page_num: int | None = None,
    strength: str = "supporting",
) -> Citation:
    """Create a citation with minimal parameters."""
    return Citation(
        doc_id=doc_id,
        node_id=node_id,
        quote=quote,
        page_num=page_num,
        strength=CitationStrength(strength),
    )


def format_citations_for_display(citations: list[Citation]) -> str:
    """Format citations for user display."""
    if not citations:
        return "No citations available."
    
    lines = ["**Sources:**", ""]
    
    for i, citation in enumerate(citations, 1):
        lines.append(f"{i}. {citation.to_formatted_string()}")
        lines.append("")
    
    return "\n".join(lines)
