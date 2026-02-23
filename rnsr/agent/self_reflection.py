"""
RNSR Self-Reflection Loop

Implements iterative self-correction where the system:
1. Generates an initial answer
2. Critiques its own answer
3. If issues found, re-navigates with critique as context
4. Repeats until confident or max iterations

Based on self-reflection patterns from:
- Reflexion (Shinn et al.)
- Self-Refine (Madaan et al.)
- Constitutional AI principles

Key insight: LLMs can often identify problems in their own outputs
that they couldn't avoid in initial generation.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Self-Reflection Prompts
# =============================================================================

CRITIQUE_PROMPT = """You are a critical reviewer. Analyze this answer for potential issues.

QUESTION: {question}

ANSWER: {answer}

EVIDENCE USED:
{evidence}

Critically evaluate:
1. ACCURACY: Does the evidence actually support this answer?
2. COMPLETENESS: Is anything important missing?
3. CONTRADICTIONS: Does any evidence contradict the answer?
4. SPECIFICITY: Is the answer too vague or too specific?
5. ASSUMPTIONS: Are there unstated assumptions?

If you find issues, explain them clearly.
If the answer is good, say "NO ISSUES FOUND".

Respond in JSON:
{{
    "has_issues": true/false,
    "issues": [
        {{"type": "accuracy|completeness|contradiction|specificity|assumption", "description": "...", "severity": "high|medium|low"}}
    ],
    "confidence_in_critique": 0.0-1.0,
    "suggested_improvements": ["..."],
    "should_retry": true/false
}}"""


REFINEMENT_PROMPT = """You are improving an answer based on feedback.

ORIGINAL QUESTION: {question}

PREVIOUS ANSWER: {previous_answer}

CRITIQUE/ISSUES FOUND:
{critique}

EVIDENCE AVAILABLE:
{evidence}

Generate an IMPROVED answer that addresses the identified issues.
Be specific and directly address each criticism.

Respond with ONLY the improved answer, no meta-commentary."""


VERIFICATION_PROMPT = """Compare these two answers and determine which is better.

QUESTION: {question}

ANSWER A (Original):
{answer_a}

ANSWER B (Refined):
{answer_b}

Which answer is:
1. More accurate?
2. More complete?
3. Better supported by evidence?

Respond in JSON:
{{
    "better_answer": "A" or "B",
    "confidence": 0.0-1.0,
    "reasoning": "..."
}}"""


# =============================================================================
# Data Models
# =============================================================================

class IssueType(str, Enum):
    """Types of issues that can be identified."""
    
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONTRADICTION = "contradiction"
    SPECIFICITY = "specificity"
    ASSUMPTION = "assumption"
    HALLUCINATION = "hallucination"


class IssueSeverity(str, Enum):
    """Severity of identified issues."""
    
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Issue:
    """An issue identified during self-critique."""
    
    type: IssueType
    description: str
    severity: IssueSeverity
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "description": self.description,
            "severity": self.severity.value,
        }


@dataclass
class CritiqueResult:
    """Result of self-critique."""
    
    has_issues: bool = False
    issues: list[Issue] = field(default_factory=list)
    confidence: float = 0.5
    suggested_improvements: list[str] = field(default_factory=list)
    should_retry: bool = False
    raw_response: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_issues": self.has_issues,
            "issues": [i.to_dict() for i in self.issues],
            "confidence": self.confidence,
            "suggested_improvements": self.suggested_improvements,
            "should_retry": self.should_retry,
        }


@dataclass
class ReflectionIteration:
    """One iteration of the reflection loop."""
    
    iteration: int
    answer: str
    critique: CritiqueResult | None = None
    improved_answer: str | None = None
    improvement_accepted: bool = False
    duration_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "answer": self.answer,
            "critique": self.critique.to_dict() if self.critique else None,
            "improved_answer": self.improved_answer,
            "improvement_accepted": self.improvement_accepted,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ReflectionResult:
    """Complete result of self-reflection process."""
    
    original_answer: str = ""
    final_answer: str = ""
    question: str = ""
    
    # Iterations
    iterations: list[ReflectionIteration] = field(default_factory=list)
    total_iterations: int = 0
    
    # Outcome
    improved: bool = False
    final_confidence: float = 0.0
    all_issues: list[Issue] = field(default_factory=list)
    
    # Timing
    total_duration_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_answer": self.original_answer,
            "final_answer": self.final_answer,
            "question": self.question,
            "iterations": [i.to_dict() for i in self.iterations],
            "total_iterations": self.total_iterations,
            "improved": self.improved,
            "final_confidence": self.final_confidence,
            "all_issues": [i.to_dict() for i in self.all_issues],
            "total_duration_ms": self.total_duration_ms,
        }


# =============================================================================
# Self-Reflection Engine
# =============================================================================


class SelfReflectionEngine:
    """
    Implements iterative self-correction for answers.
    
    Flow:
    1. Take initial answer
    2. Generate critique (what could be wrong?)
    3. If issues found, generate improved answer
    4. Verify improvement is actually better
    5. Repeat until confident or max iterations
    """
    
    def __init__(
        self,
        llm_fn: Callable[[str], str] | None = None,
        max_iterations: int = 3,
        min_confidence_threshold: float = 0.8,
        accept_improvement_threshold: float = 0.6,
        enable_verification: bool = True,
    ):
        """
        Initialize the self-reflection engine.
        
        Args:
            llm_fn: LLM function for critique and refinement.
            max_iterations: Maximum reflection iterations.
            min_confidence_threshold: Stop if critique confidence exceeds this.
            accept_improvement_threshold: Accept improvement if confidence exceeds this.
            enable_verification: Verify improvements are actually better.
        """
        self.llm_fn = llm_fn
        self.max_iterations = max_iterations
        self.min_confidence_threshold = min_confidence_threshold
        self.accept_improvement_threshold = accept_improvement_threshold
        self.enable_verification = enable_verification
        
        # Learning: track which issues are commonly found
        self._issue_stats: dict[str, int] = {}
    
    def set_llm_function(self, llm_fn: Callable[[str], str]) -> None:
        """Set the LLM function."""
        self.llm_fn = llm_fn
    
    def reflect(
        self,
        answer: str,
        question: str,
        evidence: str = "",
        navigate_fn: Callable[[str], str] | None = None,
    ) -> ReflectionResult:
        """
        Perform self-reflection on an answer.
        
        Args:
            answer: The initial answer to reflect on.
            question: The original question.
            evidence: Evidence that was used to generate the answer.
            navigate_fn: Optional function to re-navigate with new context.
            
        Returns:
            ReflectionResult with final answer and iteration history.
        """
        if self.llm_fn is None:
            logger.warning("no_llm_configured_for_reflection")
            return ReflectionResult(
                original_answer=answer,
                final_answer=answer,
                question=question,
            )
        
        start_time = time.time()
        
        result = ReflectionResult(
            original_answer=answer,
            final_answer=answer,
            question=question,
        )
        
        current_answer = answer
        
        for iteration in range(self.max_iterations):
            iter_start = time.time()
            
            logger.info(
                "reflection_iteration_started",
                iteration=iteration + 1,
                max=self.max_iterations,
            )
            
            # Step 1: Critique the current answer
            critique = self._critique(current_answer, question, evidence)
            
            iter_result = ReflectionIteration(
                iteration=iteration + 1,
                answer=current_answer,
                critique=critique,
            )
            
            # Collect issues for statistics
            for issue in critique.issues:
                self._issue_stats[issue.type.value] = \
                    self._issue_stats.get(issue.type.value, 0) + 1
                result.all_issues.append(issue)
            
            # Check if we should stop
            if not critique.has_issues or not critique.should_retry:
                logger.info(
                    "reflection_no_issues",
                    iteration=iteration + 1,
                    confidence=critique.confidence,
                )
                iter_result.duration_ms = (time.time() - iter_start) * 1000
                result.iterations.append(iter_result)
                break
            
            # Step 2: Generate improved answer
            improved_answer = self._refine(
                current_answer, question, critique, evidence
            )
            
            iter_result.improved_answer = improved_answer
            
            # Step 3: Verify improvement (optional)
            if self.enable_verification and improved_answer:
                is_better = self._verify_improvement(
                    question, current_answer, improved_answer
                )
                iter_result.improvement_accepted = is_better
                
                if is_better:
                    current_answer = improved_answer
                    result.improved = True
                    logger.info(
                        "improvement_accepted",
                        iteration=iteration + 1,
                    )
                else:
                    logger.info(
                        "improvement_rejected",
                        iteration=iteration + 1,
                    )
            elif improved_answer:
                # Accept without verification
                current_answer = improved_answer
                iter_result.improvement_accepted = True
                result.improved = True
            
            iter_result.duration_ms = (time.time() - iter_start) * 1000
            result.iterations.append(iter_result)
            
            # Check confidence threshold
            if critique.confidence >= self.min_confidence_threshold:
                logger.info(
                    "confidence_threshold_reached",
                    confidence=critique.confidence,
                    threshold=self.min_confidence_threshold,
                )
                break
        
        result.final_answer = current_answer
        result.total_iterations = len(result.iterations)
        result.total_duration_ms = (time.time() - start_time) * 1000
        
        # Calculate final confidence
        if result.iterations and result.iterations[-1].critique:
            last_critique = result.iterations[-1].critique
            result.final_confidence = last_critique.confidence if not last_critique.has_issues else 1.0 - (len(last_critique.issues) * 0.1)
        else:
            result.final_confidence = 0.7  # Default
        
        logger.info(
            "reflection_complete",
            iterations=result.total_iterations,
            improved=result.improved,
            final_confidence=result.final_confidence,
            duration_ms=result.total_duration_ms,
        )
        
        return result
    
    def _critique(
        self,
        answer: str,
        question: str,
        evidence: str,
    ) -> CritiqueResult:
        """Generate a critique of the answer."""
        prompt = CRITIQUE_PROMPT.format(
            question=question,
            answer=answer,
            evidence=evidence[:2000] if evidence else "No specific evidence provided.",
        )
        
        try:
            response = self.llm_fn(prompt)
            return self._parse_critique(response)
            
        except Exception as e:
            logger.warning("critique_failed", error=str(e))
            return CritiqueResult(
                has_issues=False,
                confidence=0.5,
                raw_response=str(e),
            )
    
    def _parse_critique(self, response: str) -> CritiqueResult:
        """Parse critique response into structured format."""
        result = CritiqueResult(raw_response=response)
        
        # Check for "NO ISSUES FOUND"
        if "NO ISSUES FOUND" in response.upper():
            result.has_issues = False
            result.confidence = 0.9
            return result
        
        # Parse JSON
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                return result
            
            data = json.loads(json_match.group())
            
            result.has_issues = data.get("has_issues", False)
            result.confidence = data.get("confidence_in_critique", 0.5)
            result.should_retry = data.get("should_retry", False)
            result.suggested_improvements = data.get("suggested_improvements", [])
            
            for issue_data in data.get("issues", []):
                try:
                    issue = Issue(
                        type=IssueType(issue_data.get("type", "accuracy")),
                        description=issue_data.get("description", ""),
                        severity=IssueSeverity(issue_data.get("severity", "medium")),
                    )
                    result.issues.append(issue)
                except ValueError:
                    pass
            
        except json.JSONDecodeError:
            # If JSON parsing fails, look for issue indicators
            if any(word in response.lower() for word in ["issue", "problem", "incorrect", "missing"]):
                result.has_issues = True
                result.should_retry = True
        
        return result
    
    def _refine(
        self,
        answer: str,
        question: str,
        critique: CritiqueResult,
        evidence: str,
    ) -> str:
        """Generate an improved answer based on critique."""
        # Format critique for prompt
        critique_text = []
        for issue in critique.issues:
            critique_text.append(f"- [{issue.severity.value.upper()}] {issue.type.value}: {issue.description}")
        
        if critique.suggested_improvements:
            critique_text.append("\nSuggested improvements:")
            for suggestion in critique.suggested_improvements:
                critique_text.append(f"- {suggestion}")
        
        prompt = REFINEMENT_PROMPT.format(
            question=question,
            previous_answer=answer,
            critique="\n".join(critique_text) if critique_text else "No specific issues identified.",
            evidence=evidence[:2000] if evidence else "Use your knowledge to improve the answer.",
        )
        
        try:
            response = self.llm_fn(prompt)
            return response.strip()
            
        except Exception as e:
            logger.warning("refinement_failed", error=str(e))
            return ""
    
    def _verify_improvement(
        self,
        question: str,
        original: str,
        improved: str,
    ) -> bool:
        """Verify that the improved answer is actually better."""
        if not improved:
            return False
        
        prompt = VERIFICATION_PROMPT.format(
            question=question,
            answer_a=original,
            answer_b=improved,
        )
        
        try:
            response = self.llm_fn(prompt)
            
            # Parse response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if not json_match:
                # Default to accepting improvement
                return True
            
            data = json.loads(json_match.group())
            better = data.get("better_answer", "B")
            confidence = data.get("confidence", 0.5)
            
            # Accept B (improved) if confident enough
            return better == "B" and confidence >= self.accept_improvement_threshold
            
        except Exception as e:
            logger.warning("verification_failed", error=str(e))
            # Default to accepting improvement
            return True
    
    def get_issue_stats(self) -> dict[str, int]:
        """Get statistics on issues found across all reflections."""
        return dict(self._issue_stats)


# =============================================================================
# Convenience Functions
# =============================================================================


def reflect_on_answer(
    answer: str,
    question: str,
    evidence: str = "",
    llm_fn: Callable[[str], str] | None = None,
    max_iterations: int = 2,
) -> ReflectionResult:
    """
    Perform self-reflection on an answer.
    
    Simple interface for one-off reflection.
    
    Args:
        answer: The answer to reflect on.
        question: The original question.
        evidence: Evidence used.
        llm_fn: LLM function (uses default if not provided).
        max_iterations: Maximum iterations.
        
    Returns:
        ReflectionResult with final answer.
    """
    if llm_fn is None:
        try:
            from rnsr.llm import get_llm
            llm = get_llm()
            llm_fn = lambda p: str(llm.complete(p))
        except Exception as e:
            logger.warning("no_llm_available", error=str(e))
            return ReflectionResult(
                original_answer=answer,
                final_answer=answer,
                question=question,
            )
    
    engine = SelfReflectionEngine(
        llm_fn=llm_fn,
        max_iterations=max_iterations,
    )
    
    return engine.reflect(answer, question, evidence)


# =============================================================================
# Strict Answer Verification (Critic Loop)
# =============================================================================

STRICT_VERIFICATION_PROMPT = """Check whether every factual claim in the answer is directly supported by the source text. Respond with ONLY a single-line JSON object — nothing else.

Question: {question}

Answer: {answer}

Source text:
{sources}

JSON: {{"verified": true/false, "unsupported_claims": [], "rejection_reason": "", "confidence": 0.0-1.0}}"""


@dataclass
class VerificationResult:
    """Result of strict answer verification."""
    
    verified: bool = False
    claims_analyzed: list[dict[str, Any]] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    rejection_reason: str = ""
    confidence: float = 0.0
    raw_response: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "verified": self.verified,
            "claims_analyzed": self.claims_analyzed,
            "unsupported_claims": self.unsupported_claims,
            "rejection_reason": self.rejection_reason,
            "confidence": self.confidence,
        }


class StrictAnswerVerifier:
    """
    Harsh critic that tries to disprove answers.
    
    This implements the "Red Team" verification step that aggressively
    looks for any claim in the answer that is NOT directly supported
    by the source text.
    
    Key principle: An answer is guilty until proven innocent.
    """
    
    def __init__(
        self,
        llm_fn: Callable[[str], str] | None = None,
        max_unsupported_claims: int = 0,
        min_verification_confidence: float = 0.7,
    ):
        """
        Initialize the strict verifier.
        
        Args:
            llm_fn: LLM function for verification.
            max_unsupported_claims: Maximum unsupported claims allowed (0 = strict mode).
            min_verification_confidence: Minimum confidence to accept verification result.
        """
        self.llm_fn = llm_fn
        self.max_unsupported_claims = max_unsupported_claims
        self.min_verification_confidence = min_verification_confidence
    
    def set_llm_function(self, llm_fn: Callable[[str], str]) -> None:
        """Set the LLM function."""
        self.llm_fn = llm_fn
    
    def verify(
        self,
        answer: str,
        sources: str,
        question: str = "",
    ) -> VerificationResult:
        """
        Verify an answer against source text.
        
        Args:
            answer: The answer to verify.
            sources: The source text (evidence) to check against.
            question: The original question (for context).
            
        Returns:
            VerificationResult indicating if answer is verified or rejected.
        """
        if self.llm_fn is None:
            logger.warning("no_llm_configured_for_strict_verification")
            return VerificationResult(
                verified=False,
                rejection_reason="No LLM available for verification",
            )
        
        # Handle "cannot determine" answers - these are acceptable
        answer_lower = answer.lower().strip()
        if any(phrase in answer_lower for phrase in [
            "cannot answer",
            "cannot determine",
            "not found",
            "i don't know",
            "no information",
            "not available",
            "unable to find",
        ]):
            logger.info("answer_is_explicit_unknown", answer=answer[:100])
            return VerificationResult(
                verified=True,
                confidence=1.0,
                rejection_reason="",
            )
        
        prompt = STRICT_VERIFICATION_PROMPT.format(
            question=question,
            answer=answer,
            sources=sources[:8000] if sources else "No source text provided.",
        )
        
        try:
            response = self.llm_fn(prompt)
            result = self._parse_verification(response)
            
            # Apply strict rules
            if len(result.unsupported_claims) > self.max_unsupported_claims:
                result.verified = False
                if not result.rejection_reason:
                    result.rejection_reason = f"Found {len(result.unsupported_claims)} unsupported claims"
            
            logger.info(
                "strict_verification_complete",
                verified=result.verified,
                claims_count=len(result.claims_analyzed),
                unsupported_count=len(result.unsupported_claims),
                confidence=result.confidence,
            )
            
            return result
            
        except Exception as e:
            logger.warning("strict_verification_failed", error=str(e))
            return VerificationResult(
                verified=False,
                rejection_reason=f"Verification failed: {str(e)}",
            )
    
    def _parse_verification(self, response: str) -> VerificationResult:
        """Parse verification response into structured format."""
        result = VerificationResult(raw_response=response)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            data = None
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            if data is None:
                data = self._recover_truncated_verification(response)

            if data is None:
                result.verified = False
                result.rejection_reason = "Could not parse verification response"
                return result

            result.verified = data.get("verified", False)
            result.claims_analyzed = data.get("claims_analyzed", [])
            result.unsupported_claims = data.get("unsupported_claims", [])
            result.rejection_reason = data.get("rejection_reason", "")
            result.confidence = data.get("confidence", 0.5)
            
        except Exception:
            result.verified = False
            result.rejection_reason = "Invalid JSON in verification response"
        
        return result

    @staticmethod
    def _recover_truncated_verification(response: str) -> dict[str, Any] | None:
        """Recover key fields from a truncated verification JSON response."""
        verified_match = re.search(
            r'"verified"\s*:\s*(true|false)', response, re.IGNORECASE
        )
        conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', response)
        if verified_match is None and conf_match is None:
            return None

        verified = (
            verified_match.group(1).lower() == "true" if verified_match else False
        )
        try:
            confidence = float(conf_match.group(1)) if conf_match else 0.5
        except ValueError:
            confidence = 0.5

        logger.info(
            "critic_verification_recovered_from_truncated_json",
            verified=verified,
            confidence=confidence,
        )
        return {
            "verified": verified,
            "confidence": confidence,
            "claims_analyzed": [],
            "unsupported_claims": [],
            "rejection_reason": "" if verified else "Truncated response (recovered)",
        }


def strict_verify_answer(
    answer: str,
    sources: str,
    question: str = "",
    llm_fn: Callable[[str], str] | None = None,
    max_unsupported_claims: int = 0,
) -> VerificationResult:
    """
    Strictly verify an answer against source text.
    
    This is the main entry point for the critic loop. It attempts to
    disprove the answer by finding any claims not directly supported
    by the source text.
    
    Args:
        answer: The answer to verify.
        sources: The source text (evidence).
        question: The original question.
        llm_fn: LLM function (uses default if not provided).
        max_unsupported_claims: Max unsupported claims allowed (0 = strict).
        
    Returns:
        VerificationResult with verified flag and rejection reason.
    """
    if llm_fn is None:
        try:
            from rnsr.llm import get_llm
            llm = get_llm()
            llm_fn = lambda p: str(llm.complete(p))
        except Exception as e:
            logger.warning("no_llm_available_for_verification", error=str(e))
            return VerificationResult(
                verified=False,
                rejection_reason=f"No LLM available: {str(e)}",
            )
    
    verifier = StrictAnswerVerifier(
        llm_fn=llm_fn,
        max_unsupported_claims=max_unsupported_claims,
    )
    
    return verifier.verify(answer, sources, question)
