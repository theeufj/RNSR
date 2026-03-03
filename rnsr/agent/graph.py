"""
Agent Graph - DEPRECATED LangGraph State Machine for Document Navigation

NOTE: The primary navigator is now RLMNavigator (rnsr/agent/rlm_navigator.py).
run_navigator() delegates to RLMNavigator.navigate() as of the consolidation.
The LangGraph graph builder and node functions below are retained for
backward compatibility but are no longer the active code path.

Key capabilities (arithmetic synthesis, vision augmentation, header-match
fallback, short-answer extraction) have been ported to rlm_navigator.py.

Original description:
Implements the Navigator Agent with full RLM (Recursive Language Model) support:

1. Decomposes queries into sub-questions (Section 2.2 - Recursive Loop)
2. Navigates the document tree (expand/traverse decisions)
3. Stores findings as pointers (Variable Stitching - Section 2.2)
4. Synthesizes final answer from stored pointers
5. Tree of Thoughts prompting (Section 7.2)
6. Recursive sub-LLM invocation for complex queries

Agent State follows Appendix C specification:
- question: Current question being answered
- sub_questions: Decomposed sub-questions (via LLM)
- current_node_id: Where we are in the document tree
- visited_nodes: Navigation history
- variables: Stored findings as $POINTER -> content
- pending_questions: Sub-questions not yet answered
- context: Accumulated context (pointers only!)
- answer: Final synthesized answer
- trace: Full retrieval trace for transparency
"""

from __future__ import annotations

import ast
import math
import operator as _operator
import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional, TypedDict, cast

import structlog

from rnsr.agent.variable_store import VariableStore, generate_pointer_name
from rnsr.indexing.kv_store import KVStore
from rnsr.indexing.semantic_search import SemanticSearcher
from rnsr.models import SkeletonNode, TraceEntry

logger = structlog.get_logger(__name__)


# =============================================================================
# Agent State Definition (Appendix C)
# =============================================================================


class AgentState(TypedDict):
    """
    State for the RNSR Navigator Agent.
    
    All fields use pointer-based Variable Stitching:
    - Full content stored in VariableStore
    - Only $POINTER names in state fields
    """
    
    # Query processing
    question: str
    sub_questions: list[str]
    pending_questions: list[str]
    current_sub_question: Optional[str]
    
    # Navigation state
    current_node_id: Optional[str]
    visited_nodes: list[str]
    navigation_path: list[str]
    
    # Tree of Thoughts (ToT) state - Section 7.2
    nodes_to_visit: list[str] # Queue for parallel exploration
    scored_candidates: list[dict[str, Any]]  # [{node_id, score, reasoning}]
    backtrack_stack: list[str]  # Stack of parent node IDs for backtracking
    dead_ends: list[str]  # Nodes marked as dead ends
    top_k: int  # Number of top candidates to explore
    tot_selection_threshold: float  # Minimum probability for selection
    tot_dead_end_threshold: float   # Probability threshold for dead end
    
    # Variable stitching (pointers only!)
    variables: list[str]  # List of $POINTER names
    context: str  # Contains pointers, not full content
    
    # Output
    answer: Optional[str]
    confidence: float
    
    # Question metadata (e.g., multiple-choice options)
    metadata: dict[str, Any]
    
    # Traceability
    trace: list[dict[str, Any]]
    iteration: int
    max_iterations: int


# =============================================================================
# Navigator Tools
# =============================================================================


def create_navigator_tools(
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    variable_store: VariableStore,
) -> dict[str, Any]:
    """
    Create tools for the Navigator Agent.
    
    Args:
        skeleton: Skeleton index (node_id -> SkeletonNode).
        kv_store: KV store with full content.
        variable_store: Variable store for findings.
        
    Returns:
        Dictionary of tool functions.
    """
    
    def get_node_summary(node_id: str) -> str:
        """Get the summary of a skeleton node."""
        node = skeleton.get(node_id)
        if node is None:
            return f"Node {node_id} not found"
        
        children_info = ""
        if node.child_ids:
            children = [skeleton.get(cid) for cid in node.child_ids]
            children_info = "\nChildren:\n" + "\n".join(
                f"  - {c.node_id}: {c.header}" for c in children if c
            )
        
        return f"""
Node: {node.node_id}
Header: {node.header}
Level: {node.level}
Summary: {node.summary}
{children_info}
"""
    
    def navigate_to_child(node_id: str, child_id: str) -> str:
        """Navigate to a child node."""
        node = skeleton.get(node_id)
        if node is None:
            return f"Node {node_id} not found"
        
        if child_id not in node.child_ids:
            return f"{child_id} is not a child of {node_id}"
        
        child = skeleton.get(child_id)
        if child is None:
            return f"Child {child_id} not found"
        
        return get_node_summary(child_id)
    
    def expand_node(node_id: str) -> str:
        """
        EXPAND: Fetch full content and store as variable.
        
        Use when the summary answers the question.
        """
        node = skeleton.get(node_id)
        if node is None:
            return f"Node {node_id} not found"
        
        # Fetch full content from KV store
        content = kv_store.get(node_id)
        if content is None:
            return f"No content found for {node_id}"
        
        # Generate pointer name (dedup to avoid overwriting)
        pointer = generate_pointer_name(node.header)
        base_pointer = pointer
        counter = 2
        while variable_store.exists(pointer):
            pointer = f"{base_pointer}_{counter}"
            counter += 1
        
        # Store as variable
        variable_store.assign(pointer, content, node_id)
        
        logger.info(
            "node_expanded",
            node_id=node_id,
            pointer=pointer,
            chars=len(content),
        )
        
        return f"Stored content as {pointer} ({len(content)} chars)"
    
    def store_finding(
        pointer_name: str,
        content: str,
        source_node_id: str = "",
    ) -> str:
        """
        Store a finding as a pointer variable.
        
        Args:
            pointer_name: Name like $LIABILITY_CLAUSE (must start with $)
            content: Full text content to store
            source_node_id: Source node for traceability
        """
        if not pointer_name.startswith("$"):
            pointer_name = "$" + pointer_name.upper()
        
        try:
            meta = variable_store.assign(pointer_name, content, source_node_id)
            return f"Stored as {pointer_name} ({meta.char_count} chars)"
        except Exception as e:
            return f"Error storing: {e}"
    
    def compare_variables(*pointers: str) -> str:
        """
        Compare multiple stored variables.
        
        Resolves pointers and returns content for comparison.
        Use during synthesis phase only.
        """
        results = []
        for pointer in pointers:
            content = variable_store.resolve(pointer)
            if content:
                results.append(f"=== {pointer} ===\n{content}")
            else:
                results.append(f"=== {pointer} ===\n[Not found]")
        
        return "\n\n".join(results)
    
    def list_stored_variables() -> str:
        """List all stored variables with metadata."""
        variables = variable_store.list_variables()
        if not variables:
            return "No variables stored yet."
        
        lines = ["Stored Variables:"]
        for v in variables:
            lines.append(f"  {v.pointer}: {v.char_count} chars from {v.source_node_id}")
        
        return "\n".join(lines)
    
    def synthesize_from_variables() -> str:
        """
        Get all stored content for final synthesis.
        
        Call this at the end to get full content for answer generation.
        """
        variables = variable_store.list_variables()
        if not variables:
            return "No variables to synthesize from."
        
        parts = []
        for v in variables:
            content = variable_store.resolve(v.pointer)
            if content:
                parts.append(f"=== {v.pointer} ===\n{content}")
        
        return "\n\n".join(parts)
    
    return {
        "get_node_summary": get_node_summary,
        "navigate_to_child": navigate_to_child,
        "expand_node": expand_node,
        "store_finding": store_finding,
        "compare_variables": compare_variables,
        "list_stored_variables": list_stored_variables,
        "synthesize_from_variables": synthesize_from_variables,
    }


# =============================================================================
# State Management Functions
# =============================================================================


def create_initial_state(
    question: str,
    root_node_id: str,
    max_iterations: int = 20,
    top_k: int = 3,
    metadata: dict[str, Any] | None = None,
    tot_selection_threshold: float = 0.4,
    tot_dead_end_threshold: float = 0.1,
) -> AgentState:
    """Create the initial agent state with ToT support."""
    return AgentState(
        question=question,
        sub_questions=[],
        pending_questions=[],
        current_sub_question=None,
        current_node_id=root_node_id,
        visited_nodes=[],
        navigation_path=[root_node_id],
        # Tree of Thoughts state
        nodes_to_visit=[],
        scored_candidates=[],
        backtrack_stack=[],
        dead_ends=[],
        top_k=top_k,
        tot_selection_threshold=tot_selection_threshold,
        tot_dead_end_threshold=tot_dead_end_threshold,
        # Variable stitching
        variables=[],
        context="",
        answer=None,
        confidence=0.0,
        metadata=metadata or {},
        trace=[],
        iteration=0,
        max_iterations=max_iterations,
    )


def add_trace_entry(
    state: AgentState,
    node_type: Literal["decomposition", "navigation", "variable_stitching", "synthesis"],
    action: str,
    details: dict | None = None,
) -> None:
    """Add a trace entry to the state."""
    entry = TraceEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        node_type=node_type,
        action=action,
        details=details or {},
    )
    state["trace"].append(entry.model_dump())


# =============================================================================
# Tree of Thoughts (ToT) Prompting - Section 7.2
# =============================================================================

# ToT System Prompt as specified in the research paper
TOT_SYSTEM_PROMPT = """You are a Deep Research Agent navigating a document tree.

You are currently at Node: {current_node_summary}
Children Nodes: {children_summaries}
Your Goal: {query}

EVALUATION TASK:
For each child node, estimate the probability (0.0 to 1.0) that it contains relevant information for the goal.

INSTRUCTIONS:
1. Evaluate: For each child node, analyze its summary and estimate relevance probability.
2. Be OPEN-MINDED: Select nodes with probability > {selection_threshold} (moderate evidence of relevance).
3. Look for matches: Prefer nodes with facts/entities mentioned in the query, but also consider broad thematic matches.
4. Balance PRECISION and RECALL: Do not prune branches too early. If unsure, include the node.
5. Plan: Select the top-{top_k} most promising nodes.
6. Reasoning: Explain briefly what SPECIFIC content in the summary makes it relevant.
7. Backtrack Signal: If NO child seems relevant (all probabilities < {dead_end_threshold}), report "DEAD_END".

OUTPUT FORMAT (JSON):
{{
    "evaluations": [
        {{"node_id": "...", "probability": 0.85, "reasoning": "Summary mentions X which directly relates to query about Y"}},
        {{"node_id": "...", "probability": 0.60, "reasoning": "Contains information about Z"}},
        ...
    ],
    "selected_nodes": ["node_id_1", "node_id_2", ...],
    "is_dead_end": false,
    "backtrack_reason": null
}}

If this is a dead end:
{{
    "evaluations": [...],
    "selected_nodes": [],
    "is_dead_end": true,
    "backtrack_reason": "None of the children appear to contain information about X."
}}

Respond ONLY with the JSON, no other text."""


def _skeleton_has_multiple_levels(skeleton: dict[str, SkeletonNode]) -> bool:
    """True if the tree has more than one level (e.g. root + sections). Used to disable summary-based pruning."""
    if not skeleton:
        return False
    levels = {n.level for n in skeleton.values()}
    return len(levels) >= 2


def _format_node_summary(node: SkeletonNode) -> str:
    """Format a node's summary for the ToT prompt."""
    return f"[{node.node_id}] {node.header}: {node.summary or '(no summary)'}"


def _is_trivial_summary(node: SkeletonNode) -> bool:
    """True if the node's summary is empty, very short, or just repeats the header."""
    s = (node.summary or "").strip()
    if not s or len(s) < 20:
        return True
    if node.header and s.lower() == node.header.lower().strip():
        return True
    return False


def _content_preview_from_kv(
    node: SkeletonNode,
    kv_store: Any,
    words_per_child: int = 100,
) -> str:
    """Build a content preview from KV store content for this node's children.

    All children are included.  Each child contributes the first
    *words_per_child* words so the preview remains readable without
    silently dropping sections.
    """
    if not node.child_ids or not kv_store:
        return ""
    parts = []
    for child_id in node.child_ids:
        content = kv_store.get(child_id) or ""
        words = content.split()[:words_per_child]
        snippet = " ".join(words) if words else ""
        if snippet:
            parts.append(snippet)
    if not parts:
        return ""
    return "Content preview: " + " | ".join(parts)


def _format_node_for_tot(
    skeleton: dict[str, SkeletonNode],
    node: SkeletonNode,
    kv_store: Any = None,
) -> str:
    """
    Format a single node for the ToT children list. For internal nodes with trivial
    summary, show raw content preview from kv_store (no summary field) to avoid
    any risk of feeding LLM-generated or hallucinated text.
    """
    if not node.child_ids or not _is_trivial_summary(node):
        return _format_node_summary(node)
    preview = _content_preview_from_kv(node, kv_store)
    if preview:
        return f"[{node.node_id}] {node.header}: {preview}"
    return _format_node_summary(node)


def _format_children_summaries(
    skeleton: dict[str, SkeletonNode],
    child_ids: list[str],
    kv_store: Any = None,
) -> str:
    """Format all children summaries for the ToT prompt."""
    if not child_ids:
        return "(no children - this is a leaf node)"
    
    lines = []
    for child_id in child_ids:
        child = skeleton.get(child_id)
        if child:
            lines.append(f"  - {_format_node_for_tot(skeleton, child, kv_store)}")
    
    return "\n".join(lines) if lines else "(no children found)"


def _verify_node_relevance(
    node_id: str,
    query: str,
    kv_store: Any,
    min_keyword_overlap: float = 0.2,
) -> tuple[bool, float]:
    """
    Verify that a node's full content is actually relevant to the query.
    
    This prevents ToT from selecting nodes based on misleading summaries.
    
    Args:
        node_id: The node ID to verify.
        query: The query to check relevance against.
        kv_store: Key-value store to fetch full content.
        min_keyword_overlap: Minimum fraction of query keywords that must appear.
        
    Returns:
        Tuple of (is_relevant, relevance_score).
    """
    if not kv_store or not query:
        return True, 1.0  # Can't verify without store or query
    
    content = kv_store.get(node_id) or ""
    if not content:
        return False, 0.0
    
    # Extract keywords from query (basic tokenization)
    query_lower = query.lower()
    stop_words = {
        "what", "is", "the", "a", "an", "of", "in", "to", "for", "and", "or",
        "how", "when", "where", "who", "which", "that", "this", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "can",
        "with", "from", "by", "on", "at", "as", "it", "its", "their", "they",
    }
    query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower)) - stop_words
    
    if not query_words:
        return True, 1.0  # No meaningful keywords
    
    # Check content for query keywords
    content_lower = content.lower()
    matched_keywords = sum(1 for kw in query_words if kw in content_lower)
    overlap_score = matched_keywords / len(query_words)
    
    is_relevant = overlap_score >= min_keyword_overlap
    
    return is_relevant, overlap_score


def evaluate_children_with_tot(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
    top_k_override: int | None = None,
    kv_store: Any = None,
) -> dict[str, Any]:
    """
    Use Tree of Thoughts prompting to evaluate child nodes.
    
    This implements Section 7.2 of the research paper:
    "We use Tree of Thoughts (ToT) which explicitly encourages the model 
    to explore multiple branches of the index before committing to a path."
    
    Args:
        state: Current agent state.
        skeleton: Skeleton index.
        top_k_override: Optional override for adaptive exploration.
        kv_store: Optional key-value store for content verification.
        
    Returns:
        Dictionary with evaluations, selected_nodes, is_dead_end, and backtrack_reason.
    """
    node_id = state.get("current_node_id")
    if node_id is None:
        return {"evaluations": [], "selected_nodes": [], "is_dead_end": True, "backtrack_reason": "No current node"}
    
    node = skeleton.get(node_id)
    if node is None:
        return {"evaluations": [], "selected_nodes": [], "is_dead_end": True, "backtrack_reason": "Node not found"}
    
    # If no children, this is a leaf - expand instead of traverse
    if not node.child_ids:
        return {"evaluations": [], "selected_nodes": [], "is_dead_end": False, "backtrack_reason": None, "is_leaf": True}
    
    # For nodes with very few children (1-3), skip ToT entirely -- there's no benefit
    # to pruning when there are so few options, and ToT can incorrectly mark them as dead ends.
    if len(node.child_ids) <= 3:
        return {
            "evaluations": [{"node_id": cid, "probability": 1.0, "reasoning": "Few children: no pruning needed"} for cid in node.child_ids],
            "selected_nodes": list(node.child_ids),
            "is_dead_end": False,
            "backtrack_reason": None,
        }
    
    # For multi-level trees, use ToT with content previews but be more permissive
    # (higher top_k, lower threshold) to avoid aggressive pruning from hallucinated summaries.
    is_multi_level = _skeleton_has_multiple_levels(skeleton)
    
    # Format the ToT prompt
    current_summary = _format_node_summary(node)
    children_summaries = _format_children_summaries(skeleton, node.child_ids, kv_store)
    query = state.get("current_sub_question") or state.get("question", "")
    if is_multi_level:
        # Multi-level: select focused subset, capped at 5 to avoid
        # flooding synthesis with the entire document
        top_k = min(max(top_k_override or 5, 3), 5)
        selection_threshold = 0.3
        dead_end_threshold = 0.05
    else:
        top_k = top_k_override if top_k_override is not None else state.get("top_k", 3)
        selection_threshold = state.get("tot_selection_threshold", 0.4)
        dead_end_threshold = state.get("tot_dead_end_threshold", 0.1)
    
    prompt = TOT_SYSTEM_PROMPT.format(
        current_node_summary=current_summary,
        children_summaries=children_summaries,
        query=query,
        top_k=top_k,
        selection_threshold=selection_threshold,
        dead_end_threshold=dead_end_threshold,
    )
    
    # Call the LLM
    try:
        from rnsr.llm import get_llm
        import json
        
        llm = get_llm()
        response = llm.complete(prompt)
        response_text = str(response).strip()
        
        # Parse JSON response with robust error handling
        def _sanitize_json_backslashes(text: str) -> str:
            """Escape lone backslashes that aren't valid JSON escape sequences.

            LLMs often embed LaTeX (e.g. $80\\%$) or other raw backslash
            sequences inside JSON string values, causing ``json.loads`` to
            fail with 'Invalid \\escape'.  This helper replaces every
            backslash that is *not* followed by a valid JSON escape character
            (``" \\ / b f n r t u``) with a double-backslash.
            """
            return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)

        try:
            # Handle potential markdown code blocks
            if "```" in response_text:
                match = re.search(r'\{[\s\S]*\}', response_text)
                if match:
                    response_text = match.group(0)

            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                result = json.loads(_sanitize_json_backslashes(response_text))
        except json.JSONDecodeError:
            logger.warning("tot_json_repair_attempt", original_response=response_text)
            # Fallback to asking the LLM to fix the JSON
            repair_prompt = f"""The following text is a malformed JSON object. Please fix it and return ONLY the corrected JSON. Do not add any commentary.

Malformed JSON:
{response_text}"""
            from rnsr.llm import get_llm

            llm = get_llm()
            repaired_response_text = str(llm.complete(repair_prompt)).strip()
            
            # Final attempt to parse the repaired JSON
            if "```" in repaired_response_text:
                 match = re.search(r'\{[\s\S]*\}', repaired_response_text)
                 if match:
                    repaired_response_text = match.group(0)

            try:
                result = json.loads(repaired_response_text)
            except json.JSONDecodeError:
                result = json.loads(_sanitize_json_backslashes(repaired_response_text))
        
        # CONTENT VERIFICATION: Verify selected nodes actually contain relevant content
        # This prevents ToT from being misled by misleading summaries
        if kv_store and result.get("selected_nodes"):
            query = state.get("current_sub_question") or state.get("question", "")
            original_selected = result["selected_nodes"]
            verified_selected = []
            unverified = []
            
            for node_id in original_selected:
                is_relevant, score = _verify_node_relevance(node_id, query, kv_store)
                if is_relevant:
                    verified_selected.append(node_id)
                else:
                    unverified.append({"node_id": node_id, "score": score})
            
            if unverified:
                logger.warning(
                    "tot_nodes_failed_content_verification",
                    original_count=len(original_selected),
                    verified_count=len(verified_selected),
                    unverified=unverified,
                )
            
            # Update result with verified nodes
            result["selected_nodes"] = verified_selected
            result["unverified_nodes"] = [u["node_id"] for u in unverified]
            
            # If all nodes failed verification, mark as potential dead end
            if not verified_selected and original_selected:
                logger.warning(
                    "tot_all_nodes_failed_verification",
                    original_nodes=original_selected,
                    query=query[:100],
                )
                # Don't mark as dead end immediately - fallback to exploring unverified
                # nodes since summaries might still be useful
                result["selected_nodes"] = original_selected[:2]  # Keep top 2 as fallback
                result["verification_warning"] = "All selected nodes failed content verification"
        
        # --- TABLE priority: for arithmetic questions, ensure TABLE nodes are selected ---
        metadata = state.get("metadata", {})
        _requires_arith = metadata.get("requires_arithmetic", False) or metadata.get(
            "answer_type", ""
        ) in ("arithmetic", "counting", "multi-span")
        if _requires_arith and result.get("selected_nodes") is not None:
            selected_set = set(result["selected_nodes"])
            table_children = []
            for cid in node.child_ids:
                child = skeleton.get(cid)
                if child is None:
                    continue
                hdr = (child.header or "").upper()
                nid = (child.node_id or "").upper()
                if "TABLE" in hdr or "TABLE" in nid:
                    table_children.append(cid)

            missing_tables = [t for t in table_children if t not in selected_set]
            if missing_tables:
                result["selected_nodes"].extend(missing_tables)
                logger.info(
                    "tot_table_priority_injected",
                    injected=missing_tables,
                    question=query[:80],
                )

        logger.debug(
            "tot_evaluation_complete",
            node_id=node_id,
            evaluations=len(result.get("evaluations", [])),
            selected=result.get("selected_nodes", []),
            is_dead_end=result.get("is_dead_end", False),
        )
        
        return result
        
    except Exception as e:
        logger.warning("tot_evaluation_failed", error=str(e), node_id=node_id)
        
        # Fallback: use simple heuristic (select first unvisited children)
        visited = state.get("visited_nodes", [])
        dead_ends = state.get("dead_ends", [])
        
        unvisited = [
            cid for cid in node.child_ids 
            if cid not in visited and cid not in dead_ends
        ]
        
        return {
            "evaluations": [{"node_id": cid, "probability": 0.5, "reasoning": "Fallback selection"} for cid in unvisited[:top_k]],
            "selected_nodes": unvisited[:top_k],
            "is_dead_end": len(unvisited) == 0,
            "backtrack_reason": "No unvisited children" if len(unvisited) == 0 else None,
        }


def backtrack_to_parent(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
) -> AgentState:
    """
    Backtrack to parent node when current path is a dead end.
    
    Implements the backtracking logic from Section 7.2:
    "Backtrack: If a node yields no useful info, report 'Dead End' and return to parent."
    """
    new_state = cast(AgentState, dict(state))
    
    current_id = new_state["current_node_id"]
    
    # Mark current node as dead end
    if current_id and current_id not in new_state["dead_ends"]:
        new_state["dead_ends"].append(current_id)
    
    # Try to backtrack using the backtrack stack
    if new_state["backtrack_stack"]:
        parent_id = new_state["backtrack_stack"].pop()
        new_state["current_node_id"] = parent_id
        new_state["navigation_path"].append(parent_id)
        
        add_trace_entry(
            new_state,
            "navigation",
            f"Backtracked to {parent_id} (dead end at {current_id})",
            {"from": current_id, "to": parent_id, "reason": "dead_end"},
        )
        
        logger.debug("backtrack_success", from_node=current_id, to_node=parent_id)
    else:
        # No parent to backtrack to - we're done exploring
        add_trace_entry(
            new_state,
            "navigation",
            "Cannot backtrack - at root or stack empty",
            {"current": current_id},
        )
        logger.debug("backtrack_failed", reason="empty_stack")
    
    new_state["iteration"] = new_state["iteration"] + 1
    return new_state


# =============================================================================
# LangGraph Node Functions
# =============================================================================


# Decomposition prompt for RLM recursive sub-task generation
DECOMPOSITION_PROMPT = """You are analyzing a complex query to decompose it into sub-tasks.

Query: {query}

Document Structure (top-level sections):
{structure}

TASK: Decompose this query into specific sub-tasks that can be executed independently.

RULES:
1. Each sub-task should target a specific section or piece of information
2. Sub-tasks should be answerable by reading specific document sections
3. For comparison queries, create one sub-task per item being compared
4. For multi-hop queries, create sequential sub-tasks (find A, then use A to find B)
5. Maximum 5 sub-tasks to maintain efficiency

OUTPUT FORMAT (JSON):
{{
    "sub_tasks": [
        {{"id": 1, "task": "Find X in section Y", "target_section": "section_hint"}},
        {{"id": 2, "task": "Extract Z from W", "target_section": "section_hint"}}
    ],
    "synthesis_plan": "How to combine sub-task results into final answer"
}}

Return ONLY valid JSON."""


def decompose_query(state: AgentState) -> AgentState:
    """
    Decompose the main question into sub-questions using LLM.
    
    Implements Section 2.2 "The Recursive Loop":
    "The model's capability to divide a complex reasoning task into 
    smaller, manageable sub-tasks and invoke instances of itself to solve them."
    """
    new_state = cast(AgentState, dict(state))  # Copy
    question = new_state["question"]
    
    # Try LLM-based decomposition
    try:
        from rnsr.llm import get_llm
        import json
        
        llm = get_llm()
        
        # Get document structure for context
        # Note: skeleton is not available here, so we'll do basic decomposition
        structure_hint = "Document sections available for navigation"
        
        prompt = DECOMPOSITION_PROMPT.format(
            query=question,
            structure=structure_hint,
        )
        
        response = llm.complete(prompt)
        response_text = str(response).strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            decomposition = json.loads(json_match.group())
            sub_tasks = decomposition.get("sub_tasks", [])
            
            if sub_tasks:
                sub_questions = [t["task"] for t in sub_tasks]
                new_state["sub_questions"] = sub_questions
                new_state["pending_questions"] = sub_questions.copy()
                new_state["current_sub_question"] = sub_questions[0]
                
                add_trace_entry(
                    new_state,
                    "decomposition",
                    f"LLM decomposed into {len(sub_questions)} sub-tasks",
                    {
                        "sub_questions": sub_questions,
                        "synthesis_plan": decomposition.get("synthesis_plan", ""),
                    },
                )
                
                new_state["iteration"] = new_state["iteration"] + 1
                return new_state
    
    except Exception as e:
        logger.warning("llm_decomposition_failed", error=str(e))
    
    # Fallback: simple decomposition patterns
    sub_questions = _simple_decompose(question)
    new_state["sub_questions"] = sub_questions
    new_state["pending_questions"] = sub_questions.copy()
    new_state["current_sub_question"] = sub_questions[0] if sub_questions else question
    
    add_trace_entry(
        new_state,
        "decomposition",
        f"Decomposed into {len(sub_questions)} sub-questions (fallback)",
        {"sub_questions": sub_questions},
    )
    
    new_state["iteration"] = new_state["iteration"] + 1
    return new_state


def _simple_decompose(question: str) -> list[str]:
    """
    Simple pattern-based decomposition fallback.
    
    Handles common query patterns without LLM.
    """
    question_lower = question.lower()
    
    # Pattern: "Compare X and Y" or "X vs Y"
    compare_patterns = [
        r"compare\s+(.+?)\s+(?:and|with|to|vs\.?)\s+(.+)",
        r"difference(?:s)?\s+between\s+(.+?)\s+and\s+(.+)",
        r"(.+?)\s+vs\.?\s+(.+)",
    ]
    
    for pattern in compare_patterns:
        match = re.search(pattern, question_lower)
        if match:
            item1, item2 = match.groups()
            return [
                f"Find information about {item1.strip()}",
                f"Find information about {item2.strip()}",
            ]
    
    # Pattern: "What are the X in sections A, B, and C"
    multi_section = re.search(
        r"in\s+(?:sections?\s+)?(.+?,\s*.+?(?:,\s*.+)*)",
        question_lower,
    )
    if multi_section:
        sections = [s.strip() for s in multi_section.group(1).split(",")]
        return [f"Find relevant information in {s}" for s in sections[:5]]
    
    # Pattern: "List all X" or "Find all X" - may need iteration
    if re.search(r"(list|find|show)\s+all", question_lower):
        return [
            f"Search for all relevant sections",
            question,  # Original as synthesis query
        ]
    
    # Default: single question
    return [question]


def navigate_tree(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
) -> AgentState:
    """
    Navigate the document tree based on current sub-question.
    
    This function now handles the node visitation queue for multi-path exploration.
    """
    new_state = cast(AgentState, dict(state))

    # Add detailed logging to debug navigation loops
    logger.debug(
        "navigate_step_start",
        iteration=new_state["iteration"],
        current_node=new_state["current_node_id"],
        queue=new_state["nodes_to_visit"],
        visited=new_state["visited_nodes"],
    )

    # If current node is None, try to pop from the visit queue.
    # Prefer leaf nodes (no children) so we expand and fetch content before running out of iterations.
    # Skip any already-visited nodes (can happen with duplicate segment IDs or re-queued nodes).
    if new_state["current_node_id"] is None and new_state["nodes_to_visit"]:
        queue = new_state["nodes_to_visit"]
        visited = set(new_state.get("visited_nodes", []))
        
        # First pass: prefer an unvisited leaf
        leaf_idx = next(
            (i for i, nid in enumerate(queue)
             if nid not in visited and skeleton.get(nid) and not skeleton.get(nid).child_ids),
            None,
        )
        if leaf_idx is not None:
            next_node_id = queue.pop(leaf_idx)
        else:
            # Second pass: prefer any unvisited node
            unvisited_idx = next(
                (i for i, nid in enumerate(queue) if nid not in visited),
                None,
            )
            if unvisited_idx is not None:
                next_node_id = queue.pop(unvisited_idx)
            else:
                # All queued nodes are already visited — drain the queue and leave current_node_id as None
                skipped = len(queue)
                queue.clear()
                logger.debug("queue_all_visited", skipped=skipped)
                next_node_id = None
        
        if next_node_id is not None:
            new_state["current_node_id"] = next_node_id
            
            # Also add to navigation path
            if next_node_id not in new_state["navigation_path"]:
                new_state["navigation_path"].append(next_node_id)
                
            add_trace_entry(
                new_state,
                "navigation",
                f"Visiting next node from queue: {next_node_id}",
                {"queue_size": len(new_state["nodes_to_visit"])},
            )
    
    node_id = new_state["current_node_id"]
    if node_id is None:
        add_trace_entry(new_state, "navigation", "No current node to navigate to and queue is empty")
        return new_state
    
    node = skeleton.get(node_id)
    
    if node is None:
        add_trace_entry(new_state, "navigation", f"Node {node_id} not found")
        return new_state
    
    # Don't add to visited here - let expand/traverse handle that
    # to avoid preventing expansion of newly queued nodes
    
    add_trace_entry(
        new_state,
        "navigation",
        f"At node {node_id}: {node.header}",
        {"summary": node.summary, "children": node.child_ids},
    )
    
    new_state["iteration"] = new_state["iteration"] + 1
    return new_state


def should_expand(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
    kv_store: Any = None,
) -> Literal["expand", "traverse", "backtrack", "done"]:
    """
    Decide whether to expand (fetch content), traverse (go to children), 
    backtrack (dead end), or finish.
    
    Uses Tree of Thoughts evaluation to make intelligent decisions.
    """
    node_id = state.get("current_node_id")
    if node_id is None:
        return "done"
    
    node = skeleton.get(node_id)
    if node is None:
        return "done"
    
    # Check iteration limit
    if state.get("iteration", 0) >= state.get("max_iterations", 20):
        queue = state.get("nodes_to_visit", [])
        if queue:
            logger.warning(
                "done_with_queue_non_empty",
                iteration=state.get("iteration"),
                max_iterations=state.get("max_iterations"),
                queue_size=len(queue),
            )
        return "done"
    
    # Removed variable count limit - iteration limit is sufficient
    
    # If no children, must expand (leaf node)
    # But only if not already visited
    visited = state.get("visited_nodes", [])
    if not node.child_ids:
        if node_id in visited:
            # Already expanded this leaf, done with it
            return "done"
        return "expand"
    
    # Check if all children are visited or dead ends
    visited = state.get("visited_nodes", [])
    dead_ends = state.get("dead_ends", [])
    unvisited_children = [
        cid for cid in node.child_ids 
        if cid not in visited and cid not in dead_ends
    ]
    
    if not unvisited_children:
        # All children explored - backtrack or done. Prefer backtrack if queue has pending nodes.
        queue = state.get("nodes_to_visit", [])
        if state.get("backtrack_stack"):
            return "backtrack"
        if queue:
            logger.warning(
                "done_with_queue_non_empty",
                node_id=node_id,
                queue_size=len(queue),
                note="All children explored but nodes_to_visit non-empty",
            )
        return "done"
    
    # Adaptive exploration: increase top_k if we haven't found much information yet
    base_top_k = state.get("top_k", 3)
    variables_found = len(state.get("variables", []))
    iteration = state.get("iteration", 0)
    
    if iteration > 5 and variables_found == 0:
        adaptive_top_k = min(len(node.child_ids), base_top_k * 3)
    elif iteration > 3 and variables_found < 2:
        adaptive_top_k = min(len(node.child_ids), base_top_k * 2)
    else:
        adaptive_top_k = base_top_k

    # Use ToT evaluation to decide
    tot_result = evaluate_children_with_tot(state, skeleton, top_k_override=adaptive_top_k, kv_store=kv_store)
    
    # Check for dead end signal from ToT
    if tot_result.get("is_dead_end", False):
        if state.get("backtrack_stack"):
            return "backtrack"
        return "done"
    
    # Check if ToT says this is a leaf (should expand)
    if tot_result.get("is_leaf", False):
        return "expand"
    
    # If ToT selected nodes, traverse
    if tot_result.get("selected_nodes"):
        return "traverse"
    
    # Fallback: expand if at deep level (H3+)
    if node.level >= 3:
        return "expand"
    
    # Otherwise traverse to children
    return "traverse"


def expand_current_node(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    variable_store: VariableStore,
) -> AgentState:
    """
    EXPAND: Fetch full content and store as variable.
    
    If the node has an associated image (stored in kv_store via put_image),
    uses VisionLLM to produce a text analysis of the image, which is appended
    to the node content and stored as a variable. This enables vision-augmented
    Tree of Thoughts navigation without losing structural context.
    """
    new_state = cast(AgentState, dict(state))
    
    node_id = new_state["current_node_id"]
    if node_id is None:
        return new_state
    
    # Check if already visited to prevent infinite loops
    if node_id in new_state.get("visited_nodes", []):
        add_trace_entry(
            new_state,
            "navigation", 
            f"Skipping already-visited node {node_id}",
            {"node_id": node_id},
        )
        new_state["current_node_id"] = None
        new_state["iteration"] = new_state["iteration"] + 1
        return new_state
    
    node = skeleton.get(node_id)
    
    if node is None:
        new_state["current_node_id"] = None
        new_state["iteration"] = new_state["iteration"] + 1
        return new_state
    
    # Fetch full content
    content = kv_store.get(node_id)
    if content is None:
        add_trace_entry(new_state, "variable_stitching", f"No content for {node_id}")
        new_state["current_node_id"] = None
        new_state["iteration"] = new_state["iteration"] + 1
        return new_state
    
    # === Vision Augmentation ===
    # If the node has an associated image, analyze it with VisionLLM
    # and append the analysis to the content. This creates a text mapping
    # of the image tied to this specific tree node.
    image_bytes = None
    if hasattr(kv_store, "get_image"):
        image_bytes = kv_store.get_image(node_id)
    
    if image_bytes is not None:
        try:
            from rnsr.ingestion.vision_retrieval import VisionLLM, VisionConfig
            
            question = new_state.get("question", "")
            vision_prompt = (
                f"Analyze this document image. For any charts or graphs, "
                f"extract EXACT data values at each labeled point — do not "
                f"approximate or round. Read axis labels, tick marks, and data "
                f"points precisely. For tables, extract all rows and columns as "
                f"pipe-separated text (header | col1 | col2 ...). For forms, "
                f"extract all field labels and values. For any other text, "
                f"transcribe it exactly.\n\n"
                f"Question to keep in mind: {question}"
            )
            
            vision_llm = VisionLLM(VisionConfig())
            vision_analysis = vision_llm.analyze_image(image_bytes, vision_prompt)
            
            # Append vision analysis to the text content
            content = (
                f"{content}\n\n"
                f"[VISION ANALYSIS]\n{vision_analysis}"
            )
            
            add_trace_entry(
                new_state,
                "variable_stitching",
                f"Vision analysis for node {node_id}",
                {"node_id": node_id, "analysis_chars": len(vision_analysis), "vision": True},
            )
        except Exception as e:
            logger.warning(
                "vision_analysis_failed",
                node_id=node_id,
                error=str(e),
            )
            add_trace_entry(
                new_state,
                "navigation",
                f"Vision analysis failed for {node_id}: {e}",
                {"node_id": node_id, "error": str(e), "vision": True},
            )
    
    # Generate and store as variable (dedup to avoid overwriting)
    pointer = generate_pointer_name(node.header)
    base_pointer = pointer
    counter = 2
    while variable_store.exists(pointer):
        pointer = f"{base_pointer}_{counter}"
        counter += 1
    variable_store.assign(pointer, content, node_id)
    
    new_state["variables"].append(pointer)
    new_state["context"] += f"\nFound: {pointer} (from {node.header})"
    
    # Mark node as visited and clear current node
    if node_id not in new_state["visited_nodes"]:
        new_state["visited_nodes"].append(node_id)
    new_state["current_node_id"] = None  # Process queue next
    
    add_trace_entry(
        new_state,
        "variable_stitching",
        f"Stored {pointer}",
        {"node_id": node_id, "header": node.header, "chars": len(content)},
    )
    
    new_state["iteration"] = new_state["iteration"] + 1
    return new_state


def traverse_to_children(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
    semantic_searcher: SemanticSearcher | None = None,
    kv_store: Any = None,
) -> AgentState:
    """
    TRAVERSE: Navigate to child nodes using Tree of Thoughts reasoning.
    
    Primary Strategy: Tree of Thoughts (ToT) with adaptive exploration.
    - Uses LLM reasoning to evaluate child nodes based on summaries
    - Dynamically increases exploration when insufficient variables found
    - Preserves document structure and context (per research paper Section 7.2)
    
    Optional: Semantic search as shortcut for finding entry points (Section 9.1)
    """
    new_state = cast(AgentState, dict(state))
    
    node_id = new_state["current_node_id"]
    if node_id is None:
        return new_state
    
    node = skeleton.get(node_id)
    
    if node is None or not node.child_ids:
        return new_state
    
    question = new_state["question"]
    base_top_k = new_state.get("top_k", 3)
    
    # Adaptive exploration: increase top_k if we haven't found much information yet
    # This addresses the original issue: "agent should be able to explore all nodes if it needs to"
    variables_found = len(new_state.get("variables", []))
    iteration = new_state.get("iteration", 0)
    
    # Dynamic top_k based on progress
    if iteration > 5 and variables_found == 0:
        # Not finding anything - expand search significantly
        adaptive_top_k = min(len(node.child_ids), base_top_k * 3)
        add_trace_entry(
            new_state,
            "navigation",
            f"Expanding search: {variables_found} variables after {iteration} iterations",
            {"base_top_k": base_top_k, "adaptive_top_k": adaptive_top_k},
        )
    elif iteration > 3 and variables_found < 2:
        # Finding very little - expand moderately
        adaptive_top_k = min(len(node.child_ids), base_top_k * 2)
    else:
        # Normal exploration
        adaptive_top_k = base_top_k
    
    # Use Tree of Thoughts as primary navigation method (per research paper)
    tot_result = evaluate_children_with_tot(new_state, skeleton, top_k_override=adaptive_top_k, kv_store=kv_store)
    
    # Check for dead end
    if tot_result.get("is_dead_end", False):
        add_trace_entry(
            new_state,
            "navigation",
            f"Dead end detected at {node_id}: {tot_result.get('backtrack_reason', 'Unknown')}",
            {"node_id": node_id, "backtrack_reason": tot_result.get("backtrack_reason")},
        )
        # Mark as dead end and don't change current node
        if node_id not in new_state["dead_ends"]:
            new_state["dead_ends"].append(node_id)
        new_state["iteration"] = new_state["iteration"] + 1
        return new_state
    
    selected_nodes = tot_result.get("selected_nodes", [])
    evaluations = tot_result.get("evaluations", [])
    new_state["scored_candidates"] = evaluations

    # Queue selected children for visitation
    if selected_nodes:
        new_state["nodes_to_visit"].extend(selected_nodes)
        # Mark parent as visited since we've traversed its children
        if node_id not in new_state["visited_nodes"]:
            new_state["visited_nodes"].append(node_id)
        # Unset current node to force the next navigate step to pull from the queue
        new_state["current_node_id"] = None
        add_trace_entry(
            new_state,
            "navigation",
            f"Queued {len(selected_nodes)} children from {node_id}",
            {"nodes": selected_nodes, "parent": node_id},
        )
    else:
        # No candidates available - this should trigger backtracking
        add_trace_entry(
            new_state,
            "navigation",
            f"No unvisited candidates at {node_id}",
            {"node_id": node_id},
        )
    
    new_state["iteration"] = new_state["iteration"] + 1
    return new_state


# =============================================================================
# Safe Math Evaluation (AST-based, no exec/eval on untrusted code)
# =============================================================================

_SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: _operator.add,
    ast.Sub: _operator.sub,
    ast.Mult: _operator.mul,
    ast.Div: _operator.truediv,
    ast.FloorDiv: _operator.floordiv,
    ast.Mod: _operator.mod,
    ast.Pow: _operator.pow,
    ast.USub: _operator.neg,
    ast.UAdd: _operator.pos,
}

_SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "int": int,
    "float": float,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "pow": math.pow,
}


def _safe_eval_node(node: ast.AST) -> float | int:
    """Recursively evaluate an AST node using only safe operations."""
    # Numeric literal
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    # Unary operator: -x, +x
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_node(node.operand)
        return _SAFE_OPERATORS[op_type](operand)

    # Binary operator: x + y, x * y, etc.
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _SAFE_OPERATORS[op_type](left, right)

    # Function call: round(x, 2), abs(x), etc.
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed (no methods)")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise ValueError(f"Function not allowed: {func_name}")
        args = [_safe_eval_node(arg) for arg in node.args]
        return _SAFE_FUNCTIONS[func_name](*args)

    # List/Tuple literal (for sum([a, b, c]), min(a, b), etc.)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval_node(elt) for elt in node.elts]  # type: ignore[return-value]

    # Name reference — only allow math constants
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
        raise ValueError(f"Variable reference not allowed: {node.id}")

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def safe_math_eval(expr: str) -> float | int | None:
    """
    Safely evaluate a Python math expression using AST parsing.

    Only permits arithmetic operators (+, -, *, /, //, %, **),
    numeric literals, and whitelisted functions (abs, round, min, max,
    sum, int, float, sqrt, ceil, floor, log, log10, pow).

    Returns None on any failure (parse error, unsupported operation, etc.).

    Examples:
        safe_math_eval("(14740 + 1910) / 2")  -> 8325.0
        safe_math_eval("round(100 / 3, 2)")    -> 33.33
        safe_math_eval("import os")             -> None
    """
    if not expr or not expr.strip():
        return None

    expr = expr.strip()

    # Reject obviously dangerous patterns before parsing
    if any(kw in expr for kw in ("import ", "__", "exec", "eval", "open", "compile")):
        logger.warning("safe_math_eval_rejected", expr=expr[:100])
        return None

    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval_node(tree.body)
        if isinstance(result, (int, float)):
            logger.debug("safe_math_eval_success", expr=expr[:100], result=result)
            return result
        return None
    except Exception as e:
        logger.debug("safe_math_eval_failed", expr=expr[:100], error=str(e))
        return None


_NO_COMPUTE_MARKER = "__NO_COMPUTE__"


def _extract_code_block(response_text: str) -> str | None:
    """Extract a Python code block from the LLM response.

    Looks for ``CODE:`` followed by a fenced code block, or a bare
    fenced code block.
    """
    # Pattern 1: CODE: header followed by ```python ... ```
    match = re.search(
        r"CODE:\s*```(?:python)?\s*\n(.*?)```",
        response_text,
        re.S,
    )
    if match:
        return match.group(1).strip()

    # Pattern 2: Just a ```python ... ``` block anywhere
    match = re.search(
        r"```(?:python)?\s*\n(.*?)```",
        response_text,
        re.S,
    )
    if match:
        return match.group(1).strip()

    return None


def _try_compute_from_response(
    response_text: str,
    context_text: str = "",
) -> str | None:
    """
    Parse a CODE:, COMPUTE:, NO_COMPUTE:, or ANSWER: line from the LLM
    response and execute any computation via Python code.

    The preferred path is CODE: with a Python code block — this ensures
    the LLM never does mental math; all arithmetic is executed by Python.

    Returns:
        - The computed/direct answer string on success.
        - _NO_COMPUTE_MARKER if the LLM explicitly said no computation is
          needed (caller should skip further fallbacks).
        - None if nothing useful was found.
    """
    compute_expr = None
    answer_line = None
    no_compute_answer = None

    for line in response_text.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("NO_COMPUTE:"):
            no_compute_answer = stripped[len("NO_COMPUTE:"):].strip()
        elif stripped.upper().startswith("COMPUTE:"):
            compute_expr = stripped[len("COMPUTE:"):].strip()
        elif stripped.upper().startswith("ANSWER:"):
            answer_line = stripped[len("ANSWER:"):].strip()

    # 1. NO_COMPUTE — LLM says no math needed; return direct answer
    if no_compute_answer is not None:
        logger.info(
            "no_compute_signal",
            direct_answer=no_compute_answer[:100],
        )
        return no_compute_answer if no_compute_answer else _NO_COMPUTE_MARKER

    # 2. CODE block — execute Python code via REPL (preferred path)
    code_block = _extract_code_block(response_text)
    if code_block:
        try:
            from rnsr.agent.repl_env import REPLEnvironment
            from rnsr.indexing.kv_store import InMemoryKVStore

            env = REPLEnvironment(
                document_text=context_text,
                skeleton={},
                kv_store=InMemoryKVStore(),
            )
            result = env.execute(code_block)

            if result["success"] and result["output"]:
                output = str(result["output"]).strip()
                # Extract the last number printed (in case of debug prints)
                nums = re.findall(r'[-]?[\d,]+\.?\d*', output)
                if nums:
                    num_str = nums[-1].replace(",", "")
                    try:
                        val = float(num_str)
                        logger.info(
                            "code_execution_success",
                            code=code_block[:200],
                            result=num_str,
                        )
                        if val == int(val) and abs(val) < 1e15:
                            return str(int(val))
                        return str(val)
                    except ValueError:
                        pass

            logger.warning(
                "code_execution_no_result",
                code=code_block[:200],
                output=str(result.get("output", ""))[:200],
                error=str(result.get("error", ""))[:200],
            )
        except Exception as e:
            logger.warning("code_execution_error", error=str(e))

    # 3. Legacy COMPUTE — evaluate single math expression via safe_math_eval
    if compute_expr:
        result = safe_math_eval(compute_expr)
        if result is not None:
            # Format: remove trailing .0 for clean integer answers
            if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
                return str(int(result))
            return str(result)
        else:
            logger.warning(
                "compute_tool_eval_failed",
                expr=compute_expr[:200],
            )

    # 4. Fallback: if CODE/COMPUTE failed but there's an ANSWER line, try to
    #    extract a number from it
    if answer_line:
        # Clean the answer line: remove commas in numbers, $ signs, etc.
        cleaned = answer_line.replace(",", "").strip().strip("$").strip("%")
        try:
            val = float(cleaned)
            if val == int(val) and abs(val) < 1e15:
                return str(int(val))
            return str(val)
        except ValueError:
            # Return raw answer line if it's not parseable as a number
            return answer_line

    return None


def _try_repl_arithmetic(
    question: str,
    context_text: str,
    llm: Any,
) -> str | None:
    """
    Fallback: use REPLEnvironment for complex multi-step arithmetic.

    Asks the LLM to generate Python code that computes the answer,
    then executes it in the sandboxed REPL.

    Returns the computed answer string, or None on failure.
    """
    try:
        from rnsr.agent.repl_env import REPLEnvironment
        from rnsr.indexing.kv_store import InMemoryKVStore
    except ImportError:
        return None

    # Build a lightweight REPL with context as the document
    env = REPLEnvironment(
        document_text=context_text,
        skeleton={},
        kv_store=InMemoryKVStore(),
    )

    code_prompt = f"""You have access to a Python REPL. The variable DOC_VAR contains the document context.
Write Python code to answer this question. The code must print() the final numeric answer.

Question: {question}

Document context is available in DOC_VAR (string). You can search it with:
- DOC_VAR.find("text") to find text positions
- re.search(pattern, DOC_VAR) for regex search

Write ONLY valid Python code. Use print() to output the final answer.
Example:
```python
val1 = 14740
val2 = 1910
result = (val1 + val2) / 2
print(result)
```

Your Python code:"""

    try:
        code_response = str(llm.complete(code_prompt)).strip()
        # Clean markdown code fences
        code_response = re.sub(r"```python\s*", "", code_response)
        code_response = re.sub(r"```\s*", "", code_response)
        code_response = code_response.strip()

        if not code_response:
            return None

        result = env.execute(code_response)

        if result["success"] and result["output"]:
            output = str(result["output"]).strip()
            # Try to extract a number from the output
            match = re.search(r'[-]?[\d,]+\.?\d*', output)
            if match:
                num_str = match.group(0).replace(",", "")
                try:
                    val = float(num_str)
                    logger.info(
                        "repl_arithmetic_success",
                        question=question[:80],
                        result=num_str,
                    )
                    if val == int(val) and abs(val) < 1e15:
                        return str(int(val))
                    return str(val)
                except ValueError:
                    pass

        logger.debug(
            "repl_arithmetic_no_result",
            success=result.get("success"),
            output=str(result.get("output", ""))[:200],
            error=str(result.get("error", ""))[:200],
        )
    except Exception as e:
        logger.debug("repl_arithmetic_error", error=str(e))

    return None


_GROWTH_KEYWORDS = re.compile(
    r"growth\s*rate|current\s*rate|continues?\s*to\s*grow|projection|"
    r"will\s*(?:\w+\s+)?reach|forecast|compound|cagr",
    re.IGNORECASE,
)


def _verify_and_rerun_formula(
    question: str,
    original_code: str,
    original_answer: str,
    context_text: str,
    llm: Any,
) -> str | None:
    """Verify that the generated code uses the correct formula for growth/projection
    questions.  If the verifier detects a problem, ask the LLM to rewrite the
    code and execute the corrected version.

    Returns the corrected answer string, or None if the original is fine.
    """
    if not _GROWTH_KEYWORDS.search(question):
        return None  # Not a growth question — skip verification

    verify_prompt = f"""A Python program was written to answer this question. Your job is to check whether the formula is correct.

Question: {question}

Generated code:
```python
{original_code}
```

Computed answer: {original_answer}

IMPORTANT CHECK: For questions about growth rates or projections:
- The code MUST compute the rate FROM the data (e.g. CAGR from multiple years).
- It must NOT use a single year-over-year percentage as "the rate" — that is a common mistake.
- CAGR formula: (end_value / start_value) ** (1 / num_years) - 1

Is the formula correct? Reply with:
- CORRECT — if the formula is appropriate for the question.
- INCORRECT — followed by a corrected Python code block that uses the right formula. The code must end with print(result).
"""

    try:
        response = str(llm.complete(verify_prompt)).strip()

        if "INCORRECT" in response.upper():
            logger.info(
                "formula_verification_flagged",
                question=question[:80],
            )
            # Extract corrected code and execute it
            code_match = re.search(
                r"```(?:python)?\s*\n(.*?)```",
                response,
                re.S,
            )
            if code_match:
                corrected_code = code_match.group(1).strip()
                try:
                    from rnsr.agent.repl_env import REPLEnvironment
                    from rnsr.indexing.kv_store import InMemoryKVStore

                    env = REPLEnvironment(
                        document_text=context_text,
                        skeleton={},
                        kv_store=InMemoryKVStore(),
                    )
                    result = env.execute(corrected_code)

                    if result["success"] and result["output"]:
                        output = str(result["output"]).strip()
                        nums = re.findall(r'[-]?[\d,]+\.?\d*', output)
                        if nums:
                            num_str = nums[-1].replace(",", "")
                            val = float(num_str)
                            corrected_answer = str(int(val)) if val == int(val) and abs(val) < 1e15 else str(val)
                            logger.info(
                                "formula_verification_corrected",
                                original=original_answer,
                                corrected=corrected_answer,
                                question=question[:80],
                            )
                            return corrected_answer
                except Exception as e:
                    logger.warning("formula_verification_rerun_failed", error=str(e))
        else:
            logger.debug(
                "formula_verification_passed",
                question=question[:80],
            )
    except Exception as e:
        logger.debug("formula_verification_error", error=str(e))

    return None


def _strip_citations(text: str) -> str:
    """Remove parenthetical citations like (Source: ...) and (Source: "...") from text."""
    # Remove patterns like: (Source: "...") or (Source: $CONTEXT_1 ...) or (Source: ...)
    text = re.sub(r'\s*\(Source:?\s*[^)]*\)', '', text, flags=re.IGNORECASE)
    # Remove patterns like: (Source: "..." in $CONTEXT_1)
    text = re.sub(r'\s*\(Source\b[^)]*\)', '', text, flags=re.IGNORECASE)
    # Remove inline citation markers like: === $CONTEXT_1 ===
    text = re.sub(r'===\s*\$\w+\s*===', '', text)
    # Remove $CONTEXT_N$ or $CONTEXT_N references
    text = re.sub(r'\$CONTEXT_\d+\$?', '', text)
    # Clean up multiple spaces
    text = re.sub(r'  +', ' ', text).strip()
    return text


def _extract_numeric_answer(text: str) -> str | None:
    """Try to extract a bare numeric answer from text (for arithmetic questions)."""
    if not text:
        return None
    # First line often has the answer
    first_line = text.split("\n")[0].strip()
    # Strip citations first
    first_line = _strip_citations(first_line)
    # Try to find a number (possibly with commas, decimals, percent, $, negative)
    # Pattern: optional $, optional -, digits with commas/decimals, optional %
    match = re.search(r'[-]?\$?[\d,]+\.?\d*%?', first_line)
    if match:
        num_str = match.group(0).replace(',', '').strip('$').strip('%')
        # Validate it parses as a number
        try:
            float(num_str)
            return num_str
        except ValueError:
            pass
    return None


_UNANSWERABLE_PATTERNS = [
    "cannot answer",
    "cannot determine",
    "not possible to determine",
    "no relevant content",
    "unable to answer",
    "not enough information",
    "cannot be determined",
    "context is empty",
    "information is not available",
    "not explicitly stated",
    "does not contain",
    "not mentioned",
    "no information",
    "cannot find",
    "i cannot answer",
    "unanswerable",
    "not answerable",
    "insufficient information",
    "no answer",
    "not provided in",
    "not found in",
]


def _extract_first_answer_phrase(text: str, max_chars: int = 300, is_arithmetic: bool = False) -> str:
    """
    Extract a short answer phrase for F1-friendly evaluation (token-level F1).

    Strategy (in order):
    1. For arithmetic: try to extract a bare number.
    2. Check for unanswerable phrasings and normalize to "Unanswerable".
    3. Take the first line only (prompt guarantees line 1 is the bare answer).
    4. Strip citations for cleaner F1 scoring.
    """
    if not text or not text.strip():
        return text
    text = text.strip()

    # For arithmetic questions, try to extract a bare number first
    if is_arithmetic:
        num = _extract_numeric_answer(text)
        if num is not None:
            return num

    # Strip citations before extracting
    text = _strip_citations(text)

    # First line (up to newline) — the prompt guarantees this is the bare answer
    first_line = text.split("\n")[0].strip()
    if not first_line:
        first_line = text[:max_chars].strip()

    # Check for unanswerable phrasings and normalize
    first_line_lower = first_line.lower()
    for pattern in _UNANSWERABLE_PATTERNS:
        if pattern in first_line_lower:
            return "Unanswerable"

    # For short-answer mode, return ONLY the first line (no sentence splitting)
    # The prompt now enforces bare-answer-first-line format
    if len(first_line) <= max_chars:
        return first_line
    # Truncate at last space before max_chars to avoid cutting words
    chunk = first_line[:max_chars]
    last_space = chunk.rfind(" ")
    return (chunk[: last_space + 1] if last_space >= 0 else chunk).strip()


def _is_table_content(content: str) -> bool:
    """Heuristic: detect if content looks like a table (pipe-delimited rows, etc.)."""
    lines = content.strip().split("\n")
    if len(lines) < 2:
        return False
    pipe_lines = sum(1 for l in lines[:10] if "|" in l)
    tab_lines = sum(1 for l in lines[:10] if "\t" in l)
    return pipe_lines >= 2 or tab_lines >= 2


def _label_pointer(
    pointer: str,
    content: str,
    variable_store: VariableStore,
    is_arithmetic: bool,
) -> str:
    """
    Create a label for a context pointer in synthesis.

    For arithmetic mode, labels table content with 'TABLE: ... (USE EXACT VALUES)'
    so the LLM prefers exact numeric values from tables.
    """
    if not is_arithmetic:
        return pointer

    # Check source node header for TABLE indication
    meta = variable_store.get_metadata(pointer)
    source_id = meta.source_node_id if meta else ""
    is_table = False

    # Heuristic 1: source node id contains "table"
    if source_id and "table" in source_id.lower():
        is_table = True
    # Heuristic 2: content looks like a table
    if not is_table and _is_table_content(content):
        is_table = True

    if is_table:
        return f"TABLE: {pointer} (USE EXACT VALUES FROM THIS TABLE)"
    return pointer


# =============================================================================
# Header-Match Fallback (bypass ToT when no content found)
# =============================================================================


def _header_match_fallback(
    question: str,
    skeleton: dict[str, SkeletonNode],
    kv_store: "KVStore",
    variable_store: VariableStore,
    min_selections: int = 2,
    max_selections: int = 5,
) -> list[str]:
    """
    Fallback when primary navigation finds no content.

    Presents ALL skeleton headers to the LLM in one call and asks it to
    pick which sections are most likely to answer the question.  Then reads
    the content of those nodes directly from the KV store and stores them
    as variables so synthesis can proceed.

    Args:
        question: The user's question.
        skeleton: Full skeleton index.
        kv_store: KV store with full content.
        variable_store: Where to store retrieved content.
        min_selections: Minimum sections the LLM must pick (default 2).
        max_selections: Maximum sections the LLM may pick (default 5).

    Returns:
        List of pointer names stored (e.g. ["$CONTEXT_1", "$CONTEXT_2"]).
        Empty list if nothing useful was found.
    """
    # 1. Build header list (exclude root)
    entries: list[tuple[str, str, str]] = []  # (node_id, header, preview)
    for nid, node in skeleton.items():
        if node.parent_id is None:
            continue  # skip root
        preview = node.summary[:120] if node.summary else ""
        entries.append((nid, node.header, preview))

    if not entries:
        logger.debug("header_match_fallback_no_entries")
        return []

    # 2. Format for LLM
    header_lines = "\n".join(
        f"{i + 1}. [{entry[0]}] {entry[1]} -- {entry[2]}"
        for i, entry in enumerate(entries)
    )

    prompt = f"""Given this question, which document sections are most likely to contain the answer?
You MUST pick between {min_selections} and {max_selections} sections.

Question: {question}

Available sections:
{header_lines}

Reply with ONLY the section numbers (comma-separated), e.g.: 2, 5, 7
You MUST select at least {min_selections} sections, even if you are unsure."""

    # 3. Call LLM
    try:
        from rnsr.llm import get_llm
        llm = get_llm()
        response = str(llm.complete(prompt)).strip()
    except Exception as e:
        logger.warning("header_match_fallback_llm_failed", error=str(e))
        return []

    logger.info(
        "header_match_fallback_response",
        response=response[:200],
        question=question[:80],
    )

    # 4. Parse response — extract numbers
    selected_indices: list[int] = []
    for token in re.split(r"[,\s]+", response):
        token = token.strip().rstrip(".")
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(entries):
                selected_indices.append(idx)

    if not selected_indices:
        logger.debug("header_match_fallback_no_selections_parsed", response=response[:200])
        return []

    # Deduplicate while preserving order, cap at max_selections
    seen: set[int] = set()
    unique_indices: list[int] = []
    for idx in selected_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    unique_indices = unique_indices[:max_selections]

    # 5. Read content from KV store and store as variables
    pointers_stored: list[str] = []
    for rank, idx in enumerate(unique_indices):
        node_id, header, _preview = entries[idx - 1]  # 1-indexed → 0-indexed
        content = kv_store.get(node_id)
        if not content or len(content.strip()) < 20:
            continue
        pointer = generate_pointer_name(header)
        # Dedup to avoid overwriting if multiple nodes share a header
        base_pointer = pointer
        counter = 2
        while variable_store.exists(pointer):
            pointer = f"{base_pointer}_{counter}"
            counter += 1
        variable_store.assign(pointer, content, source_node_id=node_id)
        pointers_stored.append(pointer)

    logger.info(
        "header_match_fallback_stored",
        num_selected=len(unique_indices),
        num_stored=len(pointers_stored),
        node_ids=[entries[i - 1][0] for i in unique_indices],
    )

    return pointers_stored


def synthesize_answer(
    state: AgentState,
    variable_store: VariableStore,
    kv_store: KVStore | None = None,
) -> AgentState:
    """
    Synthesize final answer from stored variables using an LLM.
    
    Uses the configured LLM to generate a concise answer
    from the resolved variable content.  When *kv_store* is provided
    and image bytes are available for visited nodes, the synthesis
    LLM receives the image directly (multimodal) for higher precision.
    """
    new_state = cast(AgentState, dict(state))
    
    # Resolve all variables
    pointers = new_state["variables"]
    
    if not pointers:
        new_state["answer"] = "No relevant content found."
        new_state["confidence"] = 0.0
    else:
        # Determine arithmetic mode early so we can label table content
        metadata = new_state.get("metadata", {})
        requires_arithmetic = metadata.get("requires_arithmetic", False)
        answer_type = metadata.get("answer_type", "")
        _is_arith_early = requires_arithmetic or answer_type in (
            "arithmetic", "counting", "multi-span",
        ) or "hybrid-arithmetic" in metadata.get("reasoning_type", "") or "hybrid-counting" in metadata.get("reasoning_type", "")

        # Collect all content with section labels for grounding (=== $CONTEXT_1 === etc.)
        # Skip tiny segments (< 50 chars) that are typically section headers / noise
        MIN_CONTENT_CHARS = 50
        labeled_parts = []
        for pointer in pointers:
            content = variable_store.resolve(pointer)
            if content and len(content.strip()) >= MIN_CONTENT_CHARS:
                label = _label_pointer(pointer, content, variable_store, _is_arith_early)
                labeled_parts.append(f"=== {label} ===\n{content}")
        # Fallback: if all segments were tiny, include them anyway
        if not labeled_parts:
            for pointer in pointers:
                content = variable_store.resolve(pointer)
                if content:
                    label = _label_pointer(pointer, content, variable_store, _is_arith_early)
                    labeled_parts.append(f"=== {label} ===\n{content}")
        context_text = "\n\n---\n\n".join(labeled_parts)
        question = new_state["question"]
        
        # Use LLM to synthesize answer
        try:
            from rnsr.llm import get_llm

            llm = get_llm()

            options = metadata.get("options")
            is_arithmetic = _is_arith_early

            if options and isinstance(options, list) and len(options) > 0:
                # QuALITY multiple-choice question - STRICT GROUNDING MODE
                # Ensure each option is a string and format with letters
                options_text = "\n".join([f"{chr(65+i)}. {str(opt)}" for i, opt in enumerate(options)])
                prompt = f"""Answer this multiple-choice question using ONLY the provided context. Do NOT infer or guess.

STRICT GROUNDING RULES:
1. Select the option that is EXPLICITLY supported by the context
2. You MUST cite the exact text from the context that supports your choice
3. If NO option is explicitly supported, respond with: "Cannot determine from context"
4. Do NOT choose an option based on inference or general knowledge

Question: {question}

Options:
{options_text}

Context:
{context_text}

Instructions:
1. Read the context carefully
2. Find the EXACT text that supports one of the options
3. Respond with the letter, full option text, AND your supporting quote
4. Format: "X. [option text] (Source: 'exact quote from context')"

Your answer:"""
            elif is_arithmetic:
                # Arithmetic / computation question — CODE EXECUTION ONLY
                # The LLM must NEVER do mental math. It extracts values and
                # writes Python code; we execute the code for the answer.
                prompt = f"""Answer the question using the provided context.

RULES:
1. PREFER exact numeric values from tables over approximate values from narrative text (e.g. use "$14,740" from a table rather than "$14.7 million" from a paragraph).
2. If values are in "thousands" or "millions", convert to the unit the question requests. If the question says "(in million)" give the answer in millions; if "(in thousand)" give in thousands.
3. You must NEVER do arithmetic in your head. ALL computation must be done via Python code.
4. If the question asks for a year or a name (no math needed), you may answer directly.
5. GROWTH RATE / PROJECTION RULE: When the question asks about a "growth rate", "current rate", "continues to grow", or future projections, you MUST:
   a) Extract ALL available data points across multiple years — not just one year.
   b) Compute the Compound Annual Growth Rate (CAGR) from the data: CAGR = (end_value / start_value) ** (1 / num_years) - 1
   c) Apply the CAGR for the projection period. Do NOT use a single year-over-year percentage as "the rate".

DECIDE: Does this question require mathematical computation?

OPTION A — If YES (sums, averages, ratios, differences, percentages, growth rates, projections, etc.):
You MUST write Python code. Do NOT compute the answer yourself.
1. First extract the exact values from the context.
2. Then write a Python code block that computes the answer.
3. The code MUST call print() with the final numeric result.
4. The code will be executed by a Python interpreter — you do NOT need to evaluate it.

Format:
EXTRACTED VALUES:
- value_1 = <number> (source: "<quote from context>")
- value_2 = <number> (source: "<quote from context>")
... (list all relevant values)

CODE:
```python
# Step-by-step computation
value_1 = <number>
value_2 = <number>
# ... calculation logic ...
result = <computation>
print(result)
```

Example 1 — Average:
Question: "What is the average of revenue in 2019 and 2020?"
EXTRACTED VALUES:
- revenue_2019 = 14740 (source: "Revenue 2019: $14,740")
- revenue_2020 = 1910 (source: "Revenue 2020: $1,910")

CODE:
```python
revenue_2019 = 14740
revenue_2020 = 1910
average = (revenue_2019 + revenue_2020) / 2
print(average)
```

Example 2 — Growth rate projection (CAGR):
Question: "What will Distribution fees reach in 2010 if it continues to grow at its current rate?"
EXTRACTED VALUES:
- fees_2007 = 1904 (source: "Distribution fees 2007: 1,904")
- fees_2008 = 1850 (source: "Distribution fees 2008: 1,850")
- fees_2009 = 1733 (source: "Distribution fees 2009: 1,733")

CODE:
```python
# Extract ALL available years of data
fees_2007 = 1904
fees_2008 = 1850
fees_2009 = 1733

# Compute CAGR from earliest to latest available year
start_value = fees_2007
end_value = fees_2009
num_years = 2009 - 2007  # = 2
cagr = (end_value / start_value) ** (1 / num_years) - 1

# Project forward from last known year to target year
years_forward = 2010 - 2009  # = 1
projected = end_value * (1 + cagr) ** years_forward
print(round(projected, 5))
```

OPTION B — If NO (the answer is a direct lookup, comparison, "which year", "what is the name", etc.):
Output a line starting with NO_COMPUTE: followed by the direct answer from the data.
Example: NO_COMPUTE: 2019
Example: NO_COMPUTE: Revenue from operations

Question: {question}

Context:
{context_text}

Response (use CODE or NO_COMPUTE):"""
            else:
                # Standard open-ended question - STRICT GROUNDING MODE
                use_short = metadata.get("use_short_answer", False)
                if use_short:
                    # Short-answer mode: force bare answer on first line for F1
                    format_block = """CRITICAL FORMAT RULE:
Line 1 MUST be ONLY the bare minimal answer — key words/phrases only, NO full sentences, NO articles, NO explanations.
Examples of GOOD first lines: "3,044 sentences in 100 dialogs" / "TransferTransfo and Hybrid" / "Unanswerable"
Examples of BAD first lines: "The dataset consists of 3,044 sentences..." / "This work outperforms the TransferTransfo..."
If the answer cannot be found, write exactly: "Unanswerable"
Line 2+: Optional supporting evidence with citations."""
                else:
                    format_block = """FORMAT YOUR ANSWER AS:
- First line: concise direct answer (good for F1). Then optional lines with inline citations like: "The value is X" (Source: "exact quote from context")
- If multiple sources support a claim, cite each one. Section labels (e.g. === $CONTEXT_1 ===) indicate which part of the context you are citing."""
                prompt = f"""Answer the question using ONLY the provided context. Do NOT infer, guess, or add information not explicitly stated.

STRICT GROUNDING RULES:
1. Answer ONLY using information explicitly stated in the context below
2. For EVERY claim you make, cite the source by quoting the exact text in quotation marks
3. Do NOT infer, deduce, or extrapolate beyond what is explicitly written
4. If the answer is NOT explicitly stated in the context, respond with exactly: "I cannot answer this from the provided text."
5. If only a partial answer is available, state what IS explicitly mentioned and acknowledge what is missing

{format_block}

Question: {question}

Context:
{context_text}

Answer:"""

            # --- Multimodal: collect image bytes from visited nodes ---
            image_bytes = None
            if kv_store:
                for node_id in new_state.get("visited_nodes", []):
                    try:
                        img = kv_store.get_image(node_id)
                        if img:
                            image_bytes = img
                            break  # Use first image found
                    except Exception:
                        pass

            if image_bytes and hasattr(llm, "complete_with_image"):
                # Strip the [VISION ANALYSIS] text from the context so it
                # doesn't conflict with what the LLM sees directly in the
                # image.  The vision text is an approximate conversion and
                # can poison exact-value readings (e.g. 0.240 vs 0.28).
                import re as _re
                prompt = _re.sub(
                    r"\[VISION ANALYSIS\].*",
                    "[Image attached — read values directly from the image below.]",
                    prompt,
                    flags=_re.S,
                )
                # Prepend a multimodal instruction
                prompt = (
                    "IMPORTANT: An image of the document is attached. "
                    "Read ALL values directly from the image. Do NOT rely "
                    "on any approximate text descriptions — use the image "
                    "as the primary source of truth for exact numbers, "
                    "chart values, and table entries.\n\n" + prompt
                )
                logger.info(
                    "multimodal_synthesis",
                    question=question[:80],
                    image_size=len(image_bytes),
                )

                # --- Dual-read with structured chart verification ---
                # Read 1: Original prompt (main answer)
                # Read 2: Structured chart reading (nearest tick mark)
                # If both agree, use the answer. If they differ, prefer the
                # structured reading (snaps to grid lines = more reliable).
                _multi_read_responses: list[str] = []

                # Read 1: original prompt
                try:
                    _mr_resp_0 = str(llm.complete_with_image(prompt, image_bytes)).strip()
                    _multi_read_responses.append(_mr_resp_0)
                except Exception as _mr_err:
                    logger.debug("multi_read_attempt_failed", read_idx=0, error=str(_mr_err))

                # Read 2: Structured chart reading — nearest tick mark
                _structured_prompt = f"""Look at the attached chart/image carefully.

Step 1 — GRID: What are ALL the y-axis tick marks? List each value from bottom to top.
Step 2 — LOCATE: At the x-axis point asked about, which y-axis tick mark is the data point CLOSEST to? Is it (a) right on a tick mark, or (b) between two? If between, which is it CLOSER to?
Step 3 — VALUE: If the data point is on or very near a tick mark, use that tick mark's exact value. If clearly between two tick marks, give the value of the CLOSER one.

Question: {question}

Reply with the steps, ending with ANSWER: <number>"""
                _structured_val: float | None = None
                try:
                    _struct_resp = str(llm.complete_with_image(_structured_prompt, image_bytes)).strip()
                    _multi_read_responses.append(_struct_resp)
                    logger.debug(
                        "structured_chart_read",
                        response=_struct_resp[:300],
                        question=question[:80],
                    )
                    # Extract the ANSWER: value
                    _ans_match = re.search(r'ANSWER:\s*([\d,.]+)', _struct_resp)
                    if _ans_match:
                        _structured_val = float(_ans_match.group(1).replace(",", ""))
                except Exception as _sc_err:
                    logger.debug("structured_chart_read_failed", error=str(_sc_err))

                # Use the first response as the main answer text
                response = _multi_read_responses[0] if _multi_read_responses else llm.complete_with_image(prompt, image_bytes)

                # If the structured read produced a value, and it differs from
                # the original answer's numeric value, prefer the structured
                # value (it snaps to grid lines which is more reliable).
                if _structured_val is not None:
                    _numeric_extract_prompt = (
                        f"The following is an answer to the question: \"{question}\"\n"
                        "Extract ONLY the final numeric answer value. "
                        "Reply with ONLY the number, nothing else.\n\n"
                    )
                    try:
                        _resp_text = str(response)
                        _ans_extract = str(llm.complete(
                            _numeric_extract_prompt + f"Answer: {_resp_text[:500]}\n\nNumeric value:"
                        )).strip()
                        _ans_nums = re.findall(r'[-]?[\d,]+\.?\d*', _ans_extract)
                        if _ans_nums:
                            _orig_val_str = _ans_nums[0].replace(",", "")
                            _orig_val = float(_orig_val_str)
                            if abs(_orig_val - _structured_val) > 0.005:
                                # Replace the original value with the structured one
                                if _orig_val_str in _resp_text:
                                    if "." in _orig_val_str:
                                        _decimals = len(_orig_val_str.split(".")[1])
                                        _formatted = f"{_structured_val:.{_decimals}f}"
                                    else:
                                        _formatted = str(int(_structured_val)) if _structured_val == int(_structured_val) else str(_structured_val)
                                    response = _resp_text.replace(_orig_val_str, _formatted, 1)
                                    logger.info(
                                        "structured_chart_value_correction",
                                        original=_orig_val_str,
                                        structured=_formatted,
                                        question=question[:80],
                                    )
                    except Exception:
                        pass
            else:
                response = llm.complete(prompt)
            answer = str(response).strip()

            # --- Code execution: extract and run Python code for arithmetic ---
            if is_arithmetic and not options:
                # Capture the original code block for verification
                _original_code = _extract_code_block(answer)

                computed_answer = _try_compute_from_response(
                    answer, context_text=context_text,
                )
                if computed_answer is not None:
                    if computed_answer != _NO_COMPUTE_MARKER:
                        answer = computed_answer
                        logger.info(
                            "code_execution_used",
                            computed_answer=computed_answer,
                            question=question[:80],
                        )
                        # --- Formula verification for growth/projection ---
                        if _original_code:
                            verified = _verify_and_rerun_formula(
                                question, _original_code, answer,
                                context_text, llm,
                            )
                            if verified is not None:
                                answer = verified
                    else:
                        # LLM explicitly said no computation needed — keep
                        # the original answer as-is and skip REPL
                        logger.info(
                            "no_compute_skip_repl",
                            question=question[:80],
                        )
                else:
                    # CODE/COMPUTE not found or failed — try REPL as last
                    # resort (asks the LLM to generate code from scratch)
                    repl_answer = _try_repl_arithmetic(question, context_text, llm)
                    if repl_answer is not None:
                        answer = repl_answer
                        logger.info(
                            "repl_fallback_used",
                            repl_answer=repl_answer,
                            question=question[:80],
                        )

            # Optional: use first phrase only for token-level F1 (e.g. QASPER)
            if not options and metadata.get("use_short_answer"):
                answer = _extract_first_answer_phrase(answer, is_arithmetic=is_arithmetic)
            
            # Normalize multiple-choice answers
            if options:
                answer_lower = answer.lower().strip()
                matched = False
                
                for i, opt in enumerate(options):
                    letter = chr(65 + i)  # A, B, C, D
                    opt_lower = opt.lower()
                    
                    # Match patterns: "A.", "A)", "(A)", "A. answer text"
                    if (answer_lower.startswith(f"{letter.lower()}.") or 
                        answer_lower.startswith(f"{letter.lower()})") or 
                        answer_lower.startswith(f"({letter.lower()})")):
                        answer = opt
                        matched = True
                        break
                    
                    # Exact match (case insensitive)
                    if answer_lower == opt_lower:
                        answer = opt
                        matched = True
                        break
                    
                    # Check if option text is contained in answer
                    if opt_lower in answer_lower or answer_lower in opt_lower:
                        # Use similarity to pick best match if multiple partial matches
                        answer = opt
                        matched = True
                        break
                
                new_state["confidence"] = 0.8 if matched else 0.1
            else:
                new_state["confidence"] = min(1.0, len(pointers) * 0.3)
            
            new_state["answer"] = answer

        except Exception as e:
            logger.warning("llm_synthesis_failed", error=str(e))
            # Fallback to context concatenation
            new_state["answer"] = f"Context found:\n{context_text}"
            new_state["confidence"] = min(1.0, len(pointers) * 0.2)
    
    add_trace_entry(
        new_state,
        "synthesis",
        "Generated answer from variables",
        {"variables_used": pointers, "confidence": new_state["confidence"]},
    )
    
    return new_state


# =============================================================================
# RLM Recursive Execution Functions (Section 2.2)
# =============================================================================


def execute_sub_task_with_llm(
    sub_task: str,
    context: str,
    llm_fn: Any = None,
) -> str:
    """
    Execute a single sub-task using an LLM.
    
    Implements the recursive LLM pattern from Section 2.2:
    "For each contract, call the LLM API (invoking itself) with a 
    specific sub-prompt: 'Extract liability clause from this text.'"
    
    Args:
        sub_task: The sub-task description/prompt.
        context: The context to process.
        llm_fn: Optional LLM function. If None, uses default.
        
    Returns:
        LLM response as string.
    """
    if llm_fn is None:
        try:
            from rnsr.llm import get_llm
            llm = get_llm()
            llm_fn = lambda p: str(llm.complete(p))
        except Exception as e:
            logger.warning("llm_not_available", error=str(e))
            return f"[Error: LLM not available - {str(e)}]"
    
    prompt = f"""{sub_task}

Context:
{context}

Response:"""
    
    try:
        return llm_fn(prompt)
    except Exception as e:
        logger.error("sub_task_execution_failed", error=str(e))
        return f"[Error: {str(e)}]"


def batch_execute_sub_tasks(
    sub_tasks: list[str],
    contexts: list[str],
    batch_size: int = 5,
    max_parallel: int = 4,
) -> list[str]:
    """
    Execute multiple sub-tasks in parallel batches.
    
    Implements Section 2.3 "Optimization via Batching":
    "Instead of making 1,000 individual calls to summarize 1,000 paragraphs,
    the RLM writes code to group paragraphs into chunks of 5 and processes 
    them in parallel threads."
    
    Args:
        sub_tasks: List of sub-task prompts.
        contexts: List of contexts for each sub-task.
        batch_size: Items per batch (default 5).
        max_parallel: Max parallel threads (default 4).
        
    Returns:
        List of results for each sub-task.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if len(sub_tasks) != len(contexts):
        raise ValueError("sub_tasks and contexts must have same length")
    
    if not sub_tasks:
        return []
    
    logger.info(
        "batch_execution_start",
        num_tasks=len(sub_tasks),
        batch_size=batch_size,
    )
    
    # Get LLM function
    try:
        from rnsr.llm import get_llm
        llm = get_llm()
        llm_fn = lambda p: str(llm.complete(p))
    except Exception as e:
        logger.error("batch_llm_failed", error=str(e))
        return [f"[Error: {str(e)}]"] * len(sub_tasks)
    
    results: list[str] = [""] * len(sub_tasks)
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_idx: dict[Any, int] = {}
        
        for idx, (task, ctx) in enumerate(zip(sub_tasks, contexts)):
            future = executor.submit(execute_sub_task_with_llm, task, ctx, llm_fn)
            future_to_idx[future] = idx
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result(timeout=120)
            except Exception as e:
                results[idx] = f"[Error: {str(e)}]"
    
    logger.info("batch_execution_complete", num_results=len(results))
    return results


def process_pending_questions(
    state: AgentState,
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    variable_store: VariableStore,
) -> AgentState:
    """
    Process all pending sub-questions using recursive LLM calls.
    
    This implements the full RLM recursive loop:
    1. For each pending question, find relevant context
    2. Invoke sub-LLM to extract answer
    3. Store result as variable
    4. Move to next question
    """
    new_state = cast(AgentState, dict(state))
    pending = new_state.get("pending_questions", [])
    
    if not pending:
        return new_state
    
    current_question = pending[0]
    
    # Find relevant content for this question
    # Use current node and its children
    node_id = new_state.get("current_node_id") or "root"
    node = skeleton.get(node_id)
    
    if node is None:
        # Pop and continue
        new_state["pending_questions"] = pending[1:]
        return new_state
    
    # Get context from current node and children
    context_parts = []
    
    # Add current node content
    current_content = kv_store.get(node_id) if node_id else None
    if current_content:
        context_parts.append(f"[{node.header}]\n{current_content}")
    
    # Add children summaries
    for child_id in node.child_ids:
        child = skeleton.get(child_id)
        if child:
            child_content = kv_store.get(child_id)
            if child_content:
                context_parts.append(f"[{child.header}]\n{child_content}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Execute sub-task with LLM
    result = execute_sub_task_with_llm(
        sub_task=f"Answer this question: {current_question}",
        context=context,
    )
    
    # Store as variable
    pointer = generate_pointer_name(current_question[:30])
    variable_store.assign(pointer, result, node_id or "root")
    new_state["variables"].append(pointer)
    
    add_trace_entry(
        new_state,
        "decomposition",
        f"Processed sub-question via recursive LLM",
        {
            "question": current_question,
            "pointer": pointer,
            "context_length": len(context),
        },
    )
    
    # Pop processed question
    new_state["pending_questions"] = pending[1:]
    if pending[1:]:
        new_state["current_sub_question"] = pending[1]
    
    new_state["iteration"] = new_state["iteration"] + 1
    return new_state


# =============================================================================
# Graph Builder (LangGraph)
# =============================================================================


def build_navigator_graph(
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    semantic_searcher: SemanticSearcher | None = None,
) -> Any:
    """
    Build the LangGraph state machine for document navigation.
    
    Implements Tree of Thoughts (ToT) navigation with backtracking support.
    
    Returns a compiled graph that can be invoked with a question.
    
    Usage:
        graph = build_navigator_graph(skeleton, kv_store)
        result = graph.invoke({"question": "What are the payment terms?"})
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        raise ImportError(
            "LangGraph not installed. Install with: pip install langgraph"
        )
    
    variable_store = VariableStore()
    
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("decompose", decompose_query)
    
    graph.add_node(
        "navigate",
        lambda state: navigate_tree(cast(AgentState, state), skeleton),
    )
    
    graph.add_node(
        "expand",
        lambda state: expand_current_node(cast(AgentState, state), skeleton, kv_store, variable_store),
    )
    
    graph.add_node(
        "traverse",
        lambda state: traverse_to_children(
            cast(AgentState, state),
            skeleton,
            semantic_searcher,
            kv_store,
        ),
    )
    
    # Add backtrack node for ToT dead-end handling
    graph.add_node(
        "backtrack",
        lambda state: backtrack_to_parent(cast(AgentState, state), skeleton),
    )
    
    graph.add_node(
        "synthesize",
        lambda state: synthesize_answer(cast(AgentState, state), variable_store, kv_store),
    )
    
    # Add edges
    graph.add_edge("decompose", "navigate")
    
    # Conditional edge based on expand/traverse/backtrack decision
    graph.add_conditional_edges(
        "navigate",
        lambda s: should_expand(s, skeleton, kv_store),
        {
            "expand": "expand",
            "traverse": "traverse",
            "backtrack": "backtrack",
            "done": "synthesize",
        },
    )
    
    # After expand, traverse, or backtrack, always return to the main navigation
    # handler, which will decide what to do next (e.g., pull from queue)
    graph.add_edge("expand", "navigate")
    graph.add_edge("traverse", "navigate")
    graph.add_edge("backtrack", "navigate")
    
    graph.add_edge("synthesize", END)
    
    # Set entry point
    graph.set_entry_point("decompose")
    
    logger.info("navigator_graph_built", features=["ToT", "backtracking"])
    
    return graph.compile()


# =============================================================================
# High-Level API
# =============================================================================


def run_navigator(
    question: str,
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    max_iterations: int = 20,
    top_k: int | None = None,
    use_semantic_search: bool = True,
    semantic_searcher: "SemanticSearcher | None" = None,
    metadata: dict[str, Any] | None = None,
    tot_selection_threshold: float = 0.4,
    tot_dead_end_threshold: float = 0.1,
    enable_header_fallback: bool = False,
    tables: list | None = None,
) -> dict[str, Any]:
    """
    Run the navigator agent on a question.

    Delegates to RLMNavigator which combines KG-aware decomposition,
    REPL code-generation navigation, arithmetic code execution,
    vision augmentation, and header-match fallback.

    Args:
        question: User's question.
        skeleton: Skeleton index.
        kv_store: KV store with full content.
        max_iterations: Maximum navigation iterations.
        top_k: Number of top children to explore.
        use_semantic_search: Unused (kept for API compatibility).
        semantic_searcher: Unused (kept for API compatibility).
        metadata: Optional metadata (e.g., multiple-choice options).
        tot_selection_threshold: Unused (kept for API compatibility).
        tot_dead_end_threshold: Unused (kept for API compatibility).
        enable_header_fallback: Unused (always enabled in RLMNavigator).

    Returns:
        Dictionary with answer, confidence, trace.
    """
    from rnsr.agent.rlm_navigator import RLMNavigator, RLMConfig

    config = RLMConfig(
        max_iterations=max_iterations,
        max_recursion_depth=3,
        enable_pre_filtering=True,
        enable_verification=False,
    )
    if top_k is not None:
        config.top_k = top_k

    navigator = RLMNavigator(
        skeleton=skeleton,
        kv_store=kv_store,
        config=config,
        tables=tables,
    )

    return navigator.navigate(question, metadata=metadata)
