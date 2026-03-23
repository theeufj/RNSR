"""
Navigation REPL - Document Tree as Searchable Environment

Extends the base REPL with navigation-specific functions that allow the LLM
to write code that searches the document tree and decides where to navigate.

This implements the RLM pattern for NAVIGATION (not just extraction):
- LLM writes Python/regex to search nodes
- Executes against the tree structure
- Returns matching nodes with relevance scores
- LLM decides which path to explore based on results

Integration with ToT:
- REPL provides grounded search results ("I found X in node Y")
- ToT evaluates those results and assigns probabilities
- Navigator moves to the most promising node
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import structlog

from rnsr.agent.variable_store import VariableStore
from rnsr.indexing.kv_store import KVStore
from rnsr.models import SkeletonNode

logger = structlog.get_logger(__name__)


# =============================================================================
# Navigation REPL System Prompt
# =============================================================================

NAV_REPL_SYSTEM_PROMPT = """You are navigating a document tree to answer a query.
The document is stored as a hierarchical tree - you must SEARCH to find relevant sections.

## Current State Variables:
- CURRENT_NODE_ID: String ID of current node (use for store_finding)
- CURRENT_NODE: Dict with keys: id, header, level, summary, num_children, parent_id
- CHILDREN: List of dicts, each with: id, header, level, summary
- QUERY: The question you're trying to answer
- HISTORY: List of previously visited node IDs

## Search Functions:

### search_content(pattern) -> list
Search current node's content for regex pattern.
Returns: [{"text": str, "start": int, "end": int, "context": str}]

### search_children(pattern) -> list
Search direct children's headers and content.
Returns: [{"node_id": str, "header": str, "level": int, "matches": int, "score": float, "preview": str, "has_children": bool}]
Results are sorted by SCORE (higher = more specific/relevant), not just match count.

### search_tree(pattern) -> list
Search ENTIRE subtree (breadth-first, unlimited depth).
Returns: [{"node_id": str, "header": str, "level": int, "depth_from_current": int, "matches": int, "score": float, "path": str}]
Results are sorted by SCORE which favors SPECIFIC sections over broad containers.
USE THIS for broad exploration - it searches all nested sections!

### get_node_content(node_id) -> str
Get full text content of any node by ID.

### get_current_content() -> str
Get FULL text content of the CURRENT node.
ALWAYS call this after navigate_to() to retrieve the actual document text!

### get_defined_vars() -> list
List user-defined variables from previous iterations.

## Navigation Functions:

### navigate_to(node_id) -> dict
Move to a different node. Updates CURRENT_NODE, CHILDREN.
Returns: {"success": bool, "now_at": dict, "children": int, "content_preview": str, "content_length": int}
The content_preview shows the first 500 chars of the section - READ IT to see if it contains your answer!

### go_back() -> dict
Return to previous node in history.

### go_to_root() -> dict
Return to document root.

## Finding Storage:

### store_finding(name, content, source_node_id=CURRENT_NODE_ID) -> str
Store found content for answer synthesis. Use CURRENT_NODE_ID for source.

### list_findings() -> list
List names of all stored findings.

### ready_to_synthesize() -> dict
Signal that you have gathered enough evidence to answer.

## Table Query Functions (SQL-like queries over detected tables):

### list_tables() -> list
List all tables detected in the document.
Returns: [{"id": str, "node_id": str, "title": str, "headers": list, "num_rows": int, "num_cols": int}]

### get_table_schema(table_id) -> dict
Get column headers and sample data for a table.
Returns: {"id": str, "title": str, "headers": list, "num_rows": int, "sample_data": list}

### query_table(table_id, columns=None, where=None, order_by=None, limit=None) -> list
Run SQL-like query on a table.
- columns: List of column names to select (None = all)
- where: Filter as {column: value} or {column: {"op": ">=", "value": 100}}
  - ops: "==", "!=", ">", ">=", "<", "<=", "contains"
- order_by: Column name to sort (prefix with "-" for descending)
- limit: Maximum rows to return
Returns: [{"col1": val1, "col2": val2, ...}, ...]

### aggregate_table(table_id, column, operation) -> float
Run aggregation on a column.
- operation: "sum", "avg", "count", "min", "max"
Returns: Numeric result

## Table Query Example:
```python
# List detected tables
tables = list_tables()
for t in tables:
    print(f"{t['id']}: {t['title']} ({t['num_rows']} rows, {t['num_cols']} cols)")

# Query a specific table for high-value items
results = query_table(
    "table_001",
    columns=["Description", "Amount"],
    where={"Amount": {"op": ">=", "value": 1000}},
    order_by="-Amount",
    limit=5
)
for row in results:
    print(f"{row['Description']}: ${row['Amount']}")

# Aggregate values
total = aggregate_table("table_001", "Amount", "sum")
print(f"Total: ${total}")
```

## Example Workflow:
```python
# 1. Search entire document tree for relevant sections
matches = search_tree(r'(?i)(parties|agreement|between)')
if matches:
    for m in matches[:5]:
        print(f"{m['node_id']}: {m['header']} ({m['matches']} matches)")
    
    # 2. Navigate to most promising section
    result = navigate_to(matches[0]['node_id'])
    print(f"Content preview: {result['content_preview'][:200]}")
    
    # 3. Get FULL content (not just preview) and store it
    content = get_current_content()  # Always call this after navigating!
    store_finding('PARTIES', content, CURRENT_NODE_ID)
    
    # 4. Signal completion
    ready_to_synthesize()
else:
    # No matches found
    ready_to_synthesize()
```

## CRITICAL RULES:
1. ALWAYS define variables before using them (e.g., `matches = search_tree(...)`)
2. Use search_tree() for broad searches - it finds ALL nested content
3. After navigate_to(), ALWAYS call get_current_content() to get full text
4. Store the FULL CONTENT from get_current_content(), not just the header
5. If a section is just a header with no content, navigate to its children
6. Always store findings before calling ready_to_synthesize()

## COMMON MISTAKE TO AVOID:
DON'T do this:
```python
navigate_to(node_id)
store_finding('SECTION', CURRENT_NODE['header'], CURRENT_NODE_ID)  # WRONG - just stores header!
```

DO this:
```python
navigate_to(node_id)
content = get_current_content()  # Get the FULL document text
store_finding('SECTION', content, CURRENT_NODE_ID)  # RIGHT - stores actual content!
```
"""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class NodeMatch:
    """A node that matched a search pattern."""
    node_id: str
    header: str
    level: int
    matches: int  # Match count
    preview: str  # First match context
    score: float = 0.0  # Relevance score


@dataclass
class ContentMatch:
    """A match within node content."""
    text: str
    start: int
    end: int
    context: str = ""  # Surrounding text


# =============================================================================
# Search Ranking Constants
# =============================================================================

# Headers that indicate catch-all/container sections (should be penalized)
CATCH_ALL_HEADERS = frozenset([
    "content", "document", "text", "body", "main", "page", 
    "section", "untitled", "unknown", "root"
])

# Headers that indicate definitional sections (should be boosted)
DEFINITIONAL_HEADERS = frozenset([
    "information", "definition", "definitions", "details", "overview",
    "parties", "introduction", "background", "about", "summary",
    # Financial statement terms
    "balance", "sheet", "assets", "liabilities", "equity",
    "income", "revenue", "statement", "schedule",
    "property", "plant", "equipment", "cash", "financial",
])

# Scoring weights
HEADER_MATCH_WEIGHT = 15  # Header matches are VERY highly relevant (e.g., "Client Information" for parties query)
SUMMARY_MATCH_WEIGHT = 5  # Summary matches are moderately relevant
CONTENT_MATCH_WEIGHT = 1  # Content matches (raw count, can accumulate)

# Penalties and bonuses
MAX_LENGTH_PENALTY = 0.3  # Max 30% penalty for very long content
LENGTH_PENALTY_THRESHOLD = 8000  # Content longer than this gets penalized
DEPTH_BONUS_PER_LEVEL = 0.15  # Each level deep adds 15% bonus
CATCH_ALL_PENALTY = 0.3  # 30% penalty for catch-all headers


def calculate_specificity_score(
    header_matches: int,
    content_matches: int,
    content_length: int,
    depth: int,
    header: str,
    summary_matches: int = 0,
    query_keywords: list[str] | None = None,
) -> float:
    """
    Calculate a specificity-aware relevance score.
    
    This score favors:
    - Sections with header matches (more specific)
    - Deeper sections (more focused content)
    - Sections with higher match density
    - Sections that historically answered similar queries (learned patterns)
    
    And penalizes:
    - Very long content (broad container sections)
    - Generic catch-all headers
    
    Args:
        header_matches: Number of regex matches in header
        content_matches: Number of regex matches in content
        content_length: Length of content in characters
        depth: Depth from search starting point
        header: The section header text
        summary_matches: Number of regex matches in summary
        query_keywords: Optional list of query keywords for learned pattern boosting
        
    Returns:
        Relevance score (higher = more relevant)
    """
    # Base score from match counts
    # CAP content matches to prevent sections with many keyword mentions
    # from outranking definitional sections (e.g., "Provider Tools" vs "Provider Information")
    capped_content_matches = min(content_matches, 5)  # Max 5 content matches count
    
    base_score = (
        header_matches * HEADER_MATCH_WEIGHT +
        summary_matches * SUMMARY_MATCH_WEIGHT +
        capped_content_matches * CONTENT_MATCH_WEIGHT
    )
    
    if base_score == 0:
        return 0.0
    
    # Calculate length penalty (broad sections get penalized)
    if content_length > LENGTH_PENALTY_THRESHOLD:
        length_ratio = min(content_length / LENGTH_PENALTY_THRESHOLD, 3.0)
        length_penalty = min(length_ratio * 0.2, MAX_LENGTH_PENALTY)
    else:
        length_penalty = 0.0
    
    # Calculate depth bonus (deeper = more specific)
    depth_bonus = depth * DEPTH_BONUS_PER_LEVEL
    
    # Calculate match density bonus (matches per 1000 chars)
    if content_length > 0:
        match_density = base_score / max(content_length / 1000, 1)
    else:
        match_density = base_score  # No content, use base score
    
    # Check for catch-all header penalty
    header_lower = header.lower().strip()
    is_catch_all = header_lower in CATCH_ALL_HEADERS or len(header_lower) < 3
    catch_all_penalty = CATCH_ALL_PENALTY if is_catch_all else 0.0
    
    # Check for definitional header bonus (e.g., "Client Information", "Provider Details")
    header_words = set(header_lower.split())
    is_definitional = bool(header_words & DEFINITIONAL_HEADERS)
    definitional_bonus = 0.5 if is_definitional else 0.0  # 50% bonus for definitional sections
    
    # Check for learned pattern bonus (sections that historically answered similar queries)
    learned_bonus = 0.0
    if query_keywords:
        try:
            # Import locally to avoid circular import
            from rnsr.agent.rlm_navigator import get_learned_section_patterns
            section_patterns = get_learned_section_patterns()
            boosted_words = section_patterns.get_boosted_header_words(query_keywords)
            if boosted_words and (header_words & boosted_words):
                learned_bonus = 0.4  # 40% bonus for sections that historically worked
                logger.debug(
                    "learned_pattern_boost_applied",
                    header=header,
                    boosted_words=list(boosted_words & header_words),
                )
        except Exception:
            pass  # Don't let learning failures affect scoring
    
    # Final score calculation
    # Base score adjusted by penalties and bonuses
    adjusted_score = base_score * (1.0 - length_penalty - catch_all_penalty) * (1.0 + depth_bonus + definitional_bonus + learned_bonus)
    
    # Add density bonus (rewards focused matches)
    final_score = adjusted_score + (match_density * 0.5)
    
    return max(final_score, 0.1)  # Minimum score if there's any match


# =============================================================================
# Navigation REPL
# =============================================================================

@dataclass
class NavigationREPL:
    """
    REPL environment specialized for document tree navigation.
    
    Exposes the document tree as a searchable environment where the LLM
    can write code to find relevant sections and decide navigation paths.
    
    Example:
        nav_repl = NavigationREPL(skeleton, kv_store)
        nav_repl.set_query("What is the CEO's salary?")
        
        # LLM generates search code
        result = nav_repl.execute('''
            matches = search_children(r'(?i)(ceo|salary|compensation)')
            for m in matches[:3]:
                print(f"{m['node_id']}: {m['header']} ({m['matches']} matches)")
        ''')
        # Output: section_42: Executive Compensation (5 matches)
        
        # LLM navigates
        nav_repl.execute("navigate_to('section_42')")
        
        # LLM searches content
        nav_repl.execute("search_content(r'\\$[\\d,]+')")
        # Output: [{"text": "$2,750,000", "start": 142, "end": 152}]
    """
    
    skeleton: dict[str, SkeletonNode]
    kv_store: KVStore
    
    # Navigation state
    current_node_id: str = "root"
    query: str = ""
    extracted_keywords: list[str] = field(default_factory=list)
    navigation_history: list[str] = field(default_factory=list)
    
    # Storage for findings
    variable_store: VariableStore = field(default_factory=VariableStore)
    findings: list[dict[str, Any]] = field(default_factory=list)
    
    # Detected tables (set via set_tables())
    _tables: list[Any] = field(default_factory=list)
    
    # Ready flag
    _ready_to_synthesize: bool = False
    
    # Execution
    context_window: int = 100  # Chars around match for context
    
    # LLM function for sub-queries (optional)
    _llm_fn: Callable[[str], str] | None = None
    
    # Allowed nodes constraint (set by ToT pre-filtering)
    # If set, navigation and search are restricted to these nodes only
    _allowed_nodes: set[str] | None = None

    # Embedding index for O(log s) ANN search (set via set_embedding_index)
    _embedding_index: Any = None

    # Session-scoped search cache: (pattern, from_node) -> results
    _search_cache: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the execution namespace."""
        self._namespace = self._build_namespace()
        logger.info(
            "nav_repl_initialized",
            num_nodes=len(self.skeleton),
            root_children=len(self._get_children_ids(self.current_node_id)),
        )
    
    def set_allowed_nodes(self, node_ids: set[str] | list[str] | None) -> None:
        """
        Constrain REPL to only navigate/search within specified nodes.
        
        This is set by ToT pre-filtering to prevent the LLM from
        navigating to irrelevant sections of the document.
        
        Args:
            node_ids: Set of node IDs that are allowed, or None to allow all.
        """
        if node_ids is None:
            self._allowed_nodes = None
            logger.info("repl_allowed_nodes_cleared")
        else:
            # Always include root and ancestors of allowed nodes
            allowed = set(node_ids) if isinstance(node_ids, list) else node_ids.copy()
            allowed.add("root")
            
            # Add parent nodes so we can navigate the tree structure
            for node_id in list(allowed):
                node = self.skeleton.get(node_id)
                if node and node.parent_id:
                    allowed.add(node.parent_id)
            
            self._allowed_nodes = allowed
            logger.info(
                "repl_allowed_nodes_set",
                num_allowed=len(self._allowed_nodes),
                sample=list(self._allowed_nodes)[:5],
            )
    
    def _is_node_allowed(self, node_id: str) -> bool:
        """Check if a node is within the allowed set."""
        if self._allowed_nodes is None:
            return True  # No constraint
        return node_id in self._allowed_nodes
    
    def set_embedding_index(self, index) -> None:
        """Attach an embedding index for fast ANN search."""
        self._embedding_index = index

    def _embedding_search(self, query: str, top_k: int = 15, doc_id: str | None = None) -> list[dict[str, Any]]:
        """O(log s) embedding-based section search via FAISS ANN.

        Returns results in the same format as _search_tree so callers
        can use either interchangeably.
        """
        if self._embedding_index is None or not self._embedding_index.is_ready:
            return []

        raw = self._embedding_index.search(query, top_k=top_k, doc_id=doc_id)
        results = []
        for r in raw:
            node_id = r["node_id"]
            if not self._is_node_allowed(node_id):
                continue
            node = self.skeleton.get(node_id)
            if node is None:
                continue
            results.append({
                "node_id": node_id,
                "header": node.header,
                "level": node.level,
                "depth_from_current": node.level,
                "matches": 1,
                "score": round(r["score"] * 50, 2),
                "path": self._get_path_to_node(node_id),
            })
        return results

    def _build_namespace(self) -> dict[str, Any]:
        """Build Python namespace with navigation functions."""
        return {
            # State (read-only for LLM, we update these)
            "CURRENT_NODE_ID": self.current_node_id,  # String ID for convenience
            "CURRENT_NODE": self._get_current_node_info(),  # Full node info dict
            "CHILDREN": self._get_children_info(),
            "QUERY": self.query,
            "HISTORY": self.navigation_history,
            
            # Built-ins
            "re": re,
            "len": len,
            "print": print,
            "str": str,
            "list": list,
            "dict": dict,
            "sorted": sorted,
            "enumerate": enumerate,
            
            # Navigation functions
            "search_content": self._search_content,
            "search_children": self._search_children,
            "search_tree": self._search_tree,
            "get_node_content": self._get_node_content,
            "navigate_to": self._navigate_to,
            "go_back": self._go_back,
            "go_to_root": self._go_to_root,
            
            # Content helpers
            "get_current_content": self._get_current_content,
            "get_defined_vars": self._get_defined_vars,
            
            # Finding storage
            "store_finding": self._store_finding,
            "get_findings": self._get_findings,
            "list_findings": self._list_findings,
            
            # Table querying
            "list_tables": self._list_tables,
            "query_table": self._query_table,
            "get_table_schema": self._get_table_schema,
            "aggregate_table": self._aggregate_table,
            
            # Synthesis trigger
            "ready_to_synthesize": self._ready_to_synthesize_fn,
        }
    
    def _update_namespace_state(self):
        """Update state variables in namespace after navigation."""
        self._namespace["CURRENT_NODE_ID"] = self.current_node_id
        self._namespace["CURRENT_NODE"] = self._get_current_node_info()
        self._namespace["CHILDREN"] = self._get_children_info()
        self._namespace["HISTORY"] = self.navigation_history.copy()
    
    # =========================================================================
    # State Helpers
    # =========================================================================
    
    def _get_current_node_info(self) -> dict[str, Any]:
        """Get info about current node."""
        node = self.skeleton.get(self.current_node_id)
        if not node:
            return {"id": self.current_node_id, "header": "Unknown", "level": 0}
        
        return {
            "id": node.node_id,
            "header": node.header,
            "level": node.level,
            "summary": node.summary[:300] if node.summary else "",
            "num_children": len(node.child_ids),
            "parent_id": node.parent_id,
        }
    
    def _get_children_info(self) -> list[dict[str, Any]]:
        """Get info about children of current node."""
        node = self.skeleton.get(self.current_node_id)
        if not node:
            return []
        
        children = []
        for child_id in node.child_ids:
            child = self.skeleton.get(child_id)
            if child:
                children.append({
                    "id": child.node_id,
                    "header": child.header,
                    "level": child.level,
                    "has_children": len(child.child_ids) > 0,
                    "summary": child.summary[:150] if child.summary else "",
                })
        return children
    
    def _get_children_ids(self, node_id: str) -> list[str]:
        """Get child IDs for a node."""
        node = self.skeleton.get(node_id)
        return node.child_ids if node else []
    
    # =========================================================================
    # Core Execution
    # =========================================================================
    
    def execute(self, code: str) -> dict[str, Any]:
        """Execute Python code in the navigation REPL."""
        # Clean code
        code = self._clean_code(code)
        
        result = {
            "success": False,
            "output": None,
            "error": None,
            "current_node": self.current_node_id,
            "findings_count": len(self.findings),
        }
        
        try:
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = captured = io.StringIO()
            
            try:
                # Try as expression
                compiled = compile(code, "<nav_repl>", "eval")
                output = eval(compiled, self._namespace)
                result["output"] = self._format_output(output)
                result["success"] = True
            except SyntaxError:
                # Execute as statements
                exec(code, self._namespace)
                printed = captured.getvalue()
                result["output"] = printed if printed else "OK"
                result["success"] = True
            finally:
                sys.stdout = old_stdout
            
            # Update state in namespace
            self._update_namespace_state()
            result["current_node"] = self.current_node_id
            result["findings_count"] = len(self.findings)
            
        except Exception as e:
            import traceback
            result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.warning("nav_repl_error", error=str(e), code=code[:100])
        
        return result
    
    def _clean_code(self, code: str) -> str:
        """Remove markdown code blocks."""
        code = re.sub(r"```python\s*", "", code)
        code = re.sub(r"```\s*", "", code)
        return code.strip()
    
    def _format_output(self, output: Any) -> Any:
        """Format output, converting dataclasses to dicts."""
        if isinstance(output, list):
            return [self._format_output(item) for item in output]
        if hasattr(output, "__dataclass_fields__"):
            from dataclasses import asdict
            return asdict(output)
        return output
    
    # =========================================================================
    # Search Functions (exposed to LLM)
    # =========================================================================
    
    def _search_content(self, pattern: str) -> list[dict[str, Any]]:
        """
        Search current node's content for a regex pattern.
        
        Returns list of matches with positions.
        """
        content = self.kv_store.get(self.current_node_id) or ""
        
        matches = []
        try:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                # Get context around match
                start = max(0, match.start() - self.context_window)
                end = min(len(content), match.end() + self.context_window)
                context = content[start:end]
                
                matches.append({
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "context": f"...{context}...",
                })
        except re.error as e:
            return [{"error": f"Invalid regex: {e}"}]
        
        return matches
    
    def _search_children(self, pattern: str) -> list[dict[str, Any]]:
        """
        Search all children's headers AND content for a pattern.
        
        Returns nodes sorted by match count (most relevant first).
        """
        node = self.skeleton.get(self.current_node_id)
        if not node:
            return []
        
        results = []
        
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return [{"error": f"Invalid regex: {e}"}]
        
        for child_id in node.child_ids:
            child = self.skeleton.get(child_id)
            if not child:
                continue
            
            # Search header
            header_matches = len(regex.findall(child.header))
            
            # Search summary
            summary_matches = len(regex.findall(child.summary or ""))
            
            # Search full content
            content = self.kv_store.get(child_id) or ""
            content_matches = len(regex.findall(content))
            
            raw_matches = header_matches + summary_matches + content_matches
            
            if raw_matches > 0:
                # Calculate specificity score (favors specific sections)
                score = calculate_specificity_score(
                    header_matches=header_matches,
                    content_matches=content_matches,
                    content_length=len(content),
                    depth=1,  # Direct children are depth 1
                    header=child.header,
                    summary_matches=summary_matches,
                    query_keywords=self.extracted_keywords,
                )
                
                # Get first match as preview
                first_match = regex.search(content or child.summary or child.header)
                preview = ""
                if first_match and content:
                    start = max(0, first_match.start() - 50)
                    end = min(len(content), first_match.end() + 50)
                    preview = content[start:end]
                
                results.append({
                    "node_id": child_id,
                    "header": child.header,
                    "level": child.level,
                    "matches": raw_matches,  # Raw count for transparency
                    "score": round(score, 2),  # Specificity score for ranking
                    "preview": f"...{preview}..." if preview else "",
                    "has_children": len(child.child_ids) > 0,
                })
        
        # Sort by specificity score (most specific first)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    
    def clear_search_cache(self) -> None:
        """Clear the session-scoped search cache (call between queries)."""
        self._search_cache.clear()

    def _search_tree(self, pattern: str, max_depth: int = 99, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Search the entire subtree from current position.

        Uses a two-stage strategy:
        1. Fast O(log s) embedding ANN search (if index available)
        2. O(k) regex refinement over ANN candidates + BFS fallback

        Results are cached per (pattern, from_node) for the session so
        repeated searches with the same pattern are O(1).
        max_depth limits how deep the BFS descends from the current node.
        """
        cache_key = (pattern, self.current_node_id, max_depth)
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        import time as _time
        _deadline = _time.monotonic() + 60

        logger.info(
            "search_tree_start",
            pattern=pattern,
            from_node=self.current_node_id,
        )

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return [{"error": f"Invalid regex: {e}"}]

        # ------------------------------------------------------------------
        # Stage 1: Try embedding ANN search for candidate retrieval
        # ------------------------------------------------------------------
        emb_candidates: set[str] = set()
        if self._embedding_index is not None and self._embedding_index.is_ready:
            plain_query = re.sub(r"\(\?i\)", "", pattern)
            plain_query = re.sub(r"[|()\\]", " ", plain_query).strip()
            if plain_query:
                emb_results = self._embedding_search(plain_query, top_k=25)
                emb_candidates = {r["node_id"] for r in emb_results}

        # ------------------------------------------------------------------
        # Stage 2: BFS with optional candidate-set pruning
        # ------------------------------------------------------------------
        results = []
        visited: set[str] = set()
        queue = [(self.current_node_id, 0)]
        use_pruning = len(emb_candidates) >= 5

        while queue:
            if _time.monotonic() > _deadline:
                logger.warning(
                    "search_tree_timeout",
                    pattern=pattern,
                    visited=len(visited),
                    results_so_far=len(results),
                )
                break

            node_id, depth = queue.pop(0)

            if node_id in visited:
                continue
            visited.add(node_id)

            if depth > max_depth:
                continue

            node = self.skeleton.get(node_id)
            if not node:
                continue

            if not self._is_node_allowed(node_id):
                for child_id in node.child_ids:
                    if child_id not in visited:
                        queue.append((child_id, depth + 1))
                continue

            # When we have embedding candidates, skip full-content regex
            # on nodes NOT in the candidate set (O(k) instead of O(s)).
            if use_pruning and node_id not in emb_candidates:
                header_matches = len(regex.findall(node.header))
                if header_matches == 0:
                    for child_id in node.child_ids:
                        if child_id not in visited:
                            queue.append((child_id, depth + 1))
                    continue

            content = self.kv_store.get(node_id) or ""
            header_matches = len(regex.findall(node.header))
            content_matches = len(regex.findall(content))

            if header_matches > 0 or content_matches > 0:
                score = calculate_specificity_score(
                    header_matches=header_matches,
                    content_matches=content_matches,
                    content_length=len(content),
                    depth=depth,
                    header=node.header,
                    query_keywords=self.extracted_keywords,
                )

                results.append({
                    "node_id": node_id,
                    "header": node.header,
                    "level": node.level,
                    "depth_from_current": depth,
                    "matches": header_matches + content_matches,
                    "score": round(score, 2),
                    "path": self._get_path_to_node(node_id),
                })

            for child_id in node.child_ids:
                if child_id not in visited:
                    queue.append((child_id, depth + 1))

        results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            "search_tree_complete",
            pattern=pattern,
            total_matches=len(results),
            emb_pruning=use_pruning,
            top_results=[(r["header"][:40], r["score"], r["node_id"]) for r in results[:5]],
        )

        self._search_cache[cache_key] = results
        return results
    
    def _get_path_to_node(self, target_id: str) -> str:
        """Get the path from root to a node."""
        path: list[str] = []
        seen: set[str] = set()
        current: str | None = target_id
        while current and current not in seen:
            seen.add(current)
            node = self.skeleton.get(current)
            if node:
                path.insert(0, node.header[:30])
                current = node.parent_id
            else:
                break
        return " > ".join(path)
    
    # =========================================================================
    # Navigation Functions
    # =========================================================================
    
    def _get_node_content(self, node_id: str) -> str:
        """Get full content of any node."""
        return self.kv_store.get(node_id) or ""
    
    def _navigate_to(self, node_id: str) -> dict[str, Any]:
        """Navigate to a different node."""
        if node_id not in self.skeleton:
            logger.warning("navigate_to_failed", node_id=node_id, reason="not found")
            return {"error": f"Node '{node_id}' not found"}
        
        # Check if node is within allowed set (ToT constraint)
        if not self._is_node_allowed(node_id):
            logger.warning(
                "navigate_to_blocked",
                node_id=node_id,
                reason="not in allowed nodes (ToT constraint)",
            )
            return {"error": f"Node '{node_id}' is not in the pre-filtered search scope"}
        
        previous_node = self.current_node_id
        
        # Record history
        self.navigation_history.append(self.current_node_id)
        
        # Move
        self.current_node_id = node_id
        
        # Update namespace
        self._update_namespace_state()
        
        # Get content preview so LLM can see what's in this section
        node = self.skeleton.get(node_id)
        content = self.kv_store.get(node_id) or ""
        content_preview = content[:500] + "..." if len(content) > 500 else content
        
        logger.info(
            "navigated_to",
            from_node=previous_node,
            to_node=node_id,
            to_header=node.header if node else "Unknown",
            content_length=len(content),
            num_children=len(self._get_children_ids(node_id)),
        )
        
        return {
            "success": True,
            "now_at": self._get_current_node_info(),
            "children": len(self._get_children_ids(node_id)),
            "content_preview": content_preview,
            "content_length": len(content),
        }
    
    def _go_back(self) -> dict[str, Any]:
        """Go back to previous node."""
        if not self.navigation_history:
            return {"error": "No navigation history"}
        
        self.current_node_id = self.navigation_history.pop()
        self._update_namespace_state()
        
        return {
            "success": True,
            "now_at": self._get_current_node_info(),
        }
    
    def _go_to_root(self) -> dict[str, Any]:
        """Navigate back to root."""
        self.navigation_history.append(self.current_node_id)
        self.current_node_id = "root"
        self._update_namespace_state()
        
        return {
            "success": True,
            "now_at": self._get_current_node_info(),
        }
    
    # =========================================================================
    # Content Helpers
    # =========================================================================
    
    def _get_current_content(self) -> str:
        """
        Get the FULL content of the current node.
        
        This is the main way to retrieve text content after navigating to a section.
        Always call this after navigate_to() to get the actual document text.
        
        Returns:
            The full text content of the current node.
        """
        content = self.kv_store.get(self.current_node_id) or ""
        if not content:
            logger.debug("no_content_for_node", node_id=self.current_node_id)
        return content
    
    def _get_defined_vars(self) -> list[str]:
        """
        Get list of user-defined variables available in the namespace.
        
        Useful for checking what variables you've defined in previous iterations.
        
        Returns:
            List of variable names that are user-defined (not system variables).
        """
        system_keys = {
            "CURRENT_NODE_ID", "CURRENT_NODE", "CHILDREN", "QUERY", "HISTORY",
            "re", "len", "print", "str", "list", "dict", "sorted", "enumerate",
            "search_content", "search_children", "search_tree", "get_node_content",
            "navigate_to", "go_back", "go_to_root", "store_finding", "get_findings",
            "list_findings", "ready_to_synthesize", "get_current_content", "get_defined_vars",
        }
        return [k for k in self._namespace.keys() if k not in system_keys and not k.startswith("_")]
    
    # =========================================================================
    # Finding Storage
    # =========================================================================
    
    def _validate_node_id(self, node_id: str) -> bool:
        """
        Validate that a node ID exists in the skeleton.
        
        Prevents hallucinated node IDs from polluting the context.
        
        Args:
            node_id: The node ID to validate.
            
        Returns:
            True if the node exists, False otherwise.
        """
        if not node_id:
            return False
        exists = node_id in self.skeleton
        if not exists:
            logger.warning(
                "invalid_node_id",
                node_id=node_id,
                available_sample=list(self.skeleton.keys())[:5],
            )
        return exists
    
    def _check_relevance(self, content: str, min_keyword_overlap: float = 0.3) -> tuple[bool, float]:
        """
        Check if content is relevant to the current query.
        
        Uses keyword overlap to quickly determine if content addresses the query.
        This prevents storing irrelevant content that could mislead synthesis.
        
        Args:
            content: The content to check.
            min_keyword_overlap: Minimum fraction of query keywords that must appear (0.0-1.0).
            
        Returns:
            Tuple of (is_relevant, relevance_score).
        """
        if not self.query or not content:
            return True, 1.0  # No query = can't check relevance
        
        # Extract keywords from query (basic tokenization)
        query_lower = self.query.lower()
        # Remove common stop words and extract meaningful terms
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
        
        logger.debug(
            "relevance_check",
            query_keywords=list(query_words)[:10],
            matched=matched_keywords,
            total=len(query_words),
            score=round(overlap_score, 2),
            is_relevant=is_relevant,
        )
        
        return is_relevant, overlap_score
    
    def _store_finding(self, name: str, content: str, source_node_id: str | dict = None) -> str:
        """Store a finding for later synthesis with strict validation."""
        # Log the incoming finding
        logger.info(
            "store_finding_called",
            name=name,
            content_length=len(content) if content else 0,
            content_preview=content[:100] if content else "None",
            source_node_id=source_node_id,
        )
        
        # Handle case where LLM passes CURRENT_NODE dict instead of ID string
        if isinstance(source_node_id, dict):
            source_node_id = source_node_id.get("id", self.current_node_id)
        
        if source_node_id is None:
            source_node_id = self.current_node_id
        
        # VALIDATION 1: Verify node ID exists in skeleton (prevents hallucinated node IDs)
        if not self._validate_node_id(source_node_id):
            logger.warning(
                "store_finding_rejected_invalid_node",
                name=name,
                source_node_id=source_node_id,
            )
            return f"Error: Node '{source_node_id}' does not exist. Use a valid node ID from the skeleton."
        
        # GROUNDING CHECK: Fetch actual node content to verify LLM-provided content
        actual_content = self.kv_store.get(source_node_id) or ""
        
        # Verify that the LLM-provided content actually exists in the node
        # This prevents hallucination by ensuring we only store grounded content
        if content and actual_content:
            content_stripped = content.strip()
            # Check if content is a substring of actual content (grounded)
            if content_stripped and content_stripped not in actual_content:
                logger.warning(
                    "content_not_grounded",
                    provided_len=len(content),
                    actual_len=len(actual_content),
                    source=source_node_id,
                )
                # Replace with actual content to prevent hallucination
                content = actual_content
        
        # AUTO-FETCH: If content is too short (likely just a header), get full node content
        MIN_CONTENT_LENGTH = 100
        if len(content) < MIN_CONTENT_LENGTH:
            full_content = self.kv_store.get(source_node_id) or ""
            if full_content and len(full_content) > len(content):
                logger.info(
                    "auto_fetched_content",
                    original_len=len(content),
                    full_len=len(full_content),
                    source=source_node_id,
                )
                content = full_content
        
        # VALIDATION 2: Check relevance to query (prevents storing irrelevant content)
        is_relevant, relevance_score = self._check_relevance(content)
        if not is_relevant:
            logger.warning(
                "store_finding_low_relevance",
                name=name,
                relevance_score=round(relevance_score, 2),
                source_node_id=source_node_id,
                query=self.query[:100] if self.query else None,
            )
            # Don't reject outright, but warn - the content might still be useful
            # The synthesis step will use strict grounding anyway
        
        # Generate pointer name
        if not name.startswith("$"):
            name = "$" + name.upper().replace(" ", "_")[:30]
        
        # Store in variable store
        self.variable_store.assign(name, content, source_node_id=source_node_id)
        
        # Also track in findings list
        source_node = self.skeleton.get(source_node_id)
        source_header = source_node.header if source_node else "Unknown"
        self.findings.append({
            "name": name,
            "content": content[:500],  # Preview
            "source_node_id": source_node_id,
            "source_header": source_header,
        })
        
        logger.info(
            "finding_stored",
            name=name,
            source_node_id=source_node_id,
            source_header=source_header,
            content_length=len(content),
            total_findings=len(self.findings),
        )
        
        return f"Stored as {name}"
    
    def _get_findings(self) -> list[dict[str, Any]]:
        """Get all stored findings."""
        return self.findings.copy()
    
    def _list_findings(self) -> list[str]:
        """List finding names."""
        return [f["name"] for f in self.findings]
    
    def _ready_to_synthesize_fn(self) -> dict[str, Any]:
        """Signal that enough findings have been collected."""
        self._ready_to_synthesize = True
        
        logger.info(
            "ready_to_synthesize_called",
            findings_count=len(self.findings),
            findings=[f["name"] for f in self.findings],
            current_node=self.current_node_id,
        )
        
        return {
            "ready": True,
            "findings_count": len(self.findings),
            "findings": self._list_findings(),
        }
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    def set_query(self, query: str):
        """Set the current query."""
        self.query = query
        self._namespace["QUERY"] = query
    
    def set_llm_function(self, llm_fn: Callable[[str], str]):
        """Set LLM function for sub-queries."""
        self._llm_fn = llm_fn
    
    def set_tables(self, tables: list) -> None:
        """
        Set detected tables for SQL-like querying.
        
        Args:
            tables: List of DetectedTable objects from ingestion.
        """
        self._tables = tables or []
        logger.info(
            "repl_tables_set",
            table_count=len(self._tables),
            table_ids=[t.id for t in self._tables[:5]],
        )
    
    # =========================================================================
    # Table Query Functions
    # =========================================================================
    
    def _list_tables(self) -> list[dict[str, Any]]:
        """
        List all tables detected in the document.
        
        Returns:
            List of table metadata dictionaries with id, title, headers, num_rows, etc.
        """
        result = []
        for table in self._tables:
            result.append({
                "id": table.id,
                "node_id": table.node_id,
                "page_num": table.page_num,
                "title": table.title,
                "headers": table.headers,
                "num_rows": table.num_rows,
                "num_cols": table.num_cols,
            })
        
        logger.info(
            "list_tables_called",
            table_count=len(result),
        )
        return result
    
    def _get_table_schema(self, table_id: str) -> dict[str, Any] | str:
        """
        Get the schema (column headers and types) for a specific table.
        
        Args:
            table_id: The table ID to get schema for.
            
        Returns:
            Dictionary with headers and sample data, or error string.
        """
        for table in self._tables:
            if table.id == table_id:
                # Get sample data (first 3 rows)
                sample_data = table.data[:3] if table.data else []
                
                logger.info(
                    "get_table_schema_called",
                    table_id=table_id,
                    headers=table.headers,
                    num_rows=table.num_rows,
                )
                
                return {
                    "id": table.id,
                    "title": table.title,
                    "headers": table.headers,
                    "num_cols": table.num_cols,
                    "num_rows": table.num_rows,
                    "sample_data": sample_data,
                }
        
        logger.warning("table_not_found", table_id=table_id)
        return f"Error: Table '{table_id}' not found. Use list_tables() to see available tables."
    
    def _query_table(
        self,
        table_id: str,
        columns: list[str] | None = None,
        where: dict | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]] | str:
        """
        SQL-like query over a detected table.
        
        Args:
            table_id: The table ID to query.
            columns: List of column names to select (None = all).
            where: Filter conditions as {column: value} or {column: {"op": ">=", "value": 100}}.
            order_by: Column name to sort by (prefix with - for descending).
            limit: Maximum number of rows to return.
            
        Returns:
            List of row dictionaries, or error string.
            
        Example:
            query_table("table_001", columns=["Name", "Amount"], where={"Amount": {"op": ">=", "value": 1000}}, limit=10)
        """
        # Find the table
        target_table = None
        for table in self._tables:
            if table.id == table_id:
                target_table = table
                break
        
        if not target_table:
            return f"Error: Table '{table_id}' not found. Use list_tables() to see available tables."
        
        headers = target_table.headers
        data = target_table.data
        
        # Convert data rows to dicts
        rows = []
        for row_data in data:
            row_dict = {}
            for i, value in enumerate(row_data):
                col_name = headers[i] if i < len(headers) else f"col_{i}"
                row_dict[col_name] = value
            rows.append(row_dict)
        
        # Apply WHERE filter
        if where:
            filtered_rows = []
            for row in rows:
                match = True
                for col, condition in where.items():
                    cell_value = row.get(col, "")
                    
                    if isinstance(condition, dict):
                        # Complex condition: {"op": ">=", "value": 100}
                        op = condition.get("op", "==")
                        cond_value = condition.get("value")
                        
                        # Try numeric comparison
                        try:
                            cell_num = float(cell_value.replace(",", "").replace("$", "").strip())
                            cond_num = float(cond_value)
                            
                            if op == "==" and cell_num != cond_num:
                                match = False
                            elif op == "!=" and cell_num == cond_num:
                                match = False
                            elif op == ">" and cell_num <= cond_num:
                                match = False
                            elif op == ">=" and cell_num < cond_num:
                                match = False
                            elif op == "<" and cell_num >= cond_num:
                                match = False
                            elif op == "<=" and cell_num > cond_num:
                                match = False
                        except (ValueError, TypeError):
                            # String comparison
                            if op == "==" and str(cell_value) != str(cond_value):
                                match = False
                            elif op == "!=" and str(cell_value) == str(cond_value):
                                match = False
                            elif op == "contains" and str(cond_value).lower() not in str(cell_value).lower():
                                match = False
                    else:
                        # Simple equality check (case-insensitive substring)
                        if str(condition).lower() not in str(cell_value).lower():
                            match = False
                
                if match:
                    filtered_rows.append(row)
            rows = filtered_rows
        
        # Apply ORDER BY
        if order_by:
            descending = order_by.startswith("-")
            col = order_by.lstrip("-")
            
            def sort_key(row):
                val = row.get(col, "")
                # Try numeric sort
                try:
                    return float(val.replace(",", "").replace("$", "").strip())
                except (ValueError, TypeError, AttributeError):
                    return str(val).lower()
            
            rows = sorted(rows, key=sort_key, reverse=descending)
        
        # Apply LIMIT
        if limit and limit > 0:
            rows = rows[:limit]
        
        # Apply column selection
        if columns:
            rows = [{col: row.get(col, "") for col in columns} for row in rows]
        
        logger.info(
            "query_table_executed",
            table_id=table_id,
            columns=columns,
            where=where,
            result_count=len(rows),
        )
        
        return rows
    
    def _aggregate_table(
        self,
        table_id: str,
        column: str,
        operation: str,
    ) -> float | int | str:
        """
        Run an aggregation operation on a table column.
        
        Args:
            table_id: The table ID to query.
            column: The column name to aggregate.
            operation: One of "sum", "avg", "count", "min", "max".
            
        Returns:
            Aggregation result (number), or error string.
            
        Example:
            aggregate_table("table_001", "Revenue", "sum")
        """
        # Find the table
        target_table = None
        for table in self._tables:
            if table.id == table_id:
                target_table = table
                break
        
        if not target_table:
            return f"Error: Table '{table_id}' not found. Use list_tables() to see available tables."
        
        headers = target_table.headers
        data = target_table.data
        
        # Find column index
        col_idx = None
        for i, h in enumerate(headers):
            if h.lower() == column.lower():
                col_idx = i
                break
        
        if col_idx is None:
            return f"Error: Column '{column}' not found. Available columns: {headers}"
        
        # Extract numeric values from the column
        values = []
        for row_data in data:
            if col_idx < len(row_data):
                val = row_data[col_idx]
                try:
                    # Clean and parse numeric value
                    clean_val = val.replace(",", "").replace("$", "").replace("%", "").strip()
                    if clean_val:
                        values.append(float(clean_val))
                except (ValueError, TypeError, AttributeError):
                    pass  # Skip non-numeric values
        
        if not values:
            return f"Error: No numeric values found in column '{column}'."
        
        operation = operation.lower()
        
        if operation == "sum":
            result = sum(values)
        elif operation == "avg":
            result = sum(values) / len(values)
        elif operation == "count":
            result = len(values)
        elif operation == "min":
            result = min(values)
        elif operation == "max":
            result = max(values)
        else:
            return f"Error: Unknown operation '{operation}'. Use: sum, avg, count, min, max."
        
        logger.info(
            "aggregate_table_executed",
            table_id=table_id,
            column=column,
            operation=operation,
            result=result,
        )
        
        return result
    
    def reset(self, preserve_user_vars: bool = False):
        """Reset navigation state.
        
        Args:
            preserve_user_vars: If True, preserve user-defined variables from previous runs.
        """
        # Optionally capture user-defined variables before reset
        user_vars = {}
        if preserve_user_vars:
            system_keys = {
                "CURRENT_NODE_ID", "CURRENT_NODE", "CHILDREN", "QUERY", "HISTORY",
                "re", "len", "print", "str", "list", "dict", "sorted", "enumerate",
                "search_content", "search_children", "search_tree", "get_node_content",
                "navigate_to", "go_back", "go_to_root", "store_finding", "get_findings",
                "list_findings", "ready_to_synthesize", "get_current_content", "get_defined_vars",
            }
            user_vars = {
                k: v for k, v in self._namespace.items() 
                if k not in system_keys and not k.startswith("_")
            }
        
        self.current_node_id = "root"
        self.navigation_history = []
        self.findings = []
        self._ready_to_synthesize = False
        self.variable_store = VariableStore()
        self._update_namespace_state()
        
        # Restore user variables if requested
        if user_vars:
            self._namespace.update(user_vars)
            logger.debug("preserved_user_vars", count=len(user_vars), vars=list(user_vars.keys()))
    
    def get_system_prompt(self) -> str:
        """Get the navigation REPL system prompt."""
        return NAV_REPL_SYSTEM_PROMPT
    
    def is_ready_to_synthesize(self) -> bool:
        """Check if LLM has signaled readiness to synthesize."""
        return self._ready_to_synthesize
    
    def get_state(self) -> dict[str, Any]:
        """Get current navigation state."""
        return {
            "current_node_id": self.current_node_id,
            "current_node": self._get_current_node_info(),
            "history": self.navigation_history,
            "navigation_history": self.navigation_history,  # Alias for rlm_navigator
            "findings": self._get_findings(),  # Return full finding dicts, not just names
            "ready": self._ready_to_synthesize,
        }


# =============================================================================
# Factory
# =============================================================================

def create_navigation_repl(
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    query: str = "",
    tables: list | None = None,
) -> NavigationREPL:
    """
    Create a configured NavigationREPL.
    
    Args:
        skeleton: Document skeleton nodes.
        kv_store: Key-value store for node content.
        query: Initial query string.
        tables: Optional list of DetectedTable objects for SQL-like queries.
        
    Returns:
        Configured NavigationREPL instance.
    """
    repl = NavigationREPL(
        skeleton=skeleton,
        kv_store=kv_store,
        query=query,
    )
    
    if tables:
        repl.set_tables(tables)
    
    return repl
