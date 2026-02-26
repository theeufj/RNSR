"""
REPL Environment - Recursive Language Model Execution Engine

Implements Section 2 of the research paper:
"The Prompt-as-Environment Abstraction" and "The Recursive Loop"

Key Concepts:
1. DOC_VAR: The document is loaded as a persistent variable, NOT passed to LLM context
2. Code Execution: LLM generates Python code to interact with DOC_VAR
3. Variable Stitching: Intermediate results frozen into variables to prevent Context Rot
4. Recursive Calls: Agent can invoke sub-LLM instances for sub-tasks

Per Zhang et al. (2025):
"The RLM initializes a computational environment where the massive input context 
(the 'Long Prompt') is loaded into a variable, typically denoted as P or DOC_VAR.
The Neural Network itself does not ingest P. Instead, it ingests a system instruction 
that informs it of P's existence and provides a set of tools to interact with it."

This decouples Memory (REPL state) from Processing (LLM inference), enabling
effectively unbounded context lengths.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import structlog

from rnsr.agent.variable_store import VariableStore, generate_pointer_name
from rnsr.indexing.kv_store import KVStore
from rnsr.models import SkeletonNode

logger = structlog.get_logger(__name__)


# =============================================================================
# RLM System Prompt - Informs LLM of DOC_VAR and available tools
# =============================================================================

RLM_SYSTEM_PROMPT = """You are a Recursive Language Model (RLM) operating in a Python REPL environment.

CRITICAL: You do NOT have the document in your context. The document is stored in the variable DOC_VAR.
You interact with the document by writing Python code that will be executed in the REPL.

## Available Variables:
- DOC_VAR: The full document text (string). Use slicing to access portions.
- DOC_TREE: Hierarchical tree structure of the document (dict).
- VARIABLES: Dictionary storing your findings (use store_variable() to add).

## Available Functions:
- len(DOC_VAR): Get document length in characters
- DOC_VAR[i:j]: Slice document to get characters from i to j
- search_text(pattern): Search DOC_VAR for regex pattern, returns list of (start, end, match)
- split_by_regex(pattern): Split DOC_VAR by pattern, returns list of segments
- list_children(node_id): Get summaries of child nodes in DOC_TREE
- read_node(node_id): Get full text content of a node
- store_variable(name, content): Store content as $NAME for later synthesis
- get_variable(name): Retrieve stored variable content
- sub_llm(prompt, context): Invoke sub-LLM to process context with prompt (for decomposition)
- batch_sub_llm(prompts, contexts): Batch process multiple sub-tasks in parallel

## Workflow for Complex Queries:
1. First, explore the document structure: `list_children('root')`
2. Navigate to relevant sections based on query
3. For multi-part queries, decompose and use sub_llm():
   ```python
   # Example: Compare clauses across documents
   clause_2023 = sub_llm("Extract the indemnification clause", read_node('2023_legal'))
   clause_2024 = sub_llm("Extract the indemnification clause", read_node('2024_legal'))
   store_variable("CLAUSE_2023", clause_2023)
   store_variable("CLAUSE_2024", clause_2024)
   ```
4. For bulk processing, use batch_sub_llm():
   ```python
   # Process 50 contracts in parallel batches
   clauses = batch_sub_llm(
       prompts=["Extract liability clause"] * 50,
       contexts=[read_node(f'contract_{i}') for i in range(50)]
   )
   ```

## Output:
Write Python code blocks to execute. The REPL will run them and return results.
When ready to synthesize, call: synthesize_answer(query, variable_names)

IMPORTANT: 
- Always use store_variable() to save findings - this prevents Context Rot
- Prefer batch_sub_llm() over sequential sub_llm() calls for efficiency
- Target ~200k characters per batch for optimal token density
"""


# =============================================================================
# REPL Environment Class
# =============================================================================


@dataclass
class REPLEnvironment:
    """
    Python REPL Environment for Recursive Language Model execution.
    
    Implements the "Prompt-as-Environment" abstraction from Section 2.1:
    - DOC_VAR: Document loaded as persistent variable
    - Code execution sandbox for LLM-generated code
    - Variable stitching for intermediate results
    - Recursive sub-LLM invocation
    
    Example:
        env = REPLEnvironment(
            document_text=full_text,
            skeleton=skeleton_index,
            kv_store=kv_store,
        )
        
        # LLM generates code, REPL executes it
        result = env.execute("len(DOC_VAR)")
        # -> 152347
        
        result = env.execute("list_children('root')")
        # -> [{'id': 'sec_001', 'header': 'Introduction', ...}, ...]
    """
    
    # The document as persistent variable (NOT in LLM context)
    document_text: str
    skeleton: dict[str, SkeletonNode]
    kv_store: KVStore
    
    # Variable store for findings
    variable_store: VariableStore = field(default_factory=VariableStore)
    
    # Execution state
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    max_output_length: int = 10000  # Truncate long outputs
    
    # Batching configuration
    batch_size: int = 5  # Process N items per batch
    max_parallel_batches: int = 4  # Max concurrent batch calls
    optimal_chars_per_call: int = 200_000  # Target ~200k chars per LLM call
    
    # LLM function for sub-calls (injected)
    _llm_fn: Callable[[str], str] | None = None
    _async_llm_fn: Callable[[str], Any] | None = None
    
    def __post_init__(self):
        """Build the execution namespace."""
        self._namespace = self._build_namespace()
        logger.info(
            "repl_initialized",
            doc_length=len(self.document_text),
            num_nodes=len(self.skeleton),
        )
    
    def _build_namespace(self) -> dict[str, Any]:
        """Build the Python namespace for code execution."""
        return {
            # Core variables (the "environment")
            "DOC_VAR": self.document_text,
            "DOC_TREE": self._build_tree_dict(),
            "VARIABLES": {},
            
            # Built-in functions available in namespace
            "len": len,
            "print": print,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "re": re,  # For regex operations
            
            # Navigator functions
            "search_text": self._search_text,
            "split_by_regex": self._split_by_regex,
            "list_children": self._list_children,
            "read_node": self._read_node,
            "get_node_path": self._get_node_path,
            
            # Variable stitching functions
            "store_variable": self._store_variable,
            "get_variable": self._get_variable,
            "list_variables": self._list_variables,
            
            # Recursive LLM functions
            "sub_llm": self._sub_llm,
            "batch_sub_llm": self._batch_sub_llm,
            
            # Synthesis
            "synthesize_answer": self._synthesize_answer,
        }
    
    def _build_tree_dict(self) -> dict[str, Any]:
        """Convert skeleton to navigable dict structure."""
        tree = {}
        for node_id, node in self.skeleton.items():
            tree[node_id] = {
                "id": node.node_id,
                "header": node.header,
                "summary": node.summary,
                "level": node.level,
                "children": node.child_ids,
                "parent": node.parent_id,
            }
        return tree
    
    # =========================================================================
    # Core Execution
    # =========================================================================
    
    def execute(self, code: str) -> dict[str, Any]:
        """
        Execute Python code in the REPL environment.
        
        Args:
            code: Python code to execute.
            
        Returns:
            Dict with 'success', 'output', 'error', and 'variables'.
        """
        start_time = datetime.now(timezone.utc)
        
        # Clean code (remove markdown code blocks if present)
        code = self._clean_code(code)
        
        result = {
            "success": False,
            "output": None,
            "error": None,
            "variables": list(self._namespace.get("VARIABLES", {}).keys()),
            "execution_time_ms": 0,
        }
        
        try:
            # Capture output
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                # Try as expression first (returns value)
                compiled = compile(code, "<repl>", "eval")
                output = eval(compiled, self._namespace)
                result["output"] = self._format_output(output)
                result["success"] = True
            except SyntaxError:
                # Execute as statements
                exec(code, self._namespace)
                printed = captured_output.getvalue()
                result["output"] = printed if printed else "Executed successfully"
                result["success"] = True
            finally:
                sys.stdout = old_stdout
            
            # Update VARIABLES reference
            result["variables"] = list(self._namespace.get("VARIABLES", {}).keys())
            
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.warning("repl_execution_error", error=str(e), code=code[:200])
        
        # Record timing
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        result["execution_time_ms"] = round(elapsed, 2)
        
        # Record in history
        self.execution_history.append({
            "code": code,
            "result": result,
            "timestamp": start_time.isoformat(),
        })
        
        logger.debug(
            "repl_executed",
            success=result["success"],
            output_length=len(str(result["output"])) if result["output"] else 0,
        )
        
        return result
    
    def _clean_code(self, code: str) -> str:
        """Remove markdown code blocks and clean whitespace."""
        # Remove ```python ... ``` blocks
        code = re.sub(r"```python\s*", "", code)
        code = re.sub(r"```\s*", "", code)
        return code.strip()
    
    def _format_output(self, output: Any) -> str:
        """Format output for display, truncating if needed."""
        if output is None:
            return "None"
        
        output_str = str(output)
        
        if len(output_str) > self.max_output_length:
            truncated = output_str[:self.max_output_length]
            return f"{truncated}\n... [truncated, {len(output_str)} total chars]"
        
        return output_str
    
    # =========================================================================
    # Navigator Functions (exposed to REPL)
    # =========================================================================
    
    def _search_text(self, pattern: str) -> list[tuple[int, int, str]]:
        """
        Search DOC_VAR for regex pattern.
        
        Returns list of (start_pos, end_pos, matched_text).
        """
        matches = []
        for match in re.finditer(pattern, self.document_text, re.IGNORECASE):
            matches.append((match.start(), match.end(), match.group()))
        return matches[:100]  # Limit results
    
    def _split_by_regex(self, pattern: str) -> list[str]:
        """Split DOC_VAR by regex pattern."""
        return re.split(pattern, self.document_text)
    
    def _list_children(self, node_id: str) -> list[dict[str, Any]]:
        """Get summaries of child nodes."""
        node = self.skeleton.get(node_id)
        if not node:
            return []
        
        children = []
        for child_id in node.child_ids:
            child = self.skeleton.get(child_id)
            if child:
                children.append({
                    "id": child.node_id,
                    "header": child.header,
                    "summary": child.summary[:200] if child.summary else "",
                    "has_children": len(child.child_ids) > 0,
                })
        return children
    
    def _read_node(self, node_id: str) -> str:
        """Get full text content of a node."""
        content = self.kv_store.get(node_id)
        if content:
            return content
        
        # Fallback to skeleton summary
        node = self.skeleton.get(node_id)
        return node.summary if node else ""
    
    def _get_node_path(self, node_id: str) -> list[str]:
        """Get path from root to node."""
        path = []
        current = node_id
        while current:
            node = self.skeleton.get(current)
            if node:
                path.insert(0, node.header)
                current = node.parent_id
            else:
                break
        return path
    
    # =========================================================================
    # Variable Stitching Functions (Section 2.2)
    # =========================================================================
    
    def _store_variable(self, name: str, content: str) -> str:
        """
        Store content as a named variable.
        
        This implements Variable Stitching to prevent Context Rot.
        Intermediate results are "frozen" into variables.
        """
        if not name.startswith("$"):
            name = "$" + name.upper().replace(" ", "_")
        
        self.variable_store.assign(name, content, source_node_id="repl")
        self._namespace["VARIABLES"][name] = content
        
        logger.info("variable_stored", name=name, length=len(content))
        return name
    
    def _get_variable(self, name: str) -> str | None:
        """Retrieve a stored variable."""
        if not name.startswith("$"):
            name = "$" + name.upper()
        
        return self.variable_store.resolve(name)
    
    def _list_variables(self) -> list[str]:
        """List all stored variables."""
        return [v.pointer for v in self.variable_store.list_variables()]
    
    # =========================================================================
    # Recursive LLM Functions (Section 2.2)
    # =========================================================================
    
    def _sub_llm(self, prompt: str, context: str = "") -> str:
        """
        Invoke a sub-LLM call for a specific sub-task.
        
        This is the recursive mechanism: the model can call itself
        with a specific sub-prompt to solve sub-problems.
        
        Example:
            liability_1 = sub_llm("Extract the liability clause", contract_1_text)
        """
        if self._llm_fn is None:
            return "[ERROR: LLM function not configured]"
        
        full_prompt = f"{prompt}\n\nContext:\n{context}" if context else prompt
        
        try:
            result = self._llm_fn(full_prompt)
            logger.debug("sub_llm_called", prompt_length=len(prompt), result_length=len(result))
            return result
        except Exception as e:
            logger.error("sub_llm_error", error=str(e))
            return f"[ERROR: {str(e)}]"
    
    def _batch_sub_llm(
        self,
        prompts: list[str],
        contexts: list[str],
    ) -> list[str]:
        """
        Batch process multiple sub-tasks in parallel.
        
        Implements Section 2.3 Optimization via Batching:
        "Instead of making 1,000 individual calls to summarize 1,000 paragraphs,
        the RLM writes code to group paragraphs into chunks of 5 and processes 
        them in parallel threads (or batched API calls)."
        
        Args:
            prompts: List of prompts for each sub-task.
            contexts: List of context strings for each sub-task.
            
        Returns:
            List of results from sub-LLM calls.
        """
        if len(prompts) != len(contexts):
            raise ValueError("prompts and contexts must have same length")
        
        if not prompts:
            return []
        
        logger.info(
            "batch_sub_llm_start",
            num_tasks=len(prompts),
            batch_size=self.batch_size,
        )
        
        # Group into batches for optimal token density
        batches = self._create_optimal_batches(prompts, contexts)
        
        # Process batches
        results = []
        for batch in batches:
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        
        logger.info("batch_sub_llm_complete", num_results=len(results))
        return results
    
    def _create_optimal_batches(
        self,
        prompts: list[str],
        contexts: list[str],
    ) -> list[list[tuple[str, str]]]:
        """
        Create batches targeting optimal token density (~200k chars).
        """
        batches = []
        current_batch = []
        current_chars = 0
        
        for prompt, context in zip(prompts, contexts):
            item_chars = len(prompt) + len(context)
            
            if current_chars + item_chars > self.optimal_chars_per_call and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_chars = 0
            
            current_batch.append((prompt, context))
            current_chars += item_chars
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _process_batch(self, batch: list[tuple[str, str]]) -> list[str]:
        """Process a batch of (prompt, context) pairs."""
        if self._llm_fn is None:
            return ["[ERROR: LLM not configured]"] * len(batch)
        
        results = []
        
        # Try parallel execution with ThreadPoolExecutor
        try:
            with ThreadPoolExecutor(max_workers=self.max_parallel_batches) as executor:
                futures = []
                for prompt, context in batch:
                    full_prompt = f"{prompt}\n\nContext:\n{context}" if context else prompt
                    futures.append(executor.submit(self._llm_fn, full_prompt))
                
                for future in futures:
                    try:
                        results.append(future.result(timeout=60))
                    except Exception as e:
                        results.append(f"[ERROR: {str(e)}]")
        except Exception as e:
            logger.error("batch_processing_error", error=str(e))
            # Fallback to sequential
            for prompt, context in batch:
                results.append(self._sub_llm(prompt, context))
        
        return results
    
    # =========================================================================
    # Synthesis
    # =========================================================================
    
    def _synthesize_answer(
        self,
        query: str,
        variable_names: list[str] | None = None,
    ) -> str:
        """
        Synthesize final answer from stored variables.
        
        This is the final step where all accumulated variables
        are resolved and compared to generate the answer.
        """
        if variable_names is None:
            variable_names = self._list_variables()
        
        # Resolve all variables
        context_parts = []
        for name in variable_names:
            content = self._get_variable(name)
            if content:
                context_parts.append(f"=== {name} ===\n{content}")
        
        if not context_parts:
            return "No relevant information found."
        
        context = "\n\n".join(context_parts)
        
        # Use LLM to synthesize
        if self._llm_fn:
            prompt = f"""Based on the following collected information, answer the query.

Query: {query}

Collected Information:
{context}

Give a DIRECT, CONCISE answer. Start with the answer itself — no preamble, analysis sections, or markdown headers. Keep it under 3 sentences unless more detail is explicitly needed."""
            
            return self._llm_fn(prompt)
        
        return f"Found {len(variable_names)} relevant sections. Variables: {variable_names}"
    
    # =========================================================================
    # Configuration
    # =========================================================================
    
    def set_llm_function(self, llm_fn: Callable[[str], str]) -> None:
        """Set the LLM function for sub-calls."""
        self._llm_fn = llm_fn
        logger.info("llm_function_configured")
    
    def get_system_prompt(self) -> str:
        """Get the RLM system prompt for the LLM."""
        return RLM_SYSTEM_PROMPT
    
    def get_state_summary(self) -> dict[str, Any]:
        """Get current REPL state summary."""
        return {
            "doc_length": len(self.document_text),
            "num_nodes": len(self.skeleton),
            "variables": self._list_variables(),
            "execution_count": len(self.execution_history),
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_repl_environment(
    document_text: str | None = None,
    skeleton: dict[str, SkeletonNode] | None = None,
    kv_store: KVStore | None = None,
    llm_provider: str | None = None,
) -> REPLEnvironment:
    """
    Create a configured REPL environment.
    
    Args:
        document_text: Full document text (DOC_VAR).
        skeleton: Skeleton index for tree navigation.
        kv_store: KV store for full node content.
        llm_provider: LLM provider for sub-calls ('openai', 'anthropic', 'gemini').
        
    Returns:
        Configured REPLEnvironment.
        
    Example:
        from rnsr import ingest_document, build_skeleton_index
        from rnsr.agent.repl_env import create_repl_environment
        
        result = ingest_document("contract.pdf")
        skeleton, kv_store = build_skeleton_index(result.tree)
        
        # Get full text
        full_text = kv_store.get("root") or ""
        
        env = create_repl_environment(
            document_text=full_text,
            skeleton=skeleton,
            kv_store=kv_store,
            llm_provider="gemini",
        )
        
        # Execute code
        env.execute("len(DOC_VAR)")  # -> 152347
        env.execute("list_children('root')")  # -> [...]
    """
    from rnsr.indexing.kv_store import InMemoryKVStore
    
    env = REPLEnvironment(
        document_text=document_text or "",
        skeleton=skeleton or {},
        kv_store=kv_store or InMemoryKVStore(),
    )
    
    # Configure LLM if provider specified
    if llm_provider:
        try:
            from rnsr.llm import get_llm, LLMProvider
            
            # Convert string to LLMProvider enum if needed
            provider_enum = LLMProvider(llm_provider) if isinstance(llm_provider, str) else llm_provider
            llm = get_llm(provider=provider_enum)
            
            def llm_fn(prompt: str) -> str:
                response = llm.complete(prompt)
                return str(response)
            
            env.set_llm_function(llm_fn)
        except Exception as e:
            logger.warning("llm_config_failed", error=str(e))
    
    return env


# =============================================================================
# Async Batch Processing
# =============================================================================


async def batch_process_async(
    env: REPLEnvironment,
    prompts: list[str],
    contexts: list[str],
) -> list[str]:
    """
    Async batch processing for maximum throughput.
    
    Uses asyncio for concurrent LLM calls when available.
    """
    if env._async_llm_fn is None:
        # Fallback to sync batch
        return env._batch_sub_llm(prompts, contexts)
    
    async_fn = env._async_llm_fn  # Capture for type narrowing
    
    async def process_one(prompt: str, context: str) -> str:
        full_prompt = f"{prompt}\n\nContext:\n{context}" if context else prompt
        try:
            return await async_fn(full_prompt)
        except Exception as e:
            return f"[ERROR: {str(e)}]"
    
    # Process all concurrently
    tasks = [
        process_one(p, c) for p, c in zip(prompts, contexts)
    ]
    
    return await asyncio.gather(*tasks)
