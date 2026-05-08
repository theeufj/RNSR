"""
RLM Navigator - Recursive Language Model Navigator with Full REPL Integration

This module implements the full RLM (Recursive Language Model) pattern from the
arxiv paper "Recursive Language Models" combined with RNSR's tree-based retrieval.

Key Features:
1. Full REPL environment with code execution for document filtering
2. Pre-LLM filtering using regex/keyword search before ToT evaluation
3. Deep recursive sub-LLM calls (configurable depth)
4. Answer verification loops
5. Async parallel sub-LLM processing
6. Adaptive learning for stop words and query patterns

This is the state-of-the-art combination of:
- PageIndex: Vectorless, reasoning-based tree search
- RLMs: REPL environment with recursive sub-LLM calls
- RNSR: Latent hierarchy reconstruction + variable stitching
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import operator as _operator
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal

import structlog

from rnsr.agent.variable_store import VariableStore, generate_pointer_name
from rnsr.agent.nav_repl import NavigationREPL, create_navigation_repl
from rnsr.agent.self_reflection import strict_verify_answer, VerificationResult
from rnsr.indexing.kv_store import KVStore
from rnsr.models import SkeletonNode, TraceEntry

logger = structlog.get_logger(__name__)


# =============================================================================
# Arithmetic Synthesis Utilities (ported from graph.py)
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
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return _SAFE_OPERATORS[op_type](_safe_eval_node(node.operand))
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        return _SAFE_OPERATORS[op_type](_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed (no methods)")
        if node.func.id not in _SAFE_FUNCTIONS:
            raise ValueError(f"Function not allowed: {node.func.id}")
        return _SAFE_FUNCTIONS[node.func.id](*[_safe_eval_node(a) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval_node(elt) for elt in node.elts]  # type: ignore[return-value]
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        if node.id == "e":
            return math.e
        raise ValueError(f"Variable reference not allowed: {node.id}")
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def safe_math_eval(expr: str) -> float | int | None:
    """Safely evaluate a Python math expression using AST parsing."""
    if not expr or not expr.strip():
        return None
    expr = expr.strip()
    if any(kw in expr for kw in ("import ", "__", "exec", "eval", "open", "compile")):
        logger.warning("safe_math_eval_rejected", expr=expr[:100])
        return None
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval_node(tree.body)
        if isinstance(result, (int, float)):
            return result
        return None
    except Exception:
        return None


_NO_COMPUTE_MARKER = "__NO_COMPUTE__"


def _extract_code_block(response_text: str) -> str | None:
    """Extract a Python code block from the LLM response."""
    match = re.search(r"CODE:\s*```(?:python)?\s*\n(.*?)```", response_text, re.S)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response_text, re.S)
    if match:
        return match.group(1).strip()
    return None


def _try_compute_from_response(response_text: str, context_text: str = "") -> str | None:
    """Parse CODE/COMPUTE/NO_COMPUTE/ANSWER from LLM response and execute."""
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

    if no_compute_answer is not None:
        return no_compute_answer if no_compute_answer else _NO_COMPUTE_MARKER

    code_block = _extract_code_block(response_text)
    if code_block:
        try:
            from rnsr.agent.repl_env import REPLEnvironment
            from rnsr.indexing.kv_store import InMemoryKVStore

            env = REPLEnvironment(document_text=context_text, skeleton={}, kv_store=InMemoryKVStore())
            result = env.execute(code_block)
            if result["success"] and result["output"]:
                nums = re.findall(r'[-]?[\d,]+\.?\d*', str(result["output"]).strip())
                if nums:
                    num_str = nums[-1].replace(",", "")
                    try:
                        val = float(num_str)
                        if val == int(val) and abs(val) < 1e15:
                            return str(int(val))
                        return str(val)
                    except ValueError:
                        pass
        except Exception as e:
            logger.warning("code_execution_error", error=str(e))

    if compute_expr:
        result = safe_math_eval(compute_expr)
        if result is not None:
            if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
                return str(int(result))
            return str(result)

    if answer_line:
        cleaned = answer_line.replace(",", "").strip().strip("$").strip("%")
        try:
            val = float(cleaned)
            if val == int(val) and abs(val) < 1e15:
                return str(int(val))
            return str(val)
        except ValueError:
            return answer_line

    return None


def _try_repl_arithmetic(question: str, context_text: str, llm_fn: Callable[[str], str]) -> str | None:
    """Fallback: generate Python code from scratch and execute for arithmetic answers."""
    try:
        from rnsr.agent.repl_env import REPLEnvironment
        from rnsr.indexing.kv_store import InMemoryKVStore
    except ImportError:
        return None

    env = REPLEnvironment(document_text=context_text, skeleton={}, kv_store=InMemoryKVStore())
    code_prompt = f"""You have access to a Python REPL. The variable DOC_VAR contains the document context.
Write Python code to answer this question. The code must print() the final numeric answer.

Question: {question}

Write ONLY valid Python code. Use print() to output the final answer.
```python
val1 = 14740
val2 = 1910
result = (val1 + val2) / 2
print(result)
```

Your Python code:"""

    try:
        code_response = llm_fn(code_prompt).strip()
        code_response = re.sub(r"```python\s*", "", code_response)
        code_response = re.sub(r"```\s*", "", code_response).strip()
        if not code_response:
            return None
        result = env.execute(code_response)
        if result["success"] and result["output"]:
            match = re.search(r'[-]?[\d,]+\.?\d*', str(result["output"]).strip())
            if match:
                num_str = match.group(0).replace(",", "")
                try:
                    val = float(num_str)
                    if val == int(val) and abs(val) < 1e15:
                        return str(int(val))
                    return str(val)
                except ValueError:
                    pass
    except Exception as e:
        logger.debug("repl_arithmetic_error", error=str(e))
    return None


_GROWTH_KEYWORDS = re.compile(
    r"growth\s*rate|current\s*rate|continues?\s*to\s*grow|projection|"
    r"will\s*(?:\w+\s+)?reach|forecast|compound|cagr",
    re.IGNORECASE,
)


def _verify_and_rerun_formula(
    question: str, original_code: str, original_answer: str,
    context_text: str, llm_fn: Callable[[str], str],
) -> str | None:
    """Verify growth/projection formulas and re-execute if incorrect."""
    if not _GROWTH_KEYWORDS.search(question):
        return None
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
        response = llm_fn(verify_prompt).strip()
        if "INCORRECT" in response.upper():
            code_match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.S)
            if code_match:
                try:
                    from rnsr.agent.repl_env import REPLEnvironment
                    from rnsr.indexing.kv_store import InMemoryKVStore
                    env = REPLEnvironment(document_text=context_text, skeleton={}, kv_store=InMemoryKVStore())
                    result = env.execute(code_match.group(1).strip())
                    if result["success"] and result["output"]:
                        nums = re.findall(r'[-]?[\d,]+\.?\d*', str(result["output"]).strip())
                        if nums:
                            num_str = nums[-1].replace(",", "")
                            val = float(num_str)
                            return str(int(val)) if val == int(val) and abs(val) < 1e15 else str(val)
                except Exception as e:
                    logger.warning("formula_verification_rerun_failed", error=str(e))
    except Exception as e:
        logger.debug("formula_verification_error", error=str(e))
    return None


# =============================================================================
# Short-Answer Extraction Utilities (ported from graph.py)
# =============================================================================

def _strip_citations(text: str) -> str:
    """Remove parenthetical citations from text."""
    text = re.sub(r'\s*\(Source:?\s*[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\(Source\b[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'===\s*\$\w+\s*===', '', text)
    text = re.sub(r'\$CONTEXT_\d+\$?', '', text)
    text = re.sub(r'  +', ' ', text).strip()
    return text


def _extract_numeric_answer(text: str) -> str | None:
    """Try to extract a bare numeric answer from text."""
    if not text:
        return None
    first_line = _strip_citations(text.split("\n")[0].strip())
    match = re.search(r'[-]?\$?[\d,]+\.?\d*%?', first_line)
    if match:
        num_str = match.group(0).replace(',', '').strip('$').strip('%')
        try:
            float(num_str)
            return num_str
        except ValueError:
            pass
    return None


_UNANSWERABLE_PATTERNS = [
    "cannot answer", "cannot determine", "not possible to determine",
    "no relevant content", "unable to answer", "not enough information",
    "cannot be determined", "context is empty", "information is not available",
    "not explicitly stated", "does not contain", "not mentioned",
    "no information", "cannot find", "i cannot answer", "unanswerable",
    "not answerable", "insufficient information", "no answer",
    "not provided in", "not found in",
]


def _extract_first_answer_phrase(text: str, max_chars: int = 300, is_arithmetic: bool = False) -> str:
    """Extract a short answer phrase for F1-friendly evaluation."""
    if not text or not text.strip():
        return text
    text = text.strip()
    if is_arithmetic:
        num = _extract_numeric_answer(text)
        if num is not None:
            return num
    text = _strip_citations(text)
    first_line = text.split("\n")[0].strip()
    if not first_line:
        first_line = text[:max_chars].strip()
    for pattern in _UNANSWERABLE_PATTERNS:
        if pattern in first_line.lower():
            return "Unanswerable"
    if len(first_line) <= max_chars:
        return first_line
    chunk = first_line[:max_chars]
    last_space = chunk.rfind(" ")
    return (chunk[:last_space + 1] if last_space >= 0 else chunk).strip()


# =============================================================================
# Header-Match Fallback (ported from graph.py)
# =============================================================================

def _header_match_fallback(
    question: str,
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    variable_store: VariableStore,
    min_selections: int = 2,
    max_selections: int = 5,
    llm_fn: Callable[[str], str] | None = None,
) -> list[str]:
    """Present ALL headers to the LLM and ask it to pick the most relevant sections.

    Last-resort fallback when tree navigation and search refinement both fail.
    Returns list of pointer names stored.
    """
    entries: list[tuple[str, str, str]] = []
    for nid, node in skeleton.items():
        if node.parent_id is None:
            continue
        preview = node.summary[:120] if node.summary else ""
        entries.append((nid, node.header, preview))

    if not entries or not llm_fn:
        return []

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

    try:
        response = llm_fn(prompt).strip()
    except Exception as e:
        logger.warning("header_match_fallback_llm_failed", error=str(e))
        return []

    selected_indices: list[int] = []
    for token in re.split(r"[,\s]+", response):
        token = token.strip().rstrip(".")
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(entries):
                selected_indices.append(idx)

    if not selected_indices:
        return []

    seen: set[int] = set()
    unique_indices: list[int] = []
    for idx in selected_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
    unique_indices = unique_indices[:max_selections]

    pointers_stored: list[str] = []
    for _rank, idx in enumerate(unique_indices):
        node_id, header, _preview = entries[idx - 1]
        content = kv_store.get(node_id)
        if not content or len(content.strip()) < 20:
            continue
        pointer = generate_pointer_name(header)
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
    )
    return pointers_stored


# Keywords that aren't useful for content matching (mostly query stop
# words and the literal verbs that appear in nearly every question).
_CONTENT_SEARCH_STOP = frozenset(
    {
        "the", "and", "for", "from", "with", "what", "was", "were", "did",
        "does", "any", "are", "this", "that", "have", "has", "had", "been",
        "much", "many", "into", "over", "than", "who", "how", "why", "when",
        "which", "their", "there", "between", "among", "across", "during",
        "respect", "based", "explain", "describe", "give", "list", "tell",
        "provide", "company", "companies",
    }
)


# Common financial / accounting acronyms in 10-K, 10-Q, and earnings
# release questions. Each query token (case-insensitive, lowercased) maps
# to additional surface forms to also search for. This was added after
# the Pfizer 10-K "PPNE" question failed because the document writes
# "PP&E" / "Property, Plant and Equipment" and never uses the literal
# string "PPNE" the question used.
_FINANCIAL_ACRONYM_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "ppne": ("pp&e", "property, plant and equipment", "property, plant", "property and equipment", "fixed assets"),
    "ppe": ("pp&e", "property, plant and equipment", "property, plant", "property and equipment"),
    "pp&e": ("ppne", "property, plant and equipment", "property, plant"),
    "capex": ("capital expenditure", "capital expenditures"),
    "opex": ("operating expense", "operating expenses"),
    "cogs": ("cost of goods sold", "cost of sales", "cost of revenue"),
    "sga": ("selling, general and administrative", "selling general and administrative"),
    "sg&a": ("selling, general and administrative", "selling general and administrative"),
    "ebitda": ("earnings before interest", "ebit", "operating income"),
    "ebit": ("operating income", "earnings before interest"),
    "eps": ("earnings per share", "diluted eps", "basic eps"),
    "fcf": ("free cash flow", "free cash"),
    "ar": ("accounts receivable", "trade receivables"),
    "ap": ("accounts payable", "trade payables"),
    "roa": ("return on assets",),
    "roe": ("return on equity",),
    "roic": ("return on invested capital",),
    "wc": ("working capital",),
    "dso": ("days sales outstanding", "days outstanding"),
    "dpo": ("days payable outstanding",),
    "agm": ("annual general meeting", "annual meeting of shareholders"),
    "10-k": ("10-k", "annual report", "form 10-k"),
    "10-q": ("10-q", "quarterly report", "form 10-q"),
    "8-k": ("8-k", "current report", "form 8-k"),
    "8k": ("8-k", "current report", "form 8-k"),
    "fy": ("fiscal year", "year ended"),
}


def _expand_financial_acronyms(keywords: list[str]) -> list[str]:
    """Expand common financial acronyms into the surface forms used in filings.

    Returns a new list with the originals followed by their expansions,
    de-duplicated (case-insensitive). Order is preserved: each original
    keyword is followed by its expansions before the next original.
    """
    expanded: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        kwl = kw.lower()
        if kwl not in seen:
            seen.add(kwl)
            expanded.append(kw)
        for extra in _FINANCIAL_ACRONYM_EXPANSIONS.get(kwl, ()):
            extra_l = extra.lower()
            if extra_l not in seen:
                seen.add(extra_l)
                expanded.append(extra)
    return expanded


def _extract_content_keywords(question: str) -> list[str]:
    """Tokenize the question into content-search keywords.

    Includes alphabetic words >= 3 chars (minus stop words), quoted
    phrases, capitalized proper nouns, currency/percent fragments, and
    numeric tokens (e.g. "2023", "10-Q") so we can match table cells
    that the navigator overlooked. Common financial acronyms (PPNE,
    EBITDA, CapEx, etc.) are expanded into the surface forms commonly
    used in filings.
    """
    keywords: list[str] = []

    for word in re.findall(r"\b[a-zA-Z]{3,}\b", question.lower()):
        if word not in _CONTENT_SEARCH_STOP:
            keywords.append(word)

    # Acronyms that include & or - are missed by the alphabetic regex
    # above (e.g. "PP&E", "10-Q", "SG&A"). Capture them explicitly so
    # both they and their expansions land in the keyword list.
    for symbolic in re.findall(r"\b[A-Za-z]+[&\-][A-Za-z0-9]+\b", question):
        keywords.append(symbolic)

    keywords.extend(re.findall(r'"([^"]+)"', question))

    for proper in re.findall(r"\b[A-Z][a-zA-Z]+\b", question):
        keywords.append(proper.lower())

    for num in re.findall(r"\b\d{2,4}(?:Q[1-4])?\b", question):
        keywords.append(num.lower())

    seen: set[str] = set()
    deduped: list[str] = []
    for kw in keywords:
        kwl = kw.lower()
        if kwl not in seen:
            seen.add(kwl)
            deduped.append(kw)

    return _expand_financial_acronyms(deduped)


# Question patterns that ask the reader to enumerate items across a
# domain and pick a superlative ("which of X had the most/lowest/etc.").
# When the navigator answers one of these, we want to ensure the cited
# evidence covers the *full* domain — not just one branch of a
# hierarchy. The JPM 10-Q "lowest segment" miss happened because the
# navigator landed on a CCB sub-segment table and never visited the
# top-level "BUSINESS SEGMENT RESULTS" parent that includes Corporate.
_ENUMERATION_SUPERLATIVES = (
    "lowest", "highest", "most", "least", "largest", "smallest",
    "biggest", "greatest", "top", "bottom", "max", "maximum",
    "min", "minimum", "best", "worst", "first", "last",
    "rank", "ranking",
)
_ENUMERATION_PROMPT_RE = re.compile(
    r"^\s*(?:which|what)\b", re.IGNORECASE
)


def _is_enumeration_question(question: str) -> bool:
    """Return True if the question asks for a superlative across a domain.

    Examples that match: "Which of JPM's business segments had the lowest
    net revenue?", "What region posted the highest growth?". These need
    retrieval to cover the *whole* domain so the comparison is sound.
    """
    if not _ENUMERATION_PROMPT_RE.match(question):
        return False
    q_lower = question.lower()
    return any(re.search(rf"\b{re.escape(t)}\b", q_lower) for t in _ENUMERATION_SUPERLATIVES)


# Words that frequently introduce a hierarchical "scope" in financial
# filings. When a question mentions any of these, retrieval should
# prefer parent-level sections whose headers match.
_ENUMERATION_DOMAIN_HINTS = (
    "segment", "subsidiary", "subsidiaries", "region", "geography",
    "geographies", "country", "countries", "product", "products",
    "service", "services", "division", "divisions", "business unit",
    "business units", "line of business", "lines of business",
    "operating segment", "reportable segment", "category", "categories",
    "brand", "brands", "channel", "channels", "market", "markets",
)


# Tokens that show up in enumeration questions but aren't useful for
# matching against section headers. They'd otherwise inflate scores on
# unrelated nodes (e.g. "2021" matching every page that says "March 31,
# 2021"; "net" matching every "net" line item).
_ENUMERATION_GENERIC_NOISE = frozenset(
    {
        # Superlatives — these are the *trigger*, not the *domain*
        "lowest", "highest", "most", "least", "largest", "smallest",
        "biggest", "greatest", "top", "bottom", "rank", "ranking",
        # Question words and verbs
        "which", "what", "had", "has", "have", "was", "were", "is", "are",
        # Generic measure tokens that match too broadly
        "net", "total", "amount", "value", "size",
    }
)


def _enumeration_domain_terms(question: str) -> tuple[list[str], list[str]]:
    """Split an enumeration question into (strong scope terms, weak terms).

    Strong terms are explicit domain hints from ``_ENUMERATION_DOMAIN_HINTS``
    (segment, subsidiary, region, etc.) — header hits on these are the
    most reliable signal that a node is a parent-level overview table.

    Weak terms are the rest of the question's content keywords minus
    superlatives, question words, and generic measure tokens. Header
    hits on weak terms are useful but get a much smaller score weight.

    Returns ``(strong_terms, weak_terms)``.
    """
    q_lower = question.lower()
    strong: list[str] = []
    for hint in _ENUMERATION_DOMAIN_HINTS:
        if hint in q_lower:
            strong.append(hint)

    weak: list[str] = []
    for kw in _extract_content_keywords(question):
        kwl = kw.lower()
        if kwl in _ENUMERATION_GENERIC_NOISE:
            continue
        if len(kwl) < 3:
            continue
        # Skip plain year tokens (4 digits, no Q-suffix) — they match
        # every page footer in a 10-K/Q.
        if re.fullmatch(r"\d{4}", kwl):
            continue
        if kwl in strong:
            continue
        weak.append(kwl)

    seen_s: set[str] = set()
    strong_dedup = [t for t in strong if not (t in seen_s or seen_s.add(t))]
    seen_w: set[str] = set()
    weak_dedup = [t for t in weak if not (t in seen_w or seen_w.add(t))]
    return strong_dedup, weak_dedup


# Patterns used to score data-density of a section's content. Sections
# containing many of these (currency amounts, dollar figures, numbers
# with thousands separators, parenthesised negatives) are likely to be
# the actual data tables an enumeration question needs, vs methodology
# or descriptive prose.
_DATA_DENSITY_RE = re.compile(
    r"\$\s*\d|\d{1,3}(?:,\d{3})+|\(\d{1,3}(?:,\d{3})*\)|\d+\.\d+\s*(?:%|million|billion|thousand)",
    re.IGNORECASE,
)


def _enumeration_scope_fallback(
    question: str,
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    variable_store: VariableStore,
    visited_node_ids: set[str] | None = None,
    max_selections: int = 6,
) -> list[str]:
    """Pull parent-level sections that look like comprehensive overview tables.

    For enumeration questions ("which of X had the most Y"), the right
    answer often lives in a top-level summary section ("BUSINESS SEGMENT
    RESULTS", "Lines of Business Information", "Segment Results –
    managed basis"), not in the deep sub-segment leaves the ToT
    navigator tends to drill into.

    Score components per candidate node:
      * ``strong_header_hits * 4.0`` — explicit domain hint (segment,
        subsidiary, region…) in the header. Strongest signal.
      * ``weak_header_hits * 0.6`` — other question keywords in the
        header. Useful but noisy on their own.
      * ``strong_summary_hits * 0.5`` — domain hint in summary.
      * Level bonus ``max(0, 4 - level)`` — prefer shallower parents.
      * Data-density bonus (capped at +3) — sections with lots of currency
        / numeric patterns are likely real tables, not methodology prose.
        This is what stops us from picking "Description of business
        segment reporting methodology" over the actual results table.
      * ``-1.0`` if the content is < 300 chars (header-only nodes).
      * ``-1.5`` if the navigator already visited this node (so the
        fallback adds *new* context, not a repeat).
    """
    strong_terms, weak_terms = _enumeration_domain_terms(question)
    if not strong_terms and not weak_terms:
        return []

    visited = visited_node_ids or set()
    candidates: list[tuple[float, str, SkeletonNode]] = []

    for node_id, node in skeleton.items():
        if node.parent_id is None:
            continue
        header_lower = (node.header or "").lower()
        summary_lower = (node.summary or "").lower()
        if not header_lower:
            continue

        strong_header_hits = sum(1 for term in strong_terms if term in header_lower)
        weak_header_hits = sum(1 for term in weak_terms if term in header_lower)
        strong_summary_hits = sum(1 for term in strong_terms if term in summary_lower)
        if strong_header_hits == 0 and weak_header_hits < 2 and strong_summary_hits < 2:
            continue

        try:
            level = int(node.level)
        except (TypeError, ValueError):
            level = 3
        level_bonus = max(0, 4 - level)
        visited_penalty = 1.5 if node_id in visited else 0.0

        content = kv_store.get(node_id) or ""
        size_penalty = 1.0 if len(content) < 300 else 0.0
        data_matches = len(_DATA_DENSITY_RE.findall(content))
        data_bonus = min(3.0, data_matches / 5.0)

        score = (
            strong_header_hits * 4.0
            + weak_header_hits * 0.6
            + strong_summary_hits * 0.5
            + level_bonus
            + data_bonus
            - visited_penalty
            - size_penalty
        )
        if score <= 0:
            continue
        candidates.append((score, node_id, node))

    if not candidates:
        return []

    candidates.sort(key=lambda c: c[0], reverse=True)

    pointers_stored: list[str] = []
    for score, node_id, node in candidates[:max_selections]:
        content = kv_store.get(node_id)
        if not content or len(content.strip()) < 20:
            continue
        pointer = generate_pointer_name(node.header)
        base_pointer = pointer
        counter = 2
        skip_assign = False
        while variable_store.exists(pointer):
            existing_meta = variable_store._metadata.get(pointer)
            if existing_meta and existing_meta.source_node_id == node_id:
                # Same source already stored — surface the existing pointer
                # so synthesis sees this section, but skip the redundant
                # assign() call.
                skip_assign = True
                break
            pointer = f"{base_pointer}_{counter}"
            counter += 1
        if not skip_assign:
            variable_store.assign(pointer, content, source_node_id=node_id)
        pointers_stored.append(pointer)
        logger.info(
            "enumeration_scope_match",
            node_id=node_id,
            header=(node.header or "")[:60],
            level=node.level,
            score=round(score, 2),
            content_chars=len(content),
            reused=skip_assign,
        )

    return pointers_stored


def _content_search_fallback(
    question: str,
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    variable_store: VariableStore,
    visited_node_ids: set[str] | None = None,
    max_selections: int = 4,
    min_distinct_matches: int = 2,
) -> list[str]:
    """Last-resort fallback that ranks raw KV-store content by keyword density.

    The header-matching fallback fails when ingestion produced poor
    headers (e.g. table cells like "2,018" or column labels like
    "currency ∆%"). Here we ignore headers entirely and score every
    leaf node by how many distinct query keywords appear in its KV
    content, breaking ties by total occurrences. Nodes already visited
    by the navigator are deprioritised so we add genuinely new context.
    """
    keywords = _extract_content_keywords(question)
    if not keywords:
        return []

    visited = visited_node_ids or set()
    candidates: list[tuple[int, int, bool, str]] = []

    for node_id, node in skeleton.items():
        if node.parent_id is None:
            continue
        content = kv_store.get(node_id)
        if not content or len(content.strip()) < 20:
            continue
        lower = content.lower()
        distinct = sum(1 for kw in keywords if kw.lower() in lower)
        if distinct < min_distinct_matches:
            continue
        total = sum(lower.count(kw.lower()) for kw in keywords)
        candidates.append((distinct, total, node_id in visited, node_id))

    if not candidates:
        return []

    candidates.sort(key=lambda c: (c[0], c[1], not c[2]), reverse=True)

    pointers_stored: list[str] = []
    for distinct, total, was_visited, node_id in candidates[:max_selections]:
        node = skeleton.get(node_id)
        if not node:
            continue
        content = kv_store.get(node_id)
        if not content:
            continue
        pointer = generate_pointer_name(node.header)
        base_pointer = pointer
        counter = 2
        skip_assign = False
        while variable_store.exists(pointer):
            existing_meta = variable_store._metadata.get(pointer)
            if existing_meta and existing_meta.source_node_id == node_id:
                # Same source already stored — surface the existing pointer
                # so synthesis sees this section, but skip the redundant
                # assign() call.
                skip_assign = True
                break
            pointer = f"{base_pointer}_{counter}"
            counter += 1
        if not skip_assign:
            variable_store.assign(pointer, content, source_node_id=node_id)
        pointers_stored.append(pointer)
        logger.info(
            "content_search_fallback_match",
            node_id=node_id,
            header=node.header[:60],
            distinct_keywords=distinct,
            total_occurrences=total,
            was_visited=was_visited,
            reused=skip_assign,
        )

    return pointers_stored


# =============================================================================
# Learned Stop Words Registry
# =============================================================================

DEFAULT_STOP_WORDS_PATH = Path.home() / ".rnsr" / "learned_stop_words.json"


class LearnedStopWords:
    """
    Registry for learning domain-specific stop words.
    
    Learns:
    - Words that are generic in your domain (should be filtered)
    - Words that seem generic but are important in your domain (should be kept)
    
    Examples:
    - Legal: "hereby", "whereas" are filler (add to stop)
    - Legal: "party" is important (remove from stop)
    """
    
    # Base stop words (always included unless explicitly removed)
    BASE_STOP_WORDS = {
        "what", "is", "the", "a", "an", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "dare", "ought", "used", "to", "of", "in", "for", "on", "with", "at",
        "by", "from", "about", "into", "through", "during", "before", "after",
        "above", "below", "between", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just", "and",
        "but", "if", "or", "because", "as", "until", "while", "this", "that",
        "these", "those", "find", "show", "list", "describe", "explain", "tell",
    }
    
    def __init__(
        self,
        storage_path: Path | str | None = None,
        auto_save: bool = True,
    ):
        """
        Initialize the learned stop words registry.
        
        Args:
            storage_path: Path to JSON file for persistence.
            auto_save: Whether to save after changes.
        """
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_STOP_WORDS_PATH
        self.auto_save = auto_save
        
        self._lock = Lock()
        self._added_stop_words: dict[str, dict[str, Any]] = {}  # Domain-specific additions
        self._removed_stop_words: dict[str, dict[str, Any]] = {}  # Words to keep despite being in base
        self._dirty = False
        
        self._load()
    
    def _load(self) -> None:
        """Load learned stop words from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            self._added_stop_words = data.get("added", {})
            self._removed_stop_words = data.get("removed", {})
            
            logger.info(
                "learned_stop_words_loaded",
                added=len(self._added_stop_words),
                removed=len(self._removed_stop_words),
            )
            
        except Exception as e:
            logger.warning("failed_to_load_stop_words", error=str(e))
    
    def _save(self) -> None:
        """Save to storage."""
        if not self._dirty:
            return
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "added": self._added_stop_words,
                "removed": self._removed_stop_words,
            }
            
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            
            self._dirty = False
            
        except Exception as e:
            logger.warning("failed_to_save_stop_words", error=str(e))
    
    def add_stop_word(
        self,
        word: str,
        domain: str = "general",
        reason: str = "",
    ) -> None:
        """
        Add a word to the stop word list.
        
        Args:
            word: Word to add.
            domain: Domain category.
            reason: Why this should be a stop word.
        """
        word = word.lower().strip()
        
        if not word or word in self.BASE_STOP_WORDS:
            return
        
        with self._lock:
            now = datetime.utcnow().isoformat()
            
            if word not in self._added_stop_words:
                self._added_stop_words[word] = {
                    "count": 0,
                    "domain": domain,
                    "reason": reason,
                    "first_seen": now,
                    "last_seen": now,
                }
                logger.info("stop_word_added", word=word)
            
            self._added_stop_words[word]["count"] += 1
            self._added_stop_words[word]["last_seen"] = now
            
            self._dirty = True
            
            if self.auto_save:
                self._save()
    
    def remove_stop_word(
        self,
        word: str,
        domain: str = "general",
        reason: str = "",
    ) -> None:
        """
        Mark a base stop word as important (should not be filtered).
        
        Args:
            word: Word to keep.
            domain: Domain where this is important.
            reason: Why this should be kept.
        """
        word = word.lower().strip()
        
        if not word or word not in self.BASE_STOP_WORDS:
            return
        
        with self._lock:
            now = datetime.utcnow().isoformat()
            
            if word not in self._removed_stop_words:
                self._removed_stop_words[word] = {
                    "count": 0,
                    "domain": domain,
                    "reason": reason,
                    "first_seen": now,
                    "last_seen": now,
                }
                logger.info("stop_word_marked_important", word=word)
            
            self._removed_stop_words[word]["count"] += 1
            self._removed_stop_words[word]["last_seen"] = now
            
            self._dirty = True
            
            if self.auto_save:
                self._save()
    
    def get_stop_words(self, min_count: int = 1) -> set[str]:
        """
        Get the effective stop word set.
        
        Returns:
            Set of words to filter (base + added - removed).
        """
        with self._lock:
            # Start with base
            result = set(self.BASE_STOP_WORDS)
            
            # Add learned additions
            for word, data in self._added_stop_words.items():
                if data["count"] >= min_count:
                    result.add(word)
            
            # Remove marked-important words
            for word, data in self._removed_stop_words.items():
                if data["count"] >= min_count:
                    result.discard(word)
            
            return result
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about stop words."""
        return {
            "base_count": len(self.BASE_STOP_WORDS),
            "added_count": len(self._added_stop_words),
            "removed_count": len(self._removed_stop_words),
            "effective_count": len(self.get_stop_words()),
        }


# Global stop words registry
_global_stop_words: LearnedStopWords | None = None


def get_learned_stop_words() -> LearnedStopWords:
    """Get the global learned stop words registry."""
    global _global_stop_words
    
    if _global_stop_words is None:
        custom_path = os.getenv("RNSR_STOP_WORDS_PATH")
        _global_stop_words = LearnedStopWords(
            storage_path=custom_path if custom_path else None
        )
    
    return _global_stop_words


# =============================================================================
# Learned Query Patterns Registry
# =============================================================================

DEFAULT_QUERY_PATTERNS_PATH = Path.home() / ".rnsr" / "learned_query_patterns.json"


class LearnedQueryPatterns:
    """
    Registry for learning successful query patterns.
    
    Tracks:
    - Query patterns that lead to high-confidence answers
    - Patterns that need decomposition vs. direct retrieval
    - Entity-focused vs. section-focused queries
    
    Used to:
    - Inform decomposition strategy
    - Adjust confidence thresholds
    - Route to specialized handlers
    """
    
    def __init__(
        self,
        storage_path: Path | str | None = None,
        auto_save: bool = True,
    ):
        """
        Initialize the query patterns registry.
        
        Args:
            storage_path: Path to JSON file for persistence.
            auto_save: Whether to save after changes.
        """
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_QUERY_PATTERNS_PATH
        self.auto_save = auto_save
        
        self._lock = Lock()
        self._patterns: dict[str, dict[str, Any]] = {}
        self._dirty = False
        
        self._load()
    
    def _load(self) -> None:
        """Load learned patterns from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            self._patterns = data.get("patterns", {})
            
            logger.info(
                "query_patterns_loaded",
                patterns=len(self._patterns),
            )
            
        except Exception as e:
            logger.warning("failed_to_load_query_patterns", error=str(e))
    
    def _save(self) -> None:
        """Save to storage."""
        if not self._dirty:
            return
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "patterns": self._patterns,
            }
            
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            
            self._dirty = False
            
        except Exception as e:
            logger.warning("failed_to_save_query_patterns", error=str(e))
    
    def record_query(
        self,
        query: str,
        pattern_type: str,
        success: bool,
        confidence: float,
        needed_decomposition: bool,
        sub_questions_count: int = 0,
        entities_involved: list[str] | None = None,
    ) -> None:
        """
        Record a query and its outcome.
        
        Args:
            query: The original query.
            pattern_type: Detected pattern type (entity_lookup, comparison, etc.)
            success: Whether the query was answered successfully.
            confidence: Answer confidence score.
            needed_decomposition: Whether decomposition was required.
            sub_questions_count: Number of sub-questions generated.
            entities_involved: Entity types involved in the query.
        """
        pattern_type = pattern_type.lower().strip()
        
        with self._lock:
            now = datetime.utcnow().isoformat()
            
            if pattern_type not in self._patterns:
                self._patterns[pattern_type] = {
                    "total_queries": 0,
                    "successful_queries": 0,
                    "total_confidence": 0.0,
                    "decomposition_count": 0,
                    "total_sub_questions": 0,
                    "entity_types": {},
                    "first_seen": now,
                    "last_seen": now,
                    "example_queries": [],
                }
                logger.info("new_query_pattern_discovered", pattern_type=pattern_type)
            
            pt = self._patterns[pattern_type]
            pt["total_queries"] += 1
            pt["total_confidence"] += confidence
            pt["last_seen"] = now
            
            if success:
                pt["successful_queries"] += 1
            
            if needed_decomposition:
                pt["decomposition_count"] += 1
                pt["total_sub_questions"] += sub_questions_count
            
            if entities_involved:
                for entity_type in entities_involved:
                    pt["entity_types"][entity_type] = pt["entity_types"].get(entity_type, 0) + 1
            
            if len(pt["example_queries"]) < 5:
                pt["example_queries"].append({
                    "query": query[:200],
                    "success": success,
                    "confidence": confidence,
                    "timestamp": now,
                })
            
            self._dirty = True
            
            if self.auto_save:
                self._save()
    
    def detect_pattern_type(self, query: str) -> str:
        """
        Detect the pattern type of a query.
        
        Args:
            query: The query to analyze.
            
        Returns:
            Detected pattern type.
        """
        query_lower = query.lower()
        
        # Pattern detection heuristics
        if any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            return "comparison"
        
        if any(word in query_lower for word in ["list", "all", "every", "enumerate"]):
            return "enumeration"
        
        if any(word in query_lower for word in ["when", "date", "time", "timeline"]):
            return "temporal"
        
        if any(word in query_lower for word in ["who", "person", "name"]):
            return "entity_person"
        
        if any(word in query_lower for word in ["company", "organization", "entity"]):
            return "entity_organization"
        
        if any(word in query_lower for word in ["how much", "amount", "price", "cost", "$"]):
            return "monetary"
        
        if any(word in query_lower for word in ["section", "clause", "paragraph", "article"]):
            return "section_lookup"
        
        if any(word in query_lower for word in ["what is", "define", "explain", "describe"]):
            return "definition"
        
        if any(word in query_lower for word in ["why", "reason", "cause"]):
            return "causal"
        
        return "general"
    
    def get_pattern_stats(self, pattern_type: str) -> dict[str, Any] | None:
        """
        Get statistics for a pattern type.
        
        Args:
            pattern_type: The pattern type to look up.
            
        Returns:
            Pattern statistics or None if not found.
        """
        pattern_type = pattern_type.lower().strip()
        
        with self._lock:
            if pattern_type not in self._patterns:
                return None
            
            pt = self._patterns[pattern_type]
            total = pt["total_queries"]
            
            return {
                "pattern_type": pattern_type,
                "total_queries": total,
                "success_rate": pt["successful_queries"] / total if total > 0 else 0,
                "avg_confidence": pt["total_confidence"] / total if total > 0 else 0,
                "decomposition_rate": pt["decomposition_count"] / total if total > 0 else 0,
                "avg_sub_questions": pt["total_sub_questions"] / pt["decomposition_count"] if pt["decomposition_count"] > 0 else 0,
                "top_entity_types": sorted(
                    pt["entity_types"].items(),
                    key=lambda x: -x[1]
                )[:5],
            }
    
    def should_decompose(self, pattern_type: str) -> bool:
        """
        Determine if a pattern type typically needs decomposition.
        
        Args:
            pattern_type: The pattern type.
            
        Returns:
            True if decomposition is recommended.
        """
        stats = self.get_pattern_stats(pattern_type)
        
        if not stats:
            # Default recommendations for unknown patterns
            always_decompose = {"comparison", "enumeration", "temporal"}
            return pattern_type.lower() in always_decompose
        
        # Recommend decomposition if historically needed > 50% of the time
        return stats["decomposition_rate"] > 0.5
    
    def get_confidence_threshold(self, pattern_type: str) -> float:
        """
        Get recommended confidence threshold for a pattern type.
        
        Args:
            pattern_type: The pattern type.
            
        Returns:
            Recommended confidence threshold.
        """
        stats = self.get_pattern_stats(pattern_type)
        
        if not stats or stats["total_queries"] < 5:
            return 0.7  # Default threshold
        
        # Use average confidence minus one standard deviation as threshold
        avg_conf = stats["avg_confidence"]
        return max(0.5, min(0.9, avg_conf - 0.1))
    
    def get_all_patterns(self) -> list[dict[str, Any]]:
        """Get statistics for all known patterns."""
        results = []
        
        with self._lock:
            for pattern_type in self._patterns:
                stats = self.get_pattern_stats(pattern_type)
                if stats:
                    results.append(stats)
        
        return sorted(results, key=lambda x: -x["total_queries"])


# Global query patterns registry
_global_query_patterns: LearnedQueryPatterns | None = None


def get_learned_query_patterns() -> LearnedQueryPatterns:
    """Get the global learned query patterns registry."""
    global _global_query_patterns
    
    if _global_query_patterns is None:
        custom_path = os.getenv("RNSR_QUERY_PATTERNS_PATH")
        _global_query_patterns = LearnedQueryPatterns(
            storage_path=custom_path if custom_path else None
        )
    
    return _global_query_patterns


# =============================================================================
# Learned Section Patterns - Maps query keywords to successful section types
# =============================================================================

DEFAULT_SECTION_PATTERNS_PATH = Path.home() / ".rnsr" / "learned_section_patterns.json"


class LearnedSectionPatterns:
    """
    Learn which section types successfully answer which query types.
    
    Example learnings:
    - "parties" queries -> "Information" sections (Client Information, Provider Information)
    - "termination" queries -> "Termination" and "Term" sections
    - "payment" queries -> "Payment", "Compensation" sections
    
    This is used to boost scoring for sections that historically answer similar queries.
    """
    
    def __init__(
        self,
        storage_path: Path | str | None = None,
        auto_save: bool = True,
    ):
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_SECTION_PATTERNS_PATH
        self.auto_save = auto_save
        
        self._lock = Lock()
        # Maps query_keyword -> {section_header_word -> success_count}
        self._patterns: dict[str, dict[str, int]] = {}
        self._dirty = False
        
        self._load()
    
    def _load(self) -> None:
        """Load learned patterns from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            self._patterns = data.get("patterns", {})
            
            logger.info(
                "section_patterns_loaded",
                query_keywords=len(self._patterns),
            )
            
        except Exception as e:
            logger.warning("failed_to_load_section_patterns", error=str(e))
    
    def _save(self) -> None:
        """Save to storage."""
        if not self._dirty:
            return
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "patterns": self._patterns,
            }
            
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            
            self._dirty = False
            
            logger.info("section_patterns_saved", patterns=len(self._patterns))
            
        except Exception as e:
            logger.warning("failed_to_save_section_patterns", error=str(e))
    
    def record_success(
        self,
        query_keywords: list[str],
        successful_section_headers: list[str],
    ) -> None:
        """
        Record that certain sections successfully answered a query.
        
        Args:
            query_keywords: Keywords from the query (e.g., ["parties", "contract"])
            successful_section_headers: Headers of sections that provided good answers
                (e.g., ["1.1 Client Information", "1.2 Provider Information"])
        """
        with self._lock:
            for keyword in query_keywords:
                keyword = keyword.lower().strip()
                if len(keyword) < 3:
                    continue
                
                if keyword not in self._patterns:
                    self._patterns[keyword] = {}
                
                # Extract significant words from section headers
                for header in successful_section_headers:
                    header_words = re.findall(r'\b[A-Za-z]{4,}\b', header)
                    for word in header_words:
                        word = word.lower()
                        self._patterns[keyword][word] = self._patterns[keyword].get(word, 0) + 1
            
            self._dirty = True
            
            if self.auto_save:
                self._save()
            
            logger.info(
                "section_pattern_recorded",
                keywords=query_keywords,
                sections=[h[:50] for h in successful_section_headers],
            )
    
    def get_boosted_header_words(self, query_keywords: list[str], min_count: int = 2) -> set[str]:
        """
        Get header words that should be boosted for this query based on past successes.
        
        Args:
            query_keywords: Keywords from the current query
            min_count: Minimum success count to include a word
            
        Returns:
            Set of header words that historically lead to successful answers
        """
        boosted_words = set()
        
        with self._lock:
            for keyword in query_keywords:
                keyword = keyword.lower().strip()
                if keyword in self._patterns:
                    for word, count in self._patterns[keyword].items():
                        if count >= min_count:
                            boosted_words.add(word)
        
        return boosted_words
    
    def get_all_patterns(self) -> dict[str, dict[str, int]]:
        """Get all learned patterns."""
        with self._lock:
            return {k: dict(v) for k, v in self._patterns.items()}


# Global section patterns registry
_global_section_patterns: LearnedSectionPatterns | None = None


def get_learned_section_patterns() -> LearnedSectionPatterns:
    """Get the global learned section patterns registry."""
    global _global_section_patterns
    
    if _global_section_patterns is None:
        custom_path = os.getenv("RNSR_SECTION_PATTERNS_PATH")
        _global_section_patterns = LearnedSectionPatterns(
            storage_path=custom_path if custom_path else None
        )
    
    return _global_section_patterns


# =============================================================================
# RLM Configuration
# =============================================================================


@dataclass
class RLMConfig:
    """Configuration for the RLM Navigator."""
    
    # Recursion control
    max_recursion_depth: int = 5  # Max depth for recursive sub-LLM calls
    max_iterations: int = 50  # Max navigation iterations
    
    # Tree of Thoughts parameters
    top_k: int = 3  # Base children to explore
    selection_threshold: float = 0.4  # Min probability for selection
    dead_end_threshold: float = 0.1  # Threshold for dead end
    
    # Pre-filtering
    enable_pre_filtering: bool = True  # Use regex/keyword filtering before ToT
    pre_filter_min_matches: int = 1  # Min keyword matches to include node
    
    # REPL execution
    enable_code_execution: bool = True  # Allow LLM to write/execute code
    max_code_execution_time: int = 30  # Seconds
    
    # Answer verification
    enable_verification: bool = True  # Verify answers with sub-LLM
    verification_retries: int = 2  # Max verification attempts
    
    # Async processing
    enable_async: bool = True  # Use async for parallel sub-LLM calls
    max_concurrent_calls: int = 5  # Max parallel LLM calls
    
    # Vision mode
    enable_vision: bool = False  # Use vision LLM for page images
    vision_model: str = "gemini-2.5-flash"  # Vision model to use
    
    # RLM Navigation Mode (LLM writes code to search document)
    use_rlm_navigation: bool = True  # Use LLM code generation for navigation
    rlm_max_search_iterations: int = 15  # Max code generation iterations per navigation
    rlm_search_depth: int = 99  # How many levels deep to search in one iteration (unlimited)
    rlm_min_content_length: int = 50  # Minimum useful content length
    rlm_max_content_for_specific: int = 3000  # Content longer than this suggests broad section
    
    # Minimum exploration requirements (prevents premature synthesis)
    min_nodes_to_visit: int = 2  # Minimum nodes to visit before allowing synthesis
    min_findings_required: int = 1  # Minimum findings before synthesis allowed (quality > quantity)

    # Answer format
    use_short_answer: bool = False  # When True, produce minimal key-phrase answers


# =============================================================================
# Pre-Filtering Engine (Before ToT Evaluation)
# =============================================================================


class PreFilterEngine:
    """
    Pre-filters nodes before expensive ToT LLM evaluation.
    
    Implements the key RLM insight: use code (regex, keywords) to filter
    before sending to LLM. This dramatically reduces LLM calls.
    
    Uses adaptive stop words that learn from domain-specific usage.
    
    Example:
        # Query: "What is the liability clause?"
        # Instead of evaluating all 50 children with LLM:
        # 1. Extract keywords: ["liability", "clause", "indemnification"]
        # 2. Regex search children summaries
        # 3. Only send matching children to ToT evaluation
    """
    
    def __init__(self, config: RLMConfig, enable_stop_word_learning: bool = True):
        self.config = config
        self._keyword_cache: dict[str, list[str]] = {}
        self._stop_word_registry = get_learned_stop_words() if enable_stop_word_learning else None
    
    def extract_keywords(self, query: str) -> list[str]:
        """Extract searchable keywords from a query."""
        if query in self._keyword_cache:
            return self._keyword_cache[query]
        
        # Get stop words (base + learned)
        if self._stop_word_registry:
            stop_words = self._stop_word_registry.get_stop_words()
        else:
            stop_words = LearnedStopWords.BASE_STOP_WORDS
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        keywords = [w for w in words if w not in stop_words]
        
        # Add quoted phrases as single keywords
        quoted = re.findall(r'"([^"]+)"', query)
        keywords.extend(quoted)
        
        # Add capitalized words (likely proper nouns)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', query)
        keywords.extend([pn.lower() for pn in proper_nouns])
        
        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        self._keyword_cache[query] = unique_keywords
        logger.debug("keywords_extracted", query=query[:50], keywords=unique_keywords)
        return unique_keywords
    
    def filter_nodes_by_keywords(
        self,
        nodes: list[SkeletonNode],
        keywords: list[str],
        min_matches: int | None = None,
    ) -> tuple[list[SkeletonNode], list[SkeletonNode]]:
        """
        Filter nodes by keyword matching.
        
        Returns:
            Tuple of (matching_nodes, remaining_nodes)
        """
        if not self.config.enable_pre_filtering:
            return nodes, []
        
        if not keywords:
            return nodes, []
        
        min_matches = min_matches or self.config.pre_filter_min_matches
        
        matching = []
        remaining = []
        
        for node in nodes:
            # Search in header and summary
            search_text = f"{node.header} {node.summary}".lower()
            
            matches = sum(1 for kw in keywords if kw in search_text)
            
            if matches >= min_matches:
                matching.append(node)
            else:
                remaining.append(node)
        
        logger.debug(
            "pre_filter_complete",
            total=len(nodes),
            matching=len(matching),
            remaining=len(remaining),
            keywords=keywords[:5],
        )
        
        return matching, remaining
    
    def regex_search_nodes(
        self,
        nodes: list[SkeletonNode],
        pattern: str,
    ) -> list[tuple[SkeletonNode, list[str]]]:
        """
        Search nodes using regex pattern.
        
        Returns:
            List of (node, matches) tuples.
        """
        results = []
        
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            logger.warning("invalid_regex_pattern", pattern=pattern, error=str(e))
            return results
        
        for node in nodes:
            search_text = f"{node.header}\n{node.summary}"
            matches = regex.findall(search_text)
            if matches:
                results.append((node, matches))
        
        return results


# =============================================================================
# Deep Recursive Sub-LLM Engine
# =============================================================================


class RecursiveSubLLMEngine:
    """
    Enables true multi-level recursive sub-LLM calls.
    
    Unlike single-level decomposition, this allows sub-LLMs to spawn
    their own sub-LLMs up to a configurable depth.
    
    Example:
        Query: "Compare the liability clauses in 2023 vs 2024 contracts"
        
        Depth 0 (Root): Decompose into sub-tasks
        ├── Depth 1: "Find 2023 liability clause"
        │   └── Depth 2: "Extract specific terms"
        └── Depth 1: "Find 2024 liability clause"
            └── Depth 2: "Extract specific terms"
    """
    
    def __init__(
        self,
        config: RLMConfig,
        llm_fn: Callable[[str], str] | None = None,
    ):
        self.config = config
        self._llm_fn = llm_fn
        self._call_count = 0
        self._depth_stats: dict[int, int] = {}
    
    def set_llm_function(self, llm_fn: Callable[[str], str]) -> None:
        """Set the LLM function for sub-calls."""
        self._llm_fn = llm_fn
    
    def recursive_call(
        self,
        prompt: str,
        context: str,
        depth: int = 0,
        allow_sub_calls: bool = True,
    ) -> str:
        """
        Execute a recursive LLM call.
        
        Args:
            prompt: The task/question for the LLM.
            context: Context to process.
            depth: Current recursion depth.
            allow_sub_calls: Whether this call can spawn sub-calls.
            
        Returns:
            LLM response.
        """
        if self._llm_fn is None:
            return "[ERROR: LLM function not configured]"
        
        if depth >= self.config.max_recursion_depth:
            allow_sub_calls = False
            logger.debug("max_recursion_depth_reached", depth=depth)
        
        # Track stats
        self._call_count += 1
        self._depth_stats[depth] = self._depth_stats.get(depth, 0) + 1
        
        # Build the prompt with recursion capability
        if allow_sub_calls:
            system_instruction = f"""You are a sub-LLM at recursion depth {depth}.
You can decompose complex tasks into sub-tasks.
If you need to process multiple items independently, list them as:
SUB_TASK[1]: <task description>
SUB_TASK[2]: <task description>
...
These will be processed by sub-LLMs and results aggregated.
"""
        else:
            system_instruction = f"""You are a sub-LLM at max recursion depth {depth}.
Provide a direct answer without further decomposition."""
        
        full_prompt = f"""{system_instruction}

Task: {prompt}

Context:
{context}

Response:"""
        
        try:
            response = self._llm_fn(full_prompt)
            
            # Check for sub-task declarations and process them
            if allow_sub_calls and "SUB_TASK[" in response:
                response = self._process_sub_tasks(response, depth + 1)
            
            return response
            
        except Exception as e:
            logger.error("recursive_call_failed", depth=depth, error=str(e))
            return f"[ERROR: {str(e)}]"
    
    def _process_sub_tasks(self, response: str, depth: int) -> str:
        """Process SUB_TASK declarations in the response."""
        # Extract sub-tasks
        sub_tasks = re.findall(r'SUB_TASK\[(\d+)\]:\s*(.+?)(?=SUB_TASK\[|$)', response, re.DOTALL)
        
        if not sub_tasks:
            return response
        
        logger.debug("processing_sub_tasks", count=len(sub_tasks), depth=depth)
        
        # Process each sub-task recursively
        results = []
        for idx, (task_num, task_desc) in enumerate(sub_tasks):
            result = self.recursive_call(
                prompt=task_desc.strip(),
                context="(inherited from parent)",
                depth=depth,
                allow_sub_calls=(depth < self.config.max_recursion_depth),
            )
            results.append(f"Result[{task_num}]: {result}")
        
        # Synthesize results
        synthesis_prompt = f"""Synthesize the following sub-task results into a coherent answer:

{chr(10).join(results)}

Original task: {response.split('SUB_TASK[')[0].strip()}

Synthesized answer:"""
        
        return self._llm_fn(synthesis_prompt) if self._llm_fn else "\n".join(results)
    
    async def async_recursive_call(
        self,
        prompt: str,
        context: str,
        depth: int = 0,
    ) -> str:
        """Async version of recursive_call for parallel processing."""
        # Run in thread pool to not block
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.recursive_call(prompt, context, depth),
        )
    
    def batch_recursive_calls(
        self,
        prompts: list[str],
        contexts: list[str],
        depth: int = 0,
    ) -> list[str]:
        """
        Execute multiple recursive calls in parallel.
        
        Uses ThreadPoolExecutor for parallel processing.
        """
        if len(prompts) != len(contexts):
            raise ValueError("prompts and contexts must have same length")
        
        if not prompts:
            return []
        
        results: list[str] = [""] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_calls) as executor:
            futures = {}
            for idx, (prompt, context) in enumerate(zip(prompts, contexts)):
                future = executor.submit(
                    self.recursive_call,
                    prompt,
                    context,
                    depth,
                )
                futures[future] = idx
            
            for future in futures:
                idx = futures[future]
                try:
                    results[idx] = future.result(timeout=60)
                except Exception as e:
                    results[idx] = f"[ERROR: {str(e)}]"
        
        return results
    
    def get_stats(self) -> dict[str, Any]:
        """Get call statistics."""
        return {
            "total_calls": self._call_count,
            "calls_by_depth": dict(self._depth_stats),
        }


# =============================================================================
# Answer Verification Engine
# =============================================================================


class AnswerVerificationEngine:
    """
    Verifies answers using sub-LLM calls.
    
    Implements the RLM pattern of using sub-LLMs to verify answers
    before returning, ensuring higher accuracy.
    """
    
    def __init__(
        self,
        config: RLMConfig,
        llm_fn: Callable[[str], str] | None = None,
    ):
        self.config = config
        self._llm_fn = llm_fn
    
    def set_llm_function(self, llm_fn: Callable[[str], str]) -> None:
        """Set the LLM function."""
        self._llm_fn = llm_fn
    
    def verify_answer(
        self,
        question: str,
        proposed_answer: str,
        evidence: list[str],
        attempt: int = 0,
    ) -> dict[str, Any]:
        """
        Verify an answer using sub-LLM evaluation.
        
        Returns:
            Dict with 'is_valid', 'confidence', 'issues', 'improved_answer'.
        """
        if not self.config.enable_verification:
            return {
                "is_valid": True,
                "confidence": 0.7,
                "issues": [],
                "improved_answer": proposed_answer,
            }
        
        if self._llm_fn is None:
            return {
                "is_valid": True,
                "confidence": 0.5,
                "issues": ["LLM not configured for verification"],
                "improved_answer": proposed_answer,
            }
        
        evidence_text = "\n---\n".join(evidence) if evidence else "(no evidence provided)"
        
        verification_prompt = f"""Verify whether the answer is supported by the evidence. Respond with ONLY a single-line JSON object — nothing else.

Question: {question}

Answer: {proposed_answer}

Evidence:
{evidence_text}

JSON: {{"is_valid": true/false, "confidence": 0.0-1.0, "issues": [], "improved_answer": null}}"""
        
        try:
            import json
            
            response = self._llm_fn(verification_prompt)
            
            # Parse JSON response
            json_match = re.search(r'\{[\s\S]*\}', response)
            result = None
            if json_match:
                try:
                    result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # Fallback: extract is_valid/confidence from truncated JSON
            # (LLM responses often get cut off mid-issues list)
            if result is None:
                result = self._parse_truncated_verification(response, proposed_answer)

            if result is not None:
                if not result.get("is_valid", True) and attempt < self.config.verification_retries:
                    logger.debug(
                        "answer_verification_failed",
                        attempt=attempt,
                        issues=result.get("issues", []),
                    )
                    if result.get("improved_answer"):
                        return self.verify_answer(
                            question,
                            result["improved_answer"],
                            evidence,
                            attempt + 1,
                        )
                return result
            else:
                logger.warning("verification_json_parse_failed", response=response[:200])
                return {
                    "is_valid": False,
                    "confidence": 0.2,
                    "issues": ["Could not parse verification response - treating as unverified"],
                    "improved_answer": proposed_answer,
                }
                
        except Exception as e:
            logger.error("verification_failed", error=str(e))
            return {
                "is_valid": False,
                "confidence": 0.1,
                "issues": [f"Verification failed: {str(e)}"],
                "improved_answer": proposed_answer,
            }

    @staticmethod
    def _parse_truncated_verification(
        response: str, proposed_answer: str
    ) -> dict[str, Any] | None:
        """Extract is_valid/confidence from a truncated LLM verification response.

        LLM responses sometimes exceed the token limit and get cut off before
        the JSON closing brace, causing full JSON parsing to fail. This method
        uses simple regex to recover the key fields so that a clearly-valid
        answer isn't rejected due to a formatting issue.
        """
        valid_match = re.search(
            r'"is_valid"\s*:\s*(true|false)', response, re.IGNORECASE
        )
        conf_match = re.search(
            r'"confidence"\s*:\s*([\d.]+)', response
        )
        if valid_match is None and conf_match is None:
            return None

        is_valid = (
            valid_match.group(1).lower() == "true" if valid_match else False
        )
        try:
            confidence = float(conf_match.group(1)) if conf_match else 0.5
        except ValueError:
            confidence = 0.5

        logger.info(
            "verification_recovered_from_truncated_json",
            is_valid=is_valid,
            confidence=confidence,
        )
        return {
            "is_valid": is_valid,
            "confidence": confidence,
            "issues": ["Verification JSON was truncated; key fields recovered"],
            "improved_answer": proposed_answer,
        }


# =============================================================================
# Enhanced RLM Navigator Agent State
# =============================================================================


class RLMAgentState:
    """
    State for the RLM Navigator Agent.
    
    Extends the base AgentState with RLM-specific fields.
    """
    
    def __init__(
        self,
        question: str,
        root_node_id: str,
        config: RLMConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.question = question
        self.config = config or RLMConfig()
        self.metadata = metadata or {}
        
        # Navigation state
        self.current_node_id: str | None = root_node_id
        self.visited_nodes: list[str] = []
        self.navigation_path: list[str] = [root_node_id]
        self.nodes_to_visit: list[str] = []
        self.dead_ends: list[str] = []
        self.backtrack_stack: list[str] = []
        
        # Variable stitching
        self.variables: list[str] = []
        self.context: str = ""
        
        # Sub-questions (RLM decomposition)
        self.sub_questions: list[str] = []
        self.pending_questions: list[str] = []
        self.current_sub_question: str | None = None
        
        # Pre-filtering state
        self.extracted_keywords: list[str] = []
        self.pre_filtered_nodes: dict[str, list[str]] = {}  # node_id -> matched keywords
        
        # Recursion tracking
        self.current_recursion_depth: int = 0
        self.recursion_call_count: int = 0
        
        # Output
        self.answer: str | None = None
        self.confidence: float = 0.0
        self.verification_result: dict[str, Any] | None = None
        
        # Traceability
        self.trace: list[dict[str, Any]] = []
        self.iteration: int = 0
    
    def add_trace(
        self,
        node_type: str,
        action: str,
        details: dict | None = None,
    ) -> None:
        """Add a trace entry."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_type": node_type,
            "action": action,
            "details": details or {},
            "iteration": self.iteration,
        }
        self.trace.append(entry)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "variables": self.variables,
            "visited_nodes": self.visited_nodes,
            "iteration": self.iteration,
            "recursion_call_count": self.recursion_call_count,
            "verification_result": self.verification_result,
            "trace": self.trace,
        }


# =============================================================================
# RLM Navigator - Main Class
# =============================================================================


class RLMNavigator:
    """
    The RLM Navigator combines:
    1. PageIndex-style tree search with reasoning
    2. RLM-style REPL environment with code execution
    3. RNSR-style variable stitching and skeleton indexing
    4. Entity-aware query decomposition (when knowledge graph available)
    
    This is the unified, state-of-the-art document retrieval agent.
    """
    
    def __init__(
        self,
        skeleton: dict[str, SkeletonNode],
        kv_store: KVStore,
        config: RLMConfig | None = None,
        knowledge_graph=None,
        tables: list | None = None,
        doc_profile: dict | None = None,
        doc_title: str | None = None,
        embedding_index=None,
    ):
        self.skeleton = skeleton
        self.kv_store = kv_store
        self.config = config or RLMConfig()
        self.knowledge_graph = knowledge_graph
        self.tables = tables or []
        self.doc_profile = doc_profile
        self.doc_title = doc_title
        
        # Initialize components
        self.variable_store = VariableStore()
        self.pre_filter = PreFilterEngine(self.config)
        self.recursive_engine = RecursiveSubLLMEngine(self.config)
        self.verification_engine = AnswerVerificationEngine(self.config)
        self.entity_decomposer = EntityAwareDecomposer(
            knowledge_graph, skeleton=skeleton, kv_store=kv_store,
        )
        
        # NavigationREPL for RLM-style code generation navigation
        self.nav_repl = create_navigation_repl(skeleton, kv_store, tables=tables)
        if embedding_index is not None:
            self.nav_repl.set_embedding_index(embedding_index)
        
        # LLM function
        self._llm_fn: Callable[[str], str] | None = None
        
        # Find root node
        self.root_id = self._find_root_id()
    
    def _find_root_id(self) -> str:
        """Find the root node ID."""
        for node in self.skeleton.values():
            if node.level == 0:
                return node.node_id
        raise ValueError("No root node found in skeleton")

    def _build_identity_block(self) -> str:
        """Build a DOCUMENT IDENTITY block from the profile and title.

        Returns an empty string when no useful identity data is available.
        """
        if not self.doc_profile and not self.doc_title:
            return ""

        parts: list[str] = ["DOCUMENT IDENTITY:"]

        title = self.doc_title or ""
        profile = self.doc_profile or {}

        citation = profile.get("citation", "")
        label = f'"{title}"'
        if citation:
            label += f" {citation}"
        parts.append(f"This document is: {label}")

        parties = profile.get("parties")
        if parties and isinstance(parties, list) and len(parties) > 0:
            parts.append(f"Parties: {', '.join(parties)}")

        doc_type = profile.get("document_type")
        if doc_type:
            parts.append(f"Type: {doc_type}")

        judge = profile.get("judge")
        if judge:
            parts.append(f"Judge: {judge}")

        parts.append(
            "Answer questions about THIS case/document. If it discusses "
            "other cases or proceedings, only report facts that pertain "
            "to the case identified above."
        )
        return "\n".join(parts)
    
    def set_llm_function(self, llm_fn: Callable[[str], str]) -> None:
        """Configure the LLM function for all components."""
        self._llm_fn = llm_fn
        self.recursive_engine.set_llm_function(llm_fn)
        self.verification_engine.set_llm_function(llm_fn)
        self.entity_decomposer.set_llm_function(llm_fn)
        self.nav_repl.set_llm_function(llm_fn)
    
    def set_knowledge_graph(self, kg) -> None:
        """Set the knowledge graph for entity-aware decomposition."""
        self.knowledge_graph = kg
        self.entity_decomposer.set_knowledge_graph(kg)
    
    def set_tables(self, tables: list) -> None:
        """
        Set detected tables for SQL-like querying during navigation.
        
        Args:
            tables: List of DetectedTable objects from ingestion.
        """
        self.tables = tables or []
        self.nav_repl.set_tables(self.tables)
    
    def _record_successful_patterns(self, state: "RLMAgentState") -> None:
        """
        Record successful query-to-section patterns for learning.
        
        This enables the system to learn which section types successfully
        answer which query types, improving future searches.
        """
        try:
            section_patterns = get_learned_section_patterns()
            
            # Get query keywords
            keywords = state.extracted_keywords or []
            if not keywords:
                # Extract basic keywords from question
                keywords = [w.lower() for w in state.question.split() if len(w) > 3]
            
            # Get successful section headers from stored variables
            successful_headers = []
            for var_name in state.variables:
                # Try to find the source node for this variable
                meta = self.variable_store.get_metadata(var_name)
                if meta and meta.source_node_id:
                    node = self.skeleton.get(meta.source_node_id)
                    if node:
                        successful_headers.append(node.header)
            
            if keywords and successful_headers:
                section_patterns.record_success(keywords, successful_headers)
                
        except Exception as e:
            logger.warning("failed_to_record_patterns", error=str(e))
    
    def navigate(
        self,
        question: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Navigate the document tree to answer a question.
        
        This is the main entry point for the RLM Navigator.
        
        Args:
            question: The user's question.
            metadata: Optional metadata (e.g., multiple choice options).
            
        Returns:
            Dict with answer, confidence, trace, etc.
        """
        # Merge use_short_answer from config into metadata so the
        # synthesis prompt picks it up automatically.
        effective_metadata = dict(metadata) if metadata else {}
        if self.config.use_short_answer:
            effective_metadata.setdefault("use_short_answer", True)

        # Initialize state
        state = RLMAgentState(
            question=question,
            root_node_id=self.root_id,
            config=self.config,
            metadata=effective_metadata,
        )

        # The navigator instance may be cached and reused across multiple
        # queries on the same document (see ``_navigator_cache`` in
        # ``RNSRClient``). The ``VariableStore`` is per-query context, so
        # reset it here to prevent variables from prior questions
        # leaking into this navigation (e.g. blocking fallback fixtures
        # because an earlier query already assigned the same pointer).
        self.variable_store = VariableStore()

        # Ensure LLM is configured
        if self._llm_fn is None:
            self._configure_default_llm()
        
        logger.info("rlm_navigation_started", question=question[:100])
        
        try:
            # Phase 0: Inject entity priority nodes from KG resolver
            entity_priority = effective_metadata.get("entity_priority_nodes")
            if entity_priority:
                valid_nodes = [
                    nid for nid in entity_priority if nid in self.skeleton
                ]
                if valid_nodes:
                    state.nodes_to_visit = valid_nodes + state.nodes_to_visit
                    logger.info(
                        "entity_priority_nodes_injected",
                        count=len(valid_nodes),
                    )

            # Phase 1: Pre-filtering with keyword extraction
            state = self._phase_pre_filter(state)
            
            # Phase 2: Query decomposition
            state = self._phase_decompose(state)
            
            # Recursive navigate-synthesize loop with adaptive retry depth.
            # Simple factual queries get fewer retries; complex analytical
            # queries keep the full budget.
            base_max = 1 + self.config.max_recursion_depth
            max_attempts = base_max
            for attempt in range(base_max):
                if attempt >= max_attempts:
                    break

                # Phase 3: Tree navigation with ToT
                state = self._phase_navigate(state)
                
                # Phase 3b: Re-rank sections and enrich with siblings
                state = self._phase_rerank_sections(state)
                
                # Phase 4: Synthesis
                state = self._phase_synthesize(state)
                
                if not self._answer_is_inconclusive(state.answer):
                    # Adaptive: if first attempt yielded good confidence,
                    # cap remaining retries to save LLM calls.
                    if attempt == 0 and state.confidence >= 0.7:
                        max_attempts = min(max_attempts, attempt + 2)
                    break
                
                if attempt < max_attempts - 1:
                    logger.info(
                        "answer_inconclusive_refining",
                        attempt=attempt + 1,
                        max_attempts=max_attempts,
                        answer_preview=state.answer[:100] if state.answer else "",
                    )
                    state = self._refine_search_strategy(state)
            
            # Low-confidence retry: if the synthesis produced an answer but
            # with low confidence, supplement with header-matched sections
            # and re-synthesize before giving up.
            low_confidence_retry = (
                state.variables
                and not self._answer_is_inconclusive(state.answer)
                and state.confidence < 0.5
            )
            if low_confidence_retry and self._llm_fn:
                logger.info(
                    "low_confidence_retry",
                    confidence=state.confidence,
                    answer_preview=(state.answer or "")[:100],
                )
                extra_pointers = _header_match_fallback(
                    question=state.question,
                    skeleton=self.skeleton,
                    kv_store=self.kv_store,
                    variable_store=self.variable_store,
                    llm_fn=self._llm_fn,
                    min_selections=2,
                    max_selections=4,
                )
                if extra_pointers:
                    state.variables.extend(extra_pointers)
                    state.add_trace(
                        "low_confidence_retry",
                        f"Added {len(extra_pointers)} sections for retry",
                    )
                    state = self._phase_synthesize(state)

            # Header-match fallback: if no variables were found OR the answer
            # is still inconclusive after all refinement attempts, present ALL
            # section headers to the LLM and let it pick the right ones.
            needs_fallback = (
                not state.variables
                or self._answer_is_inconclusive(state.answer)
            )
            if needs_fallback and self._llm_fn:
                logger.info("header_match_fallback_triggered")
                fallback_pointers = _header_match_fallback(
                    question=state.question,
                    skeleton=self.skeleton,
                    kv_store=self.kv_store,
                    variable_store=self.variable_store,
                    llm_fn=self._llm_fn,
                )
                if fallback_pointers:
                    state.variables.extend(fallback_pointers)
                    state.add_trace(
                        "navigation",
                        f"Header-match fallback stored {len(fallback_pointers)} sections",
                    )
                    state = self._phase_synthesize(state)

            # Content-search fallback: when header-based fallbacks still
            # leave the answer inconclusive (e.g. ingestion produced
            # numeric/column-label headers in tables), score raw KV
            # content by query-keyword density and inject the best
            # matches before the final synthesis.
            still_needs_fallback = (
                not state.variables
                or self._answer_is_inconclusive(state.answer)
            )
            if still_needs_fallback:
                logger.info("content_search_fallback_triggered")
                content_pointers = _content_search_fallback(
                    question=state.question,
                    skeleton=self.skeleton,
                    kv_store=self.kv_store,
                    variable_store=self.variable_store,
                    visited_node_ids=set(state.visited_nodes),
                )
                if content_pointers:
                    state.variables.extend(content_pointers)
                    state.add_trace(
                        "navigation",
                        f"Content-search fallback stored {len(content_pointers)} sections",
                        {"new_pointers": content_pointers[:8]},
                    )
                    state = self._phase_synthesize(state)

            # Enumeration-scope fallback: questions of the form "which of
            # X had the most/least/lowest/highest Y" need retrieval to
            # cover the *full* domain. Even a confident-sounding answer
            # can be wrong if the navigator drilled into one branch of a
            # hierarchy and never visited the parent overview table
            # (this is exactly the JPM 10-Q "Home Lending vs Corporate"
            # miss). We always run this for enumeration questions; the
            # re-synthesis is cheap relative to the cost of a wrong
            # answer.
            if _is_enumeration_question(state.question):
                logger.info(
                    "enumeration_scope_fallback_triggered",
                    answer_preview=(state.answer or "")[:120],
                )
                scope_pointers = _enumeration_scope_fallback(
                    question=state.question,
                    skeleton=self.skeleton,
                    kv_store=self.kv_store,
                    variable_store=self.variable_store,
                    visited_node_ids=set(state.visited_nodes),
                )
                if scope_pointers:
                    new_pointers = [p for p in scope_pointers if p not in state.variables]
                    if new_pointers:
                        state.variables.extend(new_pointers)
                        state.add_trace(
                            "navigation",
                            f"Enumeration-scope fallback stored {len(new_pointers)} parent-level sections",
                            {"new_pointers": new_pointers[:8]},
                        )
                        state = self._phase_synthesize(state)
            
            # Phase 5: Verification (if enabled)
            if self.config.enable_verification:
                state = self._phase_verify(state)
            
            logger.info(
                "rlm_navigation_complete",
                confidence=state.confidence,
                variables=len(state.variables),
                iterations=state.iteration,
            )
            
            # Record successful patterns for learning
            if state.confidence >= 0.5 and state.variables:
                self._record_successful_patterns(state)
            
            return state.to_dict()
            
        except Exception as e:
            logger.error("rlm_navigation_failed", error=str(e))
            state.answer = f"Error during navigation: {str(e)}"
            state.confidence = 0.0
            return state.to_dict()
    
    def _configure_default_llm(self) -> None:
        """Configure the default LLM if none set."""
        try:
            from rnsr.llm import get_llm
            llm = get_llm()
            self.set_llm_function(lambda p: str(llm.complete(p)))
        except Exception as e:
            logger.warning("default_llm_config_failed", error=str(e))
    
    def _phase_pre_filter(self, state: RLMAgentState) -> RLMAgentState:
        """Phase 1: Extract keywords and pre-filter nodes."""
        state.add_trace("pre_filter", "Extracting keywords from query")
        
        # Extract keywords
        keywords = self.pre_filter.extract_keywords(state.question)
        state.extracted_keywords = keywords
        
        if not keywords:
            state.add_trace("pre_filter", "No keywords extracted, skipping pre-filter")
            return state
        
        # Pre-filter all leaf nodes
        all_nodes = list(self.skeleton.values())
        matching, remaining = self.pre_filter.filter_nodes_by_keywords(all_nodes, keywords)
        
        # Store which nodes matched which keywords
        for node in matching:
            search_text = f"{node.header} {node.summary}".lower()
            matched_keywords = [kw for kw in keywords if kw in search_text]
            state.pre_filtered_nodes[node.node_id] = matched_keywords
        
        # DON'T hard-restrict allowed nodes based on keyword pre-filter
        # The LLM-generated search patterns will find relevant sections
        # that don't match simple keywords (e.g., "Client Information" 
        # when searching for "parties")
        # 
        # Pre-filter is used for PRIORITIZATION, not hard blocking
        self.nav_repl.set_allowed_nodes(None)  # Allow searching all nodes
        
        state.add_trace(
            "pre_filter",
            f"Pre-filtered {len(matching)}/{len(all_nodes)} nodes",
            {"keywords": keywords, "matching_nodes": len(matching)},
        )
        
        return state
    
    def _phase_decompose(self, state: RLMAgentState) -> RLMAgentState:
        """Phase 2: Decompose query into sub-questions with entity awareness."""
        state.add_trace("decomposition", "Analyzing query for decomposition")
        
        if self._llm_fn is None:
            state.sub_questions = [state.question]
            state.pending_questions = [state.question]
            return state
        
        # Try entity-aware decomposition first if knowledge graph is available
        if self.knowledge_graph:
            try:
                entity_result = self.entity_decomposer.decompose_with_entities(
                    state.question
                )
                
                if entity_result.get("entities_found"):
                    # Store entity information in state
                    state.metadata["entities_found"] = entity_result.get("entities_found", [])
                    state.metadata["entity_nodes"] = entity_result.get("entity_nodes", {})
                    state.metadata["retrieval_plan"] = entity_result.get("retrieval_plan", [])
                    state.metadata["relationships"] = entity_result.get("relationships", [])
                    
                    sub_tasks = entity_result.get("sub_queries", [state.question])
                    state.sub_questions = sub_tasks
                    state.pending_questions = sub_tasks.copy()
                    state.current_sub_question = sub_tasks[0] if sub_tasks else state.question
                    
                    # Prioritize nodes from retrieval plan in pre-filtering
                    for item in entity_result.get("retrieval_plan", []):
                        node_id = item.get("node_id")
                        if node_id and node_id not in state.pre_filtered_nodes:
                            state.pre_filtered_nodes[node_id] = ["entity_match"]
                    
                    state.add_trace(
                        "decomposition",
                        f"Entity-aware decomposition: {len(sub_tasks)} sub-tasks, {len(entity_result.get('entities_found', []))} entities",
                        {
                            "sub_tasks": sub_tasks,
                            "entities": [e.canonical_name for e in entity_result.get("entities_found", [])],
                        },
                    )
                    
                    return state
                    
            except Exception as e:
                logger.debug("entity_aware_decomposition_failed", error=str(e))
                # Fall through to standard decomposition
        
        # Standard LLM decomposition
        decomposition_prompt = f"""Analyze this query and decompose it into specific sub-tasks.

Query: {state.question}

Available document sections (pre-filtered matches):
{chr(10).join(f"- {self.skeleton[nid].header}" for nid in list(state.pre_filtered_nodes.keys())[:10])}

RULES:
1. Each sub-task should target a specific piece of information
2. For comparison queries, create one sub-task per item
3. Maximum 5 sub-tasks
4. If the query is simple, return just one sub-task

OUTPUT FORMAT (JSON):
{{
    "sub_tasks": ["task1", "task2", ...],
    "synthesis_plan": "how to combine results"
}}

Respond with JSON only:"""
        
        try:
            import json
            
            response = self._llm_fn(decomposition_prompt)
            json_match = re.search(r'\{[\s\S]*\}', response)
            
            if json_match:
                result = json.loads(json_match.group())
                sub_tasks = result.get("sub_tasks", [state.question])
                state.sub_questions = sub_tasks
                state.pending_questions = sub_tasks.copy()
                state.current_sub_question = sub_tasks[0] if sub_tasks else state.question
                
                state.add_trace(
                    "decomposition",
                    f"Decomposed into {len(sub_tasks)} sub-tasks",
                    {"sub_tasks": sub_tasks},
                )
            else:
                state.sub_questions = [state.question]
                state.pending_questions = [state.question]
                
        except Exception as e:
            logger.warning("decomposition_failed", error=str(e))
            state.sub_questions = [state.question]
            state.pending_questions = [state.question]
        
        return state
    
    def _phase_navigate(self, state: RLMAgentState) -> RLMAgentState:
        """Phase 3: Navigate the tree using ToT with pre-filtering.
        
        Iterates over all decomposed sub-questions so that multi-part
        queries (e.g. "revenue / avg PP&E") navigate to each required
        section independently.
        """
        state.add_trace("navigation", "Starting tree navigation")
        
        sub_questions = list(state.sub_questions) if state.sub_questions else [state.question]
        
        for sq_idx, sub_q in enumerate(sub_questions):
            state.current_sub_question = sub_q
            state.current_node_id = self.root_id
            state.nodes_to_visit = []
            self.nav_repl.reset(preserve_user_vars=True)
            
            logger.info(
                "navigating_sub_question",
                index=sq_idx + 1,
                total=len(sub_questions),
                sub_question=sub_q[:100],
            )
            
            while state.iteration < self.config.max_iterations:
                state.iteration += 1
                
                if state.current_node_id is None and not state.nodes_to_visit:
                    break
                
                if state.current_node_id is None and state.nodes_to_visit:
                    state.current_node_id = state.nodes_to_visit.pop(0)
                
                if state.current_node_id is None:
                    break
                
                node = self.skeleton.get(state.current_node_id)
                if node is None:
                    state.current_node_id = None
                    continue
                
                if state.current_node_id in state.visited_nodes:
                    state.current_node_id = None
                    continue
                
                action = self._decide_action(state, node)
                
                if action == "expand":
                    state = self._do_expand(state, node)
                elif action == "traverse":
                    state = self._do_traverse(state, node)
                elif action == "backtrack":
                    state = self._do_backtrack(state)
                else:
                    break
            
            if sub_q in state.pending_questions:
                state.pending_questions.remove(sub_q)
        
        state.add_trace(
            "navigation",
            f"Navigation complete after {state.iteration} iterations, {len(sub_questions)} sub-questions",
            {"variables_found": len(state.variables)},
        )
        
        return state

    # ------------------------------------------------------------------
    # Phase 3b – Section re-ranking & sibling enrichment
    # ------------------------------------------------------------------

    def _phase_rerank_sections(self, state: RLMAgentState) -> RLMAgentState:
        """Re-rank collected sections against the question and add siblings.

        After navigation stores candidate sections as variables, this phase
        uses a single lightweight LLM call to rank them by relevance and
        injects adjacent sibling nodes of the best match so the synthesis
        prompt has broader local context (e.g. the costs order sitting
        right after the main judgment section).
        """
        if not state.variables or not self._llm_fn:
            return state

        # Collect section metadata for re-ranking
        section_info: list[tuple[str, str, str, str | None]] = []
        for pointer in state.variables:
            stored = self.variable_store._metadata.get(pointer)
            if not stored:
                continue
            node = self.skeleton.get(stored.source_node_id)
            if not node:
                continue
            content = self.variable_store.resolve(pointer) or ""
            preview = content[:200].replace("\n", " ")
            section_info.append(
                (pointer, node.header, preview, stored.source_node_id)
            )

        if len(section_info) <= 1:
            # Nothing to re-rank; still try sibling enrichment
            if section_info:
                self._enrich_with_siblings(state, section_info[0][3])
            return state

        # Build a lightweight ranking prompt
        lines = []
        for i, (_, header, preview, _) in enumerate(section_info):
            lines.append(f"{i + 1}. {header}: {preview}")

        rank_prompt = (
            "Rank these document sections by relevance to the question. "
            "Return ONLY the numbers in order of relevance (most relevant first), "
            "comma-separated.\n\n"
            f"Question: {state.question}\n\n"
            "Sections:\n" + "\n".join(lines) + "\n\nRanking:"
        )

        try:
            response = self._llm_fn(rank_prompt).strip()
            indices: list[int] = []
            for tok in re.split(r"[,\s]+", response):
                tok = tok.strip().rstrip(".")
                if tok.isdigit():
                    idx = int(tok)
                    if 1 <= idx <= len(section_info) and idx not in indices:
                        indices.append(idx)

            if indices:
                # Reorder variables to match the LLM's ranking
                ranked_pointers = [section_info[i - 1][0] for i in indices]
                for ptr in state.variables:
                    if ptr not in ranked_pointers:
                        ranked_pointers.append(ptr)
                state.variables = ranked_pointers

                # Enrich with siblings of the top-ranked section
                best_node_id = section_info[indices[0] - 1][3]
                self._enrich_with_siblings(state, best_node_id)

                logger.info(
                    "sections_reranked",
                    original_order=[s[1][:40] for s in section_info],
                    new_order=[
                        section_info[i - 1][1][:40] for i in indices
                    ],
                )
        except Exception as exc:
            logger.warning("section_reranking_failed", error=str(exc))

        return state

    def _enrich_with_siblings(
        self, state: RLMAgentState, node_id: str | None
    ) -> None:
        """Add immediately adjacent sibling nodes to state variables."""
        if not node_id:
            return
        node = self.skeleton.get(node_id)
        if not node or not node.parent_id:
            return
        parent = self.skeleton.get(node.parent_id)
        if not parent:
            return

        try:
            idx = parent.child_ids.index(node_id)
        except ValueError:
            return

        siblings_to_add = []
        if idx > 0:
            siblings_to_add.append(parent.child_ids[idx - 1])
        if idx < len(parent.child_ids) - 1:
            siblings_to_add.append(parent.child_ids[idx + 1])

        for sib_id in siblings_to_add:
            sib_node = self.skeleton.get(sib_id)
            if not sib_node:
                continue
            content = self.kv_store.get(sib_id)
            if not content or len(content.strip()) < 20:
                continue
            pointer = generate_pointer_name(sib_node.header)
            if self.variable_store.exists(pointer):
                continue
            self.variable_store.assign(pointer, content, source_node_id=sib_id)
            state.variables.append(pointer)
            if sib_id not in state.visited_nodes:
                state.visited_nodes.append(sib_id)

        if siblings_to_add:
            logger.info(
                "siblings_enriched",
                target_node=node_id,
                siblings_added=len(siblings_to_add),
            )

    def _decide_action(
        self,
        state: RLMAgentState,
        node: SkeletonNode,
    ) -> Literal["expand", "traverse", "backtrack", "done"]:
        """Decide what action to take at current node."""
        # Leaf node -> expand
        if not node.child_ids:
            if node.node_id in state.visited_nodes:
                return "done"
            return "expand"
        
        # Check unvisited children
        unvisited = [
            cid for cid in node.child_ids
            if cid not in state.visited_nodes and cid not in state.dead_ends
        ]
        
        if not unvisited:
            if state.backtrack_stack:
                return "backtrack"
            return "done"
        
        # Has unvisited children -> traverse
        return "traverse"
    
    def _do_expand(self, state: RLMAgentState, node: SkeletonNode) -> RLMAgentState:
        """Expand current node: fetch content, optionally run vision analysis, and store as variable."""
        content = self.kv_store.get(node.node_id)
        
        if content:
            # Vision augmentation: if node has an associated image, analyze it
            if hasattr(self.kv_store, "get_image"):
                try:
                    image_bytes = self.kv_store.get_image(node.node_id)
                    if image_bytes:
                        from rnsr.ingestion.vision_retrieval import VisionLLM, VisionConfig
                        vision_prompt = (
                            f"Analyze this document image. For charts/graphs, extract EXACT data values. "
                            f"For tables, extract all rows and columns as pipe-separated text. "
                            f"For forms, extract all field labels and values. "
                            f"Question context: {state.question}"
                        )
                        vision_analysis = VisionLLM(VisionConfig()).analyze_image(image_bytes, vision_prompt)
                        content = f"{content}\n\n[VISION ANALYSIS]\n{vision_analysis}"
                        state.add_trace(
                            "variable_stitching",
                            f"Vision analysis for node {node.node_id}",
                            {"node_id": node.node_id, "analysis_chars": len(vision_analysis)},
                        )
                except Exception as e:
                    logger.debug("vision_analysis_skipped", node_id=node.node_id, error=str(e))

            pointer = generate_pointer_name(node.header)
            self.variable_store.assign(pointer, content, node.node_id)
            state.variables.append(pointer)
            state.context += f"\nFound: {pointer} (from {node.header})"
            
            state.add_trace(
                "variable_stitching",
                f"Stored {pointer}",
                {"node": node.node_id, "chars": len(content)},
            )
        
        state.visited_nodes.append(node.node_id)
        state.current_node_id = None
        return state
    
    def _do_traverse(self, state: RLMAgentState, node: SkeletonNode) -> RLMAgentState:
        """Traverse to children using deterministic navigation with pre-filtering."""
        # Use deterministic navigation based on pre-filter results
        if self.config.use_rlm_navigation and self._llm_fn:
            # First try deterministic navigation using search results
            state = self._deterministic_navigate(state)
            
            # Check if we got good results
            if state.variables and len(state.variables) > 0:
                state.visited_nodes.append(node.node_id)
                state.current_node_id = None
                return state
            
            # Fallback to RLM code generation if deterministic failed
            logger.info("deterministic_nav_failed_fallback_to_rlm")
            self.nav_repl._navigate_to(node.node_id)
            state = self._rlm_navigate(state)
            
            state.visited_nodes.append(node.node_id)
            state.current_node_id = None
            return state
        
        # Fallback: Traditional ToT with keyword pre-filtering
        children = [self.skeleton.get(cid) for cid in node.child_ids]
        children = [c for c in children if c is not None]
        
        # Apply pre-filtering
        if state.extracted_keywords and self.config.enable_pre_filtering:
            matching, remaining = self.pre_filter.filter_nodes_by_keywords(
                children,
                state.extracted_keywords,
            )
            
            # If we have matching nodes, prioritize them
            if matching:
                selected = matching[:self.config.top_k]
                state.add_trace(
                    "navigation",
                    f"Pre-filter selected {len(selected)}/{len(children)} children",
                    {"selected": [n.node_id for n in selected]},
                )
            else:
                # Fall back to ToT evaluation
                selected = self._tot_evaluate_children(state, children)
        else:
            # Use ToT evaluation
            selected = self._tot_evaluate_children(state, children)
        
        # Queue selected children
        if selected:
            for child in selected:
                if child.node_id not in state.nodes_to_visit:
                    state.nodes_to_visit.append(child.node_id)
            
            # Push current node to backtrack stack
            state.backtrack_stack.append(node.node_id)
        else:
            # Dead end
            state.dead_ends.append(node.node_id)
        
        state.visited_nodes.append(node.node_id)
        state.current_node_id = None
        return state
    
    def _llm_generate_search_patterns(self, query: str) -> list[str]:
        """
        Use LLM to generate intelligent search patterns for a query.
        
        The LLM understands the semantic intent and generates patterns that
        will find relevant sections, including related terms the user didn't
        explicitly mention.
        
        Returns a list of regex patterns to search the document.
        """
        if not self._llm_fn:
            return []
        
        prompt = f"""You are a document search expert. Given a user query, generate SIMPLE regex search patterns that will find ALL relevant sections.

USER QUERY: {query}

CRITICAL: Generate SIMPLE patterns that match document section headers and content.
Use only basic regex: word1|word2|word3 format with (?i) prefix for case-insensitive.

Examples of GOOD patterns:
- (?i)(parties|client|provider|company|inc|llc)
- (?i)(termination|term|expire|end|cancel)
- (?i)(payment|price|cost|fee|amount|value)
- (?i)(deliverable|phase|milestone|due date)

Examples of BAD patterns (too complex, won't match):
- (?i)(?:parties|agreement).*?(?:between|to)
- (?i)(\\w+)\\s+(?:shall|must)

Think about:
1. What words appear in section HEADERS? (e.g., "Client Information", "Provider Details")
2. What synonyms and related terms exist?
3. What proper nouns might appear? (company names, people)

Generate 2-3 SIMPLE patterns, one per line:"""

        try:
            response = self._llm_fn(prompt)
            patterns = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Clean up the pattern
                    if line.startswith('```'):
                        continue
                    # Remove list markers like "- ", "* ", "1. "
                    line = re.sub(r'^[-*]\s+', '', line)
                    line = re.sub(r'^\d+\.\s+', '', line)
                    # Remove quotes if present
                    line = line.strip('"\'`')
                    if line:
                        patterns.append(line)
            
            logger.info(
                "llm_search_patterns_generated",
                query=query,
                patterns=patterns[:10],
            )
            return patterns[:10]  # Generous pattern limit
        except Exception as e:
            logger.warning("llm_pattern_generation_failed", error=str(e))
            return []
    
    def _deterministic_navigate(self, state: RLMAgentState) -> RLMAgentState:
        """
        Hybrid navigation: LLM generates search patterns, ToT executes them.
        
        Flow:
        1. LLM generates intelligent search patterns (semantic understanding)
        2. Patterns are executed against ToT (deterministic search)
        3. Content is extracted directly from found nodes (no hallucination)
        
        This combines LLM intelligence for query understanding with
        deterministic execution for reliability.
        """
        query = state.current_sub_question or state.question
        keywords = state.extracted_keywords or []
        
        logger.info(
            "deterministic_nav_start",
            query=query,
            keywords=keywords,
        )
        
        # Reset REPL state
        self.nav_repl.reset()
        self.nav_repl.set_query(query)
        self.nav_repl.extracted_keywords = keywords  # Sync keywords for learned pattern boosting
        
        # Step 1: Use LLM to generate intelligent search patterns
        llm_patterns = self._llm_generate_search_patterns(query)
        
        # Step 2: Combine LLM patterns with keyword-based patterns as fallback
        all_patterns = []
        
        # Add LLM-generated patterns first (higher quality)
        all_patterns.extend(llm_patterns)
        
        # Add keyword-based pattern as fallback
        if keywords:
            keyword_pattern = r'(?i)(' + '|'.join(re.escape(k) for k in keywords if len(k) > 2) + ')'
            all_patterns.append(keyword_pattern)
        else:
            query_words = [w for w in query.lower().split() if len(w) > 3]
            if query_words:
                word_pattern = r'(?i)(' + '|'.join(re.escape(w) for w in query_words) + ')'
                all_patterns.append(word_pattern)
        
        if not all_patterns:
            logger.warning("deterministic_nav_no_patterns")
            return state
        
        # Step 3: Execute all patterns and collect unique results
        all_results = {}  # node_id -> result (dedup)
        for pattern in all_patterns:
            try:
                search_results = self.nav_repl._search_tree(pattern)
                for result in search_results:
                    node_id = result["node_id"]
                    # Keep highest score for each node
                    if node_id not in all_results or result["score"] > all_results[node_id]["score"]:
                        all_results[node_id] = result
            except Exception as e:
                logger.warning("pattern_search_failed", pattern=pattern, error=str(e))
                continue
        
        # Convert back to sorted list
        search_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        
        if not search_results:
            logger.warning("deterministic_nav_no_results", pattern=pattern)
            return state
        
        # Step 4: Include sibling sections for context completeness
        # If we found section 4.2 and 4.3, also include 4.1 (same parent)
        max_candidates = max(self.config.top_k * 2, 10)  # At least 10 candidates
        sibling_results = {}
        for result in search_results[:max_candidates]:  # Check top matches for siblings
            node_id = result["node_id"]
            node = self.skeleton.get(node_id)
            if node and node.parent_id:
                parent = self.skeleton.get(node.parent_id)
                if parent:
                    # Add all siblings of matched nodes
                    for sibling_id in parent.child_ids:
                        if sibling_id not in all_results and sibling_id not in sibling_results:
                            sibling_node = self.skeleton.get(sibling_id)
                            if sibling_node:
                                sibling_results[sibling_id] = {
                                    "node_id": sibling_id,
                                    "header": sibling_node.header,
                                    "level": sibling_node.level,
                                    "depth_from_current": 0,
                                    "matches": 0,
                                    "score": result["score"] * 0.5,  # Lower score for siblings
                                    "path": [],
                                    "is_sibling": True,
                                }
        
        # Add siblings to results
        if sibling_results:
            logger.info(
                "sibling_sections_added",
                count=len(sibling_results),
                siblings=[r["header"] for r in sibling_results.values()],
            )
            all_results.update(sibling_results)
            search_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        
        # Log what we found
        logger.info(
            "deterministic_nav_search_results",
            num_results=len(search_results),
            top_results=[(r["header"], r["score"], r["node_id"]) for r in search_results[:5]],
        )
        
        # Take top-ranked sections - use higher limit to include siblings
        # We process more candidates but limit actual findings stored
        top_sections = search_results[:max_candidates]
        
        findings_stored = 0
        max_findings = max(self.config.top_k * 3, 15)  # Scale with config
        for result in top_sections:
            node_id = result["node_id"]
            header = result["header"]
            score = result["score"]
            is_sibling = result.get("is_sibling", False)
            
            # Skip relevance validation for sibling sections (included for context)
            if not is_sibling:
                # Validate relevance: check that section header or content matches query intent
                if not self._validate_section_relevance(query, keywords, result):
                    logger.debug(
                        "section_relevance_rejected",
                        node_id=node_id,
                        header=header,
                        reason="low relevance to query",
                    )
                    continue
            
            # Get full content directly from kv_store (no LLM involved)
            content = self.nav_repl.kv_store.get(node_id) or ""
            
            # Check if this is a parent/container node with children but little
            # direct content.  When the user asks about "primary applicant" and
            # we match the header "PRIMARY APPLICANT DETAILS" which is a parent
            # section, we need to expand into its children (Personal Information,
            # Contact Details, etc.) where the actual data lives.
            node_obj = self.skeleton.get(node_id)
            is_truncated_parent = (
                node_obj
                and node_obj.child_ids
                and (content.rstrip().endswith("...") or len(content) < 500)
            )
            if is_truncated_parent:
                max_children_per_parent = 5
                logger.info(
                    "expanding_parent_into_children",
                    parent=header,
                    num_children=len(node_obj.child_ids),
                    expanding=min(len(node_obj.child_ids), max_children_per_parent),
                )
                for child_id in node_obj.child_ids[:max_children_per_parent]:
                    child_node = self.skeleton.get(child_id)
                    if not child_node:
                        continue
                    child_content = self.nav_repl.kv_store.get(child_id) or ""
                    if not child_content or len(child_content) < 20:
                        continue
                    child_pointer = generate_pointer_name(child_node.header)
                    
                    self.variable_store.assign(child_pointer, child_content, child_id)
                    if child_id not in state.visited_nodes:
                        state.visited_nodes.append(child_id)
                    if child_pointer not in state.variables:
                        state.variables.append(child_pointer)
                        state.context += f"\n{child_pointer}: {child_content}"
                    
                    findings_stored += 1
                    logger.info(
                        "child_finding_stored",
                        pointer=child_pointer,
                        node_id=child_id,
                        header=child_node.header,
                        parent=header,
                        content_length=len(child_content),
                    )
                    if findings_stored >= max_findings:
                        break
                
                state.visited_nodes.append(node_id)
                if findings_stored >= max_findings:
                    break
                # Fall through to also store parent content if useful
            
            # For sibling sections, use lower content threshold (they provide context)
            min_length = 20 if is_sibling else self.config.rlm_min_content_length
            if not content or len(content) < min_length:
                logger.debug(
                    "section_skipped_short_content",
                    node_id=node_id,
                    header=header,
                    content_length=len(content),
                    min_required=min_length,
                )
                continue
            
            # Create pointer name
            pointer_name = generate_pointer_name(header)
            findings_stored += 1
            
            # Store DIRECTLY in navigator's variable_store (not REPL's)
            self.variable_store.assign(pointer_name, content, node_id)
            
            # Track in state
            if node_id not in state.visited_nodes:
                state.visited_nodes.append(node_id)
            if pointer_name not in state.variables:
                state.variables.append(pointer_name)
                state.context += f"\n{pointer_name}: {content}"
            
            logger.info(
                "deterministic_finding_stored",
                pointer=pointer_name,
                node_id=node_id,
                header=header,
                score=score,
                content_length=len(content),
            )
            
            # Stop after finding enough relevant sections
            if findings_stored >= max_findings:
                break
        
        logger.info(
            "deterministic_nav_complete",
            findings_stored=findings_stored,
            variables=state.variables,
        )
        
        return state
    
    def _validate_section_relevance(
        self, 
        query: str, 
        keywords: list[str], 
        search_result: dict
    ) -> bool:
        """
        Validate that a section is actually relevant to the query.
        
        This prevents storing content from irrelevant sections just because
        they matched a keyword tangentially.
        """
        header = search_result.get("header", "").lower()
        node_id = search_result.get("node_id", "")
        score = search_result.get("score", 0)
        
        # High score sections are likely relevant
        if score >= 10.0:
            return True
        
        # Check if query terms appear in header (strong signal)
        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 3]
        header_matches = sum(1 for w in query_words if w in header)
        
        if header_matches >= 2:
            return True
        
        # Check if keywords match header
        keyword_matches = sum(1 for k in keywords if k.lower() in header)
        if keyword_matches >= 1:
            return True
        
        # Check content for keyword density
        content = self.nav_repl.kv_store.get(node_id) or ""
        if content:
            content_lower = content.lower()
            keyword_count = sum(content_lower.count(k.lower()) for k in keywords if k)
            # Require reasonable keyword density
            if keyword_count >= 2:
                return True
        
        # Low relevance
        return False
    
    def _rlm_navigate(self, state: RLMAgentState) -> RLMAgentState:
        """
        RLM-style navigation: LLM writes Python code to search the document.
        
        Instead of keyword matching, the LLM generates search code that:
        1. Uses regex patterns to find relevant content
        2. Navigates the document tree programmatically
        3. Stores findings as it discovers them
        4. Signals when ready to synthesize an answer
        
        This is the core RLM pattern - treating the document as an environment
        the LLM can explore through code execution.
        """
        if not self._llm_fn:
            logger.warning("No LLM function set for RLM navigation")
            return state
        
        # Reset and configure NavigationREPL
        self.nav_repl.reset()
        self.nav_repl.set_query(state.current_sub_question or state.question)
        self.nav_repl.extracted_keywords = state.extracted_keywords or []  # Sync for learned pattern boosting
        
        # Get the system prompt
        system_prompt = self.nav_repl.get_system_prompt()
        
        state.add_trace(
            "rlm_navigation",
            "Starting RLM navigation with code generation",
            {"query": state.question, "max_iterations": self.config.rlm_max_search_iterations},
        )
        
        # Track errors for feedback to LLM
        last_error = None
        last_code = None
        
        # Iterative code generation loop
        consecutive_empty = 0
        for iteration in range(self.config.rlm_max_search_iterations):
            # Get current REPL state
            repl_state = self.nav_repl.get_state()
            current_node = self.skeleton.get(repl_state["current_node_id"])
            findings_count = len(repl_state.get("findings", []))

            # Hard exit: if we've had 6+ consecutive iterations with no findings,
            # the document genuinely doesn't contain relevant content
            if consecutive_empty >= 6 and findings_count == 0:
                logger.info(
                    "rlm_exhausted_search",
                    iterations=iteration,
                    consecutive_empty=consecutive_empty,
                )
                break
            
            # Log iteration start with full state
            logger.info(
                "rlm_iteration_start",
                iteration=iteration,
                current_node=current_node.header if current_node else "root",
                current_node_id=repl_state["current_node_id"],
                findings_count=findings_count,
                visited_nodes=len(state.visited_nodes),
                nav_history=len(repl_state.get("navigation_history", [])),
            )
            
            # Build error feedback if there was an error
            error_feedback = ""
            if last_error:
                error_feedback = f"""
PREVIOUS ERROR: Your last code failed:
```python
{last_code[:300] if last_code else ""}
```
Error: {last_error}

FIX REQUIRED:
1. ALWAYS define variables before using them
2. Correct pattern: `matches = search_tree(pattern)` THEN `if matches:`
3. Do NOT reference 'matches' unless you just called search_tree/search_children
4. Try again with COMPLETE, working code
"""
            
            # Check if current findings are too broad and provide guidance
            broad_content_guidance = ""
            findings = repl_state.get("findings", [])
            if findings:
                # Check if any findings are from broad sections
                for finding in findings:
                    source_id = finding.get("source_node_id", "")
                    source_node = self.skeleton.get(source_id)
                    if source_node:
                        content = self.nav_repl.kv_store.get(source_id) or ""
                        if len(content) > self.config.rlm_max_content_for_specific:
                            broad_content_guidance = f"""
NOTE: You stored content from '{source_node.header}' which is a BROAD section ({len(content)} chars).
This section likely contains everything but isn't specific enough.
TRY: Navigate to its children for more focused content, or search within this section.
"""
                            break
            
            # If no findings yet and we've done iterations, encourage persistence
            persistence_guidance = ""
            if iteration > 0 and not findings:
                persistence_guidance = """
NOTE: No findings stored yet. Keep searching!
- Try different search patterns (synonyms, related terms)
- Search children of sections that had matches
- Look for sections with matching HEADERS (more specific than content matches)
"""
            
            # Build prompt for code generation
            code_gen_prompt = f"""{system_prompt}

CURRENT STATE:
- Location: {current_node.header if current_node else "root"}
- Children: {len(current_node.child_ids) if current_node else 0} sections
- Findings so far: {len(findings)}
- Navigation history: {len(repl_state.get("navigation_history", []))} moves
- Iteration: {iteration + 1} of {self.config.rlm_max_search_iterations}
{error_feedback}{broad_content_guidance}{persistence_guidance}
QUERY: {state.current_sub_question or state.question}
{self._build_identity_block()}

Your task: Write Python code to search for information relevant to the query.
Use the available functions to search content, navigate to relevant sections, 
and store important findings.

IMPORTANT:
- Write COMPLETE Python code - define all variables before using them
- Use search_tree() or search_children() to find relevant sections
- ALWAYS assign the result to a variable: `matches = search_tree(...)`
- Results are sorted by SCORE - higher scores mean MORE SPECIFIC sections
- PREFER sections with header matches over content-only matches
- If a section has children, DRILL DOWN into them for more specific content
- Use navigate_to() to move to promising sections  
- Use get_current_content() after navigating to get full text
- Use store_finding() when you find relevant information
- Call ready_to_synthesize() when you have enough information
- Keep searching until you find SPECIFIC content that answers the query

Generate Python code only, no explanations:
```python
"""
            
            try:
                response = self._llm_fn(code_gen_prompt)
                
                # Extract code from response
                code = self._extract_code_from_response(response)
                
                # Log the generated code (full code for debugging)
                logger.info(
                    "rlm_code_generated",
                    iteration=iteration,
                    code_length=len(code) if code else 0,
                    code_full=code if code and len(code) < 2000 else (code[:1000] + "... [truncated]" if code else "None"),
                )
                
                if not code:
                    logger.warning(f"RLM iteration {iteration}: No code generated")
                    state.add_trace(
                        "rlm_navigation",
                        f"Iteration {iteration}: No code generated",
                        {"response_preview": response[:200]},
                    )
                    last_error = "No valid Python code was generated"
                    last_code = response[:200]
                    continue
                
                # Execute the code
                exec_result = self.nav_repl.execute(code)
                
                # Log execution result in detail
                logger.info(
                    "rlm_code_executed",
                    iteration=iteration,
                    success=exec_result.get("success", False),
                    error=exec_result.get("error"),
                    output_preview=str(exec_result.get("output", ""))[:200],
                    current_node=exec_result.get("current_node"),
                    findings_count=exec_result.get("findings_count", 0),
                )
                
                # Track code and any errors for next iteration
                last_code = code
                if exec_result.get("error"):
                    last_error = exec_result["error"]
                else:
                    last_error = None  # Clear error on success

                # Track consecutive iterations with no findings
                current_findings = len(self.nav_repl._get_findings())
                if current_findings == 0:
                    consecutive_empty += 1
                else:
                    consecutive_empty = 0
                
                state.add_trace(
                    "rlm_navigation",
                    f"Iteration {iteration}: Code executed",
                    {
                        "code_preview": code[:200],
                        "output_preview": str(exec_result.get("output", ""))[:200],
                        "error": exec_result.get("error"),
                    },
                )
                
                # Sync REPL navigation with state.visited_nodes
                # Include both navigation_history (previous nodes) and current location
                repl_state = self.nav_repl.get_state()
                current_loc = repl_state.get("current_node_id", "")
                if current_loc and current_loc not in state.visited_nodes:
                    state.visited_nodes.append(current_loc)
                for visited_id in repl_state.get("navigation_history", []):
                    if visited_id not in state.visited_nodes:
                        state.visited_nodes.append(visited_id)
                
                # Check if ready to synthesize
                # BUT don't allow early exit if there are no findings and there was an error
                if self.nav_repl.is_ready_to_synthesize():
                    findings_count = len(self.nav_repl._get_findings())
                    nodes_visited = len(state.visited_nodes)
                    
                    # Don't allow premature exit if no findings and had errors
                    if findings_count == 0 and last_error:
                        logger.warning(
                            "ignoring_premature_ready",
                            reason="No findings stored but had errors",
                            iteration=iteration,
                        )
                        # Reset the flag and continue searching
                        self.nav_repl._ready_to_synthesize = False
                        continue
                    
                    # Don't allow exit on first iteration if no findings (likely incomplete)
                    if findings_count == 0 and iteration == 0:
                        logger.warning(
                            "ignoring_premature_ready",
                            reason="No findings on first iteration",
                        )
                        self.nav_repl._ready_to_synthesize = False
                        continue
                    
                    # Enforce minimum exploration requirements
                    # Primary requirement: must have enough findings
                    # Secondary: if no findings yet, must explore more nodes
                    needs_more_findings = findings_count < self.config.min_findings_required
                    needs_more_nodes = findings_count == 0 and nodes_visited < self.config.min_nodes_to_visit

                    # Allow exit after 3+ iterations even with 0 findings -
                    # the document genuinely may not contain relevant content
                    exhausted_search = iteration >= 3 and findings_count == 0
                    
                    if (needs_more_findings or needs_more_nodes) and not exhausted_search:
                        logger.debug(
                            "forcing_more_exploration",
                            nodes=nodes_visited,
                            min_nodes=self.config.min_nodes_to_visit,
                            findings=findings_count,
                            min_findings=self.config.min_findings_required,
                        )
                        self.nav_repl._ready_to_synthesize = False
                        continue
                    
                    logger.info(f"RLM navigation complete after {iteration + 1} iterations")
                    break
                    
            except Exception as e:
                logger.error(f"RLM navigation error: {e}")
                state.add_trace("rlm_navigation", f"Error: {str(e)}", {})
                last_error = str(e)
                last_code = code if 'code' in dir() else None
        
        # Process findings into state
        # findings is a list of dicts: [{"name": ..., "content": ..., "source_node_id": ...}, ...]
        findings = self.nav_repl._get_findings()
        
        if findings:
            state.add_trace(
                "rlm_navigation",
                f"Found {len(findings)} relevant pieces of information",
                {"finding_names": [f.get("name", "") for f in findings]},
            )
            
            # Add findings to context and variables
            # NOTE: nav_repl._store_finding() already stores FULL content in variable_store
            # The findings list only contains a truncated preview - do NOT overwrite!
            for finding in findings:
                name = finding.get("name", "")
                node_id = finding.get("source_node_id")
                
                # The pointer name is already correct from nav_repl
                pointer = name if name.startswith("$") else generate_pointer_name(name)
                
                # Add pointer to variables if not already added (node may already be in visited_nodes)
                if pointer not in state.variables:
                    state.variables.append(pointer)
                    
                    # Also ensure node is in visited_nodes
                    if node_id and node_id not in state.visited_nodes:
                        state.visited_nodes.append(node_id)
                    
                    # Get FULL content from variable_store (already stored by nav_repl)
                    # Do NOT use finding.get("content") - that's truncated!
                    full_content = self.variable_store.resolve(pointer)
                    if not full_content:
                        # Fallback: get directly from kv_store
                        full_content = self.nav_repl.kv_store.get(node_id) or ""
                        if full_content:
                            self.variable_store.assign(pointer, full_content, node_id)
                    
                    # Add preview to context for synthesis prompt
                    preview = full_content[:500] if full_content else finding.get("content", "")
                    state.context += f"\n{pointer}: {preview}"
                    
                    logger.info(
                        "variable_assigned",
                        pointer=pointer,
                        chars=len(full_content) if full_content else 0,
                        source=node_id,
                    )
        else:
            state.add_trace(
                "rlm_navigation",
                "No findings from RLM navigation",
                {},
            )
        
        return state
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code block
        code_match = re.search(r"```(?:python)?\s*(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # If no code block, try to find code-like content
        lines = response.strip().split("\n")
        code_lines = []
        in_code = False
        
        for line in lines:
            # Skip markdown and explanatory text
            if line.startswith("#") and not line.startswith("# "):
                continue
            if any(line.strip().startswith(kw) for kw in ["search_", "navigate_", "store_", "ready_", "get_", "print(", "for ", "if ", "while ", "result"]):
                in_code = True
            if in_code:
                code_lines.append(line)
        
        return "\n".join(code_lines).strip() if code_lines else response.strip()
    
    def _tot_evaluate_children(
        self,
        state: RLMAgentState,
        children: list[SkeletonNode],
    ) -> list[SkeletonNode]:
        """Use Tree of Thoughts to evaluate children."""
        if not self._llm_fn or not children:
            return children[:self.config.top_k]
        
        # Format children for evaluation, including table hints
        def _format_child(c):
            table_hint = ""
            if hasattr(c, "metadata") and c.metadata and c.metadata.get("has_tables"):
                table_hint = " [HAS TABLES]"
            return f"  - [{c.node_id}] {c.header}{table_hint}: {c.summary[:150]}"
        
        children_text = "\n".join(_format_child(c) for c in children)
        
        current_node = self.skeleton.get(state.current_node_id or self.root_id)
        current_summary = f"{current_node.header}: {current_node.summary}" if current_node else ""
        
        tot_prompt = f"""You are evaluating document sections for relevance.

Current location: {current_summary}

Children sections:
{children_text}

Query: {state.current_sub_question or state.question}

TASK: Evaluate each child's probability (0.0-1.0) of containing relevant information.

OUTPUT FORMAT (JSON):
{{
    "evaluations": [
        {{"node_id": "...", "probability": 0.85, "reasoning": "..."}}
    ],
    "selected_nodes": ["node_id_1", "node_id_2"],
    "is_dead_end": false
}}

JSON only:"""
        
        try:
            import json
            
            response = self._llm_fn(tot_prompt)
            json_match = re.search(r'\{[\s\S]*\}', response)
            
            if json_match:
                result = json.loads(json_match.group())
                selected_ids = result.get("selected_nodes", [])
                
                # Map back to nodes
                selected = [c for c in children if c.node_id in selected_ids]
                
                if not selected and not result.get("is_dead_end", False):
                    # Fallback: take top-k by probability
                    evaluations = result.get("evaluations", [])
                    sorted_evals = sorted(
                        evaluations,
                        key=lambda x: x.get("probability", 0),
                        reverse=True,
                    )
                    top_ids = [e["node_id"] for e in sorted_evals[:self.config.top_k]]
                    selected = [c for c in children if c.node_id in top_ids]
                
                return selected
                
        except Exception as e:
            logger.warning("tot_evaluation_failed", error=str(e))
        
        # Fallback: return first top_k children
        return children[:self.config.top_k]
    
    def _do_backtrack(self, state: RLMAgentState) -> RLMAgentState:
        """Backtrack to previous node."""
        if state.backtrack_stack:
            parent_id = state.backtrack_stack.pop()
            state.dead_ends.append(state.current_node_id or "")
            state.current_node_id = parent_id
            
            state.add_trace(
                "navigation",
                f"Backtracked to {parent_id}",
                {"from": state.current_node_id},
            )
        else:
            state.current_node_id = None
        
        return state
    
    def _phase_synthesize(self, state: RLMAgentState) -> RLMAgentState:
        """Phase 4: Synthesize answer from variables."""
        state.add_trace("synthesis", "Synthesizing answer from variables")
        
        if not state.variables:
            state.answer = "No relevant content found in the document."
            state.confidence = 0.0
            return state
        
        # Collect all variable content, labelling each with its real
        # section header so the LLM can cite by section name.
        # Also extract explicit paragraph numbers from the content so the
        # LLM can cite them accurately instead of guessing.
        contents = []
        for pointer in state.variables:
            content = self.variable_store.resolve(pointer)
            if content:
                # Resolve the original section header from the skeleton
                section_label = pointer
                stored = self.variable_store._metadata.get(pointer)
                if stored:
                    node = self.skeleton.get(stored.source_node_id)
                    if node:
                        # Clean header: strip leading paragraph numbers that
                        # leaked into the header text (e.g. "11 In particular…")
                        clean_hdr = re.sub(
                            r"^\s*\[?\d{1,4}\]?[\.\)\s]+", "", node.header
                        ).strip()
                        if not clean_hdr:
                            clean_hdr = node.header.strip()
                        # Truncate overly long headers
                        if len(clean_hdr) > 80:
                            clean_hdr = clean_hdr[:77] + "..."
                        section_label = f"Section: {clean_hdr}"

                # Extract explicit paragraph/numbered markers from content.
                # Common legal formats: "11  text", "[11] text", "11. text"
                # Limit to 1-3 digits to avoid matching years (2012, etc.)
                para_nums: list[str] = []
                for m in re.finditer(
                    r"(?:^|\n)\s*\[?(\d{1,3})\]?(?:\.|\s{2,})", content
                ):
                    num = m.group(1)
                    if num not in para_nums:
                        para_nums.append(num)

                if para_nums:
                    section_label += f"  [contains ¶{', ¶'.join(para_nums)}]"

                contents.append(f"=== {section_label} ===\n{content}")
        
        # Auto-inject structured table data for visited nodes
        if self.tables and state.visited_nodes:
            visited_set = set(state.visited_nodes)
            relevant_tables = [
                t for t in self.tables
                if getattr(t, "node_id", None) in visited_set
            ]
            if relevant_tables:
                table_parts = []
                for t in relevant_tables[:10]:
                    title = getattr(t, "title", "") or f"Table from {t.node_id}"
                    headers = getattr(t, "headers", [])
                    data = getattr(t, "data", [])
                    header_row = " | ".join(headers) if headers else ""
                    rows = "\n".join(" | ".join(row) for row in data[:50])
                    table_parts.append(f"--- {title} ---\n{header_row}\n{rows}")
                if table_parts:
                    contents.append(
                        "=== Structured Table Data ===\n" + "\n\n".join(table_parts)
                    )
        
        context_text = "\n\n".join(contents)
        
        if not self._llm_fn:
            state.answer = context_text
            state.confidence = 0.5
            return state
        
        # Detect question type from metadata
        options = state.metadata.get("options")
        metadata = state.metadata
        requires_arithmetic = metadata.get("requires_arithmetic", False)
        answer_type = metadata.get("answer_type", "")
        is_arithmetic = (
            requires_arithmetic
            or answer_type in ("arithmetic", "counting", "multi-span")
            or "hybrid-arithmetic" in metadata.get("reasoning_type", "")
            or "hybrid-counting" in metadata.get("reasoning_type", "")
        )

        # --- Vision augmentation: collect image bytes from visited nodes ---
        image_bytes = None
        if state.visited_nodes:
            for node_id in state.visited_nodes:
                try:
                    if hasattr(self.kv_store, "get_image"):
                        img = self.kv_store.get_image(node_id)
                        if img:
                            image_bytes = img
                            break
                except Exception:
                    pass

        if options:
            options_text = "\n".join(f"{chr(65+i)}. {opt}" for i, opt in enumerate(options))
            synthesis_prompt = f"""Based on the context, answer this multiple-choice question.

Question: {state.question}

Options:
{options_text}

Context:
{context_text}

Respond with ONLY the letter and full option text (e.g., "A. [option text]"):"""
        elif is_arithmetic:
            synthesis_prompt = f"""Answer the question using the provided context.

RULES:
1. PREFER exact numeric values from tables over approximate values from narrative text.
2. If values are in "thousands" or "millions", convert to the unit the question requests.
3. You must NEVER do arithmetic in your head. ALL computation must be done via Python code.
4. If the question asks for a year or a name (no math needed), you may answer directly.
5. GROWTH RATE / PROJECTION RULE: When the question asks about a "growth rate", "current rate", "continues to grow", or future projections, you MUST:
   a) Extract ALL available data points across multiple years.
   b) Compute the Compound Annual Growth Rate (CAGR) from the data.
   c) Apply the CAGR for the projection period.

DECIDE: Does this question require mathematical computation?

OPTION A — If YES:
You MUST write Python code. Do NOT compute the answer yourself.
1. Extract the exact values from the context.
2. Write a Python code block that computes the answer.
3. The code MUST call print() with the final numeric result.

Format:
EXTRACTED VALUES:
- value_1 = <number> (source: "<quote>")

CODE:
```python
value_1 = <number>
result = <computation>
print(result)
```

OPTION B — If NO (direct lookup, comparison, "which year", etc.):
Output: NO_COMPUTE: <direct answer>

Question: {state.question}

Context:
{context_text}

Response (use CODE or NO_COMPUTE):"""
        else:
            # Build entity context from KG if available
            entity_context_text = ""
            if self.knowledge_graph and state.metadata.get("entities_found"):
                entity_lines = []
                for entity in state.metadata["entities_found"][:5]:
                    rels = self.knowledge_graph.get_entity_relationships(entity.id)
                    if rels:
                        for rel in rels[:10]:
                            entity_lines.append(
                                f"- {rel.source_id} → {rel.type.value} → {rel.target_id}"
                            )
                    co = self.knowledge_graph.get_entities_mentioned_together(entity.id)
                    if co:
                        related = [e.canonical_name for e, _ in co[:5]]
                        entity_lines.append(
                            f"- {entity.canonical_name} is related to: {', '.join(related)}"
                        )
                if entity_lines:
                    entity_context_text = (
                        "\n\nKnowledge Graph Context (entity relationships from this document):\n"
                        + "\n".join(entity_lines)
                    )

            use_short = metadata.get("use_short_answer", False)
            if use_short:
                format_block = """CRITICAL FORMAT RULE:
Line 1 MUST be ONLY the bare minimal answer — key words/phrases only, NO full sentences.
If the answer cannot be found, write exactly: "Unanswerable"
Line 2+: Optional supporting evidence with citations."""
            else:
                format_block = """FORMAT RULES:
- Start with the direct answer in 1-2 sentences. No preamble, no headers, no markdown formatting.
- If you quote from the document, cite the section header.
- Do NOT produce bullet lists, tables, or multi-section analyses unless the question asks for a list.
- Keep your total response under 200 words."""

            identity_block = self._build_identity_block()
            identity_section = f"\n{identity_block}\n" if identity_block else ""

            synthesis_prompt = f"""You have access to the following document sections. Answer the question using ONLY these sections.
{identity_section}
GROUNDING RULES:
1. Every claim MUST be supported by text from the sections below. Never infer or assume facts not explicitly stated.
2. Section headers ARE factual content — a section labelled "MAGISTRATES COURT of WESTERN AUSTRALIA" means that text appears in the document and can be stated as a fact.
3. You MAY combine information from adjacent or related sections.
4. Do NOT use any knowledge outside these sections. If the document discusses multiple cases or sub-matters, answer ONLY about the case the question asks about.
5. Give a DIRECT answer first. Do NOT hedge or say "cannot be determined" when the information IS present.
6. If the answer is a name, date, number, or short phrase — just state it.

{format_block}

Question: {state.question}

Document Sections:
{context_text}{entity_context_text}

Answer:"""
        
        logger.info(
            "synthesis_start",
            question=state.question,
            num_variables=len(state.variables),
            context_length=len(context_text),
            is_arithmetic=is_arithmetic,
        )
        
        try:
            # --- Multimodal synthesis: use image if available ---
            if image_bytes:
                try:
                    from rnsr.llm import get_llm
                    llm_obj = get_llm()
                    if hasattr(llm_obj, "complete_with_image"):
                        answer = str(llm_obj.complete_with_image(synthesis_prompt, image_bytes)).strip()
                        logger.info("multimodal_synthesis_used", question=state.question[:80])
                    else:
                        answer = self._llm_fn(synthesis_prompt).strip()
                except Exception:
                    answer = self._llm_fn(synthesis_prompt).strip()
            else:
                answer = self._llm_fn(synthesis_prompt).strip()
            
            state.answer = answer
            state.confidence = min(0.7, 0.3 + len(state.variables) * 0.1)
            
            # --- Arithmetic code execution pipeline ---
            if is_arithmetic and not options and state.answer:
                original_code = _extract_code_block(state.answer)
                computed = _try_compute_from_response(state.answer, context_text=context_text)
                if computed is not None:
                    if computed != _NO_COMPUTE_MARKER:
                        state.answer = computed
                        logger.info("arithmetic_code_execution_used", result=computed)
                        if original_code:
                            verified = _verify_and_rerun_formula(
                                state.question, original_code, state.answer,
                                context_text, self._llm_fn,
                            )
                            if verified is not None:
                                state.answer = verified
                else:
                    repl_answer = _try_repl_arithmetic(state.question, context_text, self._llm_fn)
                    if repl_answer is not None:
                        state.answer = repl_answer
                        logger.info("repl_arithmetic_fallback_used", result=repl_answer)
            
            # Normalize multiple choice answer
            if options:
                state.answer = self._normalize_mc_answer(state.answer, options)
            
            # Short-answer extraction for benchmark compatibility
            if not options and metadata.get("use_short_answer"):
                state.answer = _extract_first_answer_phrase(
                    state.answer, is_arithmetic=is_arithmetic,
                )
            
            logger.info(
                "synthesis_complete",
                question=state.question,
                answer_length=len(state.answer) if state.answer else 0,
                answer_preview=(state.answer or "")[:300],
                initial_confidence=state.confidence,
            )
            
            # Post-synthesis grounding check
            if self.config.enable_verification:
                grounded, issues = self._verify_answer_grounded(state.answer, context_text)
                if not grounded:
                    logger.warning("answer_grounding_issues", issues=issues)
                    state.confidence = max(0.3, state.confidence - 0.2)
                
        except Exception as e:
            logger.error("synthesis_failed", error=str(e))
            state.answer = f"Error during synthesis: {str(e)}"
            state.confidence = 0.0
        
        return state
    
    def _normalize_mc_answer(self, answer: str, options: list) -> str:
        """Normalize multiple choice answer to match option text."""
        answer_lower = answer.lower().strip()
        
        for i, opt in enumerate(options):
            letter = chr(65 + i)
            opt_lower = opt.lower()
            
            if (answer_lower.startswith(f"{letter.lower()}.") or
                answer_lower.startswith(f"{letter.lower()})") or
                opt_lower in answer_lower):
                return opt
        
        return answer
    
    # Phrases that always mean "I tried to answer but the relevant data
    # was missing from what I was given." These should trigger a fallback
    # retrieval pass even when the answer is otherwise long and confident
    # (e.g. the LLM admits the gap and then volunteers adjacent context).
    _INCONCLUSIVE_PHRASES_STRONG: tuple[str, ...] = (
        "provided sections do not contain",
        "sections do not contain this information",
        "do not contain this information",
        "do not explicitly state",
        "does not explicitly state",
        "is not explicitly stated",
        "is not provided in",
        "is not available in the provided",
        "cannot be determined from the provided",
        "cannot be determined from the document",
        "no relevant content found",
        "error during",
        "i cannot answer",
        "i don't have enough information",
        "the document does not provide",
        # Phrases observed in Pfizer 10-K PP&E miss (May 2026 large-doc run)
        # and other "I can see the section but the numbers aren't here" cases.
        "is not possible to determine",
        "it is not possible to determine",
        "not possible to answer",
        "only contains introductory text",
        "does not provide the actual",
        "does not contain the actual",
        "does not contain numerical",
        "does not contain the numerical",
        "no specific data",
        "no specific numbers",
        "no specific figure",
        "the relevant section was not found",
        "the relevant data is not present",
        "insufficient information",
        "insufficient context",
        # Phrases observed in Pfizer 10-K PP&E miss after header-match
        # fallback (May 2026): the synthesis acknowledges retrieving the
        # *header* but admits the *numbers* aren't in the chunks.
        "it cannot be determined",
        "it could not be determined",
        "cannot be determined if",
        "actual numerical values",
        "actual numerical value",
        "actual numbers are not",
        "are not included in the provided",
        "is not included in the provided",
        "not included in the provided text",
        "are not in the provided",
        "is not in the provided",
        "financial balances for",
        "the actual values",
    )

    def _answer_is_inconclusive(self, answer: str) -> bool:
        """Check if the synthesis answer indicates failure to find information.

        Long answers that *also* contain a strong "data not present"
        phrase still count as inconclusive, because the LLM is signalling
        that retrieval missed the relevant section even if it then
        volunteers tangential content.
        """
        if not answer:
            return True
        lower = answer.lower()
        if any(phrase in lower for phrase in self._INCONCLUSIVE_PHRASES_STRONG):
            return True
        return False
    
    def _refine_search_strategy(self, state: RLMAgentState) -> RLMAgentState:
        """Generate refined search patterns after an inconclusive answer.
        
        Uses the LLM to produce alternative regex patterns informed by
        what the previous attempt found (or didn't find).
        """
        state.add_trace("refinement", "Refining search strategy after inconclusive answer")
        
        # Reset navigation state for re-navigation but keep findings so far
        state.current_node_id = self.root_id
        state.nodes_to_visit = []
        state.iteration = 0
        
        if not self._llm_fn:
            return state
        
        visited_headers = []
        for nid in state.visited_nodes[:20]:
            node = self.skeleton.get(nid)
            if node:
                visited_headers.append(node.header)
        
        refine_prompt = f"""The previous search did not find enough information to answer the question.

Question: {state.question}

Previous answer attempt: {state.answer[:300] if state.answer else "No answer produced"}

Sections already visited:
{chr(10).join(f"- {h}" for h in visited_headers[:15])}

Generate 3-5 alternative regex search patterns to find the missing information.
Focus on:
- Synonyms or alternative terms for the key concepts
- Financial statement section names (e.g. balance sheet, income statement, PP&E schedule)
- Specific table headers or data labels

OUTPUT FORMAT (JSON):
{{"patterns": ["pattern1", "pattern2", ...], "reasoning": "brief explanation"}}

JSON only:"""
        
        try:
            import json as json_mod
            response = self._llm_fn(refine_prompt)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json_mod.loads(json_match.group())
                new_patterns = result.get("patterns", [])
                if new_patterns:
                    # Store as new sub-questions for the next navigate pass
                    state.sub_questions = new_patterns
                    state.pending_questions = new_patterns.copy()
                    state.current_sub_question = new_patterns[0]
                    logger.info(
                        "search_strategy_refined",
                        new_patterns=new_patterns,
                        reasoning=result.get("reasoning", ""),
                    )
        except Exception as e:
            logger.warning("search_refinement_failed", error=str(e))
        
        return state
    
    def _verify_answer_grounded(self, answer: str, context: str) -> tuple[bool, str]:
        """
        Verify that key claims in the answer are grounded in the source context.
        
        This prevents hallucination by checking that quoted text actually exists
        in the source material.
        
        Args:
            answer: The synthesized answer
            context: The source context used for synthesis
            
        Returns:
            Tuple of (is_grounded, issues_description)
        """
        import re
        
        # Strip markdown formatting from context for comparison
        # This handles cases like **$750,000 USD** matching $750,000 USD
        context_clean = re.sub(r'\*+', '', context)  # Remove markdown bold/italic
        context_clean = re.sub(r'_+', '', context_clean)  # Remove markdown underlines
        context_clean = re.sub(r'`+', '', context_clean)  # Remove code formatting
        context_lower = context_clean.lower()
        
        # Extract quoted text from answer (text in quotes)
        quotes = re.findall(r'"([^"]+)"', answer)
        
        # Also extract text that looks like specific claims (names, numbers, dates)
        # These patterns catch specific facts that should be verifiable
        specific_patterns = [
            r'\$[\d,]+(?:\.\d{2})?',  # Money amounts
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Dates
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',  # Written dates
        ]
        
        specific_claims = []
        for pattern in specific_patterns:
            specific_claims.extend(re.findall(pattern, answer))
        
        ungrounded_quotes = []
        ungrounded_claims = []
        
        # Check if quoted text exists in context (more lenient matching)
        for quote in quotes:
            quote_clean = quote.lower().strip()
            if len(quote_clean) > 10:  # Only check longer quotes
                # Check if core content exists (allow for partial matches)
                # Split into words and check if most words are present
                words = quote_clean.split()
                found_words = sum(1 for w in words if len(w) > 3 and w in context_lower)
                if found_words < len([w for w in words if len(w) > 3]) * 0.7:  # 70% threshold
                    ungrounded_quotes.append(quote[:50])
        
        # Check if specific claims exist in context
        for claim in specific_claims:
            claim_clean = claim.replace(',', '')  # Handle $750,000 vs $750000
            if claim_clean.lower() not in context_lower and claim.lower() not in context_lower:
                ungrounded_claims.append(claim)
        
        if ungrounded_quotes or ungrounded_claims:
            issues = []
            if ungrounded_quotes:
                issues.append(f"Ungrounded quotes: {ungrounded_quotes[:2]}")
            if ungrounded_claims:
                issues.append(f"Ungrounded claims: {ungrounded_claims[:2]}")
            
            logger.warning(
                "grounding_check_failed",
                ungrounded_quotes=len(ungrounded_quotes),
                ungrounded_claims=len(ungrounded_claims),
            )
            return False, "; ".join(issues)
        
        return True, ""
    
    def _phase_verify(self, state: RLMAgentState) -> RLMAgentState:
        """Phase 5: Verify the answer and REJECT if not reliable.
        
        Implements a multi-stage verification:
        1. Standard verification engine checks
        2. STRICT CRITIC LOOP: Harsh critic tries to disprove the answer
        3. Only accept if both verifications pass
        """
        state.add_trace("verification", "Verifying answer")
        
        # Collect evidence
        evidence = [
            self.variable_store.resolve(p) or ""
            for p in state.variables
        ]
        evidence_text = "\n\n---\n\n".join(e for e in evidence if e)
        
        # Stage 1: Standard verification
        result = self.verification_engine.verify_answer(
            state.question,
            state.answer or "",
            evidence,
        )
        
        state.verification_result = result
        
        # Get verification results
        is_valid = result.get("is_valid", False)  # Default to False, not True
        confidence = result.get("confidence", 0.0)
        issues = result.get("issues", [])
        
        # RAISED THRESHOLD: More strict to prevent hallucinations
        min_confidence_threshold = 0.7
        
        # Stage 2: STRICT CRITIC LOOP (Red Team verification)
        # Skip the expensive critic call entirely when standard verification
        # already shows very high confidence and the answer is substantive.
        critic_passed = True
        critic_result = None

        answer_is_unknown = self._answer_is_inconclusive(state.answer)
        skip_critic = (
            is_valid
            and confidence >= 0.95
            and not answer_is_unknown
        )
        if skip_critic:
            logger.info(
                "critic_skipped_high_confidence",
                confidence=confidence,
            )

        if is_valid and confidence >= min_confidence_threshold and state.answer and not skip_critic:
            state.add_trace("verification", "Running strict critic loop")
            
            # Use LLM function for strict verification
            llm_fn = None
            if self._llm_fn:
                llm_fn = self._llm_fn
            
            critic_result = strict_verify_answer(
                answer=state.answer,
                sources=evidence_text,
                question=state.question,
                llm_fn=llm_fn,
                max_unsupported_claims=0,  # Strict mode: no unsupported claims allowed
            )
            
            critic_passed = critic_result.verified
            
            # Cap confidence with the critic's confidence (e.g. "can't find"
            # answers get verified=True but confidence=0.3, preventing
            # downstream early termination on non-answers).
            if critic_result.confidence is not None:
                confidence = min(confidence, critic_result.confidence)
            
            if not critic_passed:
                logger.warning(
                    "critic_loop_rejected_answer",
                    unsupported_claims=critic_result.unsupported_claims,
                    rejection_reason=critic_result.rejection_reason,
                )
                state.add_trace(
                    "verification",
                    f"CRITIC REJECTED: {critic_result.rejection_reason}",
                    {
                        "unsupported_claims": critic_result.unsupported_claims,
                        "claims_analyzed": len(critic_result.claims_analyzed),
                    },
                )
                # Reduce confidence since critic found issues
                confidence = min(confidence * 0.5, 0.3)
        
        # Final decision: both stages must pass for full confidence.
        # When rejected, keep the original answer at low confidence so
        # cross-document synthesis can still use it as evidence.
        if not is_valid or confidence < min_confidence_threshold or not critic_passed:
            rejection_reason = []
            if not is_valid:
                rejection_reason.append("validation failed")
            if confidence < min_confidence_threshold:
                rejection_reason.append(f"low confidence ({confidence:.2f})")
            if not critic_passed and critic_result:
                rejection_reason.append(f"critic rejected: {critic_result.rejection_reason}")

            state.confidence = 0.15
            state.add_trace(
                "verification",
                f"Answer LOW-CONFIDENCE: {', '.join(rejection_reason)}",
                {"issues": issues, "rejected": True, "critic_passed": critic_passed},
            )
            logger.info(
                "answer_low_confidence",
                is_valid=is_valid,
                confidence=confidence,
                critic_passed=critic_passed,
                issues=issues,
            )
        else:
            # Accept the answer - both verification stages passed
            if result.get("improved_answer"):
                state.answer = result["improved_answer"]
            state.confidence = confidence
            state.add_trace(
                "verification",
                f"Answer ACCEPTED: valid={is_valid}, confidence={confidence:.2f}, critic_passed={critic_passed}",
                {"issues": issues},
            )
            logger.info(
                "answer_accepted",
                confidence=confidence,
                critic_verified=critic_passed,
            )
        
        return state


# =============================================================================
# Entity-Aware Query Decomposition
# =============================================================================


class EntityAwareDecomposer:
    """
    Enhances query decomposition by leveraging entity relationships
    from the knowledge graph.
    
    This allows the navigator to:
    1. Identify entities mentioned in the query
    2. Look up related entities via the knowledge graph
    3. Plan retrieval based on entity relationships
    4. Generate entity-focused sub-queries
    """
    
    def __init__(
        self,
        knowledge_graph=None,
        llm_fn: Callable[[str], str] | None = None,
        skeleton: dict | None = None,
        kv_store=None,
    ):
        """
        Initialize the entity-aware decomposer.
        
        Args:
            knowledge_graph: Optional knowledge graph for entity lookup.
            llm_fn: LLM function for query analysis.
            skeleton: Skeleton index for section header search.
            kv_store: KV store for reading section content.
        """
        self.kg = knowledge_graph
        self._llm_fn = llm_fn
        self._skeleton = skeleton
        self._kv_store = kv_store
    
    def set_llm_function(self, llm_fn: Callable[[str], str]) -> None:
        """Set the LLM function."""
        self._llm_fn = llm_fn
    
    def set_knowledge_graph(self, kg) -> None:
        """Set the knowledge graph."""
        self.kg = kg
    
    def decompose_with_entities(
        self,
        query: str,
        doc_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Decompose a query using entity awareness.
        
        When the query references a term like "primary applicant", the method:
        1. Looks up KG entities matching the name
        2. Also searches skeleton headers for matching sections
        3. Includes child nodes of matching sections so the navigator
           can reach leaf content (e.g. Personal Information under
           PRIMARY APPLICANT DETAILS)
        
        Args:
            query: The user's query.
            doc_id: Optional document ID to scope entity lookup.
            
        Returns:
            Dict with sub_queries, entities_found, and retrieval_plan.
        """
        result = {
            "original_query": query,
            "sub_queries": [query],
            "entities_found": [],
            "entity_nodes": {},
            "retrieval_plan": [],
        }
        
        if not self.kg:
            return result
        
        # Step 1: Extract entity names from query
        entity_names = self._extract_entity_names(query)
        
        if not entity_names:
            return result
        
        # Step 2: Look up entities in knowledge graph
        entities_found = []
        entity_nodes: dict[str, list[str]] = {}
        
        for name in entity_names:
            matches = self.kg.find_entities_by_name(name, fuzzy=True)
            
            # Filter by document if specified
            if doc_id:
                matches = [e for e in matches if doc_id in e.document_ids]
            
            for entity in matches:
                if entity not in entities_found:
                    entities_found.append(entity)
                    # Get nodes where this entity is mentioned
                    entity_nodes[entity.id] = list(entity.node_ids)
        
        # Step 2b: Also search skeleton headers for matching terms.
        # This catches cases like "primary applicant" → section header
        # "PRIMARY APPLICANT DETAILS" which contains children with the
        # actual details.
        if hasattr(self, "_skeleton") and self._skeleton:
            q_lower = query.lower()
            for node_id, node in self._skeleton.items():
                header_lower = node.header.lower()
                # Check if any extracted entity name appears in the header
                for name in entity_names:
                    if name.lower() in header_lower:
                        # Add this node and all its children to the retrieval plan
                        if node.child_ids:
                            for child_id in node.child_ids:
                                result["retrieval_plan"].append({
                                    "node_id": child_id,
                                    "reason": f"child of section '{node.header}' matching '{name}'",
                                })
                        else:
                            result["retrieval_plan"].append({
                                "node_id": node_id,
                                "reason": f"section header matches '{name}'",
                            })
                        break
        
        result["entities_found"] = entities_found
        result["entity_nodes"] = entity_nodes
        
        if not entities_found and not result["retrieval_plan"]:
            return result
        
        # Step 3: Get related entities and relationships
        related_entities = []
        relationships = []
        
        for entity in entities_found:
            # Get entities co-mentioned with this one
            co_mentions = self.kg.get_entities_mentioned_together(entity.id)
            for related, count in co_mentions[:5]:  # Top 5 co-mentions
                if related not in related_entities:
                    related_entities.append(related)
            
            # Get relationships
            rels = self.kg.get_entity_relationships(entity.id)
            relationships.extend(rels)
        
        # Step 4: Generate entity-focused sub-queries
        sub_queries = self._generate_entity_sub_queries(
            query, entities_found, related_entities, relationships
        )
        
        result["sub_queries"] = sub_queries
        result["related_entities"] = related_entities
        result["relationships"] = relationships
        
        # Step 5: Create retrieval plan
        result["retrieval_plan"] = self._create_retrieval_plan(
            query, entities_found, entity_nodes, relationships
        )
        
        logger.debug(
            "entity_aware_decomposition",
            entities=len(entities_found),
            sub_queries=len(sub_queries),
            relationships=len(relationships),
        )
        
        return result
    
    def _extract_entity_names(self, query: str) -> list[str]:
        """Extract potential entity names from a query."""
        entity_names = []
        
        # Use LLM if available
        if self._llm_fn:
            try:
                prompt = f"""Extract entity names (people, organizations, places, documents) from this query.

Query: {query}

Return as JSON array of names:
["Name 1", "Name 2"]

JSON only:"""
                
                response = self._llm_fn(prompt)
                json_match = re.search(r'\[[\s\S]*?\]', response)
                if json_match:
                    import json
                    entity_names = json.loads(json_match.group())
                    
            except Exception as e:
                logger.debug("entity_extraction_llm_failed", error=str(e))
        
        # Fallback: extract capitalized phrases
        if not entity_names:
            # Find capitalized words (likely proper nouns)
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
            entity_names = proper_nouns
        
        return entity_names
    
    def _generate_entity_sub_queries(
        self,
        query: str,
        entities: list,
        related: list,
        relationships: list,
    ) -> list[str]:
        """Generate sub-queries focused on entities."""
        sub_queries = []
        
        if not self._llm_fn:
            # Simple decomposition: one query per entity
            for entity in entities[:3]:
                sub_queries.append(
                    f"Find information about {entity.canonical_name}: {query}"
                )
            return sub_queries if sub_queries else [query]
        
        # Use LLM for intelligent decomposition
        try:
            entity_names = [e.canonical_name for e in entities]
            related_names = [e.canonical_name for e in related[:5]]
            
            rel_descriptions = []
            for rel in relationships[:10]:
                rel_descriptions.append(
                    f"- {rel.source_id} {rel.type.value} {rel.target_id}"
                )
            
            prompt = f"""Decompose this query into focused sub-queries based on the entities.

IMPORTANT: Stay focused on the ORIGINAL question. Do NOT expand into unrelated aspects of the entities. If the question is a simple factual lookup, return it as-is or with minimal decomposition (1 sub-query). Every sub-query MUST help answer the original question — do not generate broad exploratory questions about the entities.

Query: {query}

Key entities found: {', '.join(entity_names)}
Related entities: {', '.join(related_names)}

Known relationships:
{chr(10).join(rel_descriptions) if rel_descriptions else '(none)'}

Generate 1-3 focused sub-queries that directly address the original question. Fewer is better.

Return as JSON:
{{"sub_queries": ["query 1", "query 2"]}}

JSON only:"""
            
            response = self._llm_fn(prompt)
            json_match = re.search(r'\{[\s\S]*?\}', response)
            if json_match:
                import json
                result = json.loads(json_match.group())
                sub_queries = result.get("sub_queries", [])
                
        except Exception as e:
            logger.debug("sub_query_generation_failed", error=str(e))
        
        return sub_queries if sub_queries else [query]
    
    def _create_retrieval_plan(
        self,
        query: str,
        entities: list,
        entity_nodes: dict[str, list[str]],
        relationships: list,
    ) -> list[dict[str, Any]]:
        """Create a retrieval plan based on entities."""
        plan = []
        
        # Priority 1: Nodes with direct entity mentions
        priority_nodes = set()
        for entity in entities:
            nodes = entity_nodes.get(entity.id, [])
            for node_id in nodes:
                priority_nodes.add(node_id)
                plan.append({
                    "node_id": node_id,
                    "priority": 1,
                    "reason": f"Contains {entity.canonical_name}",
                    "entity_id": entity.id,
                })
        
        # Priority 2: Nodes involved in relationships
        for rel in relationships:
            if rel.source_type == "node" and rel.source_id not in priority_nodes:
                plan.append({
                    "node_id": rel.source_id,
                    "priority": 2,
                    "reason": f"Related via {rel.type.value}",
                    "relationship_id": rel.id,
                })
            if rel.target_type == "node" and rel.target_id not in priority_nodes:
                plan.append({
                    "node_id": rel.target_id,
                    "priority": 2,
                    "reason": f"Related via {rel.type.value}",
                    "relationship_id": rel.id,
                })
        
        # Sort by priority
        plan.sort(key=lambda x: x["priority"])
        
        return plan


# =============================================================================
# Factory Function
# =============================================================================


def create_rlm_navigator(
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    config: RLMConfig | None = None,
    knowledge_graph=None,
) -> RLMNavigator:
    """
    Create an RLM Navigator instance.
    
    Args:
        skeleton: Skeleton index.
        kv_store: KV store with full content.
        config: Optional configuration.
        knowledge_graph: Optional knowledge graph for entity-aware queries.
        
    Returns:
        Configured RLMNavigator.
        
    Example:
        from rnsr import ingest_document, build_skeleton_index
        from rnsr.agent.rlm_navigator import create_rlm_navigator, RLMConfig
        from rnsr.indexing.knowledge_graph import KnowledgeGraph
        
        result = ingest_document("contract.pdf")
        skeleton, kv_store = build_skeleton_index(result.tree)
        
        # With knowledge graph for entity-aware queries
        kg = KnowledgeGraph("./data/kg.db")
        
        # With custom config
        config = RLMConfig(
            max_recursion_depth=3,
            enable_pre_filtering=True,
            enable_verification=True,
        )
        
        navigator = create_rlm_navigator(skeleton, kv_store, config, kg)
        result = navigator.navigate("What are the liability terms?")
        print(result["answer"])
    """
    nav = RLMNavigator(skeleton, kv_store, config, knowledge_graph)
    
    # Configure LLM
    try:
        from rnsr.llm import get_llm
        llm = get_llm()
        nav.set_llm_function(lambda p: str(llm.complete(p)))
    except Exception as e:
        logger.warning("llm_config_failed", error=str(e))
    
    return nav


def run_rlm_navigator(
    question: str,
    skeleton: dict[str, SkeletonNode],
    kv_store: KVStore,
    config: RLMConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the RLM Navigator on a question.
    
    Convenience function that creates and runs the navigator.
    
    Args:
        question: The user's question.
        skeleton: Skeleton index.
        kv_store: KV store.
        config: Optional configuration.
        metadata: Optional metadata.
        
    Returns:
        Dict with answer, confidence, trace.
    """
    navigator = create_rlm_navigator(skeleton, kv_store, config)
    return navigator.navigate(question, metadata)
