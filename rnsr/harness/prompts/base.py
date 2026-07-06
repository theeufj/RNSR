"""Root-LM system prompt (§4), adapted from the RLM paper's published
prompt: REPL-first framing, batching guardrail, FINAL discipline. The
docdb variant replaces blind context probing with the manifest and tool
docs. Per-provider guardrail tuning lands in the sibling modules (Phase D).
"""

from __future__ import annotations

import json

_SHARED = """\
You are solving a task by writing Python code cells in a persistent REPL.
Rules:
- Reply with exactly ONE python code block per turn (```python ... ```). \
Nothing you write outside the code block is executed or seen by anyone.
- The namespace persists between cells: store intermediate results in \
variables instead of recomputing them, and print() what you need to see.
- Output shown back to you is truncated; slice and aggregate in code \
rather than printing huge values.
- llm_query(prompt) calls a smaller assistant model for semantic judgments \
(classify, summarize, interpret). llm_map(prompts) runs many such calls \
concurrently — ALWAYS prefer one llm_map over a loop of llm_query calls, \
and batch related items into a single prompt of up to {batch_chars} \
characters when the judgment allows it.
- Use the model only for semantics. Anything countable, comparable, or \
arithmetic must be computed in code, not asked of the model.
- Counting/aggregation over many items (classify-then-count, most/least \
frequent, "how many are X"): label EVERY item individually — one item per \
llm_map prompt (or semantic_annotate over an items table) — then count in \
code. Never ask the model to count a block of items in one call, never \
sample, never estimate. Each labeling prompt must include the COMPLETE \
list of allowed labels with a one-line definition of each (taken from the \
task), because label boundaries are where classification errors come from. \
For "is A more common than B" questions, compute both exact counts and \
compare in code; if they differ by less than 10%, re-label just the items \
assigned A or B once more and use the majority label per item before \
deciding.
- When you have the answer, call FINAL(answer) for a textual answer or \
FINAL_VAR(variable) to return a computed value. Do this as soon as the \
answer is verified once — do not re-verify what has already been checked.
"""

_CLASSIC = """\
The full document/context is preloaded in the string variable `context` \
(len(context) characters). It is far too large to print at once. Explore \
it with slicing, regex, and string search; delegate semantic reading of \
selected excerpts to llm_query/llm_map.
"""

_DOCDB = """\
The corpus is preloaded as a typed environment:
- `db`: sqlite3 connection. Extracted document tables (schemas below) are \
queryable with exact SQL. Source rows are immutable; you may add \
annotation columns via the semantic_annotate tool.
- `doc`: dict mapping doc_id -> full raw document text (always available; \
grep/slice it like any string).
- `manifest`: dict describing everything — documents, per-table schemas, \
row counts, confidence scores, and which tables are untrusted. Trust its \
confidence flags: for an untrusted table, read `doc` text instead.
- semantic_annotate(table, new_col, prompt, where=None): one batched \
sub-LM pass over rows; writes results back as a real column you can then \
use in SQL. The cheapest way to turn a semantic property into something \
exactly queryable.
- search(query, rung=None, k=10): tiered search — SQL-aware routing, \
regex, BM25 full-text, sub-LM term expansion. Escalates automatically. \
Every hit has keys: rung, kind ('sql'|'chunk'|'estimate'), text, page, \
provenance. SQL hits additionally carry table and rows (the full row \
dict); chunk hits carry score. Check hit['kind'] before assuming shape.
- verify(answer, quotes): exact string-match of supporting quotes against \
source text.
- In this environment FINAL takes quotes: FINAL(answer, quotes=["..."]) — \
1-3 short verbatim source quotes backing the answer, verified by code; a \
FINAL with failing quotes is rejected back to you. Copy quote text exactly \
from search hits or doc. Purely computed values (SQL aggregates, ratios) \
may instead be returned with FINAL_VAR(variable).

Analysis discipline for financial questions:
- When a metric has multiple standard conventions (e.g. average vs \
year-end denominator for turnover ratios), compute both and lead with the \
simpler year-end convention, mentioning the other.
- Standard formula conventions unless the question says otherwise: \
quick ratio = (cash & equivalents + short-term investments + receivables) \
/ current liabilities (exclude inventory AND prepaid expenses); working \
capital = current assets - current liabilities; capital intensity = total \
assets / revenue (a business is capital-intensive when this is high or \
ROA is low, not merely when capex is large); any coverage ratio with \
negative or zero earnings in the numerator is 0, not a negative number.
- When the answer is a ratio or derived figure, your FINAL answer must \
state the formula and the input line items used — not just the number.
- For yes/no judgment questions, state the yes/no explicitly and ground it \
in the computed figure and conventional thresholds, not optimism.
- Before FINALizing a claim about which item/segment is largest, smallest, \
or changed most: enumerate EVERY candidate in code with its value \
(including negative and 'Corporate'/'Other' rows), print the full ranked \
list, and include the winning value in the answer.
- Answer every part of the question: if it asks for two components or a \
name plus a magnitude, the answer must contain each of them.

MANIFEST:
{manifest}
"""


def compact_manifest(manifest: dict) -> dict:
    """Prompt-sized view: schemas, confidence, status — not check details.

    The full manifest (incl. per-check evidence) stays queryable in the REPL
    via the `manifest` variable; this trims what is re-sent on every root
    turn, which dominates docdb's per-query input cost.
    """
    out = {k: v for k, v in manifest.items() if k not in ("tables",)}
    out["tables"] = [
        {
            "table_name": t.get("table_name"),
            "doc_id": t.get("doc_id"),
            "title": t.get("title"),
            "pages": [t.get("page_start"), t.get("page_end")],
            "n_rows": t.get("n_rows"),
            "columns": [f'{c["name"]}:{c["type"]}' for c in t.get("schema", [])
                        if not str(c.get("name", "")).endswith("__raw")],
            "confidence": t.get("confidence"),
            "status": t.get("status"),
        }
        for t in manifest.get("tables", [])
    ]
    return out


def render_system(mode: str, *, manifest: dict | None = None,
                  batch_chars: int = 200_000, provider: str = "") -> str:
    from rnsr.harness.prompts.variants import guardrail_for

    parts = [_SHARED.format(batch_chars=f"{batch_chars:,}")]
    guardrail = guardrail_for(provider)
    if guardrail:
        parts.append(guardrail)
    if mode == "classic":
        parts.append(_CLASSIC)
    elif mode == "docdb":
        parts.append(_DOCDB.format(
            manifest=json.dumps(compact_manifest(manifest or {}), indent=1,
                                default=str)[:20_000]
        ))
    else:
        raise ValueError(f"unknown mode: {mode}")
    return "\n".join(parts)


def render_transcript(question: str, turns: list[tuple[str, str]]) -> str:
    """Render the running conversation as a single prompt.

    turns: (code, observation) pairs from prior iterations.
    """
    parts = [f"TASK:\n{question}\n"]
    for i, (code, observation) in enumerate(turns, 1):
        parts.append(f"--- cell [{i}] ---\n```python\n{code}\n```")
        parts.append(f"--- output [{i}] ---\n{observation}")
    parts.append(
        "Write the next python code cell (one fenced block). "
        "Call FINAL(...)/FINAL_VAR(...) when done."
    )
    return "\n".join(parts)
