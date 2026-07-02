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
regex, BM25 full-text, sub-LM term expansion. Escalates automatically; \
returns hits with page/offset provenance.
- verify(answer, quotes): exact string-match of supporting quotes against \
source text. Your FINAL answer should be backed by quotes that pass.

MANIFEST:
{manifest}
"""


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
            manifest=json.dumps(manifest or {}, indent=1, default=str)[:20_000]
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
