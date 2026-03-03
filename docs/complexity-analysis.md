# RNSR Complexity Analysis

Where **n** = total content size (tokens across all documents), **s** = number of sections, **d** = number of documents, **q** = number of questions, **w** = parallel workers, **k** = top-k sections selected per query, **r** = max recursion depth (bounded constant).

## Ingestion (one-time cost)

| Step | Complexity | Notes |
|------|------------|-------|
| Text extraction / OCR | O(n) | Linear scan of all pages |
| Tree construction | O(n) | Proportional to content |
| KG extraction | O(s) LLM calls | One call per section, parallelised across `w` workers → wall-clock O(s/w) |

**Total ingestion tokens processed by LLMs: O(n)** — every section's content is sent to the LLM once.

## Per-Query

| Step | Complexity | Notes |
|------|------------|-------|
| KG Resolver / Profile check | O(1) | Direct lookup against extracted profiles |
| Cross-doc routing | O(d) | Score each document for relevance |
| Tree search (BFS + regex) | **O(s)** | Scans all section headers/content with keyword matching |
| Section selection | O(k), k ≪ s | Only top-k sections are read (typically 5–10) |
| Sub-question decomposition | O(r) iterations | r = max recursion depth (bounded constant, ~50) |
| LLM synthesis | O(1) | Bounded context window, only selected sections |
| Verification / re-ranking | O(1) | Single LLM call each |

**Per-query total: O(r × s)** for search operations, but **O(k)** for actual LLM token consumption.

Since `r` is bounded by a constant (`max_recursion_depth`), the dominant factor is **O(s)** per query for tree search. Because `s ∝ n`, search is **O(n) per query**.

## Tree Navigation vs Full Context Window

The core efficiency advantage of RNSR's tree navigation over loading entire documents into a single LLM context window:

| Approach | Tokens sent to LLM per query |
|----------|------------------------------|
| Full context window | O(n) — all content |
| RNSR tree navigation | **O(k × avg_section)** — only selected sections |

The **search phase is O(n)** (regex BFS over section headers), but the **expensive part** (LLM token processing) is **sub-linear in n** because only a small fraction of sections are read and synthesised.

## End-to-End

$$\text{Total} = \underbrace{O(n)}_{\text{ingestion}} + \underbrace{O(q \times s)}_{\text{querying}}$$

- Ingestion dominates for small `q`.
- Querying dominates as `q` grows.
- The practical bottleneck is LLM API latency (60–100s per question), not the O(s) tree search which completes near-instantly.
