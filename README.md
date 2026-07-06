# RNSR — DocDB-RLM

A typed-environment recursive language model (RLM) system for deep document
retrieval. Documents are ingested once into a single self-contained SQLite
artifact (`corpus.db`) — typed tables with row-level provenance, FTS5 chunks,
a machine-derived manifest, all over fully retained source text — and queried
by a depth-1 RLM loop over a sandboxed Python REPL. LLM calls are reserved
for semantic judgments; everything countable is computed exactly.

The authoritative design is [`docdb-rlm-design-spec.md`](docdb-rlm-design-spec.md).

## Results

**FinanceBench** (numeric needles in real SEC filings; 114 reachable
questions, 36 of 150 excluded as undownloadable):

| Metric | Result |
|---|---|
| Overall accuracy | **88.6%** |
| Numeric questions | **97.4%** (37/38) |
| Textual questions | 84.2% |
| Cost per question | p50 $0.25 · p95 $0.65 |
| Terminal status | 112 final · 2 recovered · 0 errors |

Published GPT-4-with-retrieval baselines on this benchmark sit near 50%.

**Go/no-go gate** (spec §8: DocDB vs RLM-classic — same loop and budgets,
flat-string environment — on a numeric-needle set with exact golds): **PASS**.
DocDB 89% vs classic 78% accuracy, at lower cost (p50 $0.055 vs $0.075;
p95 $0.106 vs $0.225 — classic re-reads its giant context every turn).

**Legal benchmarks** (30-question samples; scoring as above; ContractNLI
and LegalBench items carry no document, so classic and DocDB environments
are identical for them by construction):

| Benchmark | DocDB | RLM-classic (same questions) |
|---|---|---|
| CUAD clause extraction | **90%** (27/30) | 83% (25/30) |
| — of which absent-clause questions | **20/23** | 18/23 |
| ContractNLI (3-way NLI, clause-scale) | 67% | — (identical env) |
| LegalBench (4-task slice) | 84% | — (identical env) |

On CUAD the A/B split is informative: extraction parity (7/7 both) —
sampled contracts are small (median 23k chars) and fit one context
window — but DocDB is better at *absent* clauses (all four disagreements
were absent-clause questions, 3–1 DocDB), i.e. it resists finding
plausible-but-nonexistent clauses. Classic is cheaper on sub-window
documents ($0.10 vs $0.19/question). On a long-contract cut (230–300k-char
agreements, ~60–75k tokens): **no accuracy advantage** — classic 7/10,
DocDB 6/10 (n too small to rank either way; two misses shared, incl. one
contestable gold), classic slightly cheaper, DocDB 2× faster (p50 114s vs
230s — classic re-reads ~75k tokens every turn). Note CUAD cannot test the beyond-window regime for modern
models: its largest contract (~190k tokens) still fits Claude's window —
beyond-window means multi-document corpora, covered by the needle gate
and FinanceBench. These are QA-protocol numbers on samples, not the
official CUAD span-AUPR metric — not leaderboard-comparable.

**OOLONG** (Phase B harness acceptance; `oolongbench/oolong-synth`
trec_coarse, 50 questions, 1k–65k-token contexts): RLM-classic scores
**60%/58%** across two runs, in the ballpark the RLM paper reports —
closing the reproduction gate. DocDB progressed 56% → 62% → **64%**
through targeted fixes (lines-table for semantic_annotate, per-item
labeling rubric, majority voting); the residual misses are dominated by
per-item label ambiguity that OOLONG's aggregation amplifies (a 97.8%
per-line labeler still miscounts), plus a measured share of contestable
golds. Flat-text costs ~1.5× classic: when the whole context fits in one
window and carries no structure, structure buys nothing (§1.3) — the
harness exposes classic mode as a flag for exactly that regime. Bonus
finding, measured live: annotation columns persist in the artifact, so a
second run over the same corpora answered at **half cost with a median of
1 sub-call per question** — semantic work amortizes across queries (§4.1).

**Real-filing ingestion health**: 87% table-validation pass rate on two 3M
10-Ks (412 pages, 228 detected tables), against the spec's 70% stop threshold.
Scoring: exact string/numeric match first; sub-model equivalence judge only
on string failure. Answers carry code-verified quotes (§6) — supporting
quotes are string-matched against retained source text, with failures fed
back into the loop.

**Matter files** (the multi-document legal regime: a synthetic
commercial-dispute matter as real PDFs — MSA + overriding amendments,
~40 invoices, breach/cure correspondence, a superseded draft with wrong
numbers, bulk file notes; 79 docs, ~213k tokens — beyond one context
window, so prompt-stuffing is disqualified at the door):

| System | Matter v1 + v2 (24 questions) | Cost/q (v2) | Latency p50 |
|---|---|---|---|
| **DocDB** | **24/24** | $0.17 | **25s** |
| RLM-classic | 23/24 | $0.07 | 39s |
| BM25-RAG (top-12 chunks, one call) | 19/24 | $0.006 | 4s |
| Rerank-RAG (top-60 pool, LLM reranker, top-12) | 7/12 (v2) | $0.018 | 7s |

RAG aced the retrieval-friendly v1 (12/12 at 1/30th the cost — stated
plainly) and collapsed to 7/12 on the realistic v2: its five misses are
structural, not marginal — every invoice aggregation (the set exceeds any
top-k), the amendment-override question (it confidently answered the
*original* MSA terms, never seeing the amendment), and a timeline it
failed to retrieve. Adding an LLM reranker over a top-60 pool changed
NOTHING — the identical 7/12, same misses, at 3× the cost: the failures
are architectural (sets beyond k, versions, absence), not retrieval
precision. Classic's single miss (v1) was a silent arithmetic
slip while aggregating by reading — the failure class SQL makes a
non-event. DocDB was perfect across both versions with every answer
carrying verified quotes.

**The honest regime map**, consistent across every controlled A/B:

| Regime | Verdict |
|---|---|
| Multi-document / beyond-window corpora (matter files) | DocDB 24/24; classic strong but slips silently on aggregation; RAG structurally fails aggregation/override/absence |
| Single documents fitting one context window (incl. 75k-token contracts) | **No accuracy advantage.** Classic somewhat cheaper; DocDB ~2× faster at depth |
| Absent-needle questions ("no such clause") | DocDB's verification discipline resists confabulation |
| Many questions per document | DocDB amortizes: ingest + annotations pay once |
| Any regime | Only DocDB returns code-verified quotes with char offsets |

## Install

```bash
pip install -e .              # query-time core (a prebuilt corpus.db is enough)
pip install -e ".[ingest]"    # + Docling parsing stack (heavy) for ingestion
pip install -e ".[eval,dev]"  # benchmarks + dev tooling
```

Set at least one provider key in `.env` (see `.env.example`):
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GOOGLE_API_KEY`.

## Usage

```bash
# Phase A: parse → typed tables → checksum validation → FTS5 → manifest.
# Deterministic and LLM-free by default; --llm enables the vision
# re-extraction rung, prose cross-checks, and VLM transcription of scanned
# pages (no OCR engine — pages without a text layer go through the vision
# model, and the resulting tables face the same checksum validation).
rnsr ingest report.pdf -o corpus.db --report report.json

# Query via the RLM loop (root model writes code against db/doc/manifest)
rnsr query corpus.db "What was FY2023 segment revenue?"

# Evaluation harness (§8): systems are flags over the same loop
rnsr eval --benchmark financebench --system docdb
rnsr eval --benchmark oolong --system rlm-classic     # Phase B acceptance
rnsr gate                                              # go/no-go vs classic
rnsr ablate corpus.db                                  # rung-4 quantization ablation
```

Cross-document joins stay explicit by design (spec §9): headers drift
between filings, so `schema_map` *proposes* column correspondences and the
root model (or you) applies them visibly — never automatically:

```python
# inside the REPL environment (rnsr query), joining 2023 vs 2024 tables
props = schema_map("t_report2023_004", "t_report2024_007")
# -> [{"a": "revenue_m", "b": "net_revenue", "confidence": 0.8, "reason": ...}]
db.execute("""
    SELECT a.segment, a.revenue_m AS fy2023, b.net_revenue AS fy2024
    FROM t_report2023_004 a JOIN t_report2024_007 b ON a.segment = b.business_unit
""").fetchall()   # the join is written out — auditable in the trajectory
```

## Architecture

```
                 ┌───────────────────────────────────────────┐
 PDF / DOCX ──▶  │ INGESTION: parse → tables → checksum-     │
                 │ validate → FTS5 → manifest    (offline)   │
                 └────────────────────┬──────────────────────┘
                                      ▼
                          corpus.db  (one SQLite file)
                                      ▼
                 ┌───────────────────────────────────────────┐
                 │ RLM LOOP: root LM ⇄ sandboxed REPL        │
                 │ db · doc · manifest · semantic_annotate   │
                 │ search ladder · verify · FINAL+quotes     │
                 └────────────────────┬──────────────────────┘
                                      ▼
                        answer + provenance record
```

- **Search ladder** (§5): SQL → regex → FTS5/BM25 → sub-model expansion →
  lazy int8 embeddings (fp32 rescore) → exhaustive sweep (opt-in, cost
  estimate first). Every rung resolves back to retained text.
- **semantic_annotate** (§4.1): one batched sub-model pass writes results
  back as a real SQL column (idempotent, audit-logged) — O(N²) reasoning
  becomes O(N) calls plus a self-join.
- **Budgets** (§7): hard caps per query (20 iterations / 300 sub-calls /
  600 s / $2), damping against re-verification loops, variable-recovery
  fallback, sandbox restart on runaway cells, root-call timeouts.
- **Sandbox**: subprocess with no network; model/embedding calls are RPC-
  brokered by the parent under a bounded-concurrency semaphore.

## Development

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[ingest,eval,dev]"
pytest            # 204 tests; LLM-free by default (live tests opt-in: -m live)
ruff check .
```

Implementation phases (spec §10) — all delivered and all gates closed:
**A** deterministic ingestion → **B** RLM harness (OOLONG reproduction
passed) → **C** fusion + go/no-go gate (passed) → **D** hardening (rung-4
embeddings + ablation, schema_map, per-provider prompts). Scanned PDFs are
supported via VLM transcription (no OCR engine). Cross-document schema
unification stays deliberately manual via `schema_map` proposals (§9);
sub-model serving remains deployment guidance (`docs/sub-lm-serving.md`).
