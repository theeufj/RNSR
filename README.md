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

**OOLONG** (Phase B harness acceptance; `oolongbench/oolong-synth`
trec_coarse, 50 questions, 1k–65k-token contexts): RLM-classic scores
**60%** with all questions reaching a clean final answer — in the ballpark
the RLM paper reports, closing the reproduction gate. DocDB scores 56% at
~1.5× cost on this flat-text benchmark — statistically indistinguishable
(2-question gap), and the expected shape: when the whole context fits in
one window and there are no tables, structure buys nothing (§1.3).

**Real-filing ingestion health**: 87% table-validation pass rate on two 3M
10-Ks (412 pages, 228 detected tables), against the spec's 70% stop threshold.
Scoring: exact string/numeric match first; sub-model equivalence judge only
on string failure. Answers carry code-verified quotes (§6) — supporting
quotes are string-matched against retained source text, with failures fed
back into the loop.

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
