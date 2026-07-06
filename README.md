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
# re-extraction rung and prose cross-checks.
rnsr ingest report.pdf -o corpus.db --report report.json

# Query via the RLM loop (root model writes code against db/doc/manifest)
rnsr query corpus.db "What was FY2023 segment revenue?"

# Evaluation harness (§8): systems are flags over the same loop
rnsr eval --benchmark financebench --system docdb
rnsr eval --benchmark oolong --system rlm-classic     # Phase B acceptance
rnsr gate                                              # go/no-go vs classic
rnsr ablate corpus.db                                  # rung-4 quantization ablation
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

Implementation phases (spec §10) — all delivered: **A** deterministic
ingestion → **B** RLM harness → **C** fusion + go/no-go gate (passed) →
**D** hardening (rung-4 embeddings + ablation, schema_map, per-provider
prompts). Open items: OOLONG Phase B reproduction; OCR for scanned PDFs
(currently disabled); cross-document schema unification stays deliberately
manual via `schema_map` proposals.
