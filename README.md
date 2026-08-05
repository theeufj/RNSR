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

Across **three independently generated matters** (36 questions/system;
by question class):

| System | Total | Aggregations (9) | Timelines (6) | Other (21) | Cost/q |
|---|---|---|---|---|---|
| **DocDB** | **34/36 (94%)** | 7/9 | **6/6** | **21/21** | $0.17 |
| RLM-classic | 32/36 (89%) | 7/9 | 4/6 | **21/21** | $0.07 |
| Vector-RAG (gemini-embedding-2) | 27/36 (75%) | **0/9** | 6/6 | **18/18** | $0.005 |
| Graph-RAG (lean reimpl., entity graph + community summaries) | 30/36 (83%) | 3/9 | 6/6 | **21/21** | $0.17 incl. index |
| BM25-RAG / +LLM reranker (1 matter) | 7/12 both | 0/3 | — | — | $0.006/$0.018 |

RAG aced the retrieval-friendly v1 (12/12 at 1/30th the cost — stated
plainly) and collapsed to 7/12 on the realistic v2: its five misses are
structural, not marginal — every invoice aggregation (the set exceeds any
top-k), the amendment-override question (it confidently answered the
*original* MSA terms, never seeing the amendment), and a timeline it
failed to retrieve. The RAG ladder decomposes the failure honestly:
an LLM reranker over a top-60 lexical pool changed nothing (identical
7/12 at 3× cost), but semantic embeddings (gemini-embedding-2) fixed both
retrieval-quality misses — the amendment-override and the timeline —
reaching 9/12. What NO retrieval flavor fixed is the aggregation class:
39 invoices cannot occupy 12 excerpt slots regardless of ranking quality.
That is the architectural boundary of single-shot retrieval:
vector-RAG missed ALL NINE aggregations across three seeds while scoring
perfectly on everything else. GraphRAG — the incumbent answer to exactly
this criticism — raises the ceiling (30/36; its community summaries
sometimes carry the needed totals, 3/9) but every one of its six misses
is still an aggregation: a knowledge graph's summaries contain whatever
the index-time LLM happened to compile, not computation. The retrieval
ladder climbs — BM25 7 → rerank 7 → vector 9 → graph 10 per matter —
and every rung stops at the same wall, the one SQL walks through. Honest ledger for the leaders too: DocDB's
two misses are SQL slips (a unit error; a double-count from summing
invoice tables including their TOTAL rows — a fair real-world trap), and
classic twice answered a timeline with "28 days from the notice" instead
of computing the date. Classic's single miss (v1) was a silent arithmetic
slip while aggregating by reading — the failure class SQL makes a
non-event. DocDB was perfect across both versions with every answer
carrying verified quotes.


**Batched answering** (many related questions over one corpus — the
form-fill regime): `answer-csv` groups consecutive questions into shared
RLM loops (`--batch-size`, default 8) submitting via `FINAL_BATCH`; one
exploration of the corpus serves the whole group, and any question a
batch fails to answer is retried in its own loop automatically. Measured
on a real family-law matter (49 form fields, 11 documents, scored against
golden answers):

| Mode | Correct | Wall time | LLM spend |
|---|---|---|---|
| **Batched (8/loop)** | **39/49** | **7m 12s** | **$5.46** |
| One loop per question | 30/49 | 26m 55s | $23.11 |

3.7× faster and 4.2× cheaper — and *more accurate*, for a structural
reason: forms carry mutually-exclusive field groups (radio buttons,
checkbox families), and a solo loop seeing only its own field happily
answers "yes" to every sibling option. A batched loop sees the whole
group in one context and picks one. Budgets scale sub-linearly with
batch size (each extra question adds half a single question's caps), so
a confused batch cannot burn n questions' worth of spend.

**Fix-and-confirm cycle** (post three-seed autopsy): the two DocDB SQL
slips were converted into prompt disciplines (trust document units/headers;
sum line items OR total rows, never both; reconcile computed aggregates
against document-stated figures; compute calendar dates). Retested on the
originating questions: all six misses across systems converted. Confirmed
on a fresh unseen matter (seed 8): **DocDB 12/12**; classic 11/12 — its
remaining miss answered $0.00 for a $2.4M invoice total, the silent
grep-aggregation failure again. The disciplines generalize for the
architecture that computes; they cannot rescue the one that greps.

**The honest regime map**, consistent across every controlled A/B:

| Regime | Verdict |
|---|---|
| Multi-document / beyond-window corpora (matter files) | DocDB 24/24; classic strong but slips silently on aggregation; RAG structurally fails aggregation/override/absence |
| Single documents fitting one context window (incl. 75k-token contracts) | **No accuracy advantage.** Classic somewhat cheaper; DocDB ~2× faster at depth |
| Absent-needle questions ("no such clause") | DocDB's verification discipline resists confabulation |
| Many questions per document | DocDB amortizes: ingest + annotations pay once |
| Any regime | Only DocDB returns code-verified quotes with char offsets |

## When to use it (legal decision guide)

**Just load the document into model context** when all of these hold: a
single document under ~150k tokens; one-off questions; lookup/reading
answers rather than computation; no independently verifiable citation
required. We measured accuracy parity there, and stuffing is cheaper.

**Use DocDB** when any of these hold — each measured in the results above:

| Trigger | Measured basis |
|---|---|
| Multi-document matter / exceeds the context window | Stuffing disqualified at ~200k tokens; a modest matter file is already there |
| Answers computed over sets (invoice totals, counts, chronologies) | 0/3 for every RAG flavor; silent arithmetic slip for the flat-string loop; SQL makes it exact |
| Absence must be provable ("no such clause/guarantee") | Retrieval can't tell not-found from not-there; DocDB enumerates |
| Superseded versions in the file (drafts, amendments) | RAG confidently returned pre-amendment terms; DocDB date-orders the file |
| Working sessions: many questions per matter | Ingest + annotations amortize — second pass at half cost, median 1 model call; batched loops answer 8 questions per exploration (3.7× faster, 4.2× cheaper, more accurate on form-field groups) |
| Citations that survive scrutiny | Code-verified quotes with character offsets — unique to DocDB |

One line: **read a document → context window; interrogate a matter → DocDB.**

## Install

```bash
pip install -e .              # query-time core (a prebuilt corpus.db is enough)
pip install -e ".[ingest]"    # + parsing stack (Docling for PDF, anydoc for office)
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
# Formats: PDF (Docling), Word/Excel/PowerPoint/OpenDocument/RTF/EPUB/CSV
# (anydoc), and .md/.txt/.eml (built-in) — dispatched by extension.
rnsr ingest report.pdf exhibits.docx ledger.xlsx -o corpus.db --report report.json

# Query via the RLM loop (root model writes code against db/doc/manifest)
rnsr query corpus.db "What was FY2023 segment revenue?"

# Many questions over one corpus (CSV in, CSV out): ingest once (cached),
# then answer in batched loops — consecutive questions share one
# exploration via FINAL_BATCH; unanswered ones are retried solo.
# Checkpointed: rerunning resumes instead of re-paying.
rnsr answer-csv --corpus matter_dir/ --questions questions.csv --output out/
#   --batch-size 8    questions per shared loop (1 = solo loops)
#   --concurrency 4   loops in flight at once

# Evaluation harness (§8): systems are flags over the same loop
rnsr eval --benchmark financebench --system docdb
rnsr eval --benchmark oolong --system rlm-classic     # Phase B acceptance
rnsr eval --benchmark cuad --system docdb -c 4        # items in parallel
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
 PDF / Office ─▶ │ INGESTION: parse → tables → checksum-     │
 MD / TXT / EML  │ validate → FTS5 → manifest    (offline)   │
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
  fallback, sandbox restart on runaway cells, root-call timeouts. Batched
  loops scale every cap by 1 + 0.5·(n−1) for n questions.
- **Sandbox**: subprocess with no network; model/embedding calls are RPC-
  brokered by the parent under a bounded-concurrency semaphore.

## Development

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[ingest,eval,dev]"
pytest            # 268 tests; LLM-free by default (live tests opt-in: -m live)
ruff check .
```

Implementation phases (spec §10) — all delivered and all gates closed:
**A** deterministic ingestion → **B** RLM harness (OOLONG reproduction
passed) → **C** fusion + go/no-go gate (passed) → **D** hardening (rung-4
embeddings + ablation, schema_map, per-provider prompts). Scanned PDFs are
supported via VLM transcription (no OCR engine). Cross-document schema
unification stays deliberately manual via `schema_map` proposals (§9);
sub-model serving remains deployment guidance (`docs/sub-lm-serving.md`).
