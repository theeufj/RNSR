# Design Specification — DocDB-RLM
## A Typed-Environment Recursive Language Model System for Deep Document Retrieval

**Version:** 0.2 (draft for review — adds vector-index compression, the no-eviction invariant, and sub-LM serving guidance derived from PolarQuant)
**Status:** Design — pre-implementation
**Prior artifacts:** RNSR Feasibility Assessment v2; RLM paper (Zhang, Kraska & Khattab, arXiv:2512.24601); PolarQuant (Han, Kacham, Karbasi, Mirrokni & Zandieh, arXiv:2502.02617)

---

## 1. Purpose and scope

DocDB-RLM is a retrieval and question-answering system for large, messy document corpora (financial reports, contracts, technical manuals, merged PDF bundles) that solves the needle-in-a-haystack problem — including its hardest variant, *numeric* needles — by converting documents into a typed, queryable environment rather than a flat string or a vector index.

The system rests on three evidence-backed commitments derived from the RLM paper and the preceding feasibility work:

1. **The REPL is the workhorse.** A depth-1 RLM loop (root orchestrator LM + cheap sub-LM calls) over an interactive environment beats summarization, retrieval agents, and base long-context models on long, dense tasks. We build that loop, not a fixed traversal pipeline.
2. **LLM calls are for semantic transformations only.** Anything downstream of a semantic judgment is frozen into structured data (SQLite) and computed exactly. This is "variable stitching" taken to its logical conclusion.
3. **Structure is an accelerant, not a foundation.** The system must work on the raw text alone (grep + sub-LM sweep guarantees recall); tables, indexes, and annotations exist to make the common case fast and the numeric case exact.
4. **Compression is allowed; eviction is not.** PolarQuant's needle-in-a-haystack results (Figure 3 of that paper) show that at equal compression ratios, lossy-representation-of-everything (quantization: KIVI 0.984, PolarQuant 0.991 recall) decisively beats lossless-representation-of-a-subset (token eviction: SnapKV 0.858, PyramidKV 0.891). Needles look unimportant until queried, so any mechanism that discards content based on presumed importance is structurally hostile to needle retrieval. No component of DocDB-RLM may make source content unreachable: indexes, quantized vectors, summaries, and SQL tables are all *additional* compressed views over the full retained text, never replacements for it. (The RLM paper's summary-agent baseline, which this system's design rejects, is the eviction analogue at the agent level.)

Out of scope for v1: cross-document schema unification (see §9), multi-modal figures beyond page-image fallback, real-time/streaming ingestion, and recursion depth greater than one.

## 2. System overview

```
                    ┌──────────────────────────────────────────┐
                    │              INGESTION (offline)          │
  PDF/DOCX/HTML ──▶ │  parse → extract tables → validate →     │
                    │  build FTS index → write manifest         │
                    └───────────────────┬──────────────────────┘
                                        ▼
                            corpus.db (SQLite artifact)
                    ┌──────────────────────────────────────────┐
                    │  doc_text · tables_* · provenance ·       │
                    │  fts_chunks · annotations · manifest      │
                    └───────────────────┬──────────────────────┘
                                        ▼
                    ┌──────────────────────────────────────────┐
                    │           RLM LOOP (query time)           │
                    │  root LM ⇄ sandboxed Python REPL          │
                    │  preloaded: db, doc, manifest,            │
                    │  llm_query(), semantic_annotate(),        │
                    │  search ladder helpers, verify()          │
                    └───────────────────┬──────────────────────┘
                                        ▼
                          answer + provenance record
```

Two lifecycle phases. **Ingestion** runs once per document and produces a single self-contained SQLite file — this artifact *is* the persistent index, giving amortization across queries for free. **Query** spins up a sandboxed REPL, loads the artifact, and runs the depth-1 RLM loop until a final answer is produced or budget caps trigger.

## 3. Ingestion pipeline

### 3.1 Parsing

Primary parser: Docling or Unstructured (`hi_res`) for layout-aware extraction of text blocks and table candidates, retaining page numbers and bounding boxes for every element. Fallback chain per element type: primary parser → alternate parser (e.g., Camelot/pdfplumber for tables) → vision sub-LM call on the rasterized page crop. Every extracted element records which rung of the fallback chain produced it.

### 3.2 Table extraction into SQLite

Each detected table becomes a SQL table named `t_{doc_id}_{seq}` with columns inferred from the header row. Column typing is conservative: attempt numeric coercion per column; if ≥95% of non-null cells coerce, store as REAL/INTEGER with the raw string preserved in a shadow column (`{col}__raw`); otherwise TEXT. Currency symbols, thousands separators, parenthesized negatives, and percentage signs are normalized during coercion, with the normalization rule recorded in table metadata. Multi-page tables detected by header repetition are merged into one SQL table with a `source_page` column per row.

Every table row carries three provenance columns: `_page`, `_bbox`, `_extractor` (which fallback rung).

### 3.3 Self-validation (checksum pass)

Documents contain internal redundancy; we exploit it as automatic ground truth. For each extracted table, the validator runs, in order:

**Arithmetic checks.** Detect candidate total/subtotal rows (label heuristics: "total", "sum", "net", bold styling metadata where available). Verify `SUM(line items) == total` within a configurable tolerance (default 0.5% or 1 unit, whichever is larger, to absorb rounding). Percentage columns are checked to sum to ~100 where a total row implies it.

**Structural checks.** Row/column count sanity vs. the parser's detected grid; no duplicated header rows inside the body; monotonic date columns where a date type was inferred.

**Prose cross-check (sampled, sub-LM).** For a sample of k numeric cells (default k=3 per table), a sub-LM is asked whether any prose within ±1 page states or implies the value; agreement raises confidence, contradiction flags the table.

Output: a per-table `confidence` score in `manifest.tables` with the individual check results. Tables scoring below threshold (default 0.7) are automatically re-extracted via the next fallback rung; if all rungs fail validation, the table is marked `untrusted` and the RLM is told so in the manifest — it may still read the raw text region instead. No silent failures: every table is trusted, retried, or explicitly flagged.

### 3.4 Text indexing

Full document text is chunked (structure-aware where headings exist; fixed 1,500-char windows with 200 overlap otherwise) into `chunks(chunk_id, doc_id, page, char_start, char_end, text)` and indexed with **FTS5** (BM25 ranking, porter tokenizer). Embeddings are *not* computed at ingestion; see the search ladder (§5, rung 4) for lazy on-demand embedding with write-back caching into the same .db.

### 3.5 Manifest

A `manifest` table (and a JSON view of it) summarizes the environment: document list, per-table schema + row counts + confidence + page ranges, chunk statistics, and any untrusted-element flags. The manifest is the first thing the root LM sees — it replaces the RLM paper's blind `print(context[:500])` probing with immediate structural awareness.

## 4. Query-time environment

The REPL is a sandboxed Python session (no network egress; CPU/memory/time caps) preloaded with:

```python
db                # sqlite3 connection (read-write: annotations allowed, source tables immutable via triggers)
doc               # dict: doc_id -> full raw text (the RLM-classic flat string, always available)
manifest          # dict form of the manifest table
llm_query(prompt, model="sub") -> str        # depth-1 sub-LM call (async batched under the hood)
semantic_annotate(table, new_col, prompt,    # batched sub-LM pass over rows; writes column back
                  where=None, batch_size=40) # returns coverage + sample for root LM inspection
search(query, rung=None) -> results          # the tiered ladder, §5; auto-escalates unless rung pinned
verify(answer, quotes) -> report             # string-matches quotes against source text; exact check
```

Root-LM system prompt: adapted from the paper's published GPT-5 prompt, extended with the manifest description, tool docs, and the batching guardrail (tuned per model — the paper showed this line is model-specific). Answer return uses `FINAL()`/`FINAL_VAR()` **plus** the variable-recovery fallback: if the loop ends without a valid FINAL, the harness inspects the REPL namespace for the most recently constructed answer-like variable and asks the root LM one final time to confirm or reject it (mitigates the paper's B.2 failure, where a correct answer was built then abandoned).

### 4.1 semantic_annotate — the semantic ETL primitive

The single most important tool. Contract: one batched sub-LM pass over selected rows; results written back as a real column; idempotent (re-running with same args is a no-op unless `force=True`); every annotated value stores the sub-model + prompt hash in an `annotation_log` table for auditability. Batching defaults follow the paper's guardrail (~200k chars per sub-call as a starting point, tunable per model).

Why it matters: it converts O(N²)-in-LLM-reasoning problems into O(N) semantic calls plus exact SQL. Worked example (OOLONG-Pairs Task 3 class):

```python
semantic_annotate("t_qs", "label",
    "Classify this question's answer type as one of: numeric value, entity, "
    "location, description and abstract concept, abbreviation, human being. "
    "Return only the label.")
pairs = db.execute("""
  WITH hits AS (SELECT DISTINCT user_id FROM t_qs
                WHERE label IN ('description and abstract concept','abbreviation'))
  SELECT a.user_id, b.user_id FROM hits a JOIN hits b ON a.user_id < b.user_id
  ORDER BY a.user_id""").fetchall()
FINAL_VAR(pairs)
```

The quadratic part is a self-join: exact, instant, free.

## 5. The tiered search ladder

`search()` escalates through rungs, returning as soon as a rung yields results the root LM accepts. Each rung logs its cost. The ladder is the system's answer to "when do we pay for LLM calls?"

**Rung 0 — SQL.** If the manifest suggests the target is tabular (numbers, dates, entities in known columns), query the tables first. Exact; handles the numeric-needle class embeddings cannot.

**Rung 1 — grep with priors.** The root LM generates candidate surface forms (synonyms, abbreviations, formatting variants, translations) and regex-scans `doc` programmatically. Free; this is the strategy GPT-5 improvised in the paper's B.1 trajectory, now first-class.

**Rung 2 — FTS5 MATCH.** BM25-ranked lexical search over chunks. Milliseconds, no model calls.

**Rung 3 — expansion loop.** A sub-LM reads rung 1–2 near-misses and proposes new search terms; a frontier queue of terms is iterated (max 3 rounds default). Cheap, bounded.

**Rung 4 — lazy embeddings (quantized cache).** Embed chunks on demand, cache vectors back into the .db, run semantic top-k via asymmetric search: stored vectors are quantized, the query vector stays full precision, and scoring dequantizes on the fly (the pattern behind PolarQuant's `K̂·q` kernel). First use pays embedding cost once per corpus; thereafter cached. Compression here *is* lookup speed — brute-force scan is memory-bandwidth-bound, so a 4× smaller cache is roughly a 4× faster scan and a 4× smaller portable artifact.

Two tiers, chosen by corpus scale:

*Default (int8):* symmetric int8 quantization with fp32 rescoring of the top ~4k candidates, using sqlite-vec's native support. Zero custom code; near-lossless recall at typical corpus scale (10k–500k chunks); ~4× compression before metadata overhead.

*Upgrade path (polar quantization):* for large artifacts (millions of chunks) or embedding models whose distributions quantize poorly, adopt the PolarQuant recipe: apply a fixed random rotation as preconditioning, transform to polar coordinates via the recursive pairing scheme (L=4 levels), and quantize angles against an **offline, input-independent codebook**. Two properties motivate the extra machinery. First, *no per-block quantization constants*: conventional schemes store scale/zero-point per block in full precision, costing >1 extra bit per number; preconditioning makes the angle distribution analytically known (concentrated around π/4, variance O(1/√d)), so no normalization metadata is needed — bit budget bFPN+46 per 16-coordinate block, ~3.9 bits/coordinate at 4.1×+ compression. Second, *zero calibration at ingestion*: the paper's offline codebook scored within 0.7 LongBench points of per-input online clustering, so one shipped codebook serves every corpus, keeping Phase A fully deterministic and LLM-free. The rotation matrix and codebook are stored once in the manifest.

Rung-4 recall@k (quantized vs fp16) is a required ablation before either tier ships (§8).

**Rung 5 — exhaustive sub-LM sweep.** The paper's chunk-and-query strategy over all chunks. Guaranteed-recall last resort with known linear cost; requires root-LM opt-in above a cost estimate threshold.

**Ladder invariant (from §1, commitment 4).** Every rung is a compressed *view*; no rung may evict. Concretely: FTS tokenization, vector quantization, chunk boundaries, and manifest summaries are all derived from — and resolvable back to — the fully retained source text via provenance offsets. Rung 5 exists precisely so that recall degrades to "expensive," never to "impossible."

## 6. Provenance and verification

Every answer path is data lineage, not narrative. SQL results carry `_page`/`_bbox` from their rows; text hits carry chunk offsets; annotations carry their log entries. The `verify()` tool enforces the discipline: final answers must include supporting quotes, which are string-matched (with normalization) against source text by *code* — a check no LLM can hand-wave. Answers whose quotes fail verification are returned to the loop with the failure report. The full trajectory (code cells, sub-calls, costs) is persisted per query for audit and for the evaluation harness.

## 7. Budgets, caps, and failure handling

Hard caps, all configurable: max root iterations (default 20), max sub-LM calls per query (default 300), max wall-clock (default 10 min), max spend per query (default $2). On cap breach: variable-recovery fallback fires, and the answer is labeled `budget_exhausted` with partial provenance. Sub-calls are async with bounded concurrency (default 16) — the paper's stated biggest inefficiency was sequential calls; we do not repeat it. Repeated-verification loops (the B.2 death-spiral) are damped by a rule in the harness, not the prompt: after the same FINAL-candidate value is recomputed twice, the harness forces a confirm-or-reject turn.

Expect long-tailed cost distributions (paper Figures 3, 7, 8): all budgeting and reporting is at the median *and* 95th percentile.

## 8. Evaluation plan

**Benchmarks:** OOLONG (linear density), OOLONG-Pairs (quadratic; published queries), BrowseComp-Plus subsets (multi-hop, large corpus), LongBench-v2 CodeQA, FinanceBench (numeric needles in real filings — the headline demo for the SQL path), plus a custom numeric-needle set generated from our own corpora (perturb one figure in a filing; ask for it).

**Baselines:** base long-context model; RLM-classic (flat string, no db — i.e., the paper's system reproduced); vector RAG (hybrid BM25+embeddings, top-k). DocDB-RLM must beat RLM-classic on numeric tasks and match it elsewhere, at equal or lower cost, to justify the ingestion layer — this is the go/no-go gate.

**Metrics:** accuracy per task class; cost/query and latency at p50 and p95; sub-calls/query; ladder-rung histogram (where do needles die?); table-validation pass rate and re-extraction rate; verification pass rate on final answers.

**Rung-4 quantization ablation (cheap, required):** recall@k (k=10, 50) and end-to-end answer accuracy with fp16 vs int8 vs polar-quantized vectors, on the custom numeric-needle set and one BrowseComp-Plus subset. Acceptance: quantized recall@50 within 1% of fp16. PolarQuant's own NIAH result (0.991 vs exact 0.995 at 4× compression) sets the expectation that this bar is achievable; if int8 already clears it at target corpus scale, the polar upgrade path stays dormant.

## 9. Risks and open questions

**Table extraction quality is the load-bearing risk.** Mitigated by the fallback chain and checksum validation, but corpora with mostly-image tables or exotic layouts will degrade to vision calls (cost) or untrusted flags (coverage). Measure the validation pass rate early on representative documents; below ~70%, invest in extraction before anything else.

**Schema drift and cross-document joins (deferred).** Joining the 2023 table against the 2024 table when headers drifted is entity resolution requiring sub-LM judgment; errors concentrate here. v1 keeps per-document tables and exposes a sub-LM `schema_map()` helper that *proposes* column correspondences for the root LM to apply explicitly — visible, auditable, not automatic.

**Coercion hazards.** Aggressive numeric normalization can corrupt meaning (e.g., "1,234" as European decimal). Shadow raw columns and the checksum pass are the guards; any coercion rule that causes checksum failures is rolled back per column.

**Root-model dependence.** The paper found weak coding models fail as RLMs and prompts don't transfer across models. Budget for per-model prompt tuning; treat model swaps as re-evaluation events, not drop-ins.

**Manifest as attack/error surface.** The root LM trusts the manifest; a wrong confidence score misroutes reasoning. Manifest claims are therefore all machine-derived (no LLM-generated summaries in v1 manifest) so they cannot hallucinate.

## 10. Implementation phases

**Phase A — Deterministic core (1–2 weeks).** Ingestion pipeline: parse → tables-to-SQLite with provenance → checksum validation → FTS5 → manifest. Fully testable without any LLM. Deliverable: `ingest(pdf) -> corpus.db` + validation report.

**Phase B — RLM harness (1–2 weeks).** Sandboxed REPL, root loop, async `llm_query`, caps, variable-recovery fallback, trajectory logging. Reproduce RLM-classic results on OOLONG as the harness acceptance test.

**Phase C — Fusion (1 week).** Preload Phase A artifacts into Phase B environment; implement `semantic_annotate`, `search` ladder, `verify`. Run the go/no-go evaluation (§8).

**Phase D — Hardening (ongoing, conditional on C).** Per-model prompt tuning, lazy-embedding cache with the rung-4 quantization ablation, schema_map helper, cost-tail optimization, custom numeric-needle benchmark expansion, and — if sub-LM volume justifies self-hosting — the Appendix A serving configuration.

Total to go/no-go: roughly 4–5 weeks of focused work.

---

## Appendix A — Sub-LM serving economics (PolarQuant, used literally)

The system's cost centers are `semantic_annotate` and rung-5 sweeps: embarrassingly parallel batch calls over ~200k-char contexts to a cheap sub-LM, and the source of the RLM paper's long-tailed p95 costs. If query volume justifies self-hosting that sub-LM, PolarQuant applies *as written* — quantizing the transformer's internal KV cache during inference, which is a different "KV" from the corpus store but the same paper.

The trade, taken from the paper's own numbers: 4.2×+ KV-cache compression means roughly 4× longer contexts or 4× more concurrent sequences per GPU, against ~14% slower per-sequence generation (43.7s vs 38.4s exact on their 16k-prefill/1k-generate benchmark) and quality within ~0.3 points of exact on LongBench (45.45 vs 45.71). For interactive root-LM traffic that latency penalty might matter; for our batch sub-LM workload, throughput-per-dollar dominates, and 4× batch density wins decisively. Configuration guidance: use the offline codebook variant (prefill 3.4s vs 11.6s for online clustering — the online variant's per-prompt clustering cost is unacceptable at our call volumes); quality cost of offline vs online is ~0.7 LongBench points, acceptable for sub-LM duty.

This appendix is deliberately last: it is a deployment optimization, orthogonal to system correctness, and only relevant past a volume threshold. But it closes the loop on a design theme — the same principle (precondition so the data distribution becomes predictable, then exploit predictability with a fixed cheap scheme) appears in rung-4 vector compression, in checksum-validated table extraction, and here in serving. Spend structure once; compute cheaply forever after.
