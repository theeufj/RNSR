# Corpus Storage Upgrade (staged, on proven primitives)

Status: v2 — revised after feasibility assessment, 2026-08-12. Supersedes the from-scratch Rust engine plan (v1, in git history of this file).

## Decision record

v1 of this plan proposed a from-scratch Rust engine (pager, ARIES WAL, B+tree, SQL frontend, native BM25/ANN). A feasibility assessment argued — and we agree — that:

- The kernel and SQL-dialect-parity work is 18-30 solo person-months reproducing solved problems (redb and Turso are the cautionary precedents).
- SQL parity is unbounded because the query stream is LLM-generated SQLite dialect: the "subset" is whatever the model emits (`json_extract`, window functions, type-affinity quirks), not what we choose.
- Nearly every pain point is a schema or library problem fixable on the existing stack in weeks.
- The one genuinely novel idea — hierarchical TOC region routing — needs no new engine to validate, only a Python prototype and a recall ablation.

Fact-check note: the assessment's "NOT FOUND" items (`LazyDoc`, `heading_path`, `documents.sha256`, `questions_enriched.csv`) all exist in source — it could not read the code and worked from the spec/README, which are partly stale. Its effort/opportunity-cost argument stands regardless.

**Pivot: staged upgrades on SQLite + mature libraries (Tantivy, usearch), TOC prototyped in Python. Format adoption (Lance) deferred to a conditional Stage 4. From-scratch engine rejected absent a funded team.**

Requirements preserved from v1:

- `db.execute(sql)` agent contract unchanged (trivially — it stays real SQLite).
- Source documents never discarded: reference + sha256 pin, retained media, provenance from any derived artifact back to source location.
- 100k-doc scale as the measurement target (validated with synthetic corpora before any format decision).
- PoC isolated in a git worktree; eval-harness parity gate at every stage.

## Pain points -> fixes

- `t_*` explosion + un-indexed rung-0 `LIKE` sweeps (`rnsr/env/search.py`) -> derived indexed `cells` table + zone-map stats (Stage 1)
- Text materialized in RAM (`rnsr/env/lazydoc.py` mitigates, doesn't solve) -> mmap-backed text access (Stage 1)
- FTS5 scaling / bolt-on status -> Tantivy via `tantivy-py` (Stage 2)
- sqlite-vec brute-force + NumPy fp32 rescore dies at 1-10M chunks (`rnsr/env/embeddings.py`) -> usearch ANN, int8 storage + fp32 rescore, view-from-disk (Stage 2)
- Scanned-page images discarded after VLM transcription (`rnsr/ingest/llm_hooks.py`) -> content-addressed media retention + query-time re-inspection (Stage 2)
- No corpus-level navigation; manifest dominates prompt cost -> queryable TOC + region routing prototype (Stage 3)

## Stages

### Stage 0 — Worktree setup

`git worktree add ../RNSR-engine-poc -b engine-poc`. Main checkout and live runs untouched; nothing merges until the parity gate passes.

### Stage 1 — Cheap fixes on the existing stack (days-2 weeks)

1. **Baseline benchmark first**: rung-0 sweep latency, FTS/vector latency, ingest throughput, peak RSS on the largest available corpus (18k-file HANDOFF corpus if accessible, else largest `runs/` artifact plus synthetic scale-up).
2. **Derived `cells` table**: `(doc_id, table_name, row_idx, col_name, text_value, num_value)`, populated at ingest, indexed on `text_value`/`num_value`. Rung-0 sweeps query this one indexed table instead of `LIKE` over every `t_*` table. The `t_*` tables stay — they are the agent-facing typed-SQL contract and must not change.
3. **Zone-map stats**: per-column min/max/distinct-sample in `manifest_tables`, used to prune which tables a sweep or the agent touches.
4. **mmap text**: retained text served via mmap behind the existing `LazyDoc` interface.
5. Re-benchmark. Exit criterion: measured improvement, or a concrete bottleneck for Stage 2 to target.

### Stage 2 — Mature libraries behind the search ladder (2-4 weeks)

1. **Tantivy** as the BM25 provider behind rung 2, built at ingest alongside (then instead of) FTS5.
2. **usearch** as the ANN provider behind rung 4: int8 storage, fp32 rescore, view-from-disk. LanceDB is the alternate if we want vectors + media in one dependency.
3. **Pluggable index providers**: a thin provider interface in `rnsr/env/search.py` / `rnsr/db/` so old and new providers A/B behind a flag. This replaces v1's whole-backend abstraction.
4. **Media + provenance**: content-addressed (sha256) media store retaining page images with `(media_sha, bbox)` transcription provenance; retrieval API for query-time VLM re-inspection; fix the fast-ingest path to record true sha256 (`rnsr/ingest/fast_parse.py` currently writes a stat-based identity into the sha256 column).
5. **Artifact packaging**: the corpus becomes a bundle (`corpus.db` + Tantivy dir + usearch file + media store), with `pack`/`unpack` (zip container) if single-file portability is required.
6. Exit criterion: recall and latency at 1M+ chunks measured against the Stage-1 baseline.

### Stage 3 — TOC region routing, research spike (2-4 weeks)

1. **Python TOC prototype** over the Stage-2 stack: hierarchy from existing `heading_path` (corpus > document > section > chunk/table spans); per-node signatures — centroid embedding, top-terms/bloom lexical signature, rolled-up zone maps.
2. **Routing**: coarse-to-fine scoring that scopes rung 2/4 searches to selected regions; exposed to the agent as a queryable `toc` table plus a `route(query, k)` tool. Routing prioritizes — it never removes the exhaustive fallback rungs, so the no-eviction invariant holds.
3. **Recall ablation (the go/no-go)**: routed vs flat retrieval at target scale. Threshold: recall within ~1% of flat. Pass -> productionize; fail -> drop routing and keep the TOC only as the queryable replacement for the prompt-inlined manifest.

### Stage 4 — Format adoption (conditional, not scheduled)

Only if Stages 1-3 hit a hard wall (bundle portability or a unified planner becomes a real need): spike adopting the **Lance** format (versioned columnar + IVF-PQ ANN + multimodal blobs), keeping SQLite/DuckDB for relational SQL. A from-scratch engine is explicitly rejected absent a funded multi-person team.

## De-risking

- Every stage lands behind a flag; the SQLite/FTS5/sqlite-vec paths keep working until the replacement proves out.
- Parity gate at each stage: `rnsr eval` / `rnsr gate` over `testMatter/questions_enriched.csv` must be equal on old vs new providers.
- Benchmarks are stage entry/exit criteria, not afterthoughts; the 100k-doc envelope is validated with synthetic corpora before any format decision.

## Key files

- Modified: `rnsr/env/search.py`, `rnsr/env/embeddings.py`, `rnsr/env/lazydoc.py`, `rnsr/db/schema.py`, `rnsr/ingest/pipeline.py`, `rnsr/ingest/tables.py`, `rnsr/ingest/fast_parse.py`, `rnsr/ingest/llm_hooks.py`, `pyproject.toml` (add `tantivy`, `usearch`)
- New: index-provider interface in `rnsr/db/`, `rnsr/env/toc.py` (Stage 3), benchmark scripts under `benchmarks/`

## Rough timeline

- Stage 1: days to 2 weeks
- Stage 2: 2-4 weeks
- Stage 3: 2-4 weeks (research spike with an explicit kill criterion)
- Total: ~6-10 weeks solo, vs ~18-30 person-months for the v1 from-scratch engine
