# RNSR Roadmap

> 100% FinanceBench accuracy, 0 hallucinations. Now prove it generalizes and make it production-ready.

---

## Phase 1: Prove It Generalizes (Benchmarks)

The single most important thing before a Show HN post. "Does it generalize?" is the first question.

### 1.1 MultiHiertt — Multi-Step Hierarchical Table Reasoning
- **What:** Financial QA requiring multi-step arithmetic across hierarchical tables
- **Why:** Proves RNSR handles structured numerical reasoning, not just text retrieval
- **Dataset:** https://github.com/psunlpgroup/MultiHiertt
- **Key metrics:** Exact match, F1, execution accuracy
- **Status:** [x] ✅ Completed — `MultiHierttLoader` in `rnsr/benchmarks/multihiertt_bench.py`

### 1.2 TAT-QA — Tabular and Textual Financial QA
- **What:** Questions requiring joint reasoning over tables and text in financial reports
- **Why:** Validates hybrid table+text retrieval — core RNSR strength
- **Dataset:** https://nextplusplus.github.io/TAT-QA/
- **Key metrics:** Exact match, F1 (arithmetic vs. span vs. count)
- **Status:** [x] ✅ Completed — `TATQALoader` in `rnsr/benchmarks/tatqa_bench.py`

### 1.3 QASPER — Scientific Paper QA
- **What:** Questions over full NLP research papers (long docs, cross-section reasoning)
- **Why:** Proves RNSR works beyond finance — generalizes to academic/technical docs
- **Dataset:** https://allenai.org/data/qasper
- **Key metrics:** F1, answer evidence retrieval accuracy
- **Status:** [x] ✅ Completed — `QASPERLoader` in `rnsr/benchmarks/qasper_bench.py`

### 1.4 DocVQA — Visual Document Understanding
- **What:** Questions over document images (forms, receipts, reports)
- **Why:** Validates RNSR's vision/layout capabilities (LayoutLM, chart/table parsing)
- **Dataset:** https://www.docvqa.org/
- **Key metrics:** ANLS (Average Normalized Levenshtein Similarity)
- **Status:** [x] ✅ Completed — `DocVQALoader` in `rnsr/benchmarks/docvqa_bench.py`

---

## Phase 2: Reduce Friction (Adoption Blockers)

### 2.1 Make `torch` an Optional Dependency
- **What:** Move `torch`, `transformers`, `torchvision` to `rnsr[vision]` extra
- **Why:** Default `pip install rnsr` downloads multi-GB — users bounce
- **Target:**
  - `pip install rnsr` → lightweight, font-based analysis only
  - `pip install rnsr[vision]` → adds LayoutLM, torch, vision features
- **Status:** [x] ✅ Completed — torch/transformers/torchvision moved to `[vision]` extra in `pyproject.toml`

### 2.2 Fix Duplicate `pymupdf` in Dependencies
- **What:** `pymupdf>=1.23.0` is listed twice in `pyproject.toml`
- **Status:** [x] ✅ Completed — duplicate removed from `pyproject.toml`

### 2.3 Remove Deprecated Extractors from Public API
- **What:** `EntityExtractor` and `RelationshipExtractor` are deprecated but still exported
- **Status:** [x] ✅ Completed — removed from `rnsr/extraction/__init__.py` exports; utility functions (`merge_entities`, `extract_implicit_relationships`) kept

---

## Phase 3: Trust & Quality (CI/CD, Docs, Testing)

### 3.1 GitHub Actions CI/CD
- **What:** Automated pipeline on every push/PR
- **Steps:**
  - Lint (ruff)
  - Type check (mypy)
  - Test (pytest, fast subset)
  - Build package
  - Publish on tag (to PyPI)
- **Status:** [ ] Not started

### 3.2 Pre-Commit Hooks
- **What:** `.pre-commit-config.yaml` for ruff, mypy, trailing whitespace
- **Status:** [ ] Not started

### 3.3 API Documentation Site
- **What:** MkDocs or Sphinx site with full API reference, tutorials, architecture diagrams
- **Host:** GitHub Pages
- **Status:** [ ] Not started

### 3.4 Integration Tests with Real LLMs
- **What:** A small suite that actually calls an LLM (gated behind env var / CI secret)
- **Why:** Unit tests with mocks don't catch real-world regressions
- **Status:** [ ] Not started

---

## Phase 4: Production Readiness

### 4.1 Async Client API + Streaming
- **What:** `AsyncRNSRClient` with `async def query()` and streaming partial answers
- **Why:** Required for web apps, API servers, real-time UIs
- **Status:** [ ] Not started

### 4.2 Docker Image
- **What:** Dockerfile + docker-compose for running RNSR as a local API server
- **Target:** `docker run -p 8000:8000 -e OPENAI_API_KEY=... rnsr/server`
- **Status:** [ ] Not started

### 4.3 Rate Limiting & Retry Logic
- **What:** Built-in exponential backoff for LLM API calls
- **Status:** [ ] Not started

### 4.4 OpenTelemetry / Observability
- **What:** Tracing spans for each retrieval step (ingest → index → navigate → answer)
- **Why:** Enterprise users need to understand *why* an answer was produced
- **Status:** [ ] Not started

### 4.5 Thread Safety
- **What:** Fix global mutable `_llm_cache` and `_reasoning_memory` in agent module
- **Status:** [ ] Not started

---

## Phase 5: Expand Capabilities

### 5.1 Multi-Format Ingestion
- **What:** Support DOCX, HTML, PPTX, Markdown
- **Why:** Financial/legal workflows involve Word docs, SEC EDGAR HTML filings
- **Status:** [ ] Not started

### 5.2 Hosted Demo (Hugging Face Spaces)
- **What:** Deploy `demo.py` as a Gradio app on HF Spaces
- **Why:** "Try it yourself" is worth 1000 benchmark tables
- **Status:** [ ] Not started

### 5.3 Multi-Document Reasoning
- **What:** Lean into `cross_doc_navigator.py` — "Upload 10 annual reports, ask a question that spans all of them"
- **Status:** [ ] Not started

### 5.4 Embeddings for Anthropic/Gemini
- **What:** Currently only OpenAI embeddings are configured
- **Status:** [ ] Not started

### 5.5 Split LLM — Lightweight Nav Model + Full Synthesis Model
- **What:** Allow separate LLM configs for navigation/decomposition steps vs final synthesis. E.g. `OLLAMA_NAV_MODEL=qwen2.5-coder:7b` for pattern generation, decomposition, verification, entity extraction; `OLLAMA_MODEL=qwen2.5-coder:14b` for synthesis only.
- **Why:** Navigation steps produce short structured output (JSON, regex patterns) and don't need a large model. Running a smaller model for ~80% of LLM calls dramatically reduces latency on local hardware (e.g. single GPU) while preserving answer quality. Precursor to 6.1.
- **Scope:** Refactor `set_llm_function()` on navigator components to accept per-component LLM functions; add env vars for nav vs synthesis model; wire through `RNSRClient`.
- **Status:** [ ] Not started

---

## Phase 6: Research & Moat

### 6.1 Fine-Tuned Small Navigation Model
- **What:** Use provenance traces + successful navigation chains from `reasoning_memory` as training data to distill a small, fast model
- **Why:** Replace GPT-4 for navigation → 10x cost reduction
- **Status:** [ ] Not started

### 6.2 Publish a Paper
- **What:** Write up the RNSR architecture + benchmark results for arXiv / ACL / EMNLP
- **Why:** Academic credibility, citations, community adoption
- **Status:** [ ] Not started

---

## Priority Order

| Priority | Task | Impact |
|----------|------|--------|
| ✅ ~~P0~~ | ~~1.1–1.4 Run additional benchmarks~~ | ~~Credibility — blocks Show HN~~ **DONE** |
| ✅ ~~P0~~ | ~~2.1 Make torch optional~~ | ~~Adoption — users bounce on install~~ **DONE** |
| 🟠 P1 | 3.1 GitHub Actions CI/CD | Trust — prevent regressions |
| 🟠 P1 | 5.2 Hosted demo | Adoption — let people try it |
| 🟡 P2 | 4.1 Async client + streaming | Production use cases |
| 🟡 P2 | 3.3 API docs site | Developer experience |
| 🟡 P2 | 4.2 Docker image | Deployment simplicity |
| 🟢 P3 | 5.1 Multi-format ingestion | Feature expansion |
| 🟢 P3 | 5.5 Split LLM (nav + synthesis) | Local performance / precursor to 6.1 |
| 🟢 P3 | 6.1 Fine-tuned small model | Cost reduction / moat |
| 🟢 P3 | 6.2 Publish a paper | Long-term credibility |
