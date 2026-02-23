# Hacker News Post

## Title

Show HN: RNSR – 100% on FinanceBench with 0% hallucination by navigating documents like a human

## URL

https://github.com/theeufj/RNSR

## Text (paste this into the HN text box)

Hi HN,

I built RNSR (Recursive Neural-Symbolic Retriever), an open-source document retrieval system that scores 100% accuracy with 0% hallucinations on FinanceBench — the industry-standard benchmark for financial document Q&A, where GPT-4 RAG scores ~60% and Claude RAG ~65%.

**The core insight:** traditional RAG chunks documents and retrieves by vector similarity, which destroys hierarchical structure. RNSR instead preserves the full document tree and has the LLM *write code* to navigate it — like a human scanning a table of contents, drilling into sections, and cross-referencing.

**How it works:**

1. **Font Histogram → Document Tree.** We detect heading levels from font sizes (no ML, just statistics) and build a hierarchical tree. Section 4.2 knows it lives under Section 4.

2. **LLM writes navigation code.** Instead of embedding similarity, the LLM generates Python that calls `search_tree()`, `navigate_to()`, and `get_node_content()` in an iterative REPL loop. It drills deeper until it finds what it needs or honestly says "not found."

3. **Knowledge Graph + Grounding.** Entities and relationships are extracted in parallel (8 threads), stored in a KG, and verified against source text. If an entity can't be found as a substring in the original document, it's discarded.

4. **Zero hallucination by design.** Every date is regex-pre-scanned from the source text before the LLM sees it. Every answer carries provenance citations. If the LLM can't find reliable information, the system returns "unable to find" rather than guessing.

**Benchmarks (all reproducible with `make benchmark-compare`):**

| Method | Correctness | Hallucination |
|--------|------------|---------------|
| RNSR | 100% | 0% |
| Long Context LLM | 75% | 0% |
| Naive RAG | 50% | 50% |

We also benchmark timeline extraction (100% recall on 26/26 events across legal documents) and contradiction detection (100% recall on 11/11 known contradictions across single-doc and cross-doc scenarios).

**What makes this different from other RAG frameworks:**

- Not a wrapper around LangChain/LlamaIndex chunking. The document tree is the retrieval unit.
- The LLM navigates by writing code, not by embedding cosine similarity. This means it can handle "compare section 3.2 with section 5.1" — queries that break chunk-based RAG.
- Cross-document entity linking finds that "G. Sorenssen" in Document A is "GeoV William Sorenssen" in Document B.
- Six-strategy contradiction detection across documents (KG relationships, subject-gated heuristics, LLM semantic, structure-parallel section matching, entity-centric comparison, and relationship divergence).
- Works with any LLM provider (OpenAI, Anthropic, Gemini) with automatic fallback.

**Limitations / honest caveats:**

- Slower than naive RAG (~10s vs ~3s per query) because navigation is iterative.
- Accuracy on small-sample academic benchmarks (TAT-QA, QASPER, DocVQA) is 67% — failures are formatting/OCR issues, not retrieval failures. Extractive question types score 100%.
- FinanceBench results are on the standard dataset but our comparison benchmark uses a smaller document set. We'd love for others to reproduce and challenge these numbers.
- The font histogram approach works well for structured PDFs but degrades on poorly-formatted documents. We fall back to LLM-based hierarchical clustering when font signals are weak.

Stack: Python 3.9+, LlamaIndex, LangGraph, SQLite for KV/KG storage. No vector database required.

Would love feedback from anyone working on document understanding, RAG, or legal/financial AI. What benchmarks should we add? What failure modes have you seen in production RAG systems?

GitHub: https://github.com/theeufj/RNSR
