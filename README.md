# RNSR — DocDB-RLM

A typed-environment recursive language model (RLM) system for deep document
retrieval. Documents are ingested into a single self-contained SQLite artifact
(`corpus.db`) — typed tables with provenance, FTS5 chunks, a machine-derived
manifest — and queried by a depth-1 RLM loop over a sandboxed Python REPL.

The authoritative design is [`docdb-rlm-design-spec.md`](docdb-rlm-design-spec.md).

## Install

```bash
pip install -e .              # query-time core (a prebuilt corpus.db is enough)
pip install -e ".[ingest]"    # + Docling parsing stack (heavy) for ingestion
pip install -e ".[eval,dev]"  # benchmarks + dev tooling
```

## Usage

```bash
rnsr ingest report.pdf -o corpus.db     # Phase A: parse -> tables -> validate -> FTS -> manifest
rnsr query corpus.db "What was FY2023 segment revenue?"   # Phase B/C
rnsr eval --benchmark financebench --system docdb          # §8 harness
```

## Development

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[ingest,eval,dev]"
pytest            # LLM-free by default; live tests are opt-in (-m live)
ruff check .
```

Implementation phases (spec §10): **A** deterministic ingestion → **B** RLM
harness → **C** fusion + go/no-go evaluation → **D** hardening (quantized
rung-4 embeddings, schema_map, per-model prompts).
