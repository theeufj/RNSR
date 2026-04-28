#!/usr/bin/env python3
"""
Matter AI Tests — Baseline Benchmark

Ingests documents from matterAiTests/v0.2 from scratch and benchmarks
indexation + search performance using the standard FAISS IndexFlatIP
section embeddings against real QA pairs.

Usage:
    python scripts/benchmark_matter_surfaces.py --limit 3
    python scripts/benchmark_matter_surfaces.py --limit 0   # all cases
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

logger = structlog.get_logger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".md", ".txt", ".text", ".markdown", ".docx",
    ".xlsx", ".xls", ".csv", ".msg",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
}
STORE_DIR_NAME = ".rnsr_store"


def discover_case_dirs(root: Path) -> list[Path]:
    case_dirs: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        if any(f.lower() == "questions.csv" for f in filenames):
            case_dirs.append(Path(dirpath))
    case_dirs.sort(key=lambda p: str(p).lower())
    return case_dirs


def read_questions(csv_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            q, a = row[0].strip(), row[1].strip()
            if not q or q.lower() == "question":
                continue
            pairs.append((q, a))
    return pairs


def find_questions_csv(directory: Path) -> Path | None:
    for f in directory.iterdir():
        if f.is_file() and f.name.lower() == "questions.csv":
            return f
    return None


def collect_ingestible_files(directory: Path) -> list[Path]:
    files: list[Path] = []
    for f in sorted(directory.iterdir()):
        if not f.is_file():
            continue
        if "question" in f.name.lower():
            continue
        if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if f.name.startswith("."):
            continue
        files.append(f)
    return files


@dataclass
class CaseResult:
    case_name: str
    num_documents: int = 0
    num_sections: int = 0
    num_questions: int = 0
    ingestion_time_ms: float = 0.0
    build_time_ms: float = 0.0
    memory_bytes: int = 0
    avg_search_latency_ms: float = 0.0
    p50_search_latency_ms: float = 0.0
    p95_search_latency_ms: float = 0.0
    avg_results_returned: float = 0.0


def benchmark_case(case_dir: Path, qa_pairs: list[tuple[str, str]]) -> CaseResult | None:
    from rnsr import DocumentStore
    from rnsr.indexing.section_embeddings import SectionEmbeddingIndex

    store_path = case_dir / STORE_DIR_NAME
    store = DocumentStore(str(store_path))

    if len(store) == 0:
        files = collect_ingestible_files(case_dir)
        if not files:
            return None

        # Ingest
        t0 = time.perf_counter()
        result = store.batch_ingest(sources=files, build_kg=False, skip_existing=False)
        ingest_ms = (time.perf_counter() - t0) * 1000

        errors = result.errors if hasattr(result, "errors") else []
        if errors:
            for e in errors:
                print(f"      INGEST ERROR: {e.get('file', '?')}: {e.get('error', '?')}")

        # Reopen to pick up catalog
        del store
        store = DocumentStore(str(store_path))
    else:
        ingest_ms = 0.0
        print(f"    Using existing store ({len(store)} docs)")
    num_docs = len(store)
    if num_docs == 0:
        return None

    doc_ids = list(store._catalog.keys()) if hasattr(store, "_catalog") else []
    skeletons: dict[str, dict] = {}
    kv_stores: dict[str, Any] = {}
    total_sections = 0
    for doc_id in doc_ids:
        try:
            doc_data = store.get_document(doc_id)
            if doc_data is None:
                continue
            skeletons[doc_id] = doc_data[0]
            kv_stores[doc_id] = doc_data[1]
            total_sections += len(doc_data[0])
        except Exception as e:
            print(f"      WARN: could not load {doc_id}: {e}")

    if not skeletons:
        return None

    print(f"    Ingested {num_docs} docs, {total_sections} sections in {ingest_ms:.0f} ms")

    # Build embedding index
    emb_index = SectionEmbeddingIndex(store_path)
    t0 = time.perf_counter()
    for doc_id in doc_ids:
        if doc_id in skeletons:
            emb_index.build(skeletons[doc_id], kv_stores[doc_id], doc_id, replace=True)
    build_ms = (time.perf_counter() - t0) * 1000

    memory = 0
    if emb_index._index is not None:
        memory = emb_index._index.ntotal * 384 * 4

    # Search
    queries = [q for q, _ in qa_pairs]
    latencies = []
    result_counts = []

    for q in queries:
        t0 = time.perf_counter()
        results = emb_index.search(q, top_k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
        result_counts.append(len(results))

    return CaseResult(
        case_name=case_dir.name,
        num_documents=num_docs,
        num_sections=total_sections,
        num_questions=len(queries),
        ingestion_time_ms=ingest_ms,
        build_time_ms=build_ms,
        memory_bytes=memory,
        avg_search_latency_ms=float(np.mean(latencies)) if latencies else 0,
        p50_search_latency_ms=float(np.median(latencies)) if latencies else 0,
        p95_search_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0,
        avg_results_returned=float(np.mean(result_counts)) if result_counts else 0,
    )


def print_report(results: list[CaseResult]):
    print(f"\n{'='*100}")
    print(f"  Matter AI Tests — Baseline FAISS Benchmark")
    print(f"{'='*100}")

    print(f"\n  {'Case':<50} {'Docs':>5} {'Sects':>6} {'Qs':>4} "
          f"{'Ingest ms':>10} {'Build ms':>10} {'Memory':>10} "
          f"{'Srch p50':>10} {'Srch p95':>10}")
    print(f"  {'─'*95}")

    for r in results:
        mem_kb = r.memory_bytes / 1024
        print(f"  {r.case_name:<50} {r.num_documents:>5} {r.num_sections:>6} "
              f"{r.num_questions:>4} {r.ingestion_time_ms:>10.0f} "
              f"{r.build_time_ms:>10.1f} {mem_kb:>8.1f}KB "
              f"{r.p50_search_latency_ms:>10.3f} {r.p95_search_latency_ms:>10.3f}")

    print(f"\n  {'─'*95}")
    print(f"  {'AGGREGATE':<50} "
          f"{sum(r.num_documents for r in results):>5} "
          f"{sum(r.num_sections for r in results):>6} "
          f"{sum(r.num_questions for r in results):>4} "
          f"{np.mean([r.ingestion_time_ms for r in results]):>10.0f} "
          f"{np.mean([r.build_time_ms for r in results]):>10.1f} "
          f"{sum(r.memory_bytes for r in results)/1024:>8.1f}KB "
          f"{np.mean([r.p50_search_latency_ms for r in results]):>10.3f} "
          f"{np.mean([r.p95_search_latency_ms for r in results]):>10.3f}")
    print(f"{'='*100}\n")


def save_results(results: list[CaseResult], output_path: Path):
    data = {
        "benchmark": "baseline_faiss",
        "cases": [asdict(r) for r in results],
        "aggregate": {
            "total_cases": len(results),
            "total_documents": sum(r.num_documents for r in results),
            "total_sections": sum(r.num_sections for r in results),
            "total_questions": sum(r.num_questions for r in results),
            "avg_ingestion_ms": float(np.mean([r.ingestion_time_ms for r in results])),
            "avg_build_ms": float(np.mean([r.build_time_ms for r in results])),
            "avg_search_p50_ms": float(np.mean([r.p50_search_latency_ms for r in results])),
            "avg_search_p95_ms": float(np.mean([r.p95_search_latency_ms for r in results])),
            "total_memory_kb": sum(r.memory_bytes for r in results) / 1024,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Matter AI Tests — Baseline Benchmark")
    parser.add_argument("--test-dir", type=str, default="matterAiTests/v0.2")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cases", nargs="+", default=None)
    args = parser.parse_args()

    test_root = Path(args.test_dir)
    if not test_root.exists():
        print(f"ERROR: {test_root} does not exist")
        sys.exit(1)

    case_dirs = discover_case_dirs(test_root)
    print(f"Discovered {len(case_dirs)} cases in {test_root}")

    if args.cases:
        case_dirs = [d for d in case_dirs if d.name in args.cases]
    elif args.limit > 0:
        case_dirs = case_dirs[:args.limit]

    print(f"Benchmarking {len(case_dirs)} cases\n")

    results: list[CaseResult] = []

    for i, case_dir in enumerate(case_dirs):
        qcsv = find_questions_csv(case_dir)
        if not qcsv:
            continue
        qa_pairs = read_questions(qcsv)
        if not qa_pairs:
            continue

        print(f"[{i+1}/{len(case_dirs)}] {case_dir.name} "
              f"({len(qa_pairs)} questions, {len(collect_ingestible_files(case_dir))} docs)")

        try:
            r = benchmark_case(case_dir, qa_pairs)
            if r:
                results.append(r)
        except Exception as e:
            print(f"    ERROR: {e}")

        gc.collect()

        # Force-unload heavy models between cases
        try:
            from rnsr.indexing.section_embeddings import _MODEL_CACHE
        except ImportError:
            pass

    if results:
        print_report(results)

        output_path = Path(args.output) if args.output else (
            test_root / "baseline_benchmark_results.json"
        )
        save_results(results, output_path)


if __name__ == "__main__":
    main()
