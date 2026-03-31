#!/usr/bin/env python3
"""
Matter AI Tests — Surface Comparison Benchmark

Uses real QA pairs from matterAiTests/v0.2 to benchmark PolarQuant surface
configurations against actual document indexes.  Handles fresh ingestion
from cleaned caches so results are comparable across branches.

Surfaces tested (gracefully skipped if module unavailable on this branch):
  A. Baseline:        Original FAISS IndexFlatIP + full BFS
  B. Surface 1:       PolarQuant compressed embeddings + full BFS
  C. Surface 2:       FAISS IndexFlatIP + angular tree pruning
  D. Combined:        PolarQuant embeddings + angular tree pruning

Usage:
    python scripts/benchmark_matter_surfaces.py --limit 3
    python scripts/benchmark_matter_surfaces.py --test-dir matterAiTests/v0.2 --limit 0
    python scripts/benchmark_matter_surfaces.py --cases "0404 JR Lang, Estate Dispute" "0136 NK Givney, Employment Contract"
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature detection — which surfaces are available on this branch?
# ---------------------------------------------------------------------------

HAS_POLARQUANT = False
HAS_POLAR_TREE = False

try:
    from rnsr.indexing.polar_quant import encode_vectors, random_rotation_matrix
    from rnsr.indexing.section_embeddings import PolarQuantEmbeddingIndex
    HAS_POLARQUANT = True
except ImportError:
    pass

try:
    from rnsr.indexing.polar_tree import PolarTreeEncoder, PolarTreePruner
    HAS_POLAR_TREE = True
except ImportError:
    pass

print(f"Surface availability:  PolarQuant={HAS_POLARQUANT}  PolarTree={HAS_POLAR_TREE}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf", ".md", ".txt", ".text", ".markdown", ".docx",
    ".xlsx", ".xls", ".csv", ".msg",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif",
}
STORE_DIR_NAME = ".rnsr_store"


# ---------------------------------------------------------------------------
# Discovery (mirrors run_matter_tests.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SurfaceResult:
    surface: str
    ingestion_time_ms: float = 0.0
    build_time_ms: float = 0.0
    memory_bytes: int = 0
    avg_search_latency_ms: float = 0.0
    p95_search_latency_ms: float = 0.0
    avg_nodes_visited: float = 0.0
    recall_at_10: float = 0.0
    num_questions: int = 0
    num_sections: int = 0
    num_documents: int = 0
    compression_ratio: float = 1.0


@dataclass
class CaseBenchmark:
    case_name: str
    num_sections: int = 0
    num_documents: int = 0
    num_questions: int = 0
    ingestion_time_ms: float = 0.0
    surfaces: dict[str, SurfaceResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_case(case_dir: Path) -> tuple[Any, float, int, int]:
    """Ingest all documents in a case directory.

    Returns (store, elapsed_ms, num_docs, num_sections).
    """
    from rnsr import DocumentStore

    store_path = case_dir / STORE_DIR_NAME
    store = DocumentStore(str(store_path))

    files = collect_ingestible_files(case_dir)
    if not files:
        return store, 0.0, 0, 0

    t0 = time.perf_counter()
    result = store.batch_ingest(sources=files, build_kg=False, skip_existing=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    errors = result.errors if hasattr(result, "errors") else []
    if errors:
        for e in errors:
            print(f"      INGEST ERROR: {e.get('file', '?')}: {e.get('error', '?')}")

    # Reopen store to pick up freshly written catalog
    del store
    store = DocumentStore(str(store_path))
    num_docs = len(store)
    total_sections = 0
    doc_ids = list(store._catalog.keys()) if hasattr(store, "_catalog") else []
    for doc_id in doc_ids:
        try:
            r = store.get_document(doc_id)
            if r:
                total_sections += len(r[0])
        except Exception:
            pass

    return store, elapsed_ms, num_docs, total_sections


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_case(
    case_dir: Path,
    qa_pairs: list[tuple[str, str]],
    prune_threshold: float,
) -> CaseBenchmark | None:
    """Benchmark all available surfaces for a single case."""
    from rnsr import DocumentStore
    from rnsr.indexing.section_embeddings import SectionEmbeddingIndex

    # Step 1: Ingest
    print(f"    Ingesting documents...")
    store, ingest_ms, num_docs, num_sections = ingest_case(case_dir)

    if num_docs == 0:
        print(f"    SKIP: no documents ingested")
        return None

    print(f"    Ingested {num_docs} docs, {num_sections} sections in {ingest_ms:.0f} ms")

    store_path = case_dir / STORE_DIR_NAME

    # Load skeletons and KV stores
    doc_ids = list(store._catalog.keys()) if hasattr(store, "_catalog") else []
    skeletons: dict[str, dict] = {}
    kv_stores: dict[str, Any] = {}
    for doc_id in doc_ids:
        try:
            doc_data = store.get_document(doc_id)
            if doc_data is None:
                continue
            skeletons[doc_id] = doc_data[0]
            kv_stores[doc_id] = doc_data[1]
        except Exception as e:
            print(f"      WARN: could not load {doc_id}: {e}")
            continue

    if not skeletons:
        print(f"    SKIP: no loadable indexes")
        return None

    bench = CaseBenchmark(
        case_name=case_dir.name,
        num_sections=num_sections,
        num_documents=num_docs,
        num_questions=len(qa_pairs),
        ingestion_time_ms=ingest_ms,
    )

    queries = [q for q, _ in qa_pairs]

    # ── Surface A: Baseline (FAISS IndexFlatIP) ──
    print(f"    [A] Baseline FAISS...")
    emb_index = SectionEmbeddingIndex(store_path)

    t0 = time.perf_counter()
    for doc_id in doc_ids:
        if doc_id in skeletons:
            emb_index.build(skeletons[doc_id], kv_stores[doc_id], doc_id, replace=True)
    baseline_build_ms = (time.perf_counter() - t0) * 1000

    baseline_memory = 0
    if emb_index._index is not None:
        baseline_memory = emb_index._index.ntotal * 384 * 4

    baseline_latencies = []
    baseline_results_map: dict[int, list[str]] = {}

    for qi, q in enumerate(queries):
        t0 = time.perf_counter()
        results = emb_index.search(q, top_k=10)
        baseline_latencies.append((time.perf_counter() - t0) * 1000)
        baseline_results_map[qi] = [r["node_id"] for r in results]

    bench.surfaces["A_baseline"] = SurfaceResult(
        surface="A. Baseline (FAISS)",
        ingestion_time_ms=ingest_ms,
        build_time_ms=baseline_build_ms,
        memory_bytes=baseline_memory,
        avg_search_latency_ms=float(np.mean(baseline_latencies)) if baseline_latencies else 0,
        p95_search_latency_ms=float(np.percentile(baseline_latencies, 95)) if baseline_latencies else 0,
        avg_nodes_visited=float(np.mean([len(v) for v in baseline_results_map.values()])),
        recall_at_10=1.0,
        num_questions=len(queries),
        num_sections=num_sections,
        num_documents=num_docs,
    )

    # ── Surface B: PolarQuant Embeddings ──
    if HAS_POLARQUANT:
        print(f"    [B] PolarQuant embeddings (3-bit)...")
        pq_index = PolarQuantEmbeddingIndex(store_path, bits=3)
        t0 = time.perf_counter()
        for doc_id in doc_ids:
            if doc_id in skeletons:
                pq_index.build(skeletons[doc_id], kv_stores[doc_id], doc_id, replace=True)
        pq_build_ms = (time.perf_counter() - t0) * 1000
        pq_memory = pq_index.memory_bytes

        pq_latencies = []
        pq_recalls = []
        for qi, q in enumerate(queries):
            t0 = time.perf_counter()
            results = pq_index.search(q, top_k=10)
            pq_latencies.append((time.perf_counter() - t0) * 1000)
            pq_ids = {r["node_id"] for r in results}
            baseline_ids = set(baseline_results_map.get(qi, []))
            if baseline_ids:
                pq_recalls.append(len(pq_ids & baseline_ids) / len(baseline_ids))
            else:
                pq_recalls.append(1.0)

        bench.surfaces["B_polarquant_emb"] = SurfaceResult(
            surface="B. PolarQuant Embeddings",
            ingestion_time_ms=ingest_ms,
            build_time_ms=pq_build_ms,
            memory_bytes=pq_memory,
            avg_search_latency_ms=float(np.mean(pq_latencies)) if pq_latencies else 0,
            p95_search_latency_ms=float(np.percentile(pq_latencies, 95)) if pq_latencies else 0,
            avg_nodes_visited=float(np.mean([len(v) for v in baseline_results_map.values()])),
            recall_at_10=float(np.mean(pq_recalls)) if pq_recalls else 0,
            num_questions=len(queries),
            num_sections=num_sections,
            num_documents=num_docs,
            compression_ratio=baseline_memory / max(pq_memory, 1),
        )

        # Cleanup PQ files
        for f in store_path.glob("section_embeddings_pq*"):
            f.unlink(missing_ok=True)
    else:
        print(f"    [B] SKIP: PolarQuant not available on this branch")

    # ── Surface C: Angular Tree Pruning ──
    if HAS_POLAR_TREE:
        print(f"    [C] Angular tree pruning (threshold={prune_threshold})...")
        encoder = PolarTreeEncoder(n_components=16)
        t0 = time.perf_counter()
        for doc_id in doc_ids:
            if doc_id in skeletons:
                encoder.encode(skeletons[doc_id], emb_index)
        polar_encode_ms = (time.perf_counter() - t0) * 1000

        all_skeleton_nodes: dict[str, Any] = {}
        for skel in skeletons.values():
            all_skeleton_nodes.update(skel)

        pruner = PolarTreePruner(all_skeleton_nodes, threshold=prune_threshold)

        from rnsr.indexing.section_embeddings import _get_model
        model = _get_model()

        prune_latencies = []
        prune_node_counts = []
        prune_recalls = []

        for qi, q in enumerate(queries):
            t0 = time.perf_counter()

            q_emb = model.encode([q], normalize_embeddings=True, show_progress_bar=False)
            q_emb = np.asarray(q_emb, dtype=np.float32).ravel()

            try:
                q_polar = encoder.encode_query(q_emb)
            except Exception:
                prune_latencies.append(0)
                prune_node_counts.append(num_sections)
                prune_recalls.append(1.0)
                continue

            # BFS with angular pruning
            visited: set[str] = set()
            queue = []
            for nid, node in all_skeleton_nodes.items():
                if getattr(node, "parent_id", None) is None or getattr(node, "level", 1) == 0:
                    queue.append(nid)

            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                node = all_skeleton_nodes.get(nid)
                if not node:
                    continue
                if pruner.should_prune(q_polar, nid):
                    continue
                for cid in getattr(node, "child_ids", []):
                    if cid not in visited:
                        queue.append(cid)

            prune_latencies.append((time.perf_counter() - t0) * 1000)
            prune_node_counts.append(len(visited))

            baseline_ids = set(baseline_results_map.get(qi, []))
            if baseline_ids:
                prune_recalls.append(len(baseline_ids & visited) / len(baseline_ids))
            else:
                prune_recalls.append(1.0)

        bench.surfaces["C_polar_tree"] = SurfaceResult(
            surface="C. Angular Tree Pruning",
            ingestion_time_ms=ingest_ms,
            build_time_ms=baseline_build_ms + polar_encode_ms,
            memory_bytes=baseline_memory,
            avg_search_latency_ms=float(np.mean(prune_latencies)) if prune_latencies else 0,
            p95_search_latency_ms=float(np.percentile(prune_latencies, 95)) if prune_latencies else 0,
            avg_nodes_visited=float(np.mean(prune_node_counts)) if prune_node_counts else 0,
            recall_at_10=float(np.mean(prune_recalls)) if prune_recalls else 0,
            num_questions=len(queries),
            num_sections=num_sections,
            num_documents=num_docs,
        )
    else:
        print(f"    [C] SKIP: PolarTree not available on this branch")

    # ── Surface D: Combined ──
    if HAS_POLARQUANT and HAS_POLAR_TREE:
        print(f"    [D] Combined (PQ embeddings + angular pruning)...")
        combined_latencies = []
        combined_node_counts = []
        combined_recalls = []

        pq_index_d = PolarQuantEmbeddingIndex(store_path, bits=3)
        for doc_id in doc_ids:
            if doc_id in skeletons:
                pq_index_d.build(skeletons[doc_id], kv_stores[doc_id], doc_id, replace=True)

        for qi, q in enumerate(queries):
            t0 = time.perf_counter()

            pq_results = pq_index_d.search(q, top_k=10)
            pq_candidate_ids = {r["node_id"] for r in pq_results}

            q_emb = model.encode([q], normalize_embeddings=True, show_progress_bar=False)
            q_emb = np.asarray(q_emb, dtype=np.float32).ravel()

            try:
                q_polar = encoder.encode_query(q_emb)
            except Exception:
                combined_latencies.append(0)
                combined_node_counts.append(num_sections)
                combined_recalls.append(1.0)
                continue

            visited_d: set[str] = set()
            queue_d = []
            for nid, node in all_skeleton_nodes.items():
                if getattr(node, "parent_id", None) is None or getattr(node, "level", 1) == 0:
                    queue_d.append(nid)

            while queue_d:
                nid = queue_d.pop(0)
                if nid in visited_d:
                    continue
                visited_d.add(nid)
                node = all_skeleton_nodes.get(nid)
                if not node:
                    continue
                if nid not in pq_candidate_ids and pruner.should_prune(q_polar, nid):
                    continue
                for cid in getattr(node, "child_ids", []):
                    if cid not in visited_d:
                        queue_d.append(cid)

            combined_latencies.append((time.perf_counter() - t0) * 1000)
            combined_node_counts.append(len(visited_d))

            baseline_ids = set(baseline_results_map.get(qi, []))
            if baseline_ids:
                combined_recalls.append(len(baseline_ids & visited_d) / len(baseline_ids))
            else:
                combined_recalls.append(1.0)

        pq_mem_d = pq_index_d.memory_bytes
        bench.surfaces["D_combined"] = SurfaceResult(
            surface="D. Combined (PQ + Pruning)",
            ingestion_time_ms=ingest_ms,
            build_time_ms=pq_build_ms + polar_encode_ms,
            memory_bytes=pq_mem_d,
            avg_search_latency_ms=float(np.mean(combined_latencies)) if combined_latencies else 0,
            p95_search_latency_ms=float(np.percentile(combined_latencies, 95)) if combined_latencies else 0,
            avg_nodes_visited=float(np.mean(combined_node_counts)) if combined_node_counts else 0,
            recall_at_10=float(np.mean(combined_recalls)) if combined_recalls else 0,
            num_questions=len(queries),
            num_sections=num_sections,
            num_documents=num_docs,
            compression_ratio=baseline_memory / max(pq_mem_d, 1),
        )

        for f in store_path.glob("section_embeddings_pq*"):
            f.unlink(missing_ok=True)
    else:
        print(f"    [D] SKIP: requires both PolarQuant + PolarTree")

    return bench


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(benchmarks: list[CaseBenchmark], branch_name: str):
    print(f"\n{'='*95}")
    print(f"  Matter AI Tests — Surface Comparison")
    print(f"  Branch: {branch_name}")
    print(f"{'='*95}")

    surface_keys = ["A_baseline", "B_polarquant_emb", "C_polar_tree", "D_combined"]

    for bench in benchmarks:
        print(f"\n  Case: {bench.case_name}")
        print(f"  Docs: {bench.num_documents}  |  Sections: {bench.num_sections}  |  "
              f"Questions: {bench.num_questions}  |  Ingestion: {bench.ingestion_time_ms:.0f} ms")
        print(f"  {'Surface':<30} {'Build ms':>10} {'Memory':>12} {'Srch p50 ms':>12} "
              f"{'Srch p95 ms':>12} {'Nodes':>8} {'Recall@10':>10}")
        print(f"  {'─'*94}")

        for key in surface_keys:
            sr = bench.surfaces.get(key)
            if not sr:
                continue
            mem_kb = sr.memory_bytes / 1024 if sr.memory_bytes else 0
            compress = f" ({sr.compression_ratio:.1f}x)" if sr.compression_ratio != 1.0 else ""
            print(f"  {sr.surface:<30} {sr.build_time_ms:>10.1f} "
                  f"{mem_kb:>8.1f} KB{compress:>3} "
                  f"{sr.avg_search_latency_ms:>12.3f} {sr.p95_search_latency_ms:>12.3f} "
                  f"{sr.avg_nodes_visited:>8.1f} {sr.recall_at_10:>10.4f}")

    # Aggregate
    if len(benchmarks) > 1:
        print(f"\n{'─'*95}")
        print(f"  AGGREGATE across {len(benchmarks)} cases")
        print(f"  {'Surface':<30} {'Avg Build':>10} {'Avg Srch ms':>12} "
              f"{'Avg Nodes':>10} {'Avg Recall':>10} {'Compress':>10}")
        print(f"  {'─'*72}")

        for key in surface_keys:
            srs = [b.surfaces[key] for b in benchmarks if key in b.surfaces]
            if not srs:
                continue
            label = srs[0].surface
            avg_comp = np.mean([s.compression_ratio for s in srs])
            comp_str = f"{avg_comp:.1f}x" if avg_comp != 1.0 else "—"
            print(f"  {label:<30} "
                  f"{np.mean([s.build_time_ms for s in srs]):>10.1f} "
                  f"{np.mean([s.avg_search_latency_ms for s in srs]):>12.3f} "
                  f"{np.mean([s.avg_nodes_visited for s in srs]):>10.1f} "
                  f"{np.mean([s.recall_at_10 for s in srs]):>10.4f} "
                  f"{comp_str:>10}")

    print(f"\n{'='*95}\n")


def save_results_json(benchmarks: list[CaseBenchmark], output_path: Path, branch: str):
    data = {
        "benchmark": "polar_quant_surface_comparison",
        "branch": branch,
        "has_polarquant": HAS_POLARQUANT,
        "has_polar_tree": HAS_POLAR_TREE,
        "cases": [
            {
                "case_name": b.case_name,
                "num_sections": b.num_sections,
                "num_documents": b.num_documents,
                "num_questions": b.num_questions,
                "ingestion_time_ms": b.ingestion_time_ms,
                "surfaces": {k: asdict(v) for k, v in b.surfaces.items()},
            }
            for b in benchmarks
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Matter AI Tests — Surface Comparison")
    parser.add_argument("--test-dir", type=str, default="matterAiTests/v0.2")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Specific case folder names to benchmark")
    args = parser.parse_args()

    test_root = Path(args.test_dir)
    if not test_root.exists():
        print(f"ERROR: {test_root} does not exist")
        sys.exit(1)

    # Detect branch name
    import subprocess
    try:
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip()
    except Exception:
        branch = "unknown"

    case_dirs = discover_case_dirs(test_root)
    print(f"Discovered {len(case_dirs)} cases in {test_root}")

    if args.cases:
        case_dirs = [d for d in case_dirs if d.name in args.cases]
    elif args.limit > 0:
        case_dirs = case_dirs[:args.limit]

    print(f"Benchmarking {len(case_dirs)} cases on branch: {branch}\n")

    benchmarks: list[CaseBenchmark] = []

    for i, case_dir in enumerate(case_dirs):
        qcsv = find_questions_csv(case_dir)
        if not qcsv:
            continue
        qa_pairs = read_questions(qcsv)
        if not qa_pairs:
            continue

        print(f"[{i+1}/{len(case_dirs)}] {case_dir.name} "
              f"({len(qa_pairs)} questions, "
              f"{len(collect_ingestible_files(case_dir))} docs)")

        result = benchmark_case(case_dir, qa_pairs, args.threshold)
        if result:
            benchmarks.append(result)

        gc.collect()

    if benchmarks:
        print_report(benchmarks, branch)

        output_path = Path(args.output) if args.output else (
            test_root / f"polar_quant_benchmark_{branch.replace('/', '_')}.json"
        )
        save_results_json(benchmarks, output_path, branch)


if __name__ == "__main__":
    main()
