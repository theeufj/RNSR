#!/usr/bin/env python3
"""
PolarQuant Embedding Compression Benchmark

Compares the PolarQuant-compressed embedding index against the baseline
FAISS IndexFlatIP on:
  - recall@1, recall@5, recall@10
  - index memory (bytes)
  - build time
  - search latency (p50, p95, p99)

Generates synthetic embedding data when no real index is available,
so the benchmark is self-contained and can run without API keys.

Usage:
    python scripts/benchmark_polar_embeddings.py
    python scripts/benchmark_polar_embeddings.py --vectors 5000 --dim 384
    python scripts/benchmark_polar_embeddings.py --bits 3 4
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_synthetic_data(
    n_vectors: int,
    dim: int,
    n_queries: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate L2-normalized synthetic embeddings mimicking sentence-transformers."""
    rng = np.random.RandomState(seed)
    # Clustered data to simulate real embedding distributions
    n_clusters = max(1, n_vectors // 50)
    centers = rng.randn(n_clusters, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    labels = rng.randint(0, n_clusters, size=n_vectors)
    vectors = centers[labels] + rng.randn(n_vectors, dim).astype(np.float32) * 0.15
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    q_labels = rng.randint(0, n_clusters, size=n_queries)
    queries = centers[q_labels] + rng.randn(n_queries, dim).astype(np.float32) * 0.15
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    return vectors.astype(np.float32), queries.astype(np.float32)


def baseline_faiss_search(vectors: np.ndarray, query: np.ndarray, top_k: int = 10):
    """Exact inner-product search via FAISS (ground truth)."""
    import faiss

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    scores, indices = index.search(query.reshape(1, -1), top_k)
    return indices[0], scores[0]


def polar_quant_search(
    vectors: np.ndarray,
    query: np.ndarray,
    top_k: int = 10,
    bits: int = 3,
    rotation: np.ndarray | None = None,
):
    """PolarQuant approximate search."""
    from rnsr.indexing.polar_quant import (
        approximate_inner_product,
        encode_vectors,
        random_rotation_matrix,
    )

    dim = vectors.shape[1]
    if rotation is None:
        rotation = random_rotation_matrix(dim)

    qv = encode_vectors(vectors, rotation, bits=bits)
    scores = approximate_inner_product(query, qv, rotation, dim)
    ranked = np.argsort(-scores)[:top_k]
    return ranked, scores[ranked], qv, rotation


def compute_recall(gt_indices: np.ndarray, pred_indices: np.ndarray, k: int) -> float:
    gt_set = set(gt_indices[:k].tolist())
    pred_set = set(pred_indices[:k].tolist())
    return len(gt_set & pred_set) / k


def run_benchmark(n_vectors: int, dim: int, bits_list: list[int], n_queries: int = 50):
    from rnsr.indexing.polar_quant import (
        encode_vectors,
        random_rotation_matrix,
    )

    print(f"\n{'='*70}")
    print(f"  PolarQuant Embedding Benchmark")
    print(f"  vectors={n_vectors}  dim={dim}  queries={n_queries}")
    print(f"{'='*70}")

    vectors, queries = generate_synthetic_data(n_vectors, dim, n_queries)

    # -- Baseline: FAISS IndexFlatIP --
    import faiss

    print("\n[Baseline] FAISS IndexFlatIP (float32)")
    t0 = time.perf_counter()
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(vectors)
    faiss_build_ms = (time.perf_counter() - t0) * 1000

    faiss_memory = n_vectors * dim * 4  # float32

    faiss_latencies = []
    gt_results = []
    for q in queries:
        t0 = time.perf_counter()
        indices, scores = baseline_faiss_search(vectors, q, top_k=10)
        faiss_latencies.append((time.perf_counter() - t0) * 1000)
        gt_results.append(indices)

    faiss_latencies = np.array(faiss_latencies)
    print(f"  Build time:    {faiss_build_ms:8.2f} ms")
    print(f"  Memory:        {faiss_memory:>10,} bytes ({faiss_memory/1024:.1f} KB)")
    print(f"  Latency p50:   {np.percentile(faiss_latencies, 50):8.3f} ms")
    print(f"  Latency p95:   {np.percentile(faiss_latencies, 95):8.3f} ms")
    print(f"  Latency p99:   {np.percentile(faiss_latencies, 99):8.3f} ms")

    # -- PolarQuant at each bit level --
    rotation = random_rotation_matrix(dim)

    for bits in bits_list:
        print(f"\n[PolarQuant] {bits}-bit quantization")

        t0 = time.perf_counter()
        qv = encode_vectors(vectors, rotation, bits=bits)
        pq_build_ms = (time.perf_counter() - t0) * 1000

        n = qv.radii.shape[0]
        angle_cols = qv.quantized_angles.shape[1] if qv.quantized_angles.ndim == 2 else 0
        pq_memory = (n * 2) + (n * angle_cols) + (dim * dim * 4)  # radii + angles + rotation

        pq_latencies = []
        recalls_at_1 = []
        recalls_at_5 = []
        recalls_at_10 = []

        from rnsr.indexing.polar_quant import approximate_inner_product

        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            scores = approximate_inner_product(q, qv, rotation, dim)
            ranked = np.argsort(-scores)[:10]
            pq_latencies.append((time.perf_counter() - t0) * 1000)

            gt = gt_results[i]
            recalls_at_1.append(compute_recall(gt, ranked, 1))
            recalls_at_5.append(compute_recall(gt, ranked, 5))
            recalls_at_10.append(compute_recall(gt, ranked, 10))

        pq_latencies = np.array(pq_latencies)
        compression_ratio = faiss_memory / pq_memory

        print(f"  Build time:    {pq_build_ms:8.2f} ms")
        print(f"  Memory:        {pq_memory:>10,} bytes ({pq_memory/1024:.1f} KB)")
        print(f"  Compression:   {compression_ratio:8.1f}x vs baseline")
        print(f"  Latency p50:   {np.percentile(pq_latencies, 50):8.3f} ms")
        print(f"  Latency p95:   {np.percentile(pq_latencies, 95):8.3f} ms")
        print(f"  Latency p99:   {np.percentile(pq_latencies, 99):8.3f} ms")
        print(f"  Recall@1:      {np.mean(recalls_at_1):8.4f}")
        print(f"  Recall@5:      {np.mean(recalls_at_5):8.4f}")
        print(f"  Recall@10:     {np.mean(recalls_at_10):8.4f}")

    # -- Round-trip fidelity check --
    print(f"\n{'='*70}")
    print("  Round-trip fidelity (encode -> decode -> compare)")
    print(f"{'='*70}")

    from rnsr.indexing.polar_quant import decode_vectors

    for bits in bits_list:
        qv = encode_vectors(vectors[:100], rotation, bits=bits)
        reconstructed = decode_vectors(qv, rotation, dim)
        mse = np.mean((vectors[:100] - reconstructed) ** 2)
        cosine_sims = np.sum(vectors[:100] * reconstructed, axis=1)
        print(f"  {bits}-bit: MSE={mse:.6f}  mean_cosine={np.mean(cosine_sims):.6f}")

    print(f"\n{'='*70}")
    print("  Benchmark complete.")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="PolarQuant Embedding Benchmark")
    parser.add_argument("--vectors", type=int, default=2000, help="Number of vectors")
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--queries", type=int, default=50, help="Number of queries")
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4], help="Bit widths to test")
    args = parser.parse_args()

    run_benchmark(args.vectors, args.dim, args.bits, args.queries)


if __name__ == "__main__":
    main()
