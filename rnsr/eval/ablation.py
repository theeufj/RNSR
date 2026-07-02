"""Rung-4 quantization ablation (§8, required before either tier ships).

Measures recall@k of quantized coarse+rescore retrieval against exact fp32
top-k on the same embeddings. Acceptance: quantized recall@50 within 1% of
fp32. If int8 clears the bar at target corpus scale, the polar upgrade
path stays dormant.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from rnsr.env.embeddings import EmbeddingStore, Quantizer


def run_ablation(
    corpus_db: str | Path,
    embed_fn,
    queries: list[str],
    *,
    ks: tuple[int, ...] = (10, 50),
    quantizer: Quantizer | None = None,
    rescore_pool: int = 4000,
) -> dict:
    """embed_fn: list[str] -> list[list[float]] (sync)."""
    conn = sqlite3.connect(corpus_db)
    try:
        store = EmbeddingStore(conn, quantizer=quantizer, rescore_pool=rescore_pool)
        build = store.ensure(embed_fn, model="ablation")

        recalls: dict[int, list[float]] = {k: [] for k in ks}
        for qv in embed_fn(queries):
            for k in ks:
                exact = {cid for cid, _ in store.knn(qv, k, exact=True)}
                quant = {cid for cid, _ in store.knn(qv, k)}
                if exact:
                    recalls[k].append(len(exact & quant) / len(exact))
        report = {
            "quantizer": store.quantizer.name,
            "rescore_pool": rescore_pool,
            "n_queries": len(queries),
            "build": build,
            "recall": {f"@{k}": (sum(v) / len(v) if v else None)
                       for k, v in recalls.items()},
        }
        r50 = report["recall"].get("@50")
        report["accepts"] = r50 is not None and r50 >= 0.99
        return report
    finally:
        conn.close()
