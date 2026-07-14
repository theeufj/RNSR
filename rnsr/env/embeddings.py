"""Rung 4: lazy embeddings with a quantized cache (spec §5, Phase D).

Default tier: symmetric int8 quantization in a sqlite-vec ``vec0`` table
for the coarse KNN, with fp32 blobs kept alongside for exact rescoring of
the top candidates. Embeddings are computed on demand (first use pays once
per corpus) and written back into the same corpus.db — a compressed
*additional* view, never a replacement (§1.4).

The Quantizer interface is the seam for the PolarQuant upgrade path: it
slots in behind the same store if int8 fails the §8 recall bar at target
corpus scale.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Quantizer(Protocol):
    name: str

    def quantize(self, vecs: np.ndarray) -> list[bytes]: ...
    def coarse_topk(self, conn: sqlite3.Connection, query: np.ndarray,
                    pool: int) -> list[int]: ...


@dataclass
class Int8Quantizer:
    """Unit-normalize then scale to [-127, 127]; sqlite-vec native KNN."""

    name: str = "int8"

    def quantize(self, vecs: np.ndarray) -> list[bytes]:
        out = []
        for v in vecs:
            q = np.clip(np.round(_unit(v) * 127.0), -127, 127).astype(np.int8)
            out.append(q.tobytes())
        return out

    def coarse_topk(self, conn: sqlite3.Connection, query: np.ndarray,
                    pool: int) -> list[int]:
        q = np.clip(np.round(_unit(query) * 127.0), -127, 127).astype(np.int8)
        rows = conn.execute(
            "SELECT chunk_id FROM vec_chunks WHERE embedding MATCH vec_int8(?) "
            "AND k = ? ORDER BY distance",
            (q.tobytes(), pool),
        ).fetchall()
        return [r[0] for r in rows]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


class EmbeddingStore:
    """Lazy write-back embedding cache over the corpus chunks."""

    def __init__(self, conn: sqlite3.Connection, *,
                 quantizer: Quantizer | None = None,
                 rescore_pool: int = 4000):
        self.conn = conn
        self.quantizer = quantizer or Int8Quantizer()
        self.rescore_pool = rescore_pool
        self._load_vec_extension()

    def _load_vec_extension(self) -> None:
        import sqlite_vec

        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

    def ready(self) -> bool:
        return bool(self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'vec_chunks_fp32'"
        ).fetchone())

    def ensure(self, embed_fn, model: str, *, batch: int = 64) -> dict:
        """Embed all chunks not yet cached; write int8 + fp32 back. embed_fn:
        list[str] -> list[list[float]] (sync; the child RPCs to the parent)."""
        chunks = self.conn.execute(
            "SELECT chunk_id, text FROM chunks ORDER BY chunk_id"
        ).fetchall()
        have: set[int] = set()
        if self.ready():
            have = {r[0] for r in self.conn.execute(
                "SELECT chunk_id FROM vec_chunks_fp32")}
        todo = [(cid, text) for cid, text in chunks if cid not in have]
        if not todo:
            return {"embedded": 0, "cached": len(have), "model": model}

        first = np.asarray(embed_fn([todo[0][1]])[0], dtype=np.float32)
        dim = first.shape[0]
        if not self.ready():
            self.conn.execute(
                f"CREATE VIRTUAL TABLE vec_chunks USING vec0("
                f"chunk_id INTEGER PRIMARY KEY, embedding int8[{dim}])"
            )
            self.conn.execute(
                "CREATE TABLE vec_chunks_fp32 ("
                "chunk_id INTEGER PRIMARY KEY, embedding BLOB NOT NULL)"
            )
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS embedding_meta "
                "(model TEXT, dim INTEGER, quantization TEXT)"
            )
            self.conn.execute("INSERT INTO embedding_meta VALUES (?,?,?)",
                              (model, dim, self.quantizer.name))
            self.conn.execute(
                "INSERT OR REPLACE INTO manifest VALUES ('rung4_meta', ?)",
                (json.dumps({"model": model, "dim": dim,
                             "quantization": self.quantizer.name}),),
            )

        done = 0
        vectors = [first]
        pending_ids = [todo[0][0]]
        for i in range(1, len(todo), batch):
            group = todo[i : i + batch]
            vecs = embed_fn([t for _, t in group])
            vectors += [np.asarray(v, dtype=np.float32) for v in vecs]
            pending_ids += [cid for cid, _ in group]
        arr = np.stack(vectors)
        int8_blobs = self.quantizer.quantize(arr)
        for cid, v, qb in zip(pending_ids, arr, int8_blobs, strict=True):
            self.conn.execute("INSERT INTO vec_chunks VALUES (?, vec_int8(?))",
                              (cid, qb))
            self.conn.execute("INSERT INTO vec_chunks_fp32 VALUES (?, ?)",
                              (cid, struct.pack(f"{len(v)}f", *v)))
            done += 1
        self.conn.commit()
        return {"embedded": done, "cached": len(have), "model": model, "dim": dim}

    def _fp32(self, chunk_ids: list[int]) -> dict[int, np.ndarray]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            f"SELECT chunk_id, embedding FROM vec_chunks_fp32 "
            f"WHERE chunk_id IN ({marks})", chunk_ids).fetchall()
        return {cid: np.frombuffer(blob, dtype=np.float32) for cid, blob in rows}

    def knn(self, query_vec, k: int = 10, *, exact: bool = False) -> list[tuple[int, float]]:
        """(chunk_id, cosine) top-k: quantized coarse pass + fp32 rescore.
        exact=True skips the coarse pass (ablation ground truth)."""
        q = _unit(np.asarray(query_vec, dtype=np.float32))
        if exact:
            ids = [r[0] for r in self.conn.execute(
                "SELECT chunk_id FROM vec_chunks_fp32")]
        else:
            ids = self.quantizer.coarse_topk(self.conn, q, self.rescore_pool)
        vecs = self._fp32(ids)
        scored = sorted(
            ((cid, float(np.dot(q, _unit(v)))) for cid, v in vecs.items()),
            key=lambda x: -x[1],
        )
        return scored[:k]
