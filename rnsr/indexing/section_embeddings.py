"""
Section Embedding Index — O(log s) retrieval via FAISS ANN search.

Replaces O(s) BFS + regex scans with embedding-based approximate nearest
neighbour lookup.  Embeddings are computed once at ingestion and persisted
to an on-disk FAISS index alongside the SQLite store.

Usage (ingestion):
    idx = SectionEmbeddingIndex(store_path)
    idx.build(skeleton, kv_store, doc_id)

Usage (query):
    idx = SectionEmbeddingIndex(store_path)
    results = idx.search("What is the liability clause?", top_k=10)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
_INDEX_FILENAME = "section_embeddings.faiss"
_META_FILENAME = "section_embeddings_meta.json"

_model = None
_model_lock = __import__("threading").Lock()


def _get_model():
    """Lazy-load the sentence-transformer model (singleton)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


class SectionEmbeddingIndex:
    """FAISS-backed section embedding index for O(log s) retrieval."""

    def __init__(self, store_path: Path | str):
        self._store_path = Path(store_path)
        self._index_path = self._store_path / _INDEX_FILENAME
        self._meta_path = self._store_path / _META_FILENAME
        self._index = None
        self._meta: list[dict[str, str]] = []
        self._load()

    def _load(self):
        """Load persisted FAISS index and metadata if they exist."""
        if self._index_path.exists() and self._meta_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(self._index_path))
                with open(self._meta_path) as f:
                    self._meta = json.load(f)
                logger.debug(
                    "section_embeddings_loaded",
                    vectors=self._index.ntotal,
                    path=str(self._index_path),
                )
            except Exception as e:
                logger.warning("section_embeddings_load_failed", error=str(e))
                self._index = None
                self._meta = []

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def build(
        self,
        skeleton: dict[str, Any],
        kv_store: Any,
        doc_id: str,
        *,
        replace: bool = False,
    ) -> int:
        """Embed all sections and add to the FAISS index.

        Args:
            skeleton: node_id -> SkeletonNode mapping.
            kv_store: KVStore for full-text content.
            doc_id: Document identifier.
            replace: If True, remove existing vectors for this doc_id first.

        Returns:
            Number of vectors added.
        """
        import faiss

        if replace:
            self._remove_doc(doc_id)

        if self._index is None:
            self._index = faiss.IndexFlatIP(_EMBEDDING_DIM)

        texts: list[str] = []
        metas: list[dict[str, str]] = []

        for node_id, node in skeleton.items():
            header = getattr(node, "header", "") or ""
            summary = getattr(node, "summary", "") or ""
            content = (kv_store.get(node_id) or "")[:500]
            combined = f"{header}. {summary} {content}".strip()
            if len(combined) < 10:
                continue
            texts.append(combined)
            metas.append({
                "node_id": node_id,
                "doc_id": doc_id,
                "header": header[:200],
            })

        if not texts:
            return 0

        model = _get_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        self._index.add(embeddings)
        self._meta.extend(metas)
        self._save()

        logger.info(
            "section_embeddings_built",
            doc_id=doc_id,
            vectors_added=len(texts),
            total_vectors=self._index.ntotal,
        )
        return len(texts)

    def search(
        self,
        query: str,
        top_k: int = 15,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for sections most similar to a natural-language query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            doc_id: Optional filter to a specific document.

        Returns:
            List of dicts with keys: node_id, doc_id, header, score.
        """
        if not self.is_ready:
            return []

        model = _get_model()
        q_emb = model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        q_emb = np.asarray(q_emb, dtype=np.float32)

        k = min(top_k * 3 if doc_id else top_k, self._index.ntotal)
        scores, indices = self._index.search(q_emb, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            meta = self._meta[idx]
            if doc_id and meta["doc_id"] != doc_id:
                continue
            results.append({
                "node_id": meta["node_id"],
                "doc_id": meta["doc_id"],
                "header": meta["header"],
                "score": float(score),
            })
            if len(results) >= top_k:
                break

        return results

    def _remove_doc(self, doc_id: str):
        """Remove all vectors for a document (rebuild required)."""
        if not self.is_ready:
            return
        keep = [i for i, m in enumerate(self._meta) if m["doc_id"] != doc_id]
        if len(keep) == len(self._meta):
            return
        import faiss
        old_index = self._index
        self._index = faiss.IndexFlatIP(_EMBEDDING_DIM)
        if keep:
            vecs = np.array([old_index.reconstruct(i) for i in keep], dtype=np.float32)
            self._index.add(vecs)
        self._meta = [self._meta[i] for i in keep]
        self._save()

    def _save(self):
        """Persist the FAISS index and metadata to disk."""
        if self._index is None:
            return
        try:
            import faiss
            self._store_path.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self._index_path))
            with open(self._meta_path, "w") as f:
                json.dump(self._meta, f)
        except Exception as e:
            logger.warning("section_embeddings_save_failed", error=str(e))
