"""
Expandable Skeleton - Transparent Lazy Document Loading

Wraps a collection skeleton dict so that when the navigator accesses a
doc-stub node's children, the full document skeleton is loaded from disk
and grafted into the working tree.  The navigator sees a single unified
dict[str, SkeletonNode] and does not need to know about lazy expansion.

Usage:
    skel = ExpandableSkeleton(collection_nodes, lazy_kv, store_path)
    # navigator just does skel[node_id], skel.get(node_id) as normal
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator, MutableMapping

import structlog

from rnsr.indexing.collection_skeleton import DOC_PREFIX, doc_id_from_stub, is_doc_stub
from rnsr.indexing.kv_store import LazyKVStore
from rnsr.indexing.persistence import load_index
from rnsr.models import SkeletonNode

if TYPE_CHECKING:
    from rnsr.indexing.store_db import StoreDB

logger = structlog.get_logger(__name__)

_MAX_LOADED_DOCS = 5


class ExpandableSkeleton(MutableMapping[str, SkeletonNode]):
    """Dict-like wrapper that lazily expands doc-stub nodes.

    When a ``doc:<id>`` node is accessed via ``__getitem__`` or ``get()``,
    and it hasn't been expanded yet, the document's full skeleton is loaded
    from disk, grafted into the working tree, and the stub's ``child_ids``
    are replaced with the document root's children.

    An LRU eviction policy keeps at most ``max_loaded`` documents
    expanded simultaneously to bound memory usage.
    """

    def __init__(
        self,
        collection_nodes: dict[str, SkeletonNode],
        lazy_kv: LazyKVStore,
        store_path: Path | str,
        max_loaded: int = _MAX_LOADED_DOCS,
        store_db: StoreDB | None = None,
    ):
        self._nodes: dict[str, SkeletonNode] = dict(collection_nodes)
        self._lazy_kv = lazy_kv
        self._store_path = Path(store_path)
        self._max_loaded = max_loaded
        self._store_db = store_db

        # Track which docs are currently expanded and their node IDs
        self._expanded_docs: dict[str, list[str]] = {}   # doc_id → [node_ids grafted]
        self._load_order: list[str] = []                   # LRU order

    # ------------------------------------------------------------------
    # MutableMapping interface
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> SkeletonNode:
        self._maybe_expand(key)
        return self._nodes[key]

    def __setitem__(self, key: str, value: SkeletonNode) -> None:
        self._nodes[key] = value

    def __delitem__(self, key: str) -> None:
        del self._nodes[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, key: object) -> bool:
        return key in self._nodes

    def get(self, key: str, default: SkeletonNode | None = None) -> SkeletonNode | None:  # type: ignore[override]
        self._maybe_expand(key)
        return self._nodes.get(key, default)

    # ------------------------------------------------------------------
    # Expansion logic
    # ------------------------------------------------------------------

    def _maybe_expand(self, node_id: str) -> None:
        """If *node_id* is an unexpanded doc stub, expand it now."""
        if not is_doc_stub(node_id):
            return

        doc_id = doc_id_from_stub(node_id)
        if doc_id in self._expanded_docs:
            self._touch(doc_id)
            return

        stub = self._nodes.get(node_id)
        if stub is None:
            return
        if stub.child_ids:
            return

        self._expand_document(doc_id, node_id)

    def _expand_document(self, doc_id: str, stub_nid: str) -> None:
        """Load a document skeleton and graft it into the tree."""
        skeleton = None
        kv_store = None

        if self._store_db is not None:
            try:
                result = self._store_db.load_document(doc_id)
                if result is not None:
                    skeleton, kv_store, _tables = result
            except Exception as exc:
                logger.warning("expand_doc_load_failed", doc_id=doc_id, error=str(exc))
                return

        if skeleton is None:
            index_path = self._store_path / doc_id
            if not index_path.exists():
                logger.warning("expand_doc_not_found", doc_id=doc_id, path=str(index_path))
                return
            try:
                skeleton, kv_store, _tables = load_index(index_path)
            except Exception as exc:
                logger.warning("expand_doc_load_failed", doc_id=doc_id, error=str(exc))
                return

        # Find the document's root node
        root_node: SkeletonNode | None = None
        for node in skeleton.values():
            if node.level == 0:
                root_node = node
                break

        if root_node is None:
            logger.warning("expand_doc_no_root", doc_id=doc_id)
            return

        # Evict if at capacity
        self._evict_if_needed()

        # Graft all document nodes into the working skeleton
        grafted_ids: list[str] = []
        for nid, node in skeleton.items():
            if node.level == 0:
                # Reparent root's children under the stub
                continue
            # Set parent_id of root's direct children to the stub
            if node.parent_id == root_node.node_id:
                node.parent_id = stub_nid
            self._nodes[nid] = node
            grafted_ids.append(nid)

        # Update the stub to point to the document root's children
        stub = self._nodes.get(stub_nid)
        if stub:
            stub.child_ids = list(root_node.child_ids)
            # Carry forward root summary into stub if stub summary is minimal
            if len(stub.summary) < 50 and root_node.summary:
                stub.summary = root_node.summary

        # Register node IDs in the LazyKVStore routing table
        all_doc_nids = list(skeleton.keys())
        self._lazy_kv.register_doc_nodes(doc_id, all_doc_nids)
        self._lazy_kv.register_doc_store(doc_id, kv_store)

        self._expanded_docs[doc_id] = grafted_ids
        self._load_order.append(doc_id)

        logger.info(
            "document_expanded",
            doc_id=doc_id,
            grafted_nodes=len(grafted_ids),
            total_skeleton_size=len(self._nodes),
        )

    def _evict_if_needed(self) -> None:
        """Remove the oldest expanded document if we're at capacity."""
        while len(self._expanded_docs) >= self._max_loaded and self._load_order:
            oldest_id = self._load_order.pop(0)
            self._collapse_document(oldest_id)

    def _collapse_document(self, doc_id: str) -> None:
        """Remove a document's grafted nodes and restore the stub to unexpanded state."""
        grafted = self._expanded_docs.pop(doc_id, [])
        for nid in grafted:
            self._nodes.pop(nid, None)

        stub_nid = f"{DOC_PREFIX}{doc_id}"
        stub = self._nodes.get(stub_nid)
        if stub:
            stub.child_ids = []

        self._lazy_kv.unload_document(doc_id)

        logger.info(
            "document_collapsed",
            doc_id=doc_id,
            removed_nodes=len(grafted),
            total_skeleton_size=len(self._nodes),
        )

    def _touch(self, doc_id: str) -> None:
        """Move doc_id to end of LRU list."""
        if doc_id in self._load_order:
            self._load_order.remove(doc_id)
            self._load_order.append(doc_id)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_expanded(self, doc_id: str) -> bool:
        return doc_id in self._expanded_docs

    @property
    def expanded_count(self) -> int:
        return len(self._expanded_docs)

    def expand_all(self) -> None:
        """Eagerly expand every doc stub (useful for small collections)."""
        for nid in list(self._nodes.keys()):
            if is_doc_stub(nid):
                self._maybe_expand(nid)
