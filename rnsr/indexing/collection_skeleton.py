"""
Collection Skeleton - Folder-Hierarchy Navigation for Large Document Stores

Extends RNSR's recursive navigation from within-document to across-documents
by treating the folder hierarchy as the top levels of the skeleton tree.

Architecture:
    root_folder → subfolder → document → section → content

Folder nodes carry summaries aggregated from their children. Document stubs
are lightweight references (~200 bytes each) that get expanded lazily when
the navigator decides to "enter" a document.

Two operating modes:
  - Scoped (matter folder): flat list of doc stubs under root — behaves
    identically to the previous eager-loading approach.
  - Unscoped (company directory): full folder hierarchy with aggregated
    summaries at every level.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from rnsr.models import SkeletonNode

logger = structlog.get_logger(__name__)

COLLECTION_SKELETON_FILE = "collection_skeleton.json"

# Node-ID prefixes used to distinguish collection-level nodes from
# document-internal nodes.
FOLDER_PREFIX = "folder:"
DOC_PREFIX = "doc:"


def _folder_node_id(relative_path: str) -> str:
    """Canonical node ID for a folder."""
    return f"{FOLDER_PREFIX}{relative_path}" if relative_path else f"{FOLDER_PREFIX}."


def _doc_node_id(doc_id: str) -> str:
    """Canonical node ID for a document stub."""
    return f"{DOC_PREFIX}{doc_id}"


def is_folder_node(node_id: str) -> bool:
    return node_id.startswith(FOLDER_PREFIX)


def is_doc_stub(node_id: str) -> bool:
    return node_id.startswith(DOC_PREFIX)


def doc_id_from_stub(node_id: str) -> str:
    """Extract the real doc_id from a doc stub node_id."""
    return node_id[len(DOC_PREFIX):]


# -------------------------------------------------------------------------
# Building the collection skeleton
# -------------------------------------------------------------------------

class CollectionSkeletonBuilder:
    """Build a folder-hierarchy skeleton from a document catalog.

    Parameters
    ----------
    catalog : dict[str, dict]
        Mapping of ``doc_id`` → document info dicts.  Each info dict must
        contain at least ``title`` (str).  Optional keys:

        - ``source_path`` (str) – original file path
        - ``summary`` (str) – root-node summary from the document skeleton
        - ``node_count`` (int)
    root_path : Path | None
        If supplied, folder hierarchy is derived by computing each
        document's relative path from *root_path*.  When ``None``, all
        documents are placed directly under the root (flat / scoped mode).
    """

    def __init__(
        self,
        catalog: dict[str, dict[str, Any]],
        root_path: Path | str | None = None,
    ):
        self._catalog = catalog
        self._root_path = Path(root_path).resolve() if root_path else None
        self._nodes: dict[str, SkeletonNode] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> dict[str, SkeletonNode]:
        """Build the full collection skeleton and return it."""
        self._nodes.clear()

        folder_children: dict[str, list[str]] = {}  # rel_folder → child node_ids
        folder_summaries: dict[str, list[str]] = {}  # rel_folder → child summary texts

        for doc_id, info in self._catalog.items():
            stub_nid = _doc_node_id(doc_id)
            title = info.get("title", doc_id)
            summary = info.get("summary", f"Document: {title}")

            stub = SkeletonNode(
                node_id=stub_nid,
                parent_id=None,  # set below
                level=0,         # set below
                header=title,
                summary=summary,
                child_ids=[],    # expanded lazily
                metadata={
                    "doc_id": doc_id,
                    "is_doc_stub": True,
                    "source_path": info.get("source_path", ""),
                    "node_count": info.get("node_count", 0),
                },
            )
            self._nodes[stub_nid] = stub

            rel_folder = self._relative_folder(info.get("source_path"))
            folder_children.setdefault(rel_folder, []).append(stub_nid)
            folder_summaries.setdefault(rel_folder, []).append(
                f"{title}: {summary[:120]}"
            )

        all_folders = self._collect_ancestor_folders(folder_children.keys())

        for folder in sorted(all_folders, key=lambda f: f.count("/"), reverse=True):
            folder_nid = _folder_node_id(folder)
            direct_doc_stubs = folder_children.get(folder, [])
            direct_subfolders = [
                _folder_node_id(f)
                for f in all_folders
                if self._parent_folder(f) == folder and f != folder
            ]
            child_ids = sorted(direct_subfolders) + sorted(direct_doc_stubs)

            child_summary_parts = folder_summaries.get(folder, [])
            for sf in direct_subfolders:
                sf_node = self._nodes.get(sf)
                if sf_node:
                    child_summary_parts.append(f"[{sf_node.header}] {sf_node.summary[:80]}")

            summary = "; ".join(child_summary_parts[:20])
            if len(child_summary_parts) > 20:
                summary += f" ... and {len(child_summary_parts) - 20} more"

            folder_name = Path(folder).name if folder and folder != "." else "Collection Root"
            self._nodes[folder_nid] = SkeletonNode(
                node_id=folder_nid,
                parent_id=None,  # set below
                level=0,         # set below
                header=folder_name,
                summary=summary or f"Folder: {folder_name}",
                child_ids=child_ids,
                metadata={"is_folder": True, "relative_path": folder},
            )

        self._set_parents_and_levels()
        logger.info(
            "collection_skeleton_built",
            folders=sum(1 for n in self._nodes.values() if is_folder_node(n.node_id)),
            doc_stubs=sum(1 for n in self._nodes.values() if is_doc_stub(n.node_id)),
        )
        return self._nodes

    # ------------------------------------------------------------------
    # Incremental mutation
    # ------------------------------------------------------------------

    def add_doc_stub(
        self,
        doc_id: str,
        title: str,
        summary: str,
        source_path: str | None = None,
        node_count: int = 0,
    ) -> None:
        """Add a single document stub, creating parent folders as needed."""
        stub_nid = _doc_node_id(doc_id)
        if stub_nid in self._nodes:
            self._nodes[stub_nid].header = title
            self._nodes[stub_nid].summary = summary
            return

        info = {
            "title": title,
            "summary": summary,
            "source_path": source_path or "",
            "node_count": node_count,
        }
        rel_folder = self._relative_folder(source_path)
        folder_nid = _folder_node_id(rel_folder)

        stub = SkeletonNode(
            node_id=stub_nid,
            parent_id=folder_nid,
            level=0,
            header=title,
            summary=summary,
            child_ids=[],
            metadata={
                "doc_id": doc_id,
                "is_doc_stub": True,
                "source_path": source_path or "",
                "node_count": node_count,
            },
        )
        self._nodes[stub_nid] = stub

        self._ensure_folder_chain(rel_folder)

        if stub_nid not in self._nodes[folder_nid].child_ids:
            self._nodes[folder_nid].child_ids.append(stub_nid)
            self._update_folder_summary(folder_nid)

        self._set_parents_and_levels()

    def remove_doc_stub(self, doc_id: str) -> None:
        """Remove a document stub and clean up empty parent folders."""
        stub_nid = _doc_node_id(doc_id)
        stub = self._nodes.pop(stub_nid, None)
        if stub is None:
            return

        parent_nid = stub.parent_id
        if parent_nid and parent_nid in self._nodes:
            parent = self._nodes[parent_nid]
            if stub_nid in parent.child_ids:
                parent.child_ids.remove(stub_nid)
            self._update_folder_summary(parent_nid)
            self._prune_empty_folders(parent_nid)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, store_path: Path | str) -> Path:
        """Persist the collection skeleton to JSON."""
        out = Path(store_path) / COLLECTION_SKELETON_FILE
        data = {
            nid: {
                "node_id": n.node_id,
                "parent_id": n.parent_id,
                "level": n.level,
                "header": n.header,
                "summary": n.summary,
                "child_ids": n.child_ids,
                "metadata": n.metadata,
            }
            for nid, n in self._nodes.items()
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("collection_skeleton_saved", path=str(out), nodes=len(data))
        return out

    @classmethod
    def load(cls, store_path: Path | str) -> dict[str, SkeletonNode]:
        """Load a previously saved collection skeleton from JSON."""
        path = Path(store_path) / COLLECTION_SKELETON_FILE
        if not path.exists():
            return {}
        with open(path) as f:
            data = json.load(f)
        nodes: dict[str, SkeletonNode] = {}
        for nid, d in data.items():
            nodes[nid] = SkeletonNode(
                node_id=d["node_id"],
                parent_id=d.get("parent_id"),
                level=d.get("level", 0),
                header=d.get("header", ""),
                summary=d.get("summary", ""),
                child_ids=d.get("child_ids", []),
                metadata=d.get("metadata", {}),
            )
        logger.info("collection_skeleton_loaded", path=str(path), nodes=len(nodes))
        return nodes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _relative_folder(self, source_path: str | None) -> str:
        """Compute the relative folder for a source file.

        If no root_path was set or source_path is None, everything goes
        under "." (flat / scoped mode).
        """
        if not source_path or not self._root_path:
            return "."
        try:
            rel = Path(source_path).resolve().parent.relative_to(self._root_path)
            return str(rel) if str(rel) != "." else "."
        except ValueError:
            return "."

    @staticmethod
    def _parent_folder(folder: str) -> str:
        if not folder or folder == ".":
            return ""
        parent = str(Path(folder).parent)
        return parent if parent != folder else "."

    def _collect_ancestor_folders(self, leaf_folders) -> set[str]:
        """Given a set of leaf folders, return all ancestor folders including root."""
        folders: set[str] = set()
        for f in leaf_folders:
            current = f
            while current:
                folders.add(current)
                parent = self._parent_folder(current)
                if parent == current:
                    break
                current = parent
            folders.add(".")
        return folders

    def _ensure_folder_chain(self, rel_folder: str) -> None:
        """Create folder nodes for all ancestors up to root if they don't exist."""
        current = rel_folder
        while True:
            folder_nid = _folder_node_id(current)
            if folder_nid not in self._nodes:
                folder_name = Path(current).name if current and current != "." else "Collection Root"
                self._nodes[folder_nid] = SkeletonNode(
                    node_id=folder_nid,
                    parent_id=None,
                    level=0,
                    header=folder_name,
                    summary=f"Folder: {folder_name}",
                    child_ids=[],
                    metadata={"is_folder": True, "relative_path": current},
                )
            parent_folder = self._parent_folder(current)
            if parent_folder == current or not parent_folder:
                break
            parent_nid = _folder_node_id(parent_folder)
            if folder_nid not in self._nodes.get(parent_nid, SkeletonNode(
                node_id="", parent_id=None, level=0, header="", summary="", child_ids=[]
            )).child_ids:
                self._ensure_folder_chain(parent_folder)
                if parent_nid in self._nodes and folder_nid not in self._nodes[parent_nid].child_ids:
                    self._nodes[parent_nid].child_ids.append(folder_nid)
            current = parent_folder

    def _set_parents_and_levels(self) -> None:
        """Walk the tree from root to set parent_id and level consistently."""
        root_nid = _folder_node_id(".")
        if root_nid not in self._nodes:
            return

        self._nodes[root_nid].parent_id = None
        self._nodes[root_nid].level = 0
        queue = [(root_nid, 0)]
        while queue:
            nid, level = queue.pop(0)
            node = self._nodes[nid]
            for cid in node.child_ids:
                child = self._nodes.get(cid)
                if child:
                    child.parent_id = nid
                    child.level = level + 1
                    queue.append((cid, level + 1))

    def _update_folder_summary(self, folder_nid: str) -> None:
        """Regenerate a folder node's summary from its current children."""
        folder = self._nodes.get(folder_nid)
        if not folder:
            return
        parts = []
        for cid in folder.child_ids:
            child = self._nodes.get(cid)
            if child:
                parts.append(f"{child.header}: {child.summary[:80]}")
        folder.summary = "; ".join(parts[:20]) or f"Folder: {folder.header}"

    def _prune_empty_folders(self, folder_nid: str) -> None:
        """Remove folder nodes that have no children (except root)."""
        root_nid = _folder_node_id(".")
        node = self._nodes.get(folder_nid)
        if not node or folder_nid == root_nid:
            return
        if not node.child_ids:
            parent_nid = node.parent_id
            del self._nodes[folder_nid]
            if parent_nid and parent_nid in self._nodes:
                parent = self._nodes[parent_nid]
                if folder_nid in parent.child_ids:
                    parent.child_ids.remove(folder_nid)
                self._prune_empty_folders(parent_nid)
