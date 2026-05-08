"""Document registry for the API layer.

Maps API-level ``doc_id`` (a stable UUID assigned at upload/registration time)
to the on-disk PDF path. Persisted to a JSON file so the registry survives
process restarts.

The registry intentionally does NOT keep RNSR's internal cache key — that is
recomputed on demand by ``RNSRClient`` from the file's path/size/mtime. We only
need the ``doc_id -> path`` mapping plus a bit of metadata for listing.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class DocumentRecord:
    """A single registered document."""

    doc_id: str
    name: str
    path: str
    size_bytes: int
    source: str  # "upload" or "path"
    created_at: str
    indexed: bool = False
    knowledge_graph_built: bool = False
    metadata: dict = field(default_factory=dict)


class DocumentRegistry:
    """Thread-safe registry of documents known to the API.

    Persists to ``<storage_dir>/registry.json`` on every mutation.
    """

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir = self.storage_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.storage_dir / "registry.json"
        self._lock = threading.Lock()
        self._records: dict[str, DocumentRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self._registry_path.exists():
            return
        try:
            data = json.loads(self._registry_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        for raw in data.get("documents", []):
            try:
                rec = DocumentRecord(**raw)
                self._records[rec.doc_id] = rec
            except TypeError:
                continue

    def _save(self) -> None:
        payload = {
            "documents": [asdict(r) for r in self._records.values()],
        }
        tmp = self._registry_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self._registry_path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def register_path(self, path: Path) -> DocumentRecord:
        """Register an existing on-disk PDF without copying it."""
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        with self._lock:
            for existing in self._records.values():
                if Path(existing.path) == path:
                    return existing

            record = DocumentRecord(
                doc_id=uuid.uuid4().hex[:16],
                name=path.name,
                path=str(path),
                size_bytes=path.stat().st_size,
                source="path",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._records[record.doc_id] = record
            self._save()
            return record

    def register_upload(self, filename: str, content: bytes) -> DocumentRecord:
        """Save uploaded bytes under ``uploads/<doc_id>/<filename>``."""
        with self._lock:
            doc_id = uuid.uuid4().hex[:16]
            target_dir = self.uploads_dir / doc_id
            target_dir.mkdir(parents=True, exist_ok=True)

            safe_name = Path(filename).name or "document.pdf"
            target = target_dir / safe_name
            target.write_bytes(content)

            record = DocumentRecord(
                doc_id=doc_id,
                name=safe_name,
                path=str(target),
                size_bytes=len(content),
                source="upload",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._records[doc_id] = record
            self._save()
            return record

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(doc_id)

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return list(self._records.values())

    def update(self, doc_id: str, **fields) -> DocumentRecord | None:
        with self._lock:
            rec = self._records.get(doc_id)
            if rec is None:
                return None
            for key, value in fields.items():
                if hasattr(rec, key):
                    setattr(rec, key, value)
            self._save()
            return rec

    def remove(self, doc_id: str, delete_files: bool = False) -> bool:
        with self._lock:
            rec = self._records.pop(doc_id, None)
            if rec is None:
                return False
            self._save()

        if delete_files and rec.source == "upload":
            upload_dir = self.uploads_dir / doc_id
            if upload_dir.exists():
                import shutil

                shutil.rmtree(upload_dir, ignore_errors=True)
        return True
