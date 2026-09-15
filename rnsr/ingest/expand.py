"""Expand containers (zip, email attachments) into child ParsedDocuments."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from rnsr.ingest.model import ParsedDocument

ParseFn = Callable[[Path], ParsedDocument]


def expand_document(parsed: ParsedDocument, parse: ParseFn,
                    seen_ids: set[str] | None = None) -> list[ParsedDocument]:
    """Return [parsed] plus recursively parsed attachments/zip members.

    Children keep ``parent_doc_id``. Dedupes doc_id against ``seen_ids``.
    """
    seen_ids = seen_ids if seen_ids is not None else set()
    if parsed.doc_id in seen_ids:
        parsed.doc_id = f"{parsed.doc_id}_{len(seen_ids)}"
    seen_ids.add(parsed.doc_id)
    out = [parsed]
    for name, data in parsed.pending_attachments:
        suffix = Path(name).suffix.lower() or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            child = parse(tmp_path)
        except Exception:
            continue
        finally:
            tmp_path.unlink(missing_ok=True)
        child.source_path = f"{parsed.source_path}::{name}"
        child.parent_doc_id = parsed.doc_id
        if not child.doc_id or child.doc_id == tmp_path.stem:
            stem = Path(name).stem
            child.doc_id = stem or f"{parsed.doc_id}_att"
        out.extend(expand_document(child, parse, seen_ids))
    parsed.pending_attachments = []
    return out
