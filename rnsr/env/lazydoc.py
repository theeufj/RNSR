"""LazyDoc: the `doc` mapping backed by the corpus db with an LRU cache.

At 80 documents an eager dict is fine; at 18,000 it is gigabytes inside a
memory-capped sandbox. LazyDoc keeps the same Mapping interface the tools
and the model rely on (doc[id], doc.items(), len(doc)) while holding only
`cache_size` documents' text in memory at once.
"""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from collections.abc import Iterator, Mapping


class LazyDoc(Mapping):
    def __init__(self, conn: sqlite3.Connection, cache_size: int = 32):
        self._conn = conn
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_size = cache_size
        self._ids: list[str] = [
            r[0] for r in conn.execute("SELECT doc_id FROM documents ORDER BY doc_id")
        ]

    def __getitem__(self, doc_id: str) -> str:
        if doc_id in self._cache:
            self._cache.move_to_end(doc_id)
            return self._cache[doc_id]
        rows = self._conn.execute(
            "SELECT text FROM doc_text WHERE doc_id = ? ORDER BY page", (doc_id,)
        ).fetchall()
        if not rows:
            raise KeyError(doc_id)
        text = "".join(r[0] for r in rows)
        self._cache[doc_id] = text
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return text

    def __iter__(self) -> Iterator[str]:
        return iter(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:  # keep the REPL from printing gigabytes
        return f"<LazyDoc: {len(self._ids)} documents (db-backed, lazy)>"
