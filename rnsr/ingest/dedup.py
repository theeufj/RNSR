"""Exact and near-duplicate detection over ingested document text.

Exact matches use ``content_sha256``. Near-duplicates use a 64-bit
simhash of canonical text (hamming distance ≤ ``NEAR_DUP_BITS``).
Older / earlier copies point ``duplicate_of`` at the kept document.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3

NEAR_DUP_BITS = 3
_TOKEN = re.compile(r"[A-Za-z0-9]{2,}")


def canonical_text(text: str) -> str:
    return " ".join(_TOKEN.findall((text or "").lower()))


def simhash64(text: str) -> int:
    tokens = _TOKEN.findall((text or "").lower())
    if not tokens:
        return 0
    acc = [0] * 64
    for tok in tokens:
        h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "little")
        for i in range(64):
            acc[i] += 1 if h & (1 << i) else -1
    out = 0
    for i, v in enumerate(acc):
        if v >= 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _doc_text(conn: sqlite3.Connection, doc_id: str) -> str:
    rows = conn.execute(
        "SELECT text FROM doc_text WHERE doc_id = ? ORDER BY page", (doc_id,)
    ).fetchall()
    return "\n".join(r[0] or "" for r in rows)


def detect_duplicates(conn: sqlite3.Connection, *,
                      near_bits: int = NEAR_DUP_BITS) -> list[dict]:
    """Stamp ``documents.duplicate_of`` and return the duplicate groups.

    The kept document is the last by ``modified_at`` then ``doc_id``.
    """
    conn.execute("UPDATE documents SET duplicate_of=NULL")
    docs = [
        dict(r) if isinstance(r, sqlite3.Row) else {
            "doc_id": r[0], "content_sha256": r[1], "modified_at": r[2],
        }
        for r in conn.execute(
            "SELECT doc_id, content_sha256, modified_at FROM documents "
            "ORDER BY doc_id"
        )
    ]
    if len(docs) < 2:
        return []

    texts = {d["doc_id"]: canonical_text(_doc_text(conn, d["doc_id"])) for d in docs}
    hashes = {d["doc_id"]: simhash64(texts[d["doc_id"]]) for d in docs}

    def _keep(group: list[dict]) -> str:
        return sorted(
            group,
            key=lambda d: (d.get("modified_at") or "", d["doc_id"]),
        )[-1]["doc_id"]

    assigned: dict[str, str] = {}

    by_sha: dict[str, list[dict]] = {}
    for d in docs:
        sha = d.get("content_sha256")
        if sha:
            by_sha.setdefault(sha, []).append(d)
    for group in by_sha.values():
        if len(group) < 2:
            continue
        keeper = _keep(group)
        for d in group:
            if d["doc_id"] != keeper:
                assigned[d["doc_id"]] = keeper

    remaining = [d for d in docs if d["doc_id"] not in assigned]
    used = set(assigned)
    for i, a in enumerate(remaining):
        if a["doc_id"] in used:
            continue
        cluster = [a]
        for b in remaining[i + 1:]:
            if b["doc_id"] in used:
                continue
            if not texts[a["doc_id"]] or not texts[b["doc_id"]]:
                continue
            if hamming(hashes[a["doc_id"]], hashes[b["doc_id"]]) <= near_bits:
                cluster.append(b)
        if len(cluster) < 2:
            continue
        keeper = _keep(cluster)
        for d in cluster:
            if d["doc_id"] != keeper:
                assigned[d["doc_id"]] = keeper
                used.add(d["doc_id"])
        used.add(keeper)

    for doc_id, keeper in assigned.items():
        conn.execute(
            "UPDATE documents SET duplicate_of = ? WHERE doc_id = ?",
            (keeper, doc_id),
        )

    groups: dict[str, list[str]] = {}
    for doc_id, keeper in assigned.items():
        groups.setdefault(keeper, [keeper]).append(doc_id)
    return [
        {"kept": kept, "duplicates": sorted(set(ids) - {kept}), "kind": "content"}
        for kept, ids in sorted(groups.items())
    ]
