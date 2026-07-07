"""GraphRAG-style baseline: a lean, honest reimplementation.

Index (once per corpus, cached in the corpus.db as derived tables):
  1. entity + relation extraction over every chunk (sub-model, batched)
  2. communities via union-find over relation edges
  3. one sub-model summary per community (largest first, capped)

Query: match question terms to entities -> gather their communities'
summaries + the top entity-linked chunks -> single answer call.

This mirrors the Microsoft GraphRAG local/global hybrid at benchmark
scale without its config machinery. Labeled as a reimplementation in all
reporting. Index cost is real and reported separately — amortizing it is
the pattern's own selling point.
"""

from __future__ import annotations

import json
import re
import sqlite3

from rnsr.llm.batch import map_prompts

_EXTRACT_PROMPT = """\
Extract entities and relations from this document excerpt.

Excerpt (id {chunk_id}):
{text}

Return ONLY JSON:
{{"entities": [{{"name": "...", "type": "person|org|document|amount|date|other"}}],
 "relations": [{{"src": "entity name", "rel": "verb phrase", "dst": "entity name"}}]}}
Use canonical names (e.g. "INV-2301", "Amendment No. 2"). Max 8 entities."""

_SUMMARY_PROMPT = """\
Summarize this community of related entities from a document corpus in
3-5 sentences. Include concrete facts: names, amounts, dates, statuses.

Entities: {entities}
Relations: {relations}
Sample source excerpts:
{samples}"""

_ANSWER_PROMPT = """\
Answer the question using the community summaries and source excerpts
below, drawn from a knowledge graph over the document corpus. If they do
not contain the answer, say so plainly rather than guessing.

COMMUNITY SUMMARIES:
{summaries}

SOURCE EXCERPTS:
{excerpts}

Question: {question}

Answer:"""

GRAPH_DDL = """
CREATE TABLE IF NOT EXISTS graph_entities (
  id INTEGER PRIMARY KEY, name TEXT UNIQUE, type TEXT, community INTEGER
);
CREATE TABLE IF NOT EXISTS graph_relations (
  src INTEGER, rel TEXT, dst INTEGER
);
CREATE TABLE IF NOT EXISTS graph_entity_chunks (
  entity_id INTEGER, chunk_id INTEGER
);
CREATE TABLE IF NOT EXISTS graph_summaries (
  community INTEGER PRIMARY KEY, summary TEXT, n_entities INTEGER
);
"""


def _parse_extraction(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj.get("entities"), list):
        return None
    obj.setdefault("relations", [])
    return obj


class _UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def graph_ready(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'graph_entities'").fetchone()
    if not row:
        return False
    return conn.execute("SELECT count(*) FROM graph_entities").fetchone()[0] > 0


async def build_graph_index(conn: sqlite3.Connection, client, model: str, *,
                            concurrency: int = 16,
                            max_summaries: int = 40,
                            on_usage=None) -> dict:
    """Extract -> connect -> summarize. Idempotent via graph_ready()."""
    if graph_ready(conn):
        return {"cached": True}
    conn.executescript(GRAPH_DDL)

    chunks = conn.execute(
        "SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    prompts = [_EXTRACT_PROMPT.format(chunk_id=cid, text=text[:2500])
               for cid, text in chunks]
    replies = await map_prompts(client, prompts, model=model,
                                concurrency=concurrency, max_tokens=800,
                                on_usage=on_usage)

    ids: dict[str, int] = {}
    next_id = [0]

    def entity_id(name: str, etype: str = "other") -> int:
        key = name.strip().lower()
        if key not in ids:
            next_id[0] += 1
            ids[key] = next_id[0]
            conn.execute("INSERT OR IGNORE INTO graph_entities (id, name, type) "
                         "VALUES (?, ?, ?)", (ids[key], name.strip(), etype))
        return ids[key]

    uf = _UnionFind()
    n_rel = 0
    for (cid, _), reply in zip(chunks, replies, strict=True):
        parsed = _parse_extraction(reply.text if reply else "")
        if not parsed:
            continue
        for e in parsed["entities"][:8]:
            if isinstance(e, dict) and e.get("name"):
                eid = entity_id(str(e["name"]), str(e.get("type", "other")))
                conn.execute("INSERT INTO graph_entity_chunks VALUES (?, ?)",
                             (eid, cid))
        for r in parsed["relations"]:
            if isinstance(r, dict) and r.get("src") and r.get("dst"):
                s, d = entity_id(str(r["src"])), entity_id(str(r["dst"]))
                conn.execute("INSERT INTO graph_relations VALUES (?, ?, ?)",
                             (s, str(r.get("rel", "related to")), d))
                uf.union(s, d)
                n_rel += 1

    # assign communities
    for _key, eid in ids.items():
        conn.execute("UPDATE graph_entities SET community = ? WHERE id = ?",
                     (uf.find(eid), eid))
    conn.commit()

    # summarize the largest communities
    comms = conn.execute(
        "SELECT community, count(*) n FROM graph_entities GROUP BY community "
        "ORDER BY n DESC LIMIT ?", (max_summaries,)).fetchall()
    sprompts = []
    for comm, _n in comms:
        ents = [r[0] for r in conn.execute(
            "SELECT name FROM graph_entities WHERE community = ? LIMIT 30", (comm,))]
        rels = conn.execute(
            "SELECT a.name, r.rel, b.name FROM graph_relations r "
            "JOIN graph_entities a ON a.id = r.src "
            "JOIN graph_entities b ON b.id = r.dst "
            "WHERE a.community = ? LIMIT 30", (comm,)).fetchall()
        samples = [r[0] for r in conn.execute(
            "SELECT DISTINCT c.text FROM chunks c "
            "JOIN graph_entity_chunks ec ON ec.chunk_id = c.chunk_id "
            "JOIN graph_entities e ON e.id = ec.entity_id "
            "WHERE e.community = ? LIMIT 3", (comm,))]
        sprompts.append(_SUMMARY_PROMPT.format(
            entities=", ".join(ents),
            relations="; ".join(f"{a} {rel} {b}" for a, rel, b in rels),
            samples="\n---\n".join(s[:800] for s in samples)))
    sreplies = await map_prompts(client, sprompts, model=model,
                                 concurrency=concurrency, max_tokens=400,
                                 on_usage=on_usage)
    for (comm, n), reply in zip(comms, sreplies, strict=True):
        conn.execute("INSERT OR REPLACE INTO graph_summaries VALUES (?, ?, ?)",
                     (comm, (reply.text if reply else "")[:2000], n))
    conn.commit()
    return {"cached": False, "entities": len(ids), "relations": n_rel,
            "communities": len(comms), "chunks": len(chunks)}


def graph_retrieve(conn: sqlite3.Connection, question: str, *,
                   k_chunks: int = 12, k_summaries: int = 6
                   ) -> tuple[list[str], list[tuple[str, str]]]:
    """Entity-match the question -> (community summaries, linked chunks)."""
    from rnsr.env.search import _terms

    terms = [t.lower() for t in _terms(question)]
    if not terms:
        terms = [question.lower()[:30]]
    clauses = " OR ".join("lower(name) LIKE ?" for _ in terms)
    rows = conn.execute(
        f"SELECT id, community FROM graph_entities WHERE {clauses}",
        [f"%{t}%" for t in terms]).fetchall()
    entity_ids = [r[0] for r in rows]
    comms = list({r[1] for r in rows if r[1] is not None})

    summaries = []
    if comms:
        marks = ",".join("?" * len(comms))
        summaries = [r[0] for r in conn.execute(
            f"SELECT summary FROM graph_summaries WHERE community IN ({marks}) "
            "ORDER BY n_entities DESC LIMIT ?", [*comms, k_summaries])]
    # always include the top global summaries too (GraphRAG 'global' flavor)
    for r in conn.execute(
            "SELECT summary FROM graph_summaries ORDER BY n_entities DESC LIMIT ?",
            (k_summaries,)):
        if r[0] not in summaries:
            summaries.append(r[0])
    summaries = summaries[:k_summaries]

    chunks: list[tuple[str, str]] = []
    if entity_ids:
        marks = ",".join("?" * len(entity_ids))
        chunks = conn.execute(
            f"SELECT DISTINCT c.doc_id, c.text FROM chunks c "
            f"JOIN graph_entity_chunks ec ON ec.chunk_id = c.chunk_id "
            f"WHERE ec.entity_id IN ({marks}) LIMIT ?",
            [*entity_ids, k_chunks]).fetchall()
    return summaries, chunks
