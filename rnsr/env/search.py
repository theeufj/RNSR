"""The tiered search ladder (spec §5). Runs inside the sandbox child.

Rungs: 0 SQL (manifest-guided) · 1 regex over doc · 2 FTS5/BM25 ·
3 sub-LM term expansion (bounded) · 5 exhaustive sub-LM sweep (explicit
opt-in). rung=None auto-escalates 0→3 and returns the first rung that
yields hits; instead of silently running rung 5, it returns a cost
estimate the root LM must act on. Every rung is a view over retained
text — hits carry provenance back to doc/char offsets (§1.4).
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_STOP = frozenset("the and for was were with what which how many much does did".split())


def _terms(query: str) -> list[str]:
    return [t for t in _TOKEN.findall(query) if t.lower() not in _STOP][:12]


@dataclass
class Ladder:
    conn: sqlite3.Connection
    doc: dict[str, str]
    manifest: dict
    rpc: object                    # Child.rpc for rungs 3/5
    expansion_max_rounds: int = 3
    sweep_chunk_batch: int = 20    # chunks per rung-5 sub-call

    # --- public entry --------------------------------------------------------

    def search(self, query: str, rung: int | None = None, k: int = 10) -> list[dict]:
        if rung is not None:
            return self._run_rung(rung, query, k)
        for r in (0, 1, 2, 3, 4):
            try:
                hits = self._run_rung(r, query, k)
            except Exception:
                if r == 4:      # no embed handler / extension — rung stays dormant
                    continue
                raise
            if hits:
                return hits
        n_chunks = self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return [{
            "rung": 5, "kind": "estimate",
            "estimated_sub_calls": -(-n_chunks // self.sweep_chunk_batch),
            "note": ("rungs 0-4 found nothing. The exhaustive sweep is "
                     "available via search(query, rung=5) at the estimated "
                     "cost above."),
        }]

    def _run_rung(self, rung: int, query: str, k: int) -> list[dict]:
        fn = {0: self._rung0_sql, 1: self._rung1_grep, 2: self._rung2_fts,
              3: self._rung3_expand, 4: self._rung4_semantic,
              5: self._rung5_sweep}.get(rung)
        if fn is None:
            raise ValueError(f"no such rung: {rung} (0,1,2,3,4,5)")
        hits = fn(query, k)
        self._log(rung, query, len(hits))
        return hits

    def _log(self, rung: int, query: str, n_hits: int) -> None:
        try:
            self.rpc({"op": "log", "event": "search_rung", "rung": rung,
                      "query": query[:200], "hits": n_hits})
        except Exception:
            pass  # logging must never break a search

    # --- rung 0: manifest-guided SQL ----------------------------------------

    def _rung0_sql(self, query: str, k: int) -> list[dict]:
        terms = [t.lower() for t in _terms(query)]
        numbers = [n.replace(",", "") for n in _NUMBER.findall(query)]
        hits: list[dict] = []
        for table in self.manifest.get("tables", []):
            if table.get("status") == "untrusted":
                continue
            name = table["table_name"]
            schema = table.get("schema", [])
            col_names = {c["name"] for c in schema}
            # route to tables whose column names/caption overlap the query
            caption = (table.get("title") or "").lower()
            overlap = [t for t in terms
                       if any(t in c for c in col_names) or t in caption]
            if not overlap and not numbers:
                continue
            clauses, params = [], []
            for c in schema:
                if c["type"] == "TEXT":
                    for t in terms:
                        clauses.append(f'lower("{c["name"]}") LIKE ?')
                        params.append(f"%{t}%")
                elif numbers:
                    for n in numbers:
                        clauses.append(f'"{c["name"]}" = ?')
                        params.append(float(n))
            if not clauses:
                continue
            sql = (f'SELECT rowid, * FROM "{name}" WHERE '
                   + " OR ".join(clauses) + f" LIMIT {int(k)}")
            try:
                cur = self.conn.execute(sql, params)
            except sqlite3.Error:
                continue
            cols = [d[0] for d in cur.description]
            for row in cur.fetchall():
                record = dict(zip(cols, row, strict=True))
                data_cols = {k: v for k, v in record.items()
                             if k not in ("rowid", "_page", "_bbox", "_extractor")
                             and not k.endswith("__raw")}
                hits.append({
                    "rung": 0, "kind": "sql", "table": name, "rows": record,
                    # uniform fields shared with chunk hits — the root model
                    # reads hit['text']/hit['score'] regardless of rung
                    "text": json.dumps(data_cols, default=str),
                    "score": None,
                    "page": record.get("_page"),
                    "provenance": {"table": name, "rowid": record.get("rowid"),
                                   "_bbox": record.get("_bbox")},
                })
        return hits[:k]

    # --- rung 1: grep with priors -------------------------------------------

    def _rung1_grep(self, query: str, k: int, terms: list[str] | None = None) -> list[dict]:
        hits: list[dict] = []
        for term in (terms or _terms(query)):
            try:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
            except re.error:
                continue
            for doc_id, text in self.doc.items():
                for m in pattern.finditer(text):
                    start = max(m.start() - 150, 0)
                    end = min(m.end() + 150, len(text))
                    hits.append({
                        "rung": 1, "kind": "chunk", "doc_id": doc_id,
                        "term": term, "text": text[start:end],
                        "provenance": {"doc_id": doc_id, "char_start": m.start(),
                                       "char_end": m.end()},
                    })
                    if len(hits) >= k * 3:
                        break
        # dedupe overlapping windows, keep earliest per (doc, region)
        seen: set[tuple] = set()
        unique = []
        for h in hits:
            key = (h["provenance"]["doc_id"], h["provenance"]["char_start"] // 300)
            if key not in seen:
                seen.add(key)
                unique.append(h)
        return unique[:k]

    # --- rung 2: FTS5 --------------------------------------------------------

    def _rung2_fts(self, query: str, k: int) -> list[dict]:
        from rnsr.db import fts

        terms = _terms(query)
        match_query = " OR ".join(terms) if terms else query
        return [{
            "rung": 2, "kind": "chunk", "doc_id": h["doc_id"], "page": h["page"],
            "text": h["text"], "score": h["score"],
            "heading_path": h["heading_path"],
            "provenance": {"doc_id": h["doc_id"], "chunk_id": h["chunk_id"],
                           "char_start": h["char_start"], "char_end": h["char_end"]},
        } for h in fts.match(self.conn, match_query, k)]

    # --- rung 3: sub-LM expansion loop ---------------------------------------

    def _rung3_expand(self, query: str, k: int) -> list[dict]:
        tried: set[str] = {t.lower() for t in _terms(query)}
        frontier = list(tried)
        for _ in range(self.expansion_max_rounds):
            near_misses = self._rung2_fts(" ".join(frontier), 5)
            context = "\n".join(h["text"][:300] for h in near_misses[:5])
            prompt = (
                f"Search query: {query}\n"
                f"Terms already tried: {', '.join(sorted(tried))}\n"
                f"Nearby text from the corpus:\n{context}\n\n"
                "Propose up to 5 NEW search terms (synonyms, abbreviations, "
                "formatting variants) likely to find the answer in this "
                "corpus. Return one term per line, nothing else."
            )
            reply = self.rpc({"op": "llm_batch", "prompts": [prompt],
                              "model": "sub"})["results"][0]
            new_terms = [t.strip().strip("-• ").lower()
                         for t in reply.splitlines() if t.strip()][:5]
            new_terms = [t for t in new_terms if t and t not in tried]
            if not new_terms:
                break
            tried.update(new_terms)
            hits = self._rung1_grep(query, k, terms=new_terms)
            if hits:
                for h in hits:
                    h["rung"] = 3
                return hits
            frontier = new_terms
        return []

    # --- rung 4: lazy quantized embeddings (Phase D) --------------------------

    def _rung4_semantic(self, query: str, k: int) -> list[dict]:
        """Semantic top-k over the lazy int8 cache; first use pays the
        embedding cost once per corpus (§5). Requires the parent to expose
        an 'embed' RPC handler — raises otherwise (auto-escalation skips)."""
        from rnsr.env.embeddings import EmbeddingStore

        def embed(texts: list[str]) -> list[list[float]]:
            return self.rpc({"op": "embed", "texts": texts})["vectors"]

        store = EmbeddingStore(self.conn)
        if not store.ready():
            stats = store.ensure(embed, model="role:embed")
            self._log(4, f"(built cache: {stats})", 0)
        scored = store.knn(embed([query])[0], k)
        if not scored:
            return []
        ids = [cid for cid, _ in scored]
        marks = ",".join("?" * len(ids))
        rows = {r[0]: r for r in self.conn.execute(
            f"SELECT chunk_id, doc_id, page, char_start, char_end, text "
            f"FROM chunks WHERE chunk_id IN ({marks})", ids)}
        return [{
            "rung": 4, "kind": "chunk", "doc_id": rows[cid][1],
            "page": rows[cid][2], "text": rows[cid][5], "score": score,
            "provenance": {"doc_id": rows[cid][1], "chunk_id": cid,
                           "char_start": rows[cid][3], "char_end": rows[cid][4]},
        } for cid, score in scored if cid in rows]

    # --- rung 5: exhaustive sweep (opt-in) ------------------------------------

    def _rung5_sweep(self, query: str, k: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT chunk_id, doc_id, page, text FROM chunks ORDER BY chunk_id"
        ).fetchall()
        prompts = []
        groups: list[list] = []
        for i in range(0, len(rows), self.sweep_chunk_batch):
            group = rows[i : i + self.sweep_chunk_batch]
            groups.append(group)
            numbered = "\n\n".join(
                f"[{c[0]}] (doc={c[1]}, page={c[2]})\n{c[3]}" for c in group
            )
            prompts.append(
                f"Question: {query}\n\nChunks:\n{numbered}\n\n"
                "List the chunk ids (the numbers in brackets) that contain "
                "information answering the question, one per line. If none "
                "do, reply NONE."
            )
        replies = self.rpc({"op": "llm_batch", "prompts": prompts,
                            "model": "sub"})["results"]
        chunk_by_id = {c[0]: c for c in rows}
        hits = []
        for reply in replies:
            for m in re.finditer(r"\d+", reply or ""):
                c = chunk_by_id.get(int(m.group()))
                if c and all(h["provenance"]["chunk_id"] != c[0] for h in hits):
                    hits.append({
                        "rung": 5, "kind": "chunk", "doc_id": c[1], "page": c[2],
                        "text": c[3],
                        "provenance": {"doc_id": c[1], "chunk_id": c[0]},
                    })
        return hits[:k]


def _dumps(obj) -> str:
    return json.dumps(obj, default=repr)
