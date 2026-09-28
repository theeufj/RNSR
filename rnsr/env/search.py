"""The tiered search ladder (spec §5). Runs inside the sandbox child.

Rungs: 0 SQL (manifest-guided) · 1 regex over doc · 2 FTS5/BM25 ·
3 sub-LM term expansion (bounded) · 5 exhaustive sub-LM sweep (explicit
opt-in). rung=None auto-escalates 0→3 and returns the first rung that
yields hits; instead of silently running rung 5, it returns a cost
estimate the root LM must act on. Every rung is a view over retained
text — hits carry provenance back to doc/char offsets (§1.4).
"""

from __future__ import annotations

import contextlib
import heapq
import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict

from rnsr.db.schema import PROVENANCE_COLUMNS, quote_ident
from rnsr.env.evidence import SourceContext
from rnsr.harness.prompts.search import render_expansion, render_sweep

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}|\b\d+(?:[,.]\d+)*\b")
_YEAR = re.compile(r"(?<!\d)[12]\d{3}(?!\d)")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
STOP_WORDS = frozenset(["the", "and", "for", "was", "were", "with", "what", "which", "how", "many", "much", "does", "did"])


def terms(query: str) -> list[str]:
    found = [t for t in TOKEN.findall(query) if t.lower() not in STOP_WORDS]
    # Keep numeric constraints even when they occur late in a long question.
    return list(dict.fromkeys(found[:12] + [t for t in found if t[0].isdigit()]))


class SearchHit(TypedDict):
    rung: int
    kind: Literal["sql", "chunk", "estimate"]
    text: str
    score: float | None
    provenance: dict
    doc_id: NotRequired[str]
    table: NotRequired[str]
    rows: NotRequired[dict]
    page: NotRequired[int | None]


@dataclass
class Ladder:
    conn: sqlite3.Connection
    doc: dict[str, str]
    manifest: dict
    rpc: object                    # Child.rpc for rungs 3/5
    expansion_max_rounds: int = 3
    sweep_chunk_batch: int = 20    # chunks per rung-5 sub-call
    enable_embeddings: bool = True
    rebuild_cells: bool = False  # public replay/validation seam
    _cell_relation: str | None = field(default=None, init=False, repr=False)
    _embedding_store: object | None = field(default=None, init=False, repr=False)
    _context: SourceContext = field(init=False, repr=False)

    def __post_init__(self):
        self._context = SourceContext(self.conn, self.doc)

    # --- public entry --------------------------------------------------------

    def search(self, query: str, rung: int | None = None, k: int = 10) -> list[SearchHit]:
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= 1000:
            raise ValueError("k must be an integer between 0 and 1000")
        if k == 0:
            return []
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
            "rung": 5, "kind": "estimate", "text": "", "score": None, "provenance": {},
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
        for hit in hits:
            if "source_context" not in hit and hit.get("doc_id") is not None:
                p = hit.get("provenance", {})
                if p.get("char_start") is not None and p.get("char_end") is not None:
                    hit["source_context"] = self._context(hit["doc_id"], char_start=p["char_start"],
                                                         char_end=p["char_end"])
                    hit.setdefault("heading_path", " > ".join(hit["source_context"]["heading_paths"]) or None)
            self._rank(hit, query)
            hit.setdefault("score", None)
            hit.setdefault("provenance", {})
        self._log(rung, query, len(hits))
        return hits

    def _log(self, rung: int, query: str, n_hits: int) -> None:
        # logging must never break a search
        with contextlib.suppress(Exception):
            self.rpc({"op": "log", "event": "search_rung", "rung": rung,
                      "query": query[:200], "hits": n_hits})

    # --- rung 0: manifest-guided SQL ----------------------------------------

    def _rung0_sql(self, query: str, k: int) -> list[dict]:
        query_terms = [t.lower() for t in terms(query)]
        numbers = [n.replace(",", "") for n in _NUMBER.findall(query)]
        return self._rung0_cells(query_terms, numbers, k, query=query)

    def _cells_ready(self) -> bool:
        """Whether the source carries its optional persisted derived index."""
        return bool(self.conn.execute("SELECT 1 FROM cells LIMIT 1").fetchone())

    def _cell_source(self) -> str:
        if self._cell_relation is not None:
            return self._cell_relation
        if not self.rebuild_cells and self._cells_ready():
            self._cell_relation = "main.cells"
            return self._cell_relation
        # Old/opt-out artifacts get a private derived projection, not a
        # second search algorithm. TEMP tables are writable on a ro handle.
        self.conn.execute("CREATE TEMP TABLE IF NOT EXISTS search_cells ("
                          "table_name TEXT,row_idx INTEGER,text_value TEXT,num_value REAL)")
        self.conn.execute("DELETE FROM temp.search_cells")
        for table in self.manifest.get("tables", []):
            name = table["table_name"]
            for col in table.get("schema", []):
                if col.get("annotation"):
                    continue
                value = quote_ident(col["name"])
                raw = quote_ident(col.get("raw_col") or col["name"])
                rows = self.conn.execute(
                    f"SELECT rowid,{value},{raw} FROM {quote_ident(name)}")
                self.conn.executemany("INSERT INTO temp.search_cells VALUES (?,?,?,?)", [
                    (name, rid, None if text is None else str(text).lower(),
                     val if col.get("raw_col") is not None else None)
                    for rid, val, text in rows if val is not None or text is not None])
        self._cell_relation = "temp.search_cells"
        return self._cell_relation

    def _routed_tables(self, terms: list[str], numbers: list[str]) -> list[str]:
        """Rung-0 routing gate: probe only trusted tables whose
        column names or caption overlap the query, or any trusted table
        when the query carries numbers.

        This gate is load-bearing beyond performance: natural-language
        queries that match no table must yield NO rung-0 hits so the
        auto-escalating ladder reaches FTS prose chunks — weak table-row
        hits here would stop the escalation with worse evidence (seen
        live on the golden matter: address/email answers regressed when
        the gate was dropped)."""
        routed = []
        for table in self.manifest.get("tables", []):
            if table.get("status") == "untrusted":
                continue
            col_names = {c["name"] for c in table.get("schema", [])}
            caption = (table.get("title") or "").lower()
            overlap = [t for t in terms
                       if any(t in c for c in col_names) or t in caption]
            if overlap or numbers:
                routed.append(table["table_name"])
        return routed

    def _rung0_cells(self, terms: list[str], numbers: list[str],
                     k: int, *, query: str = "") -> list[dict]:
        """One scan of the derived cells index instead of LIKE over every
        routed t_* table. Text probes match text cells; numeric probes
        match numeric cells. Keep at most k rows per table, then rank all
        table candidates with their retained section context before the
        global cap. Alphabetical table order must not hide a later year.
        """
        routed = self._routed_tables(terms, numbers)
        clauses, params = [], []
        for t in terms:
            if t[0].isdigit():
                continue
            clauses.append("(num_value IS NULL AND text_value LIKE ?)")
            params.append(f"%{t}%")
        for n in numbers:
            clauses.append("num_value = ?")
            params.append(float(n))
        if not clauses or not routed:
            return []
        # dedupe (table,row) BEFORE the window: a row matching several cells
        # would otherwise carry several rn values, and duplicates of the
        # alphabetically-first table exhaust the LIMIT before later tables
        # are reached (seen live: ten copies of one row as the whole result)
        sql = (
            "SELECT table_name, row_idx FROM ("
            "  SELECT table_name, row_idx,"
            "         ROW_NUMBER() OVER (PARTITION BY table_name"
            "                            ORDER BY row_idx) AS rn"
            f"  FROM (SELECT DISTINCT table_name, row_idx FROM {self._cell_source()}"
            "        WHERE (" + " OR ".join(clauses) + ")"
            "        AND table_name IN ("
            + ",".join("?" * len(routed)) + "))"
            ") WHERE rn <= ? ORDER BY table_name, rn"
        )
        located = self.conn.execute(
            sql, [*params, *routed, k])
        ranked = []
        serial = 0
        for table, row_idx in located:
            try:
                cur = self.conn.execute(
                    f"SELECT rowid, * FROM {quote_ident(table)} WHERE rowid = ?",
                    (row_idx,))
            except sqlite3.Error:
                continue
            row = cur.fetchone()
            if row is None:
                continue
            cols = [d[0] for d in cur.description]
            record = dict(zip(cols, row, strict=True))
            hit = self._sql_hit(table, record)
            rank = self._rank(hit, query)
            serial += 1
            item = (rank, -serial, hit)
            if len(ranked) < k:
                heapq.heappush(ranked, item)
            elif item[:2] > ranked[0][:2]:
                heapq.heapreplace(ranked, item)
        return [h for _, _, h in sorted(ranked, key=lambda item: item[:2], reverse=True)]

    def _sql_hit(self, name: str, record: dict) -> dict:
        data_cols = {k: v for k, v in record.items()
                     if k != "rowid" and k not in PROVENANCE_COLUMNS
                     and not k.endswith("__raw")}
        context = self._context(table=name, rowid=record.get("rowid"))
        headings = " > ".join(context["heading_paths"]) or "(no section heading retained)"
        label = "Section" if context["row_location"] == "exact" else "Page headings (row-to-section unverified)"
        return {
            "rung": 0, "kind": "sql", "table": name, "rows": record,
            # uniform fields shared with chunk hits — the root model
            # reads hit['text']/hit['score'] regardless of rung
            "text": (f"Source: {context['doc_id']} (page {context['page']})\n"
                     f"{label}: {headings}\n"
                     + (f"Table caption: {context['title']}\n" if context.get('title') else "")
                     + json.dumps(data_cols, default=str)),
            "doc_id": context["doc_id"],
            "source_context": context,
            "heading_path": " > ".join(context["heading_paths"]) or None,
            "score": None,
            "page": record.get("_page"),
            "provenance": {"table": name, "rowid": record.get("rowid"),
                           "doc_id": context["doc_id"], "source_path": context["source_path"],
                           "_bbox": record.get("_bbox")},
        }

    # --- rung 1: grep with priors -------------------------------------------

    def _rung1_grep(self, query: str, k: int, expanded_terms: list[str] | None = None) -> list[dict]:
        query_terms = expanded_terms or terms(query)
        if not query_terms:
            return []
        pattern = re.compile('|'.join(re.escape(t) for t in sorted(query_terms, key=len, reverse=True)),
                             re.IGNORECASE)
        ranked = []
        serial = 0
        for doc_id, text in self.doc.items():
            seen = set()
            for match in pattern.finditer(text):
                context = self._context(doc_id, char_start=match.start(), char_end=match.end(), max_chars=600)
                # A new section can begin inside the same 300-character
                # bucket: never let an earlier section suppress its hits.
                section = tuple(s['char_start'] for s in context['sections'])
                region = (section, match.start() // 300)
                if region in seen:
                    continue
                seen.add(region)
                hit = {
                    "rung": 1, "kind": "chunk", "doc_id": doc_id,
                    "term": match.group(), "text": context["text"], "page": context["page"],
                    "heading_path": " > ".join(context["heading_paths"]) or None,
                    "source_context": context,
                    "provenance": {"doc_id": doc_id, "char_start": match.start(),
                                   "char_end": match.end(), "source_path": context["source_path"]},
                }
                serial += 1
                item = (self._rank(hit, query), -serial, hit)
                if len(ranked) < k:
                    heapq.heappush(ranked, item)
                elif item[:2] > ranked[0][:2]:
                    heapq.heapreplace(ranked, item)
        return [h for _, _, h in sorted(ranked, key=lambda item: item[:2], reverse=True)]

    def _rank(self, hit: dict, query: str) -> tuple:
        """Prefer query coverage and explicit section-year alignment, without
        asserting that an unknown or conflicting section cannot be relevant.
        Multi-year sections stay visible for comparative questions/tables.
        """
        requested = set(_YEAR.findall(query))
        context = hit.get('source_context', {})
        source_years = set(_YEAR.findall(' '.join(context.get('heading_paths', []))))
        if hit.get('kind') == 'sql' and context.get('row_location') != 'exact':
            source_years = set()
        alignment = ('unknown' if not source_years or not requested else
                     'match' if requested <= source_years else 'conflict')
        hit['query_scope'] = {'requested_years': sorted(requested),
                              'section_years': sorted(source_years), 'year_alignment': alignment}
        haystack = (hit['text'] + ' ' + ' '.join(context.get('heading_paths', []))).casefold()
        coverage = sum(t.casefold() in haystack for t in terms(query))
        # No inferred year: a filename or adjacent next-section heading is not
        # evidence that the retrieved row belongs to the requested fiscal year.
        return ({'match': 2, 'unknown': 1, 'conflict': 0}[alignment], coverage)

    # --- rung 2: FTS5 --------------------------------------------------------

    def _rung2_fts(self, query: str, k: int) -> list[dict]:
        from rnsr.db import fts

        query_terms = terms(query)
        match_query = " OR ".join('"' + t.replace('"', '""') + '"' for t in query_terms) if query_terms else query
        return [{
            "rung": 2, "kind": "chunk", "doc_id": h["doc_id"], "page": h["page"],
            "text": h["text"], "score": h["score"],
            "heading_path": h["heading_path"],
            "provenance": {"doc_id": h["doc_id"], "chunk_id": h["chunk_id"],
                           "char_start": h["char_start"], "char_end": h["char_end"]},
        } for h in fts.match(self.conn, match_query, k)]

    # --- rung 3: sub-LM expansion loop ---------------------------------------

    def _rung3_expand(self, query: str, k: int) -> list[dict]:
        tried: set[str] = {t.lower() for t in terms(query)}
        frontier = list(tried)
        for _ in range(self.expansion_max_rounds):
            near_misses = self._rung2_fts(" ".join(frontier), 5)
            context = "\n".join(h["text"][:300] for h in near_misses[:5])
            prompt = render_expansion(query, tried, context)
            reply = self.rpc({"op": "llm_batch", "prompts": [prompt],
                              "model": "sub"})["results"][0]
            new_terms = [t.strip().strip("-• ").lower()
                         for t in reply.splitlines() if t.strip()][:5]
            new_terms = [t for t in new_terms if t and t not in tried]
            if not new_terms:
                break
            tried.update(new_terms)
            hits = self._rung1_grep(query, k, expanded_terms=new_terms)
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

        if not self.enable_embeddings:
            return []
        if self._embedding_store is None:
            self._embedding_store = EmbeddingStore(self.conn, cache_conn=sqlite3.connect(""))
        store = self._embedding_store
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
            "SELECT chunk_id, doc_id, page, text, char_start, char_end "
            "FROM chunks ORDER BY chunk_id"
        ).fetchall()
        prompts = []
        groups: list[list] = []
        for i in range(0, len(rows), self.sweep_chunk_batch):
            group = rows[i : i + self.sweep_chunk_batch]
            groups.append(group)
            numbered = "\n\n".join(
                f"[{c[0]}] (doc={c[1]}, page={c[2]})\n{c[3]}" for c in group
            )
            prompts.append(render_sweep(query, numbered))
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
                        "provenance": {"doc_id": c[1], "chunk_id": c[0],
                                       "char_start": c[4], "char_end": c[5]},
                    })
        return hits[:k]
