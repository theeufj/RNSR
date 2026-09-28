"""Read-only source context for retrieved rows and exact quote offsets.

Headings come from retained chunk boundaries, not a nearby following heading
or a filename date. Table rows are located in canonical page text when possible;
otherwise page-level context is explicitly marked as ambiguous.
"""
from __future__ import annotations

import sqlite3
from collections import OrderedDict
from collections.abc import Callable

from rnsr.db.metadata import decode_table_schema
from rnsr.db.schema import quote_ident


class SourceContext:
    def __init__(self, conn: sqlite3.Connection, doc=None, *, check: Callable[[], None] | None = None):
        self.conn = conn
        self._doc = doc
        self._check = check
        self._cache: OrderedDict[str, tuple] = OrderedDict()

    def _checked(self, values):
        for value in values:
            if self._check is not None:
                self._check()
            yield value
        if self._check is not None:
            self._check()

    def _find(self, text, needle, start=0):
        if self._check is None:
            return text.find(needle, start)
        # Search large retained pages in overlapping bounded windows so Python
        # work cooperates with the same parent deadline as its SQLite reads.
        while start <= len(text) - len(needle):
            self._check()
            offset = text.find(needle, start, start + 65536 + len(needle) - 1)
            if offset >= 0:
                return offset
            start += 65536
        self._check()
        return -1

    def _metadata(self, doc_id: str):
        if self._check is not None:
            self._check()
        if doc_id not in self._cache:
            source = self.conn.execute(
                "SELECT source_path FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
            if source is None:
                raise KeyError(doc_id)
            pages = list(self._checked(self.conn.execute(
                "SELECT page,char_start,char_end FROM doc_text WHERE doc_id=? ORDER BY page",
                (doc_id,))))
            sections = []
            for start, end, heading in self._checked(self.conn.execute(
                "SELECT char_start,char_end,heading_path FROM chunks "
                    "WHERE doc_id=? ORDER BY char_start,chunk_id", (doc_id,))):
                if sections and sections[-1]['heading_path'] == heading and start <= sections[-1]['char_end']:
                    sections[-1]['char_end'] = max(end, sections[-1]['char_end'])
                else:
                    sections.append({'heading_path': heading, 'char_start': start, 'char_end': end})
            self._cache[doc_id] = (source[0], pages, sections)
            while len(self._cache) > 32:
                self._cache.popitem(last=False)
        self._cache.move_to_end(doc_id)
        return self._cache[doc_id]

    def __call__(self, doc_id: str | None = None, *, char_start: int | None = None,
                 char_end: int | None = None, page: int | None = None,
                 table: str | None = None, rowid: int | None = None,
                 max_chars: int = 600) -> dict:
        if self._check is not None:
            self._check()
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 1 <= max_chars <= 4000:
            raise ValueError('max_chars must be an integer between 1 and 4000')
        if table is not None:
            if doc_id is not None or char_start is not None or char_end is not None or page is not None:
                raise ValueError('use either a table/rowid or a document/span/page')
            return self._table(table, rowid, max_chars)
        if doc_id is None:
            raise ValueError('source_context requires doc_id or table')
        source, pages, sections = self._metadata(doc_id)
        if page is not None:
            selected = next((p for p in self._checked(pages) if p[0] == page), None)
            if selected is None:
                raise ValueError('page does not exist in document')
            if char_start is None:
                char_start, char_end = selected[1:]
        if char_start is None:
            raise ValueError('supply char_start/char_end or page')
        char_end = char_start + 1 if char_end is None else char_end
        if (isinstance(char_start, bool) or isinstance(char_end, bool)
                or not isinstance(char_start, int) or not isinstance(char_end, int)
                or not 0 <= char_start < char_end <= (pages[-1][2] if pages else 0)):
            raise ValueError('invalid canonical source span')
        relevant = [s.copy() for s in self._checked(sections)
                    if s['char_start'] < char_end and s['char_end'] > char_start]
        heading_paths = list(dict.fromkeys(s['heading_path'] for s in self._checked(relevant) if s['heading_path']))
        # A snippet never crosses a section boundary just to include a nearby
        # year. If the submitted span itself crosses sections, expose both.
        lower = relevant[0]['char_start'] if relevant else 0
        upper = relevant[-1]['char_end'] if relevant else pages[-1][2]
        start = max(lower, char_start - min(150, max_chars // 4))
        end = min(upper, max(char_end, start + max_chars), start + max_chars)
        if self._doc is not None:
            text = self._doc[doc_id][start:end]
        else:
            text = ''.join(row[0] for row in self._checked(self.conn.execute(
                "SELECT text FROM doc_text WHERE doc_id=? AND char_start<? AND char_end>? ORDER BY page",
                (doc_id, end, start))))
            first_page = next(p for p in self._checked(pages) if p[1] <= start < p[2])
            text = text[start - first_page[1]:end - first_page[1]]
        actual_page = next(p[0] for p in self._checked(pages) if p[1] <= char_start < p[2])
        if self._check is not None:
            self._check()
        return {'doc_id': doc_id, 'source_path': source, 'page': actual_page,
                'char_start': char_start, 'char_end': char_end,
                'heading_paths': heading_paths, 'sections': relevant,
                'text': text, 'text_char_start': start, 'text_char_end': start + len(text)}

    def _table(self, table: str, rowid: int | None, max_chars: int) -> dict:
        meta = self.conn.execute(
            "SELECT doc_id,title,page_start,page_end,schema_json FROM manifest_tables WHERE table_name=?",
            (table,)).fetchone()
        if meta is None:
            raise KeyError(table)
        doc_id, title, page, page_end, schema_json = meta
        locations = []
        if rowid is not None:
            cur = self.conn.execute(f'SELECT * FROM {quote_ident(table)} WHERE rowid=?', (rowid,))
            row = cur.fetchone()
            if row is None:
                raise KeyError((table, rowid))
            record = dict(zip((d[0] for d in cur.description), row, strict=True))
            page = record.get('_page') or page
            columns = [c for c in decode_table_schema(schema_json).columns if not c.annotation]
            values = [record.get(c.raw_col or c.name) for c in columns]
            needle = ' | '.join('' if v is None else str(v) for v in values)
            for base, text in self._checked(self.conn.execute(
                    'SELECT char_start,text FROM doc_text WHERE doc_id=? AND page=?', (doc_id, page))):
                offset = self._find(text, needle) if needle.strip(' |') else -1
                while offset >= 0 and len(locations) < 8:
                    end = offset + len(needle)
                    # Match the whole canonical row: a value 51 must not be
                    # "located" in an unrelated row whose value is 5198.
                    if ((offset == 0 or text[offset - 1] == '\n')
                            and (end == len(text) or text[end] == '\n')):
                        locations.append(self(doc_id, char_start=base + offset,
                                              char_end=base + end, max_chars=max_chars))
                    offset = self._find(text, needle, offset + 1)
        context = locations[0].copy() if len(locations) == 1 else self(doc_id, page=page, max_chars=max_chars)
        context.update(table=table, title=title, page_start=page, page_end=page_end,
                       rowid=rowid, row_location='exact' if len(locations) == 1 else 'page_only',
                       row_locations=locations)
        return context
