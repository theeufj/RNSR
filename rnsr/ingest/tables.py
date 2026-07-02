"""RawTable -> typed SQLite table with provenance (spec §3.2).

Each detected table becomes ``t_{doc_id}_{seq:03d}``. Columns are typed by
the conservative coercion rules in coerce.py; coerced columns keep their
raw strings in ``{col}__raw`` shadow columns. Every row carries
``_page``/``_bbox``/``_extractor``. Multi-page tables detected by header
repetition are merged into one table with a ``source_page`` column.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass

from rnsr.db import schema
from rnsr.ingest.coerce import coerce_column
from rnsr.ingest.model import RawTable

_WS = re.compile(r"\s+")


def _norm_header_cell(cell: str | None) -> str:
    return _WS.sub(" ", (cell or "").strip().lower())


def headers_match(a: list[str], b: list[str]) -> bool:
    return len(a) == len(b) and all(
        _norm_header_cell(x) == _norm_header_cell(y) for x, y in zip(a, b, strict=True)
    )


def merge_multipage(tables: list[RawTable]) -> list[RawTable]:
    """Merge runs of tables with repeated headers on consecutive pages (§3.2).

    Tables are assumed to be in document order. A continuation must repeat
    the header exactly (after whitespace/case normalization) and start on
    the same page as, or the page after, the previous fragment ends.
    """
    merged: list[RawTable] = []
    for t in tables:
        prev = merged[-1] if merged else None
        prev_last_page = (
            prev.row_page(len(prev.rows) - 1) if prev and prev.rows else prev.page if prev else -1
        )
        if (
            prev is not None
            and headers_match(prev.header, t.header)
            and t.page in (prev_last_page, prev_last_page + 1)
        ):
            prev.row_pages = [prev.row_page(i) for i in range(len(prev.rows))] + [
                t.row_page(i) for i in range(len(t.rows))
            ]
            prev.row_bboxes = [prev.row_bbox(i) for i in range(len(prev.rows))] + [
                t.row_bbox(i) for i in range(len(t.rows))
            ]
            prev.rows = prev.rows + t.rows
        else:
            merged.append(t)
    return merged


@dataclass
class BuiltTable:
    """Result of writing one RawTable; feeds manifest_tables (§3.5)."""

    name: str
    doc_id: str
    page_start: int
    page_end: int
    n_rows: int
    n_cols: int
    schema_entries: list[dict]     # [{name, type, coercion_rule, raw_col}]
    extractor: str
    caption: str | None
    multipage: bool

    @property
    def schema_json(self) -> str:
        return json.dumps(self.schema_entries)


def build_data_table(
    conn: sqlite3.Connection,
    doc_id: str,
    seq: int,
    raw: RawTable,
    *,
    coerce_threshold: float = 0.95,
    style_overrides: dict[str, str] | None = None,
) -> BuiltTable:
    """Create, fill, and freeze one t_{doc_id}_{seq} table.

    `style_overrides` maps column name -> 'us'|'eu' for the §9 coercion
    rollback path (validate.py re-runs with an explicit style, or forces
    TEXT by passing style 'text').
    """
    taken: set[str] = set()
    col_names = [schema.sanitize_column_name(h, taken) for h in raw.header]

    columns: list[tuple[str, str]] = []       # DDL (name, type) incl. shadows
    schema_entries: list[dict] = []
    col_values: list[list] = []               # per data column, aligned with rows
    raw_columns: list[list[str | None] | None] = []

    overrides = style_overrides or {}
    for idx, name in enumerate(col_names):
        raw_vals = [row[idx] if idx < len(row) else None for row in raw.rows]
        override = overrides.get(name)
        if override == "text":
            coerced = coerce_column(raw_vals, threshold=2.0)  # unreachable -> TEXT
        else:
            coerced = coerce_column(raw_vals, threshold=coerce_threshold, style=override)
        if coerced.is_numeric:
            columns.append((name, coerced.sql_type))
            columns.append((f"{name}__raw", "TEXT"))
            col_values.append(coerced.values)
            raw_columns.append(raw_vals)
            rule = coerced.rule.to_dict() if coerced.rule else None
            schema_entries.append(
                {"name": name, "type": coerced.sql_type, "coercion_rule": rule,
                 "raw_col": f"{name}__raw"}
            )
        else:
            columns.append((name, "TEXT"))
            col_values.append([None if v is None else str(v) for v in raw_vals])
            raw_columns.append(None)
            schema_entries.append(
                {"name": name, "type": "TEXT", "coercion_rule": None, "raw_col": None}
            )

    multipage = raw.row_pages is not None and len(set(raw.row_pages)) > 1
    table = schema.data_table_name(doc_id, seq)
    schema.create_data_table(conn, table, columns, with_source_page=multipage)

    rows_out: list[list] = []
    for i in range(len(raw.rows)):
        row_out: list = []
        for c, vals in enumerate(col_values):
            row_out.append(vals[i])
            if raw_columns[c] is not None:
                row_out.append(raw_columns[c][i])
        if multipage:
            row_out.append(raw.row_page(i))
        bbox = raw.row_bbox(i)
        row_out += [raw.row_page(i), json.dumps(bbox) if bbox else "[]", raw.extractor]
        rows_out.append(row_out)

    width = len(columns) + (1 if multipage else 0) + len(schema.PROVENANCE_COLUMNS)
    conn.executemany(
        f"INSERT INTO {schema.quote_ident(table)} VALUES ({', '.join('?' * width)})",
        rows_out,
    )

    source_cols = [n for n, _ in columns]
    if multipage:
        source_cols.append("source_page")
    source_cols += list(schema.PROVENANCE_COLUMNS)
    schema.freeze_table(conn, table, source_columns=source_cols)

    pages = [raw.row_page(i) for i in range(len(raw.rows))] or [raw.page]
    return BuiltTable(
        name=table,
        doc_id=doc_id,
        page_start=min(pages),
        page_end=max(pages),
        n_rows=len(raw.rows),
        n_cols=len(raw.header),
        schema_entries=schema_entries,
        extractor=raw.extractor,
        caption=raw.caption,
        multipage=multipage,
    )
