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
from rnsr.ingest.coerce import caption_scale, coerce_column
from rnsr.ingest.model import RawTable
from rnsr.ingest.validate import classify_row_kind

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
            and (prev.caption or "") == (t.caption or "")
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


def _numeric_stats(values: list) -> dict:
    """Zone map for a numeric column: min/max over non-null values."""
    present = [v for v in values if v is not None]
    if not present:
        return {"n_null": len(values)}
    return {"min": min(present), "max": max(present),
            "n_null": len(values) - len(present)}


def _text_stats(values: list[str | None], sample_n: int = 3,
                sample_chars: int = 40) -> dict:
    """Zone map for a text column: distinct count plus a short sample.

    Kept small on purpose: schema entries travel in the full manifest (the
    prompt-side compact_manifest drops them, so token cost is zero there).
    """
    present = [v for v in values if v]
    distinct = list(dict.fromkeys(present))
    return {
        "n_distinct": len(distinct),
        "sample": [d[:sample_chars] for d in distinct[:sample_n]],
    }


def _populate_cells(
    conn: sqlite3.Connection,
    doc_id: str,
    table: str,
    schema_entries: list[dict],
    col_values: list[list],
    raw_columns: list[list[str | None] | None],
) -> None:
    """Write every data cell into the derived `cells` index (Stage 1).

    row_idx is the source table rowid (fresh tables insert rowids 1..n in
    order). Numeric cells carry both the coerced number and the lowered raw
    string, so a text probe for "3,400" and a SQL probe for 3400 both hit.
    """
    schema.ensure_cells_table(conn)
    rows: list[tuple] = []
    n_rows = len(col_values[0]) if col_values else 0
    for i in range(n_rows):
        for c, entry in enumerate(schema_entries):
            value = col_values[c][i]
            if entry["raw_col"] is not None:        # numeric column
                raw_v = (raw_columns[c] or [None] * n_rows)[i]
                text = None if raw_v is None else str(raw_v).lower()
                num = value
            else:
                text = None if value is None else str(value).lower()
                num = None
            if text is None and num is None:
                continue
            rows.append((doc_id, table, i + 1, entry["name"], text, num))
    if rows:
        conn.executemany("INSERT INTO cells VALUES (?,?,?,?,?,?)", rows)


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
    n_total_rows: int = 0
    n_data_rows: int = 0

    @property
    def schema_json(self) -> str:
        return json.dumps({
            "columns": self.schema_entries,
            "n_total_rows": self.n_total_rows,
            "n_data_rows": self.n_data_rows,
        })


def build_data_table(
    conn: sqlite3.Connection,
    doc_id: str,
    seq: int,
    raw: RawTable,
    *,
    coerce_threshold: float = 0.95,
    style_overrides: dict[str, str] | None = None,
    cells: bool = True,
) -> BuiltTable:
    """Create, fill, and freeze one t_{doc_id}_{seq} table.

    `style_overrides` maps column name -> 'us'|'eu' for the §9 coercion
    rollback path (validate.py re-runs with an explicit style, or forces
    TEXT by passing style 'text').

    With `cells` (Stage 1, engine-poc-plan), every data cell is also written
    to the derived `cells` index, and per-column zone-map stats (numeric
    min/max, text distinct-count + sample) are attached to the schema
    entries — both feed rung-0 sweeps and table pruning.
    """
    taken: set[str] = set()
    col_names = [schema.sanitize_column_name(h, taken) for h in raw.header]

    columns: list[tuple[str, str]] = []       # DDL (name, type) incl. shadows
    schema_entries: list[dict] = []
    col_values: list[list] = []               # per data column, aligned with rows
    raw_columns: list[list[str | None] | None] = []

    overrides = style_overrides or {}
    unit_scale = caption_scale(raw.caption)
    for idx, name in enumerate(col_names):
        raw_vals = [row[idx] if idx < len(row) else None for row in raw.rows]
        override = overrides.get(name)
        if override == "text":
            coerced = coerce_column(raw_vals, threshold=2.0)  # unreachable -> TEXT
        else:
            coerced = coerce_column(
                raw_vals, threshold=coerce_threshold, style=override,
                unit_scale=unit_scale)
        if coerced.is_numeric:
            columns.append((name, coerced.sql_type))
            columns.append((f"{name}__raw", "TEXT"))
            col_values.append(coerced.values)
            raw_columns.append(raw_vals)
            rule = coerced.rule.to_dict() if coerced.rule else None
            schema_entries.append(
                {"name": name, "type": coerced.sql_type, "coercion_rule": rule,
                 "raw_col": f"{name}__raw", "stats": _numeric_stats(coerced.values)}
            )
        else:
            text_vals = [None if v is None else str(v) for v in raw_vals]
            columns.append((name, "TEXT"))
            col_values.append(text_vals)
            raw_columns.append(None)
            schema_entries.append(
                {"name": name, "type": "TEXT", "coercion_rule": None,
                 "raw_col": None, "stats": _text_stats(text_vals)}
            )

    multipage = raw.row_pages is not None and len(set(raw.row_pages)) > 1
    table = schema.data_table_name(doc_id, seq)
    schema.create_data_table(conn, table, columns, with_source_page=multipage)

    label_idx = 0
    for i, entry in enumerate(schema_entries):
        if entry["type"] == "TEXT":
            label_idx = i
            break
    kinds = [classify_row_kind(raw.rows[i], label_idx) for i in range(len(raw.rows))]
    n_total_rows = sum(1 for k in kinds if k in ("total", "subtotal"))
    n_data_rows = sum(1 for k in kinds if k == "data")

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
        row_out += [raw.row_page(i), json.dumps(bbox) if bbox else "[]",
                    raw.extractor, kinds[i]]
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

    if cells:
        _populate_cells(conn, doc_id, table, schema_entries, col_values,
                        raw_columns)

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
        n_total_rows=n_total_rows,
        n_data_rows=n_data_rows,
    )
