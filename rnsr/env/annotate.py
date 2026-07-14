"""semantic_annotate — the semantic ETL primitive (spec §4.1).

One batched sub-LM pass over selected rows; results written back as a real
column; idempotent (same table/column/prompt/model/where is a no-op unless
force=True); every run logged to annotation_log with the prompt hash.
Converts O(N²)-in-LLM-reasoning problems into O(N) semantic calls + exact
SQL. Runs inside the sandbox child; sub-calls RPC to the parent.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime

from rnsr.db import schema

_LINE = re.compile(r"^\s*(\d+)\s*[.):\-]\s*(.+?)\s*$")


def _source_columns(conn: sqlite3.Connection, table: str, annotated: set[str]) -> list[str]:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({schema.quote_ident(table)})")]
    return [c for c in cols
            if c not in schema.PROVENANCE_COLUMNS
            and not c.endswith("__raw")
            and c != "source_page"
            and c not in annotated]


def _batch_prompt(prompt: str, rows: list[tuple[int, str]]) -> str:
    numbered = "\n".join(f"{i}. {rendered}" for i, rendered in rows)
    return (
        f"For EACH numbered row below, apply this instruction:\n{prompt}\n\n"
        f"Rows:\n{numbered}\n\n"
        f"Reply with exactly {len(rows)} lines, one per row, in the form "
        "'<row number>. <result>'. No other text."
    )


def _parse_batch(reply: str, expected: list[int]) -> dict[int, str] | None:
    out: dict[int, str] = {}
    for line in (reply or "").splitlines():
        m = _LINE.match(line)
        if m and int(m.group(1)) in set(expected):
            out[int(m.group(1))] = m.group(2)
    return out if len(out) == len(expected) else None


class Annotator:
    def __init__(self, conn: sqlite3.Connection, rpc, *,
                 char_budget: int = 200_000, default_batch_size: int = 40):
        self.conn = conn
        self.rpc = rpc
        self.char_budget = char_budget
        self.default_batch_size = default_batch_size
        self._annotated: set[str] = set()

    def _one_pass(self, prompt: str, rendered: list[tuple[int, str]],
                  batch_size: int, model: str) -> dict[int, str]:
        """One full labeling pass: batch -> llm -> parse -> strict re-ask."""
        batches: list[list[tuple[int, str]]] = [[]]
        chars = 0
        for item in rendered:
            if batches[-1] and (len(batches[-1]) >= batch_size
                                or chars + len(item[1]) > self.char_budget):
                batches.append([])
                chars = 0
            batches[-1].append(item)
            chars += len(item[1])

        prompts = [_batch_prompt(prompt, b) for b in batches]
        replies = self.rpc({"op": "llm_batch", "prompts": prompts,
                            "model": model})["results"]

        values: dict[int, str] = {}
        retry_prompts, retry_batches = [], []
        for batch, reply in zip(batches, replies, strict=True):
            parsed = _parse_batch(reply, [i for i, _ in batch])
            if parsed is None:                       # count mismatch — strict re-ask
                retry_batches.append(batch)
                retry_prompts.append(
                    _batch_prompt(prompt, batch)
                    + "\nYour previous reply did not have one line per row. "
                      "Follow the format exactly."
                )
            else:
                values.update(parsed)
        if retry_prompts:
            replies = self.rpc({"op": "llm_batch", "prompts": retry_prompts,
                                "model": model})["results"]
            for batch, reply in zip(retry_batches, replies, strict=True):
                parsed = _parse_batch(reply, [i for i, _ in batch])
                if parsed:
                    values.update(parsed)
        return values

    def annotate(self, table: str, new_col: str, prompt: str, *,
                 where: str | None = None, batch_size: int | None = None,
                 model: str = "sub", force: bool = False, votes: int = 1) -> dict:
        """votes>1 runs the labeling pass that many times with different
        (seeded) item orders and writes the per-row majority — decorrelates
        per-item classification noise, which dominates counting-task error.
        Sub-call cost scales with votes. Ties keep the first pass's label.
        """
        batch_size = batch_size or self.default_batch_size
        votes = max(1, min(int(votes), 5))
        prompt_sha = hashlib.sha256(f"{prompt}|votes={votes}".encode()).hexdigest()

        prior = self.conn.execute(
            "SELECT rows_written, rows_failed FROM annotation_log WHERE "
            "table_name=? AND column=? AND prompt_sha256=? AND model=? "
            "AND ifnull(where_clause,'')=?",
            (table, new_col, prompt_sha, model, where or ""),
        ).fetchone()
        if prior is not None and not force:
            return {"noop": True, "rows": prior[0], "failed": prior[1],
                    "coverage": None,
                    "note": "identical annotation already applied; pass force=True to redo"}

        src_cols = _source_columns(self.conn, table, self._annotated)
        sql = (f"SELECT rowid, {', '.join(schema.quote_ident(c) for c in src_cols)} "
               f"FROM {schema.quote_ident(table)}")
        if where:
            sql += f" WHERE {where}"
        rows = self.conn.execute(sql).fetchall()
        if not rows:
            return {"rows": 0, "failed": 0, "coverage": 0.0, "sample": []}

        rendered = [
            (row[0], json.dumps(dict(zip(src_cols, row[1:], strict=True)), default=str))
            for row in rows
        ]

        # At temperature 0 identical prompts give identical replies, so each
        # vote uses a different item order: different batch contexts give
        # decorrelated per-item errors that the majority can cancel.
        import random

        tallies: dict[int, list[str]] = {}
        for vote in range(votes):
            ordered = list(rendered)
            if vote > 0:
                random.Random(vote).shuffle(ordered)
            pass_values = self._one_pass(prompt, ordered, batch_size, model)
            for rowid, label in pass_values.items():
                tallies.setdefault(rowid, []).append(label)

        values: dict[int, str] = {}
        for rowid, labels in tallies.items():
            counts: dict[str, int] = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            best = max(counts.values())
            # tie -> earliest vote's label (labels[] preserves vote order)
            values[rowid] = next(la for la in labels if counts[la] == best)

        schema.add_annotation_column(self.conn, table, new_col)
        self._annotated.add(new_col)
        self.conn.executemany(
            f"UPDATE {schema.quote_ident(table)} SET {schema.quote_ident(new_col)} = ? "
            "WHERE rowid = ?",
            [(v, rowid) for rowid, v in values.items()],
        )
        failed = len(rows) - len(values)
        if force:
            self.conn.execute(
                "DELETE FROM annotation_log WHERE table_name=? AND column=? AND "
                "prompt_sha256=? AND model=? AND ifnull(where_clause,'')=?",
                (table, new_col, prompt_sha, model, where or ""),
            )
        self.conn.execute(
            "INSERT INTO annotation_log (created_at, table_name, column, prompt, "
            "prompt_sha256, model, where_clause, batch_size, rows_written, "
            "rows_failed, usage_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(UTC).isoformat(), table, new_col, prompt, prompt_sha,
             model, where, batch_size, len(values), failed, "{}"),
        )
        self.conn.commit()

        sample = list(values.items())[:5]
        return {"rows": len(values), "failed": failed,
                "coverage": round(len(values) / len(rows), 4), "sample": sample}
