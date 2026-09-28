"""semantic_annotate — the semantic ETL primitive (spec §4.1).

One batched sub-LM pass over selected rows; results written back as a real
column; idempotent (same table/column/prompt/model/where is a no-op unless
force=True); every run logged to annotation_log with the prompt hash.
Converts O(N²)-in-LLM-reasoning problems into O(N) semantic calls + exact
SQL. Runs in the trusted parent; the child requests this narrow operation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from datetime import UTC, datetime

from rnsr.db import schema

_LINE = re.compile(r"^\s*(\d+)\s*[.):\-]\s*(.+?)\s*$")
_AUDIT_LABEL_CHARS = 4096
_HISTORY_LIMIT = 3


def _allowed_labels(labels) -> tuple[str, ...] | None:
    if labels is None:
        return None
    if (not isinstance(labels, (list, tuple)) or not 1 <= len(labels) <= 256
            or any(not isinstance(label, str) or not label.strip()
                   or label != label.strip() or len(label) > 256
                   or "\n" in label or "\r" in label for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("allowed_labels must contain 1 to 256 unique, nonempty single-line labels")
    return tuple(sorted(labels))


def _audit_label(label: str) -> dict:
    """Bound free-form labels; classification labels fit without truncation."""
    value = {"label": label[:_AUDIT_LABEL_CHARS]}
    if len(label) > _AUDIT_LABEL_CHARS:
        value.update(label_truncated=True, label_chars=len(label),
                     label_sha256=hashlib.sha256(label.encode()).hexdigest())
    return value


def _response_metadata(value) -> dict:
    """Only trusted, optional response metadata; never infer a resolved model."""
    if not isinstance(value, dict):
        return {}
    out = {}
    if isinstance(value.get("model"), str):
        out["resolved_model"] = value["model"][:256]
    for name in ("input_tokens", "output_tokens", "cost_usd", "latency_s"):
        number = value.get(name)
        if (isinstance(number, (int, float)) and not isinstance(number, bool)
                and math.isfinite(number) and number >= 0):
            out[name] = number
    return out


def _previous_runs(prior) -> dict:
    old = json.loads(prior[3])
    history = old.pop("previous_runs", [])
    dropped_count = old.pop("earlier_runs_count", 0)
    dropped_digest = old.pop("earlier_runs_sha256", "")
    history.append({"created_at": prior[2], "rows_written": prior[0],
                    "rows_failed": prior[1], "usage": old})
    for dropped in history[:-_HISTORY_LIMIT]:
        dropped_digest = hashlib.sha256(
            (dropped_digest + json.dumps(dropped, sort_keys=True)).encode()).hexdigest()
        dropped_count += 1
    return {"previous_runs": history[-_HISTORY_LIMIT:], "earlier_runs_count": dropped_count,
            "earlier_runs_sha256": dropped_digest}


def _source_columns(conn: sqlite3.Connection, table: str, annotated: set[str]) -> list[str]:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({schema.quote_ident(table)})")]
    return [c for c in cols
            if c not in schema.PROVENANCE_COLUMNS
            and not c.endswith("__raw")
            and c != "source_page"
            and c not in annotated]


def _batch_prompt(prompt: str, rows: list[tuple[int, str]],
                  allowed_labels: tuple[str, ...] | None = None) -> str:
    numbered = "\n".join(f"{i}. {rendered}" for i, rendered in rows)
    contract = ("\nUse exactly one of these labels, without quotes: "
                + json.dumps(allowed_labels, ensure_ascii=False) if allowed_labels else "")
    return (
        f"For EACH numbered row below, apply this instruction:\n{prompt}{contract}\n\n"
        f"Rows:\n{numbered}\n\n"
        f"Reply with exactly {len(rows)} lines, one per row, in the form "
        "'<row number>. <result>'. No other text."
    )


def _inspect_batch(reply: str, expected: list[int],
                   allowed_labels: tuple[str, ...] | None = None) -> tuple[dict | None, dict]:
    out: dict[int, str] = {}
    expected_set = set(expected)
    duplicate_ids = set()
    invalid = {}
    for line in (reply or "").splitlines():
        m = _LINE.match(line)
        if m and int(m.group(1)) in expected_set:
            rowid, label = int(m.group(1)), m.group(2)
            if rowid in out:
                duplicate_ids.add(rowid)
            out[rowid] = label
            if allowed_labels is not None and label not in allowed_labels:
                invalid[rowid] = {"rowid": rowid, **_audit_label(label)}
    issues = {"missing_rowids": [i for i in expected if i not in out],
              "invalid_labels": list(invalid.values()),
              "duplicate_rowids": sorted(duplicate_ids)}
    # Preserve permissive free-form parsing when no contract was requested.
    valid = len(out) == len(expected) and not invalid
    if allowed_labels is not None and duplicate_ids:
        valid = False
    return (out if valid else None), issues


def _parse_batch(reply: str, expected: list[int]) -> dict[int, str] | None:
    return _inspect_batch(reply, expected)[0]


_SOURCE_GENERATION = (
    "SELECT m.schema_json, m.doc_id, d.sha256, d.content_sha256, d.ingested_at "
    "FROM manifest_tables m JOIN documents d ON d.doc_id=m.doc_id WHERE m.table_name=?"
)


def _selection_digest(generation, rows) -> str:
    payload = [list(generation), [list(row) for row in rows]]
    return hashlib.sha256(json.dumps(payload, default=str, ensure_ascii=False).encode()).hexdigest()


class Annotator:
    def __init__(self, conn: sqlite3.Connection, rpc, *,
                 char_budget: int = 200_000, default_batch_size: int = 40,
                 cancelled=lambda: False):
        self.conn = conn
        self.rpc = rpc
        self.char_budget = char_budget
        self.default_batch_size = default_batch_size
        self.cancelled = cancelled
        self.usage = {"calls": 0, "prompts": 0}

    def _ask(self, prompts: list[str], model: str) -> tuple[list[str], list[dict]]:
        if self.cancelled():
            raise RuntimeError("annotation cancelled")
        self.usage["calls"] += 1
        self.usage["prompts"] += len(prompts)
        response = self.rpc({"op": "llm_batch", "prompts": prompts, "model": model})
        replies = response["results"]
        if not isinstance(replies, list) or len(replies) != len(prompts):
            raise ValueError("annotation RPC must return one result per prompt")
        if any(not isinstance(reply, str) for reply in replies):
            raise ValueError("annotation RPC results must be text")
        metadata = response.get("response_metadata")
        if not isinstance(metadata, list) or len(metadata) != len(replies):
            metadata = [{} for _ in replies]
        return replies, [_response_metadata(item) for item in metadata]

    def _one_pass(self, prompt: str, rendered: list[tuple[int, str]],
                  batch_size: int, model: str,
                  allowed_labels: tuple[str, ...] | None = None) -> tuple[dict[int, str], list[dict]]:
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

        prompts = [_batch_prompt(prompt, b, allowed_labels) for b in batches]
        replies, metadata = self._ask(prompts, model)

        values: dict[int, str] = {}
        retry_prompts, retry_batches, attempts = [], [], []

        def inspect(batch, reply, sent_prompt, response_meta, retry):
            parsed, issues = _inspect_batch(reply, [i for i, _ in batch], allowed_labels)
            attempts.append({"rowids": [i for i, _ in batch], "retry": retry,
                             "accepted": parsed is not None,
                             "prompt_sha256": hashlib.sha256(sent_prompt.encode()).hexdigest(),
                             "response_sha256": hashlib.sha256(reply.encode()).hexdigest(),
                             **issues, **response_meta})
            return parsed

        for batch, reply, sent, meta in zip(batches, replies, prompts, metadata, strict=True):
            parsed = inspect(batch, reply, sent, meta, False)
            if parsed is None:                       # count mismatch — strict re-ask
                retry_batches.append(batch)
                retry_prompts.append(
                    _batch_prompt(prompt, batch, allowed_labels)
                    + ("\nYour previous reply did not have one valid label per row. "
                       "Use the allowed labels and follow the format exactly."
                       if allowed_labels is not None else
                       "\nYour previous reply did not have one line per row. "
                       "Follow the format exactly.")
                )
            else:
                values.update(parsed)
        if retry_prompts:
            replies, metadata = self._ask(retry_prompts, model)
            for batch, reply, sent, meta in zip(
                    retry_batches, replies, retry_prompts, metadata, strict=True):
                parsed = inspect(batch, reply, sent, meta, True)
                if parsed:
                    values.update(parsed)
        return values, attempts

    def _read_selection(self, table: str, sql: str, where: str | None):
        """A bounded predicate over this table, without subqueries or unions."""
        if where is not None and (not isinstance(where, str) or len(where) > 10_000):
            raise ValueError("annotation where must be a SQL predicate of at most 10000 characters")
        # SQLite may optimize a constant EXISTS subquery before emitting
        # a second authorizer SELECT event. Reject subquery syntax explicitly,
        # while leaving quoted data/identifiers and comments out of the check.
        predicate_sql = re.sub(
            r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^]]*\]|--[^\n]*|/\*.*?\*/",
            " ", where or "", flags=re.DOTALL)
        if re.search(r"\b(?:select|with|union|intersect|except)\b", predicate_sql, re.IGNORECASE):
            raise ValueError("annotation where supports table predicates without subqueries or unions")
        safe_functions = {"abs", "round", "lower", "upper", "length", "substr", "substring",
                          "trim", "ltrim", "rtrim", "coalesce", "ifnull", "nullif",
                          "like", "glob", "typeof", "instr", "min", "max", "count", "sum", "avg"}
        selects = 0
        def read_only(action, arg1, arg2, _db, _source):
            nonlocal selects
            if action == sqlite3.SQLITE_SELECT:
                selects += 1
                return sqlite3.SQLITE_OK if selects == 1 else sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ:
                return sqlite3.SQLITE_OK if arg1 == table else sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION:
                return (sqlite3.SQLITE_OK if (arg2 or arg1 or "").lower() in safe_functions
                        else sqlite3.SQLITE_DENY)
            return sqlite3.SQLITE_DENY
        deadline = time.monotonic() + 5.0
        self.conn.set_authorizer(read_only)
        self.conn.set_progress_handler(
            lambda: int(self.cancelled() or time.monotonic() >= deadline), 1000)
        try:
            return self.conn.execute(sql).fetchall()
        finally:
            self.conn.set_authorizer(None)
            self.conn.set_progress_handler(lambda: int(self.cancelled()), 1000)

    def annotate(self, table: str, new_col: str, prompt: str, *,
                 where: str | None = None, batch_size: int | None = None,
                 model: str = "sub", force: bool = False, votes: int = 1,
                 allowed_labels: list[str] | tuple[str, ...] | None = None) -> dict:
        """votes>1 runs the labeling pass that many times with different
        (seeded) item orders and writes the per-row majority. These passes
        share a model and prompt; their errors need not be independent.
        Sub-call cost scales with votes. Ties keep the first pass's label.
        Optional allowed_labels enforce exact membership; invalid batches
        get one strict retry. Every vote and failed attempt is audited.
        """
        from rnsr.db.metadata import decode_table_schema

        row = self.conn.execute(_SOURCE_GENERATION, (table,)).fetchone()
        if row is None:
            raise ValueError("annotations require a registered extracted table")
        columns = decode_table_schema(row[0]).columns
        annotated = {c.name for c in columns if c.annotation}
        physical = {r[1].casefold() for r in self.conn.execute(
            f"PRAGMA table_info({schema.quote_ident(table)})")}
        source = physical - {name.casefold() for name in annotated}
        source |= {"rowid", "oid", "_rowid_"}
        if (not isinstance(new_col, str) or not new_col or new_col.startswith("_")
                or new_col.endswith("__raw") or new_col.casefold() in source):
            raise ValueError("annotation cannot overwrite a source or provenance column")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("annotation prompt must be nonempty text")
        allowed_labels = _allowed_labels(allowed_labels)
        self.usage = {"calls": 0, "prompts": 0}
        batch_size = self.default_batch_size if batch_size is None else batch_size
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 1000:
            raise ValueError("annotation batch_size must be between 1 and 1000")
        votes = max(1, min(int(votes), 5))
        prompt_key = f"{prompt}|votes={votes}"
        if allowed_labels is not None:
            prompt_key += "|allowed_labels=" + json.dumps(allowed_labels, ensure_ascii=False)
        prompt_sha = hashlib.sha256(prompt_key.encode()).hexdigest()

        prior_sql = (
            "SELECT rows_written, rows_failed, created_at, usage_json FROM annotation_log WHERE "
            "table_name=? AND column=? AND prompt_sha256=? AND model=? "
            "AND ifnull(where_clause,'')=?")
        prior_key = (table, new_col, prompt_sha, model, where or "")
        prior = self.conn.execute(prior_sql, prior_key).fetchone()
        if prior is not None and not force:
            return {"noop": True, "rows": prior[0], "failed": prior[1],
                    "coverage": None,
                    "note": "identical annotation already applied; pass force=True to redo"}

        src_cols = _source_columns(self.conn, table, annotated)
        sql = (f"SELECT rowid, {', '.join(schema.quote_ident(c) for c in src_cols)} "
               f"FROM {schema.quote_ident(table)}")
        if where:
            sql += f" WHERE {where}"
        rows = self._read_selection(table, sql, where)
        if not rows:
            return {"rows": 0, "failed": 0, "coverage": 0.0, "sample": []}

        source_digest = _selection_digest(row, rows)
        rendered = [
            (row[0], json.dumps(dict(zip(src_cols, row[1:], strict=True)), default=str))
            for row in rows
        ]

        # Different row orders vary batch context. Retain every vote so any
        # benefit or correlated errors can be measured, rather than assumed.
        import random

        tallies: dict[int, list[str]] = {}
        vote_audit = []
        for vote in range(votes):
            ordered = list(rendered)
            if vote > 0:
                random.Random(vote).shuffle(ordered)
            started = time.monotonic()
            pass_values, attempts = self._one_pass(
                prompt, ordered, batch_size, model, allowed_labels)
            vote_audit.append({
                "index": vote, "shuffle_seed": vote if vote else None,
                "elapsed_s": time.monotonic() - started,
                "rows": [{"rowid": rowid, **(_audit_label(pass_values[rowid])
                         if rowid in pass_values else {"label": None})} for rowid, _ in ordered],
                "attempts": attempts,
            })
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

        self.usage["vote_audit"] = {
            "schema_version": 1, "requested_model": model,
            "allowed_labels": allowed_labels, "votes_requested": votes,
            "source": {"doc_id": row[1], "sha256": row[2], "content_sha256": row[3],
                       "selection_sha256": source_digest, "row_count": len(rows)},
            "instruction_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "aggregation": "plurality_of_valid_votes; ties_use_earliest_valid_vote",
            "partial_vote_rowids": [r[0] for r in rows if 0 < len(tallies.get(r[0], [])) < votes],
            "failed_rowids": [r[0] for r in rows if r[0] not in values],
            "votes": vote_audit,
        }
        if self.cancelled():
            raise RuntimeError("annotation cancelled")
        # Release the read snapshot before provider calls, then verify the
        # exact generation/selected source rows under the publishing write
        # lock. Rowids may be reused by a concurrent document replacement.
        if not self.conn.in_transaction:
            self.conn.execute("BEGIN IMMEDIATE")
        try:
            current = self.conn.execute(_SOURCE_GENERATION, (table,)).fetchone()
            if current is None or list(current) != list(row):
                raise ValueError("annotation source changed while labeling; retry against current source")
            current_rows = self._read_selection(table, sql, where)
            if _selection_digest(current, current_rows) != source_digest:
                raise ValueError("annotation source changed while labeling; retry against current source")
            schema.add_annotation_column(self.conn, table, new_col)
            self.conn.executemany(
                f"UPDATE {schema.quote_ident(table)} SET {schema.quote_ident(new_col)} = ? "
                "WHERE rowid = ?",
                [(v, rowid) for rowid, v in values.items()],
            )
            failed = len(rows) - len(values)
            if force:
                # Read the latest log under the publishing lock; another
                # trusted annotation may have completed during provider calls.
                prior = self.conn.execute(prior_sql, prior_key).fetchone()
                if prior is not None:
                    self.usage.update(_previous_runs(prior))
                self.conn.execute(
                    "DELETE FROM annotation_log WHERE table_name=? AND column=? AND "
                    "prompt_sha256=? AND model=? AND ifnull(where_clause,'')=?", prior_key)
            self.conn.execute(
                "INSERT INTO annotation_log (created_at, table_name, column, prompt, "
                "prompt_sha256, model, where_clause, batch_size, rows_written, "
                "rows_failed, usage_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(UTC).isoformat(), table, new_col, prompt, prompt_sha,
                 model, where, batch_size, len(values), failed, json.dumps(self.usage)),
            )
            if self.cancelled():
                raise RuntimeError("annotation cancelled")
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

        sample = list(values.items())[:5]
        return {"rows": len(values), "failed": failed,
                "coverage": round(len(values) / len(rows), 4), "sample": sample}
