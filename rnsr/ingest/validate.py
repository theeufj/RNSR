"""Self-validation checksum pass (spec §3.3) + coercion rollback (§9).

Documents contain internal redundancy; we exploit it as automatic ground
truth. Checks run on the in-memory grid *before* the table is written, so
their outcome can steer coercion (style rollback) and re-extraction.

Check groups:
  arithmetic — total/subtotal rows must equal the sum of their line items
               within max(rel_tol·|total|, abs_tol); percent columns sum
               to ~100 where a total row implies it.
  structural — grid consistency, no repeated header rows in the body,
               monotonic date-like columns.
  prose      — sampled numeric cells cross-checked against nearby prose by
               a sub-LM (skipped when no LLM client is provided; Phase A
               is fully deterministic without it).

Confidence is a weighted mean over the groups that actually applied.
No silent failures: every table ends trusted, re-extracted, or untrusted.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from rnsr.ingest.coerce import CoercedColumn, coerce_column, is_null_cell
from rnsr.ingest.model import RawTable

TOTAL_LABEL = re.compile(r"\b(total|subtotal|sum|net)\b", re.IGNORECASE)
_YEAR = re.compile(r"^(19|20)\d{2}$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_WEIGHTS = {"arithmetic": 0.5, "structural": 0.3, "prose": 0.2}

# ask(prompts) -> yes/no/None per prompt; wired to the sub-LM in Phase C
ProseChecker = Callable[[list[str]], list[bool | None]]


@dataclass
class GroupResult:
    applicable: int = 0
    passed: int = 0
    details: list[dict] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        return None if self.applicable == 0 else self.passed / self.applicable

    def to_dict(self) -> dict:
        return {"applicable": self.applicable, "passed": self.passed,
                "score": self.score, "details": self.details}


@dataclass
class TableValidation:
    confidence: float
    checks: dict[str, GroupResult]
    style_overrides: dict[str, str]     # §9 rollbacks chosen during validation

    def to_checks_json(self) -> dict:
        return {k: v.to_dict() for k, v in self.checks.items()}


def _coerce_all(raw: RawTable, threshold: float,
                overrides: dict[str, str]) -> dict[int, CoercedColumn]:
    """Coerce every column of the grid; keyed by column index."""
    out: dict[int, CoercedColumn] = {}
    for idx in range(raw.n_cols):
        vals = [row[idx] if idx < len(row) else None for row in raw.rows]
        style = overrides.get(str(idx))
        out[idx] = coerce_column(vals, threshold=threshold, style=style)
    return out


def _label_column(raw: RawTable, cols: dict[int, CoercedColumn]) -> int:
    for idx in range(raw.n_cols):
        if not cols[idx].is_numeric:
            return idx
    return 0


def _total_rows(raw: RawTable, label_col: int) -> list[int]:
    hits = []
    for i, row in enumerate(raw.rows):
        cell = row[label_col] if label_col < len(row) else None
        if cell and TOTAL_LABEL.search(str(cell)):
            hits.append(i)
    return hits


def _check_arithmetic_column(
    values: list, totals: list[int], rel_tol: float, abs_tol: float
) -> list[dict]:
    """Check each total row against the line items since the previous total."""
    results = []
    prev = -1
    for t in totals:
        expected = values[t]
        items = [v for v in values[prev + 1 : t] if v is not None]
        prev = t
        if expected is None or len(items) < 2:
            continue
        s = sum(items)
        tol = max(rel_tol * abs(expected), abs_tol)
        results.append({
            "total_row": t, "expected": expected, "sum": s,
            "tolerance": tol, "passed": abs(s - expected) <= tol,
        })
    return results


def check_arithmetic(raw: RawTable, cols: dict[int, CoercedColumn],
                     rel_tol: float, abs_tol: float) -> GroupResult:
    g = GroupResult()
    label_col = _label_column(raw, cols)
    totals = _total_rows(raw, label_col)
    for idx, col in cols.items():
        if not col.is_numeric or idx == label_col:
            continue
        checks = _check_arithmetic_column(col.values, totals, rel_tol, abs_tol)
        for c in checks:
            g.applicable += 1
            g.passed += bool(c["passed"])
            g.details.append({"column": idx, **c})
        # percent columns: items should sum to ~100 when the total row says ~100
        if col.rule and "percent" in col.rule.features:
            for c in checks:
                if c["expected"] is not None and abs(c["expected"] - 100.0) <= 1.0:
                    g.applicable += 1
                    ok = abs(c["sum"] - 100.0) <= max(100 * rel_tol, abs_tol)
                    g.passed += ok
                    g.details.append({"column": idx, "check": "pct_sums_to_100",
                                      "sum": c["sum"], "passed": ok})
    return g


def _is_date_like(values: list[str | None]) -> bool:
    non_null = [v for v in values if not is_null_cell(v)]
    if len(non_null) < 3:
        return False
    hits = sum(bool(_YEAR.match(str(v).strip()) or _ISO_DATE.match(str(v).strip()))
               for v in non_null)
    return hits / len(non_null) >= 0.95


def check_structural(raw: RawTable, cols: dict[int, CoercedColumn]) -> GroupResult:
    g = GroupResult()

    # Grid consistency: no row wider than the header.
    g.applicable += 1
    too_wide = [i for i, row in enumerate(raw.rows) if len(row) > raw.n_cols]
    g.passed += not too_wide
    g.details.append({"check": "grid_width", "rows_too_wide": too_wide,
                      "passed": not too_wide})

    # No repeated header rows inside the body (missed multi-page merge symptom).
    g.applicable += 1
    header_norm = [_norm(c) for c in raw.header]
    repeats = [
        i for i, row in enumerate(raw.rows)
        if len(row) == raw.n_cols and [_norm(c) for c in row] == header_norm
    ]
    g.passed += not repeats
    g.details.append({"check": "no_header_repeats", "rows": repeats, "passed": not repeats})

    # Monotonic date-like columns.
    for idx in range(raw.n_cols):
        vals = [row[idx] if idx < len(row) else None for row in raw.rows]
        if not _is_date_like(vals):
            continue
        seq = [str(v).strip() for v in vals if not is_null_cell(v)]
        ok = seq == sorted(seq) or seq == sorted(seq, reverse=True)
        g.applicable += 1
        g.passed += ok
        g.details.append({"check": "monotonic_dates", "column": idx, "passed": ok})
    return g


def _norm(cell: str | None) -> str:
    return re.sub(r"\s+", " ", (cell or "").strip().lower())


def check_prose(
    raw: RawTable,
    cols: dict[int, CoercedColumn],
    page_texts: dict[int, str],
    ask: ProseChecker,
    k: int,
    seed: int = 0,
) -> GroupResult:
    """Sampled sub-LM cross-check of numeric cells vs nearby prose (§3.3)."""
    g = GroupResult()
    numeric_cells = [
        (i, idx, cols[idx].values[i])
        for idx, col in cols.items() if col.is_numeric
        for i in range(len(raw.rows)) if cols[idx].values[i] is not None
    ]
    if not numeric_cells:
        return g
    rng = random.Random(seed)
    sample = rng.sample(numeric_cells, min(k, len(numeric_cells)))
    prompts = []
    for i, idx, value in sample:
        page = raw.row_page(i)
        context = "\n".join(
            page_texts.get(p, "") for p in (page - 1, page, page + 1)
        ).strip()[:12000]
        prompts.append(
            f"Document excerpt:\n{context}\n\n"
            f"Question: Does the prose above state or imply the value {value} "
            f"(from a table, column '{raw.header[idx]}')? Answer YES, NO, or UNCLEAR."
        )
    answers = ask(prompts)
    for (i, idx, value), ans in zip(sample, answers, strict=True):
        if ans is None:
            continue  # UNCLEAR — no evidence either way
        g.applicable += 1
        g.passed += bool(ans)
        g.details.append({"row": i, "column": idx, "value": value, "agrees": ans})
    return g


def _confidence(checks: dict[str, GroupResult]) -> float:
    total_w = 0.0
    acc = 0.0
    for name, group in checks.items():
        score = group.score
        if score is None:
            continue
        w = _WEIGHTS[name]
        acc += w * score
        total_w += w
    return acc / total_w if total_w else 1.0  # nothing applicable -> no evidence against


def validate_table(
    raw: RawTable,
    *,
    coerce_threshold: float = 0.95,
    rel_tol: float = 0.005,
    abs_tol: float = 1.0,
    prose_checker: ProseChecker | None = None,
    page_texts: dict[int, str] | None = None,
    prose_cells: int = 3,
    seed: int = 0,
) -> TableValidation:
    """Run the checksum pass, attempting per-column style rollback (§9).

    If a numeric column fails its arithmetic checks, the column is re-coerced
    with the opposite decimal style; if that fixes the checks, the override
    is recorded (build_data_table applies it). Anything still failing counts
    against confidence and steers re-extraction in the pipeline.
    """
    overrides: dict[str, str] = {}
    cols = _coerce_all(raw, coerce_threshold, overrides)
    arithmetic = check_arithmetic(raw, cols, rel_tol, abs_tol)

    # §9 rollback: retry failing columns with the opposite style.
    failing = {d["column"] for d in arithmetic.details if not d["passed"]}
    if failing:
        improved = False
        for idx in failing:
            col = cols[idx]
            if not col.rule:
                continue
            flipped = "eu" if col.rule.style == "us" else "us"
            retry = coerce_column(
                [row[idx] if idx < len(row) else None for row in raw.rows],
                threshold=coerce_threshold, style=flipped,
            )
            if not retry.is_numeric:
                continue
            trial = dict(cols)
            trial[idx] = retry
            re_arith = check_arithmetic(raw, trial, rel_tol, abs_tol)
            before = [d for d in arithmetic.details if d.get("column") == idx and not d["passed"]]
            after = [d for d in re_arith.details if d.get("column") == idx and not d["passed"]]
            if before and not after:
                cols[idx] = retry
                overrides[str(idx)] = flipped
                improved = True
        if improved:
            arithmetic = check_arithmetic(raw, cols, rel_tol, abs_tol)

    checks = {
        "arithmetic": arithmetic,
        "structural": check_structural(raw, cols),
        "prose": (
            check_prose(raw, cols, page_texts or {}, prose_checker, prose_cells, seed)
            if prose_checker is not None
            else GroupResult()
        ),
    }
    # Map index-keyed overrides to sanitized column names for build_data_table.
    from rnsr.db.schema import sanitize_column_name

    taken: set[str] = set()
    names = [sanitize_column_name(h, taken) for h in raw.header]
    named_overrides = {names[int(i)]: style for i, style in overrides.items()}

    return TableValidation(_confidence(checks), checks, named_overrides)
