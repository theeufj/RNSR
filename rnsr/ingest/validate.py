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
No silent failures: every table ends trusted, re-extracted, untrusted, or
unchecked (no arithmetic/prose evidence — excluded from the pass rate).
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from rnsr.ingest.coerce import CoercedColumn, caption_scale, coerce_column, is_null_cell
from rnsr.ingest.model import RawTable

# "net" is not a total: "Net income" is a line item, not a checksum row.
TOTAL_LABEL = re.compile(
    r"^(?:grand\s+)?(?:total|subtotal|sub-total|sum)\b|\b(?:total|subtotal)\s*$",
    re.IGNORECASE,
)
_SUBTOTAL = re.compile(r"\bsubtotal\b", re.IGNORECASE)
_PROSE_VERB = re.compile(
    r"\b(is|are|was|were|will|shall|must|has|have|had|includes?|exceeds?|due)\b"
    r"|\bmay\s+(?:be|have|not)\b",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(
    r"^(?:id|no\.?|(?:line|row|record|account|invoice|exhibit|reference|serial)"
    r"(?:[ _-]*(?:id|no\.?|number))?|code|zip|postal code|year|date)$", re.IGNORECASE,
)
_NON_ADDITIVE = re.compile(
    r"\b(average|avg|weighted|ratio|rate|margin|growth|yield|remaining)\b"
    r"|\bper(?:[ -]+\w+){0,3}[ -]+(?:share|unit|employee|capita)\b"
    r"|\beps\b", re.IGNORECASE,
)
_PRICE_COLUMN = re.compile(r"\bunit[ -]price\b|\bprice[ -]per\b", re.IGNORECASE)
_BALANCE_SNAPSHOT = re.compile(
    r"\b(?:opening|closing|beginning|ending|end[ -]of[ -](?:period|year|month))\s+balance\b",
    re.IGNORECASE,
)
_REMAINING_AUTHORIZATION = re.compile(r"\b(?:may|can)\s+(?:yet|still)\s+be\s+purchased\b",
                                     re.IGNORECASE)


def aggregate_kind(text: str) -> str | None:
    """Recognize short table labels, not arbitrary prose mentioning totals."""
    text = text.strip().strip(":() ")
    if len(text.split()) > 8 or _PROSE_VERB.search(text) or not TOTAL_LABEL.search(text):
        return None
    return "subtotal" if _SUBTOTAL.search(text) or text.lower().startswith("sub-total") else "total"
FOOTNOTE_LABEL = re.compile(
    r"^\s*(\*|†|‡|§|\(\d+\)|\[\d+\]|[¹²³⁴⁵⁶⁷⁸⁹⁰])"
)


def classify_row_kind(row: list, label_col: int = 0) -> str:
    """Classify a body row: data | total | subtotal | footnote | section."""
    cell = row[label_col] if label_col < len(row) else None
    text = str(cell).strip() if cell not in (None, "") else ""
    others = [
        c for i, c in enumerate(row)
        if i != label_col and c not in (None, "", "-", "–", "—")
    ]
    kind = aggregate_kind(text)
    if kind:
        return kind
    if text and FOOTNOTE_LABEL.match(text) and not others:
        return "footnote"
    if text and not others:
        return "section"
    return "data"
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

    @property
    def structural_errors(self) -> bool:
        return any(d.get("blocking") and not d["passed"]
                   for d in self.checks["structural"].details)

    @property
    def evidence(self) -> bool:
        """True when arithmetic or prose actually applied.

        Structural grid checks always run and almost always pass; they are
        not evidence the *values* are right. Tables with no totals and no
        prose check are ``unchecked``, not trusted.
        """
        return any(
            name in ("arithmetic", "prose") and g.applicable > 0
            for name, g in self.checks.items()
        )


def assign_table_status(
    validation: TableValidation,
    threshold: float,
    *,
    first_attempt: bool = True,
    reextracted: bool = False,
) -> str:
    """Map a validation result to a manifest_tables status."""
    if validation.structural_errors:
        return "untrusted"
    if not validation.evidence:
        return "unchecked"
    if validation.confidence < threshold:
        return "untrusted"
    if reextracted:
        return "reextracted"
    if first_attempt:
        return "trusted"
    return "reextracted"


def _coerce_all(raw: RawTable, threshold: float,
                overrides: dict[str, str]) -> dict[int, CoercedColumn]:
    """Coerce every column of the grid; keyed by column index."""
    out: dict[int, CoercedColumn] = {}
    unit_scale = caption_scale(raw.caption)
    for idx in range(raw.n_cols):
        vals = [row[idx] if idx < len(row) else None for row in raw.rows]
        style = overrides.get(str(idx))
        out[idx] = coerce_column(
            vals, threshold=threshold, style=style, unit_scale=unit_scale)
    return out


def label_column(raw: RawTable, numeric_columns: set[int]) -> int:
    """Use an actual populated text column; empty title columns are not labels."""
    for idx in range(raw.n_cols):
        if idx not in numeric_columns and any(
            idx < len(row) and not is_null_cell(row[idx]) for row in raw.rows
        ):
            return idx
    return 0


def _total_rows(raw: RawTable, label_col: int) -> list[int]:
    hits = []
    for i, row in enumerate(raw.rows):
        cell = row[label_col] if label_col < len(row) else None
        if cell and aggregate_kind(str(cell)):
            hits.append(i)
    return hits


def _scope_label(text: str) -> str:
    """Compare a section heading with its closing label, not its values."""
    text = re.sub(r"\s*\(\d+\)\s*$", "", text.strip())
    text = re.sub(r"^(?:(?:grand\s+)?total|subtotal|sub-total|sum)\b\s*", "", text,
                  flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" :").casefold()


def _arithmetic_plan(raw: RawTable, label_col: int) -> list[dict]:
    """Resolve checksum scope from headings and closing labels alone.

    A section's closing row replaces its components in the parent. This
    also handles subtotals without the word 'total', such as a numeric
    'Cash equivalents' row closing a 'Cash equivalents:' heading. Never
    choose a grouping because its numbers happen to add up.
    """
    root = {"label": "", "rows": [], "segment": []}
    stack = [root]
    finished: list[int] = []
    plan: list[dict] = []

    def append(node, index):
        node["rows"].append(index)
        node["segment"].append(index)

    def check(index, rows, reason=None):
        # A numeric row with no label could itself be an unmarked subtotal.
        # Its scope cannot be established from this grid.
        if any(not str(raw.rows[i][label_col] or "").strip() for i in rows):
            reason = reason or "ambiguous_unlabelled_component"
        item = {"total_row": index, "rows": list(rows)}
        if reason:
            item.update(applicable=False, skipped=reason)
        plan.append(item)

    for index, row in enumerate(raw.rows):
        label = str(row[label_col] or "") if label_col < len(row) else ""
        kind = classify_row_kind(row, label_col)
        scope = _scope_label(label)
        if kind == "footnote":
            continue
        if kind == "section":
            stack.append({"label": scope, "rows": [], "segment": []})
            continue
        matches = [i for i, node in enumerate(stack[1:], 1)
                   if scope and scope == node["label"]]
        if matches:
            target = matches[-1]
            node = stack[target]
            check(index, node["rows"],
                  "ambiguous_unclosed_section" if target != len(stack) - 1 else None)
            del stack[target:]
            append(stack[-1], index)
        elif kind == "subtotal":
            node = stack[-1]
            following = next((r for r in raw.rows[index + 1:]
                              if any(not is_null_cell(value) for value in r)
                              and classify_row_kind(r, label_col) != "footnote"), None)
            next_label = (str(following[label_col] or "")
                          if following is not None and label_col < len(following) else "")
            next_kind = classify_row_kind(following, label_col) if following else None
            closes_section = len(stack) > 1 and not scope and (
                following is None or next_kind == "section"
                or (next_kind == "total" and (
                    re.match(r"^\s*grand\s+total\b", next_label, re.IGNORECASE)
                    or any(_scope_label(next_label) == parent["label"]
                           for parent in stack[1:-1]))))
            if closes_section:
                # A generic Subtotal followed by another section (or the
                # parent's total) closes this section. Keep its value once
                # in the parent, including for a later Grand total.
                check(index, node["rows"])
                stack.pop()
                append(stack[-1], index)
                continue
            rows = node["segment"]
            check(index, rows, "ambiguous_subtotal_scope" if not rows else None)
            if rows:
                node["rows"] = node["rows"][:-len(rows)]
            else:
                # Do not count a possible parent subtotal alongside children.
                node["rows"] = []
            node["rows"].append(index)
            node["segment"] = []
        elif kind == "total":
            if len(stack) > 1:
                node = stack[-1]
                # A generic Total can close a single, otherwise isolated
                # section. Other unresolved scopes must not fail a checksum.
                isolated = len(stack) == 2 and not root["rows"] and not scope
                check(index, node["rows"],
                      None if isolated else "ambiguous_total_scope")
                del stack[1:]
                append(root, index)
            else:
                rows = root["rows"]
                grand = bool(re.match(r"^\s*grand\s+total\b", label, re.IGNORECASE))
                if grand:
                    rows = finished + rows
                check(index, rows)
                root["rows"], root["segment"] = [], []
                if grand:
                    finished = [index]
                else:
                    finished.append(index)
        else:
            append(stack[-1], index)
    return plan


def _check_arithmetic_column(
    values: list, plan: list[dict], rel_tol: float, abs_tol: float,
) -> list[dict]:
    """Evaluate the same structurally chosen checks in every numeric column."""
    results = []
    for item in plan:
        t = item["total_row"]
        expected = values[t]
        rows = [i for i in item["rows"] if values[i] is not None]
        if item.get("applicable") is False:
            results.append(item)
            continue
        if expected is None or len(rows) < 2:
            continue
        s = sum(values[i] for i in rows if values[i] is not None)
        tol = max(rel_tol * abs(expected), abs_tol)
        results.append({
            "total_row": t, "expected": expected, "sum": s,
            "tolerance": tol, "passed": abs(s - expected) <= tol, "rows": rows,
        })
    return results


def _identifier_column(raw: RawTable, idx: int) -> bool:
    return bool(_IDENTIFIER.fullmatch(raw.header[idx].strip()))


def _non_additive_column(raw: RawTable, idx: int, col: CoercedColumn) -> str | None:
    if _identifier_column(raw, idx):
        return "identifier"
    header = raw.header[idx]
    if (_NON_ADDITIVE.search(header) or _PRICE_COLUMN.search(header)
            or _BALANCE_SNAPSHOT.search(header) or _REMAINING_AUTHORIZATION.search(header)):
        return "non_additive_measure"
    percent = (col.rule and "percent" in col.rule.features) or re.search(
        r"%|\bpercent(?:age)?\b", header, re.IGNORECASE)
    if (percent or _YEAR.fullmatch(header.strip()) or header.strip().lower() in ("", "value")) \
            and _NON_ADDITIVE.search(raw.caption or ""):
        return "non_additive_measure"
    if percent and not re.search(r"\b(share|mix|composition|allocation|distribution|proportion)\b"
                                 r"|%\s*of\s+total\b",
                                 header, re.IGNORECASE):
        return "percentage_without_additive_share_semantics"
    return None


def check_arithmetic(raw: RawTable, cols: dict[int, CoercedColumn],
                     rel_tol: float, abs_tol: float) -> GroupResult:
    g = GroupResult()
    if raw.kind == "text_lines":
        return g
    label_col = label_column(raw, {idx for idx, col in cols.items() if col.is_numeric})
    plan = _arithmetic_plan(raw, label_col)
    for idx, col in cols.items():
        if not col.is_numeric or idx == label_col:
            continue
        reason = _non_additive_column(raw, idx, col)
        if reason:
            g.details.append({"column": idx, "applicable": False, "skipped": reason})
            continue
        checks = _check_arithmetic_column(col.values, plan, rel_tol, abs_tol)
        applicable_checks = []
        for c in checks:
            if c.get("applicable") is False:
                g.details.append({"column": idx, **c})
                continue
            labels = [str(raw.rows[i][label_col] or "")
                      for i in c["rows"] + [c["total_row"]]]
            if any(_NON_ADDITIVE.search(label) or _BALANCE_SNAPSHOT.search(label)
                   for label in labels):
                g.details.append({"column": idx, "total_row": c["total_row"],
                                  "applicable": False, "skipped": "non_additive_metric_rows"})
                continue
            applicable_checks.append(c)
            g.applicable += 1
            g.passed += bool(c["passed"])
            g.details.append({"column": idx, **c})
        # percent columns: items should sum to ~100 when the total row says ~100
        if col.rule and "percent" in col.rule.features:
            for c in applicable_checks:
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
                      "passed": not too_wide, "blocking": True})

    # A named, empty leading column in a numeric table may be a page title
    # captured as a header (seen on the EU ledger). Blank spacer columns and
    # text-only form layouts are not evidence of a misaligned extraction.
    # Preserve every source column; alternate extraction is still required
    # when the numeric/title pattern supplies evidence of a malformed grid.
    populated = [idx for idx in range(raw.n_cols) if any(
        idx < len(row) and not is_null_cell(row[idx]) for row in raw.rows)]
    empty_leading = list(range(min(populated))) if populated else []
    numeric_measures = any(
        col.is_numeric and not _identifier_column(raw, idx)
        for idx, col in cols.items()
    )
    suspicious = [idx for idx in empty_leading if raw.header[idx].strip()] \
        if numeric_measures else []
    g.applicable += 1
    ok = bool(raw.header) and not suspicious
    g.passed += ok
    g.details.append({"check": "no_empty_leading_columns", "columns": suspicious,
                      "retained_empty_columns": empty_leading,
                      "passed": ok, "blocking": True})

    # No repeated header rows inside the body (missed multi-page merge symptom).
    g.applicable += 1
    header_norm = [_norm(c) for c in raw.header]
    repeats = [
        i for i, row in enumerate(raw.rows)
        if len(row) == raw.n_cols and [_norm(c) for c in row] == header_norm
    ]
    g.passed += not repeats
    g.details.append({"check": "no_header_repeats", "rows": repeats,
                      "passed": not repeats, "blocking": True})

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


_PROSE_CONTEXT_CHARS = 12000


def _prose_excerpt(text: str, anchors: list[str], limit: int) -> str:
    """Bound long pages around claim identifiers instead of dropping their tail."""
    if len(text) <= limit:
        return text
    folded = text.casefold()
    spans = []
    for anchor in anchors:
        anchor = anchor.strip().casefold()
        if not anchor:
            continue
        start = 0
        for _ in range(3):
            hit = folded.find(anchor, start)
            if hit < 0:
                break
            spans.append((max(0, hit - 600), min(len(text), hit + len(anchor) + 600)))
            start = hit + len(anchor)
    if not spans:
        return text[:limit]
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return "\n[excerpt gap]\n".join(text[start:end] for start, end in merged)[:limit]


def _prose_context(page_texts: dict[int, str], page: int, anchors: list[str]) -> str:
    # The row's page takes priority; a long previous page must not displace
    # the very page whose values are being checked.
    excerpts = []
    remaining = _PROSE_CONTEXT_CHARS
    for p in (page, page - 1, page + 1):
        text = page_texts.get(p, "").strip()
        prefix = f"[Page {p}]\n"
        if not text or remaining <= len(prefix):
            continue
        excerpt = prefix + _prose_excerpt(text, anchors, remaining - len(prefix))
        excerpts.append(excerpt)
        remaining -= len(excerpt) + 2
    return "\n\n".join(excerpts)


def check_prose(
    raw: RawTable,
    cols: dict[int, CoercedColumn],
    page_texts: dict[int, str],
    ask: ProseChecker,
    k: int,
    seed: int = 0,
) -> GroupResult:
    """Cross-check claims against independent narrative, excluding table renders.

    ``page_texts`` contains prose only (the writer enforces this). Absence
    of corroboration is not a contradiction and supplies no trust evidence.
    """
    g = GroupResult()
    if raw.kind == "text_lines":
        return g
    numeric_cells = [
        (i, idx, cols[idx].values[i])
        for idx, col in cols.items() if col.is_numeric and not _identifier_column(raw, idx)
        for i in range(len(raw.rows)) if cols[idx].values[i] is not None
    ]
    if not numeric_cells:
        return g
    rng = random.Random(seed)
    sample = rng.sample(numeric_cells, min(k, len(numeric_cells)))
    prompts = []
    claims = []
    numeric_columns = {idx for idx, col in cols.items() if col.is_numeric}
    for i, idx, value in sample:
        page = raw.row_page(i)
        row_labels = [str(cell) for c, cell in enumerate(raw.rows[i])
                      if c not in numeric_columns and cell]
        raw_value = raw.rows[i][idx]
        context = _prose_context(
            page_texts, page, row_labels + [str(raw_value), raw.header[idx], raw.caption or ""])
        detail = {"row": i, "column": idx, "page": page, "value": value,
                  "raw_value": raw_value, "row_labels": row_labels,
                  "column_name": raw.header[idx]}
        if not context:
            g.details.append({**detail, "applicable": False, "skipped": "no_independent_prose"})
            continue
        claims.append(detail)
        prompts.append(
            f"Independent document prose (table renderings excluded):\n{context}\n\n"
            f"Table claim to check, not evidence: page {page}; caption {raw.caption!r}; "
            f"row labels {row_labels!r}; column {raw.header[idx]!r}; "
            f"raw cell {raw_value!r}; interpreted numeric value {value}; "
            f"caption unit multiplier {caption_scale(raw.caption)}.\n"
            "Question: Does the prose above state or imply this particular table claim? "
            "Compare the same entity, measure, period and units; a matching number alone "
            "does not establish agreement. Answer YES only for supporting prose; "
            "NO only for explicit conflicting prose about the same claim; UNCLEAR if "
            "the value is absent, the prose is unrelated, or identity/units are ambiguous. "
            "A value appearing only in the table claim is UNCLEAR, never YES or NO. "
            "Answer only YES, NO, or UNCLEAR."
        )
    if not prompts:
        return g
    answers = ask(prompts)
    for detail, ans in zip(claims, answers, strict=True):
        if ans is None:
            g.details.append({**detail, "applicable": False, "skipped": "unclear_or_no_support"})
            continue  # UNCLEAR — no evidence either way
        g.applicable += 1
        g.passed += bool(ans)
        g.details.append({**detail, "applicable": True, "agrees": ans})
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
    failing = {d["column"] for d in arithmetic.details if d.get("passed") is False}
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
                unit_scale=caption_scale(raw.caption),
            )
            if not retry.is_numeric:
                continue
            trial = dict(cols)
            trial[idx] = retry
            re_arith = check_arithmetic(raw, trial, rel_tol, abs_tol)
            before = [d for d in arithmetic.details
                      if d.get("column") == idx and d.get("passed") is False]
            after = [d for d in re_arith.details
                     if d.get("column") == idx and d.get("passed") is False]
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
