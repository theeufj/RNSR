"""Scoring and aggregate metrics (§8): accuracy per task class, cost and
latency at p50 *and* p95 (the cost tail is the story), plus the go/no-go
gate comparison."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass


@dataclass
class EvalResult:
    qid: str
    task_class: str
    predicted: str | None
    gold: str
    correct: bool
    status: str                 # QueryResult.status
    latency_s: float
    cost_usd: float
    sub_calls: int
    iterations: int
    trajectory_path: str | None = None
    scored_by: str = "string"   # 'string' | 'judge'

    def to_dict(self) -> dict:
        return asdict(self)


_NUM = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower()).strip(" .")


def _as_number(text: str) -> float | None:
    m = _NUM.search(str(text).replace(",", ""))
    try:
        return float(m.group()) if m else None
    except ValueError:
        return None


def score_answer(predicted: object, gold: str, *, numeric_rel_tol: float = 0.01) -> bool:
    """Exact normalized match, else numeric match within tolerance.

    Numeric gold answers accept magnitude-only agreement (e.g. '3,234' vs
    '3234.0'); textual answers accept containment either way after
    normalization — benchmark-specific judges can override per loader.
    """
    if predicted is None:
        return False
    p, g = _normalize(str(predicted)), _normalize(gold)
    if p == g:
        return True
    gn = _as_number(g)
    if gn is not None:
        pn = _as_number(p)
        if pn is None:
            return False
        if gn == 0:
            return pn == 0
        return abs(pn - gn) / abs(gn) <= numeric_rel_tol
    return (g in p or p in g) and len(p) < 4 * len(g)


_JUDGE_PROMPT = """\
Question: {question}

Reference answer: {gold}

Candidate answer: {predicted}

Does the candidate answer agree with the reference answer on the substance
of the question? Treat numeric values as agreeing when they match after
unit conversion and reasonable rounding. Ignore extra explanation, hedging,
or detail beyond the reference. Reply with exactly YES or NO."""


async def judge_answer(client, model: str, question: str,
                       predicted: str, gold: str) -> bool | None:
    """One sub-LM YES/NO equivalence call. None when the judge is unusable
    (call failed or reply unparseable) — callers keep the string verdict."""
    try:
        resp = await client.complete(
            _JUDGE_PROMPT.format(question=question, gold=gold, predicted=predicted),
            model=model, max_tokens=8,
        )
    except Exception:
        return None
    text = resp.text.strip().upper()
    if text.startswith("YES"):
        return True
    if text.startswith("NO"):
        return False
    return None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(results: list[EvalResult]) -> dict:
    by_class: dict[str, list[EvalResult]] = {}
    for r in results:
        by_class.setdefault(r.task_class, []).append(r)
    latencies = [r.latency_s for r in results]
    costs = [r.cost_usd for r in results]
    return {
        "n": len(results),
        "accuracy": (sum(r.correct for r in results) / len(results)) if results else 0.0,
        "accuracy_by_class": {
            c: sum(r.correct for r in rs) / len(rs) for c, rs in sorted(by_class.items())
        },
        "latency_s": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        "cost_usd": {"p50": percentile(costs, 0.5), "p95": percentile(costs, 0.95)},
        "sub_calls_mean": (sum(r.sub_calls for r in results) / len(results)) if results else 0,
        "status_counts": {
            s: sum(r.status == s for r in results)
            for s in sorted({r.status for r in results})
        },
        "scored_by_counts": {
            s: sum(r.scored_by == s for r in results)
            for s in sorted({r.scored_by for r in results})
        },
    }


def gate_report(docdb: dict, classic: dict, *, numeric_classes: tuple[str, ...] = ("numeric",),
                match_margin: float = 0.02) -> dict:
    """§8 go/no-go: DocDB must beat classic on numeric classes and match it
    elsewhere (within margin), at equal-or-lower median cost."""
    checks = {}
    for cls, acc in docdb["accuracy_by_class"].items():
        base = classic["accuracy_by_class"].get(cls)
        if base is None:
            continue
        if cls in numeric_classes:
            checks[f"beats_classic[{cls}]"] = acc > base
        else:
            checks[f"matches_classic[{cls}]"] = acc >= base - match_margin
    checks["cost_not_worse_p50"] = docdb["cost_usd"]["p50"] <= classic["cost_usd"]["p50"] * 1.05
    return {"pass": all(checks.values()), "checks": checks,
            "docdb": docdb, "classic": classic}
