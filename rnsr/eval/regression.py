"""Golden-set regression scoring.

Accuracy here depends on things outside the repository: a provider can
change a model's behaviour between two identical runs, and the enrichment
rules are tuned against observed misses. That combination needs a standing
check rather than a number in a README — otherwise a silent provider-side
change is discovered by a client.

score_run() generalizes the two per-test-set scorers: string agreement
first (free), a sub-LM equivalence judge only for string failures, and a
minimum-accuracy gate so a scheduled run can fail loudly.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

NOT_FOUND = "Not found in matter corpus"
_NEGATIVES = ("", "no", "n/a", "na", "none", "unknown", "not applicable",
              "not_applicable", "blank", "leave blank", "not reached")


def normalize(text: str) -> str:
    text = re.sub(r"[\u2610\u2611\u2612\u2713\u2717]", " ", text or "")
    return re.sub(r"\s+", " ",
                  re.sub(r"[^\w\s/@.:$%-]", " ", text.lower())).strip(" .")


def is_negative(text: str) -> bool:
    n = normalize(text)
    return n in _NEGATIVES or n.startswith(normalize(NOT_FOUND))


def string_agrees(golden: list[str] | str, answer: str) -> bool:
    """Field-level agreement before any model judgement.

    Vendors are inconsistent about how a chosen option is recorded — often
    "yes", sometimes the option's full text — so a yes/no answer and the
    corresponding option text count as agreeing.
    """
    golds = [golden] if isinstance(golden, str) else list(golden)
    golds = [g for g in golds if g is not None]
    if not golds or all(not str(g).strip() for g in golds):
        return is_negative(answer)
    a = normalize(answer)
    for gold in golds:
        g = normalize(str(gold))
        if g == a:
            return True
        if g in _NEGATIVES or g.startswith(("leave blank", "not reached")):
            if is_negative(answer):
                return True
            continue
        # a trailing "- yes"/"- no" marks how the vendor recorded the choice
        m_g = re.search(r"[-:]\s*(yes|no)$", g)
        m_a = re.search(r"[-:]\s*(yes|no)$", a)
        if (m_g and a == m_g.group(1)) or (m_a and g == m_a.group(1)):
            return True
        shorter, longer = sorted((g, a), key=len)
        if len(shorter) >= 4 and shorter in longer:
            return True
    return False


@dataclass
class FieldResult:
    field_id: str
    golden: str
    answer: str
    agrees: bool
    scored_by: str = "string"
    note: str = ""


@dataclass
class RegressionReport:
    results: list[FieldResult] = field(default_factory=list)
    min_accuracy: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(r.agrees for r in self.results)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def passed(self) -> bool:
        return self.accuracy >= self.min_accuracy

    @property
    def substantive(self) -> tuple[int, int]:
        """(correct, total) over fields whose golden holds a real value.

        Fields whose gold is blank or a bare "No" are satisfied by answering
        nothing, so raw agreement over every field flatters a cautious run.
        """
        rows = [r for r in self.results
                if r.golden.strip() and normalize(r.golden) not in _NEGATIVES]
        return sum(r.agrees for r in rows), len(rows)

    def summary(self) -> dict:
        sub_correct, sub_total = self.substantive
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "substantive_correct": sub_correct,
            "substantive_total": sub_total,
            "scored_by_judge": sum(r.scored_by == "judge" for r in self.results),
            "min_accuracy": self.min_accuracy,
            "passed": self.passed,
            "disagreements": [
                {"field_id": r.field_id, "golden": r.golden[:200],
                 "answer": r.answer[:200]}
                for r in self.results if not r.agrees
            ],
        }

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["field_id", "verdict", "scored_by", "golden", "answer"])
            for r in self.results:
                w.writerow([r.field_id, "OK" if r.agrees else "DIFF",
                            r.scored_by, r.golden, r.answer])
        (out / "regression_summary.json").write_text(
            json.dumps(self.summary(), indent=2))
        return out / "regression_summary.json"


def load_golden(path: str | Path) -> dict[str, list[str]]:
    """field_id -> golden values, from a vendor-shaped golden JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = data.get("fields") or data.get("items") or []
    out: dict[str, list[str]] = {}
    for f in fields:
        gold = f.get("golden", [])
        out[f["id"] if "id" in f else f["qid"]] = (
            [gold] if isinstance(gold, str) else list(gold or []))
    return out


def load_field_answers(path: str | Path) -> dict[str, str]:
    """field_id -> answer, from a two-column CSV (id, answer)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}
    start = 1 if rows[0] and rows[0][0].strip().lower() in (
        "field_id", "qid", "id") else 0
    return {r[0]: (r[1] if len(r) > 1 else "") for r in rows[start:] if r}


def score_run(golden: dict[str, list[str]], answers: dict[str, str], *,
              min_accuracy: float = 0.0) -> RegressionReport:
    """String-only scoring (free, deterministic). Judge separately."""
    report = RegressionReport(min_accuracy=min_accuracy)
    for field_id, gold in golden.items():
        answer = answers.get(field_id, "")
        report.results.append(FieldResult(
            field_id=field_id,
            golden="; ".join(str(g) for g in gold),
            answer=answer,
            agrees=string_agrees(gold, answer),
        ))
    return report


async def judge_disagreements(report: RegressionReport, client, model: str, *,
                              concurrency: int = 8) -> RegressionReport:
    """Ask a sub-LM whether string-failed answers are equivalent anyway.

    Long-form answers differ in wording without differing in meaning, which
    string matching cannot see. Only failures are judged, so agreement can
    only go up and the judge cannot invent a regression.
    """
    from rnsr.eval.metrics import judge_answer

    sem = asyncio.Semaphore(concurrency)
    pending = [r for r in report.results if not r.agrees and r.answer.strip()]

    async def one(r: FieldResult) -> None:
        async with sem:
            verdict = await judge_answer(client, model, r.field_id, r.answer,
                                         r.golden)
        if verdict is True:
            r.agrees, r.scored_by = True, "judge"

    await asyncio.gather(*(one(r) for r in pending))
    return report
