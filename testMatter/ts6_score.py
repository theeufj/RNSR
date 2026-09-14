"""Score a Test Set 6 answers CSV against the verification key.

String scoring first (free); a sub-LM equivalence judge decides the
string failures, mirroring the eval harness. Emits comparison.csv next to
the answers file.

Usage:
    python testMatter/ts6_score.py --answers runs/ts6/out/answers_chunk1.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
GOLDEN = (HERE / "Test Set 6 - Initiating Application (999 Documents)"
          / "ts6_golden.json")
NOT_FOUND = "Not found in matter corpus"


def norm(s: str) -> str:
    s = re.sub(r"[☑☐✓]", " ", s or "")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s/@.:$%-]", " ", s.lower())).strip()


def string_match(golden: str, answer: str) -> bool:
    g, a = norm(golden), norm(answer)
    if not g:
        return not a or a in ("not applicable", "unknown") or a.startswith(
            norm(NOT_FOUND))
    if g == a:
        return True
    negatives = ("not applicable", "n/a", "no", "blank", "not reached",
                 "unknown")
    if g in negatives or g.startswith(("leave blank", "blank ", "not reached")):
        return a in negatives or a.startswith(norm(NOT_FOUND)) or "blank" in a
    return (g in a or a in g) and min(len(g), len(a)) >= 4


async def judge_misses(rows: list[dict]) -> None:
    from rnsr.config import Settings
    from rnsr.eval.metrics import judge_answer
    from rnsr.llm.router import Router

    sub = Router(Settings.from_env()).resolve("sub")
    sem = asyncio.Semaphore(16)

    async def one(r: dict) -> None:
        async with sem:
            verdict = await judge_answer(sub.client, sub.model, r["question"],
                                         r["answer"], r["golden"])
        if verdict is True:
            r["verdict"], r["scored_by"] = "OK", "judge"

    await asyncio.gather(*(one(r) for r in rows if r["verdict"] == "DIFF"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, type=Path)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    golden = {it["qid"]: it for it in json.loads(GOLDEN.read_text())["items"]}
    with open(args.answers, newline="", encoding="utf-8") as fh:
        answers = list(csv.DictReader(fh))
    qids = list(golden)
    assert len(answers) == len(qids), (len(answers), len(qids))

    rows = []
    for qid, arow in zip(qids, answers, strict=True):
        it = golden[qid]
        ans = arow["model_answer"]
        ok = string_match(it["golden"], ans)
        rows.append({
            "qid": qid, "part": it["part"], "item": it["item"],
            "question": it["question"], "golden": it["golden"],
            "answer": ans, "note": it["note"],
            "verdict": "OK" if ok else "DIFF",
            "scored_by": "string",
        })

    if not args.no_judge and any(r["verdict"] == "DIFF" for r in rows):
        asyncio.run(judge_misses(rows))

    n_ok = sum(r["verdict"] == "OK" for r in rows)
    print(f"agreement: {n_ok}/{len(rows)}")
    for by in ("string", "judge"):
        n = sum(r["verdict"] == "OK" and r["scored_by"] == by for r in rows)
        print(f"  via {by}: {n}")
    print("\nremaining disagreements:")
    for r in rows:
        if r["verdict"] == "DIFF":
            print(f"  [{r['part']} {r['item']}] {r['question'][:70]}")
            print(f"    golden: {r['golden'][:160]!r}")
            print(f"    rnsr:   {r['answer'][:160]!r}")
            if r["note"]:
                print(f"    note:   {r['note'][:140]}")

    out = args.answers.parent / "comparison.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
