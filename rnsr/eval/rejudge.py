"""Re-judge saved eval answers offline, with repeated passes, without re-answering.

    python -m rnsr.eval.rejudge --run-dir runs/public-arms/financebench-docdb --passes 2

Reads ``results.jsonl``, re-applies the deterministic scorer and then the
current judge rubric ``--passes`` times to every row the scorer does not
settle, and writes ``results.rejudged.jsonl`` (a row is correct when the
scorer agrees or a majority of judge passes say YES) and
``rejudge_summary.json`` (per-pass counts, so judge variance is visible).
The original ``results.jsonl`` is not modified. Questions are re-read from
the run manifest's dataset when present, else from the benchmark loader.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rnsr.config import Settings
from rnsr.eval.metrics import JUDGE_VERSION, SCORER_VERSION, judge_answer, score_answer
from rnsr.sdk import make_runner


def _questions(run_dir: Path) -> dict[str, str]:
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        items = data.get("dataset", {}).get("items") or data.get("items") or []
        out = {i.get("qid"): i.get("question") for i in items if isinstance(i, dict) and i.get("qid") and i.get("question")}
        if out:
            return out
    return {}


async def rejudge(run_dir: Path, passes: int, questions: dict[str, str]) -> dict:
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    runner = make_runner(Settings.from_env())
    per_pass = [0] * passes
    out_rows = []
    sem = asyncio.Semaphore(4)

    async def one(r: dict) -> dict:
        pred, gold = r.get("predicted"), r.get("gold")
        ok = score_answer(pred, gold)
        votes = []
        if not ok and pred is not None:
            async with sem:
                for _ in range(passes):
                    votes.append(await judge_answer(runner.sub_client, runner.sub_model,
                                                    question=questions.get(r["qid"], ""), predicted=pred, gold=gold))
        for i, v in enumerate(votes):
            per_pass[i] += bool(ok or v is True)
        if ok:
            for i in range(passes):
                per_pass[i] += 0  # already counted via votes branch only when judged; add below
        yes = sum(1 for v in votes if v is True)
        correct = bool(ok) or (bool(votes) and yes * 2 > len(votes))
        return {**r, "correct": correct, "scored_by": "string" if ok else "judge", "judge_votes": votes,
                "correct_original": r.get("correct")}

    out_rows = await asyncio.gather(*(one(r) for r in rows))
    # Deterministic agreements count in every pass.
    det = sum(1 for r in out_rows if r["scored_by"] == "string" and r["correct"])
    per_pass = [p + det for p in per_pass]
    (run_dir / "results.rejudged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    summary = {"n": len(out_rows), "correct": sum(r["correct"] for r in out_rows),
               "correct_original": sum(1 for r in out_rows if r.get("correct_original")),
               "per_pass_correct": per_pass, "passes": passes, "deterministic": det,
               "scorer_version": SCORER_VERSION, "judge_version": JUDGE_VERSION,
               "questions_known": sum(1 for r in rows if r["qid"] in questions)}
    (run_dir / "rejudge_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, nargs="+")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--questions-json", type=Path, default=None,
                    help="optional {qid: question} file when the run manifest lacks questions")
    args = ap.parse_args(argv)
    extra = json.loads(args.questions_json.read_text()) if args.questions_json else {}
    for rd in args.run_dir:
        qs = {**_questions(rd), **extra}
        s = asyncio.run(rejudge(rd, args.passes, qs))
        print(f"{rd}: {s['correct_original']} -> {s['correct']} / {s['n']}  per-pass {s['per_pass_correct']}  (questions known {s['questions_known']}/{s['n']})")


if __name__ == "__main__":
    main()
