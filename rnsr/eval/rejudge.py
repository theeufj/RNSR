"""Re-judge saved eval answers offline, with repeated passes, without re-answering.

    python -m rnsr.eval.rejudge --run-dir runs/public-arms/financebench-docdb --passes 3

Reads ``results.jsonl``, re-applies the strict scorer, then the deterministic
presentation normaliser (percent/ratio, currency, scale words, unit suffixes),
then the current judge rubric ``--passes`` times (odd, so a majority exists)
to every row neither settles, and writes ``results.rejudged.jsonl`` (a row is
correct when a deterministic pass agrees or a majority of judge passes say
YES) and ``rejudge_summary.json`` (per-pass counts, so judge variance is
visible). ``--sample N`` also writes ``adjudication_sample.jsonl``: N
judge-decided rows chosen by seed, with the verdict and the votes, for a
human to mark ``human_correct`` by hand; ``--adjudicated FILE`` reads such a
file back and reports judge-vs-human agreement. The original
``results.jsonl`` is not modified.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from rnsr.config import Settings
from rnsr.eval.metrics import (
    JUDGE_VERSION,
    NORMALISER_VERSION,
    SCORER_VERSION,
    judge_answer,
    normalised_match,
    score_answer,
)
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
        how = "string" if ok else None
        if not ok and pred is not None and normalised_match(pred, gold):
            ok, how = True, "normalised"
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
        return {**r, "correct": correct, "scored_by": how or "judge", "judge_votes": votes,
                "correct_original": r.get("correct")}

    out_rows = await asyncio.gather(*(one(r) for r in rows))
    # Deterministic agreements count in every pass.
    det = sum(1 for r in out_rows if r["scored_by"] in ("string", "normalised") and r["correct"])
    per_pass = [p + det for p in per_pass]
    (run_dir / "results.rejudged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    summary = {"n": len(out_rows), "correct": sum(r["correct"] for r in out_rows),
               "correct_original": sum(1 for r in out_rows if r.get("correct_original")),
               "per_pass_correct": per_pass, "passes": passes, "deterministic": det,
               "normalised": sum(1 for r in out_rows if r["scored_by"] == "normalised"),
               "judge_split": sum(1 for r in out_rows if r["judge_votes"]
                                  and len({bool(v) for v in r["judge_votes"]}) > 1),
               "scorer_version": SCORER_VERSION, "normaliser_version": NORMALISER_VERSION,
               "judge_version": JUDGE_VERSION,
               "questions_known": sum(1 for r in rows if r["qid"] in questions)}
    (run_dir / "rejudge_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def write_sample(run_dirs: list[Path], n: int, seed: int, out: Path, questions: dict[str, str]) -> int:
    """Judge-decided rows across runs, sampled by seed, for hand adjudication."""
    pool = []
    for rd in run_dirs:
        path = rd / "results.rejudged.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("scored_by") == "judge":
                pool.append({"run": rd.name, "qid": r["qid"], "question": questions.get(r["qid"], ""),
                             "predicted": r.get("predicted"), "gold": r.get("gold"),
                             "judge_correct": r["correct"], "judge_votes": r.get("judge_votes"),
                             "human_correct": None, "human_note": ""})
    random.Random(seed).shuffle(pool)
    rows = sorted(pool[:n], key=lambda r: (r["run"], r["qid"]))
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def agreement(adjudicated: Path) -> dict:
    rows = [json.loads(line) for line in adjudicated.read_text().splitlines() if line.strip()]
    marked = [r for r in rows if r.get("human_correct") in (True, False)]
    agree = sum(1 for r in marked if r["human_correct"] == r["judge_correct"])
    return {"rows": len(rows), "marked": len(marked), "agree": agree,
            "judge_yes_human_no": sum(1 for r in marked if r["judge_correct"] and not r["human_correct"]),
            "judge_no_human_yes": sum(1 for r in marked if not r["judge_correct"] and r["human_correct"])}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, nargs="+")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--questions-json", type=Path, default=None,
                    help="optional {qid: question} file when the run manifest lacks questions")
    ap.add_argument("--sample", type=int, default=0, help="write N judge-decided rows for hand adjudication")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--sample-out", type=Path, default=None)
    ap.add_argument("--adjudicated", type=Path, default=None, help="report judge-vs-human agreement for a marked sample")
    ap.add_argument("--no-rejudge", action="store_true", help="only sample/report from existing results.rejudged.jsonl")
    args = ap.parse_args(argv)
    extra = json.loads(args.questions_json.read_text()) if args.questions_json else {}
    all_q = dict(extra)
    for rd in args.run_dir:
        qs = {**_questions(rd), **extra}
        all_q.update(qs)
        if args.no_rejudge:
            continue
        s = asyncio.run(rejudge(rd, args.passes, qs))
        print(f"{rd}: {s['correct_original']} -> {s['correct']} / {s['n']}  per-pass {s['per_pass_correct']}  "
              f"normalised {s['normalised']}  judge-split {s['judge_split']}  (questions known {s['questions_known']}/{s['n']})")
    if args.sample:
        out = args.sample_out or (args.run_dir[0].parent / "adjudication_sample.jsonl")
        print(f"sample: {write_sample(args.run_dir, args.sample, args.seed, out, all_q)} rows -> {out}")
    if args.adjudicated:
        print("agreement:", json.dumps(agreement(args.adjudicated)))


if __name__ == "__main__":
    main()
