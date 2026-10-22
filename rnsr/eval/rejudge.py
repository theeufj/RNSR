"""Re-judge saved eval answers offline, with repeated passes and more than one
judge, without re-answering.

    python -m rnsr.eval.rejudge --run-dir runs/public-arms/financebench-docdb --passes 3 \
        --judges anthropic:claude-haiku-5-5,gemini:gemini-3.8-flash

Reads ``results.jsonl``, re-applies the strict scorer, then the deterministic
presentation normaliser, then an optional gold overlay, then each judge
``--passes`` times (odd, so a majority exists) to every row nothing
deterministic settles. A row is correct when a deterministic pass agrees or
when EVERY judge's majority says YES; a row on which the judges' majorities
disagree is ``correct=False``, ``scored_by="judge_disputed"`` and goes to the
adjudication file. ``results.rejudged.jsonl`` carries per-judge votes and
``rejudge_summary.json`` carries per-judge, per-pass counts, cross-judge
agreement (raw and Cohen's kappa) and judge spend.

``--sample N`` writes ``adjudication_sample.jsonl``: every disputed row plus N
seeded judge-decided rows, with the verdicts and votes, for a human to mark
``human_correct``; ``--adjudicated FILE`` reads such a file back and reports
judge-vs-human agreement per judge. The original ``results.jsonl`` is not
modified.
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
    judge_answer_with_usage,
    normalised_match,
    score_answer,
)

DEFAULT_JUDGES = ("anthropic:claude-haiku-5-5",)


def _questions(run_dir: Path) -> dict[str, str]:
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        items = data.get("dataset", {}).get("items") or data.get("items") or []
        out = {i.get("qid"): i.get("question") for i in items if isinstance(i, dict) and i.get("qid") and i.get("question")}
        if out:
            return out
    return {}


def resolve_judges(specs: list[str]):
    """'provider:model' specs -> [(label, client, model)]. A bare model name
    uses the configured provider. Clients are governed like every other call."""
    from rnsr.llm.router import Router

    router = Router(Settings.from_env())
    out = []
    for spec in specs:
        provider, _, model = spec.rpartition(":")
        provider = provider or router.provider
        out.append((f"{provider}:{model}", router._client_for(provider), model))
    return out


def kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa between two binary verdict vectors."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


async def rejudge(run_dir: Path, passes: int, questions: dict[str, str], *,
                  judges: list[str] | None = None, overlay=None) -> dict:
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    judge_specs = list(judges or DEFAULT_JUDGES)
    resolved = resolve_judges(judge_specs)
    labels = [label for label, _, _ in resolved]
    sem = asyncio.Semaphore(4)
    spend = {label: 0.0 for label in labels}

    async def one(r: dict) -> dict:
        pred, gold = r.get("predicted"), r.get("gold")
        ok = score_answer(pred, gold)
        how = "string" if ok else None
        if not ok and pred is not None and normalised_match(pred, gold):
            ok, how = True, "normalised"
        if not ok and pred is not None and overlay is not None and overlay.accepts(r["qid"], pred):
            ok, how = True, "gold_overlay"
        votes: dict[str, list] = {label: [] for label in labels}
        if not ok and pred is not None:
            async with sem:
                for label, client, model in resolved:
                    for _ in range(passes):
                        v, cost = await judge_answer_with_usage(
                            client, model, question=questions.get(r["qid"], ""), predicted=pred, gold=gold)
                        votes[label].append(v)
                        spend[label] += cost
        majority = {label: (sum(1 for v in vs if v is True) * 2 > len(vs)) if vs else None
                    for label, vs in votes.items()}
        if ok:
            correct, scored_by = True, how
        else:
            decided = [m for m in majority.values() if m is not None]
            if not decided:
                correct, scored_by = False, "judge_unavailable" if pred is not None else "no_answer"
            elif all(decided):
                correct, scored_by = True, "judge"
            elif not any(decided):
                correct, scored_by = False, "judge"
            else:
                correct, scored_by = False, "judge_disputed"
        return {**r, "correct": correct, "scored_by": scored_by,
                "judge_votes": votes[labels[0]], "judge_votes_by": votes, "judge_majority": majority,
                "correct_original": r.get("correct")}

    out_rows = await asyncio.gather(*(one(r) for r in rows))
    det = sum(1 for r in out_rows if r["scored_by"] in ("string", "normalised", "gold_overlay") and r["correct"])
    per_judge = {}
    for label in labels:
        per_pass = [det] * passes
        for r in out_rows:
            for i, v in enumerate(r["judge_votes_by"].get(label, [])):
                per_pass[i] += bool(v is True)
        maj_correct = det + sum(1 for r in out_rows if r["judge_majority"].get(label) is True)
        split = sum(1 for r in out_rows if r["judge_votes_by"].get(label)
                    and len({bool(v) for v in r["judge_votes_by"][label]}) > 1)
        per_judge[label] = {"per_pass_correct": per_pass, "majority_correct": maj_correct,
                            "split_rows": split, "spend_usd": round(spend[label], 4)}
    agreement = {}
    if len(labels) > 1:
        judged = [r for r in out_rows if all(r["judge_majority"].get(label) is not None for label in labels)]
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                va = [bool(r["judge_majority"][a]) for r in judged]
                vb = [bool(r["judge_majority"][b]) for r in judged]
                agreement[f"{a} vs {b}"] = {
                    "judged_rows": len(judged),
                    "agree": sum(1 for x, y in zip(va, vb, strict=True) if x == y),
                    "kappa": kappa(va, vb)}
    (run_dir / "results.rejudged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    summary = {"n": len(out_rows), "correct": sum(r["correct"] for r in out_rows),
               "correct_original": sum(1 for r in out_rows if r.get("correct_original")),
               "passes": passes, "deterministic": det,
               "normalised": sum(1 for r in out_rows if r["scored_by"] == "normalised"),
               "gold_overlay": sum(1 for r in out_rows if r["scored_by"] == "gold_overlay"),
               "disputed": sum(1 for r in out_rows if r["scored_by"] == "judge_disputed"),
               "judges": labels, "per_judge": per_judge, "cross_judge": agreement,
               "judge_spend_usd": round(sum(spend.values()), 4),
               # backwards-compatible fields (first judge)
               "per_pass_correct": per_judge[labels[0]]["per_pass_correct"],
               "judge_split": per_judge[labels[0]]["split_rows"],
               "scorer_version": SCORER_VERSION, "normaliser_version": NORMALISER_VERSION,
               "judge_version": JUDGE_VERSION,
               "overlay_version": getattr(overlay, "version", None),
               "questions_known": sum(1 for r in rows if r["qid"] in questions)}
    (run_dir / "rejudge_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def write_sample(run_dirs: list[Path], n: int, seed: int, out: Path, questions: dict[str, str]) -> int:
    """Every disputed row, plus n judge-decided rows sampled by seed, for hand
    adjudication."""
    disputed, pool = [], []
    for rd in run_dirs:
        path = rd / "results.rejudged.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("scored_by") not in ("judge", "judge_disputed"):
                continue
            row = {"run": rd.name, "qid": r["qid"], "question": questions.get(r["qid"], ""),
                   "predicted": r.get("predicted"), "gold": r.get("gold"),
                   "judge_correct": r["correct"], "scored_by": r["scored_by"],
                   "judge_votes": r.get("judge_votes"),
                   "judge_votes_by": r.get("judge_votes_by") or {"judge": r.get("judge_votes")},
                   "human_correct": None, "human_note": ""}
            (disputed if r["scored_by"] == "judge_disputed" else pool).append(row)
    random.Random(seed).shuffle(pool)
    rows = sorted(disputed + pool[:n], key=lambda r: (r["run"], r["qid"]))
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def agreement(adjudicated: Path) -> dict:
    rows = [json.loads(line) for line in adjudicated.read_text().splitlines() if line.strip()]
    marked = [r for r in rows if r.get("human_correct") in (True, False)]
    out = {"rows": len(rows), "marked": len(marked),
           "agree": sum(1 for r in marked if r["human_correct"] == r["judge_correct"]),
           "judge_yes_human_no": sum(1 for r in marked if r["judge_correct"] and not r["human_correct"]),
           "judge_no_human_yes": sum(1 for r in marked if not r["judge_correct"] and r["human_correct"]),
           "per_judge": {}}
    labels = sorted({label for r in marked for label in (r.get("judge_votes_by") or {})})
    for label in labels:
        pairs = []
        for r in marked:
            vs = (r.get("judge_votes_by") or {}).get(label) or []
            if vs:
                pairs.append((sum(1 for v in vs if v is True) * 2 > len(vs), r["human_correct"]))
        out["per_judge"][label] = {"marked": len(pairs), "agree": sum(1 for j, h in pairs if j == h)}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True, nargs="+")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--judges", default=",".join(DEFAULT_JUDGES),
                    help="comma-separated provider:model judges; all must agree for a judge-scored row to count")
    ap.add_argument("--overlay", type=Path, default=None, help="gold overlay JSON (see rnsr.eval.gold_overlay)")
    ap.add_argument("--questions-json", type=Path, default=None,
                    help="optional {qid: question} file when the run manifest lacks questions")
    ap.add_argument("--sample", type=int, default=0, help="write N judge-decided rows (plus every disputed row) for hand adjudication")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--sample-out", type=Path, default=None)
    ap.add_argument("--adjudicated", type=Path, default=None, help="report judge-vs-human agreement for a marked sample")
    ap.add_argument("--no-rejudge", action="store_true", help="only sample/report from existing results.rejudged.jsonl")
    args = ap.parse_args(argv)
    extra = json.loads(args.questions_json.read_text()) if args.questions_json else {}
    overlay = None
    if args.overlay:
        from rnsr.eval.gold_overlay import GoldOverlay

        overlay = GoldOverlay.load(args.overlay)
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    all_q = dict(extra)
    for rd in args.run_dir:
        qs = {**_questions(rd), **extra}
        all_q.update(qs)
        if args.no_rejudge:
            continue
        s = asyncio.run(rejudge(rd, args.passes, qs, judges=judges, overlay=overlay))
        per = "  ".join(f"{label}: maj {v['majority_correct']} passes {v['per_pass_correct']} split {v['split_rows']} ${v['spend_usd']:.2f}"
                        for label, v in s["per_judge"].items())
        print(f"{rd}: {s['correct_original']} -> {s['correct']} / {s['n']}  det {s['deterministic']} "
              f"(normalised {s['normalised']}, overlay {s['gold_overlay']})  disputed {s['disputed']}  | {per}"
              + (f"  | cross-judge {json.dumps(s['cross_judge'])}" if s["cross_judge"] else ""))
    if args.sample:
        out = args.sample_out or (args.run_dir[0].parent / "adjudication_sample.jsonl")
        print(f"sample: {write_sample(args.run_dir, args.sample, args.seed, out, all_q)} rows -> {out}")
    if args.adjudicated:
        print("agreement:", json.dumps(agreement(args.adjudicated)))


if __name__ == "__main__":
    main()
