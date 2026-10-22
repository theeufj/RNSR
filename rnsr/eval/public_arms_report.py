"""Cross-system report for `rnsr eval` runs of one public benchmark.

    python -m rnsr.eval.public_arms_report --run-dir runs/public-arms --benchmark financebench \
        --systems stuffing,rlm-classic,dci,dci-raw,docdb

Reads ``<run-dir>/<benchmark>-<system>/results.jsonl`` for each system and
writes ``<run-dir>/<benchmark>-report.md`` and ``<benchmark>-summary.json``:
accuracy overall and by task class, spend, latency, status and judge-scored
shares, a miss matrix with every system's answer, and for DocDB the
auditability signals mined from its trajectories (verified citations,
replayable derivations on numeric questions, tiers, review halts, embedding
batches). Only qids answered by every listed system are compared.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rnsr.eval.arms_bench import DERIVATION_CHECKS, _report_facts
from rnsr.eval.stats import bootstrap_diff, mcnemar, repeat_range, wilson

LABELS = {"closed-book": "Closed book (no documents)", "stuffing": "Whole document in context", "rlm-classic": "Classic (flat string in REPL)",
          "dci": "DCI (ingested text, per file)", "dci-raw": "DCI over raw files (no ingest)", "docdb": "DocDB"}


def load_results(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[r["qid"]] = r
    return rows


def trajectory_signals(tdir: Path, qids: set[str]) -> dict:
    """Per-qid parent report facts plus run-level counters from solo trajectories."""
    out = {"facts": {}, "root_calls": 0, "embed_batches": 0, "review_halts": 0, "claim_reviews": 0,
           "certificate_nudges": 0, "cells": 0, "cells_sql": 0, "cells_calc": 0}
    if not tdir.exists():
        return out
    for f in tdir.glob("*.jsonl"):
        qid = f.name.split(".", 1)[0]
        if qid not in qids:
            continue
        last_final = None
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = r.get("kind")
                if k == "root_call_completed":
                    out["root_calls"] += 1
                elif k == "embed_batch":
                    out["embed_batches"] += 1
                elif k == "claim_review_halted":
                    out["review_halts"] += 1
                elif k == "claim_review":
                    out["claim_reviews"] += 1
                elif k == "certificate_nudge":
                    out["certificate_nudges"] += 1
                elif k == "final":
                    last_final = r.get("verification")
                if "code" in r:
                    c = r["code"]
                    out["cells"] += 1
                    out["cells_sql"] += ("db.execute" in c or "SELECT " in c.upper())
                    out["cells_calc"] += any(s in c for s in ("source_number(", "aggregate(", "calculate(", "FINAL_CALC"))
        out["facts"][qid] = _report_facts(last_final if isinstance(last_final, dict) else None)
    return out


def build(run_dir: Path, benchmark: str, systems: list[str], *, rejudged: bool = False) -> dict:
    name = "results.rejudged.jsonl" if rejudged else "results.jsonl"
    results = {s: load_results(run_dir / f"{benchmark}-{s}" / name) for s in systems}
    # Repeats: <benchmark>-<system>-r2, -r3, ... answer the same items again.
    repeats = {s: [results[s]] + [load_results(p / name) for p in sorted(run_dir.glob(f"{benchmark}-{s}-r[0-9]*"))
                                  if (p / name).exists()] for s in systems}
    common = set.intersection(*(set(r) for r in results.values() if r)) if any(results.values()) else set()
    n = len(common)
    classes = sorted({results[s][q]["task_class"] for s in systems for q in common if q in results[s]})
    totals, by_class, misses = {}, defaultdict(dict), []
    for s in systems:
        rows = [results[s][q] for q in sorted(common)]
        correct = sum(1 for r in rows if r["correct"])
        rep_counts = [sum(1 for q in common if rr.get(q, {}).get("correct")) for rr in repeats[s] if rr]
        totals[s] = {"correct": correct, "n": n, "ci95": list(wilson(correct, n)),
                     "repeats": repeat_range(rep_counts),
                     "cap_hits": dict(Counter(r.get("cap_hit") for r in rows if r.get("cap_hit"))),
                     "spend_usd": round(sum(r.get("cost_usd", 0) or 0 for r in rows), 4),
                     "latency_p50_s": round(sorted(r.get("latency_s", 0) for r in rows)[len(rows) // 2], 1) if rows else None,
                     "status": dict(Counter(r.get("status") for r in rows)),
                     "judge_scored": sum(1 for r in rows if r.get("scored_by") == "judge" and r["correct"]),
                     "normalised_scored": sum(1 for r in rows if r.get("scored_by") == "normalised" and r["correct"]),
                     "tiers": dict(Counter(r.get("tier") or "none" for r in rows))}
        for c in classes:
            cr = [r for r in rows if r["task_class"] == c]
            by_class[c][s] = [sum(1 for r in cr if r["correct"]), len(cr)]
    for q in sorted(common):
        wrong = {s: not results[s][q]["correct"] for s in systems}
        if any(wrong.values()):
            misses.append({"qid": q, "class": results[systems[0]][q]["task_class"], "gold": results[systems[0]][q]["gold"],
                           "answers": {s: {"miss": wrong[s], "answer": results[s][q].get("predicted"),
                                           "scored_by": results[s][q].get("scored_by")} for s in systems}})
    qids = sorted(common)
    pairwise = []
    for i, a in enumerate(systems):
        for b in systems[i + 1:]:
            va = [bool(results[a][q]["correct"]) for q in qids]
            vb = [bool(results[b][q]["correct"]) for q in qids]
            test = mcnemar(va, vb) if qids else {"n": 0, "a_only": 0, "b_only": 0, "diff": 0.0, "p_value": 1.0}
            lo, hi = bootstrap_diff(va, vb) if qids else (0.0, 0.0)
            pairwise.append({"a": a, "b": b, **test, "diff_ci95": [lo, hi]})
    audit = {}
    for s in systems:
        if s in ("stuffing", "closed-book"):
            continue
        sig = trajectory_signals(run_dir / f"{benchmark}-{s}" / "trajectories", common)
        numeric = [q for q in common if results[s][q]["task_class"] == "numeric"]
        facts = sig["facts"]
        audit[s] = {"verified_citation": sum(1 for q in common if facts.get(q, {}).get("verified_quotes") or facts.get(q, {}).get("check") in DERIVATION_CHECKS),
                    "replayable_numeric": sum(1 for q in numeric if facts.get(q, {}).get("check") in DERIVATION_CHECKS),
                    "numeric": len(numeric), "sub_calls": sum(int(results[s][q].get("sub_calls") or 0) for q in common),
                    **{k: v for k, v in sig.items() if k != "facts"}}
    # ---- markdown
    L = [f"# {benchmark}: cross-system comparison", "",
         f"Run dir `{run_dir}`. {n} questions answered by every system." + (" Scores are re-judged offline (majority of judge passes)." if rejudged else ""), "",
         "| System | Correct | Accuracy (95% CI) | Repeats min–max | Cap hits | Normalised / Judge-scored | Spend | Latency p50 | Status |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in systems:
        t = totals[s]
        rep = t["repeats"]
        rep_s = f"{rep['min']}–{rep['max']} ({rep['repeats']}×)" if rep["repeats"] > 1 else "–"
        caps = ", ".join(f"{k} {v}" for k, v in sorted(t["cap_hits"].items())) or "0"
        L.append(f"| {LABELS.get(s, s)} | {t['correct']} / {n} | {t['correct'] / n:.1%} ({t['ci95'][0]:.0%}–{t['ci95'][1]:.0%}) | {rep_s} | {caps} | "
                 f"{t['normalised_scored']} / {t['judge_scored']} | ${t['spend_usd']:.2f} | "
                 f"{t['latency_p50_s']} s | {t['status']} |" if n else f"| {LABELS.get(s, s)} | – |")
    L += ["", "Paired differences on identical questions (exact McNemar on discordant pairs; bootstrap 95% CI on the accuracy difference):", "",
          "| A | B | A right, B wrong | B right, A wrong | Diff (A−B) | 95% CI | p |", "|---|---|---|---|---|---|---|"]
    for pw in pairwise:
        L.append(f"| {LABELS.get(pw['a'], pw['a'])} | {LABELS.get(pw['b'], pw['b'])} | {pw['a_only']} | {pw['b_only']} | "
                 f"{pw['diff']:+.1%} | {pw['diff_ci95'][0]:+.0%}…{pw['diff_ci95'][1]:+.0%} | {pw['p_value']:.2f} |")
    L += ["", "| Task class | " + " | ".join(LABELS.get(s, s) for s in systems) + " |", "|---|" + "---|" * len(systems)]
    for c in classes:
        L.append(f"| `{c}` ({by_class[c][systems[0]][1]}) | " + " | ".join(f"{by_class[c][s][0]} / {by_class[c][s][1]}" for s in systems) + " |")
    L += ["", "## Auditability and how each arm worked", "",
          "| System | Verified citation | Replayable derivation (numeric) | Tiers | Root calls | Sub calls | Cells | SQL | calc | Embed batches | Reviews | Halts | Nudges |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in systems:
        if s not in audit:
            L.append(f"| {LABELS.get(s, s)} | 0 / {n} | 0 / {sum(1 for q in common if results[s][q]['task_class'] == 'numeric')} | – | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |")
            continue
        a = audit[s]
        tiers = totals[s]["tiers"]
        L.append(f"| {LABELS.get(s, s)} | {a['verified_citation']} / {n} | {a['replayable_numeric']} / {a['numeric']} | "
                 f"{tiers.get('high', 0)}/{tiers.get('medium', 0)}/{tiers.get('low', 0)} | {a['root_calls']} | {a['sub_calls']} | {a['cells']} | {a['cells_sql']} | "
                 f"{a['cells_calc']} | {a['embed_batches']} | {a['claim_reviews']} | {a['review_halts']} | {a['certificate_nudges']} |")
    L += ["", "## Misses (X = scored wrong)", "", "| qid | class | gold | " + " | ".join(systems) + " |", "|---|---|---|" + "---|" * len(systems)]
    for m in misses:
        L.append(f"| `{m['qid']}` | {m['class']} | {str(m['gold'])[:40]} | " + " | ".join("X" if m["answers"][s]["miss"] else "" for s in systems) + " |")
    L += ["", "### Missed answers", ""]
    for m in misses:
        L.append(f"- `{m['qid']}` gold `{str(m['gold'])[:80]}`")
        for s in systems:
            if m["answers"][s]["miss"]:
                L.append(f"  - {s}: {str(m['answers'][s]['answer'])[:200]!r}")
    (run_dir / f"{benchmark}-report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    summary = {"benchmark": benchmark, "systems": systems, "n": n, "totals": totals, "pairwise": pairwise,
               "by_class": {c: dict(by_class[c]) for c in classes}, "audit": audit, "misses": misses}
    (run_dir / f"{benchmark}-summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--systems", default="stuffing,rlm-classic,dci,dci-raw,docdb")
    ap.add_argument("--rejudged", action="store_true", help="read results.rejudged.jsonl (see rnsr.eval.rejudge)")
    args = ap.parse_args(argv)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    s = build(args.run_dir, args.benchmark, systems, rejudged=args.rejudged)
    for sys_ in systems:
        t = s["totals"][sys_]
        print(f"{LABELS.get(sys_, sys_):34} {t['correct']}/{t['n']}  ${t['spend_usd']:.2f}")
    print(f"wrote {args.run_dir / (args.benchmark + '-report.md')}")


if __name__ == "__main__":
    main()
