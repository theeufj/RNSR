"""Cross-arm comparison for the matter benchmark: accuracy, spend and wall per
arm, per matter and per question type, from each arm's ``score/comparison.csv``
and ``run_report.json``. Writes Markdown to stdout and a JSON summary.

    python -m rnsr.eval.matter_arms_report \
        --docdb runs/matters-bench --arms runs/matters-bench-arms \
        --matters-src testMatter/matters --out runs/matters-bench-arms/summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ARM_ORDER = ("stuffing", "classic", "dci", "docdb")
LABELS = {"stuffing": "Whole matter in context", "classic": "Classic (flat string in REPL)",
          "dci": "Direct corpus interaction", "docdb": "DocDB (October run)"}


def question_types(matters_src: Path, matter: str) -> dict[str, str]:
    key = json.loads((matters_src / matter / "answer_key.json").read_text())
    items = key if isinstance(key, list) else key.get("items") or key.get("questions") or []
    return {i["id"]: i.get("question_type", "unknown") for i in items}


def verdicts(score_dir: Path) -> dict[str, bool]:
    with open(score_dir / "comparison.csv", newline="", encoding="utf-8") as fh:
        return {r["field_id"]: r["verdict"] == "OK" for r in csv.DictReader(fh)}


def report_numbers(path: Path) -> dict:
    if not path.exists():
        return {}
    r = json.loads(path.read_text())
    prov = r.get("provider") or {}
    return {"spend": prov.get("spend_usd", 0.0), "requests": prov.get("requests", 0),
            "wall_s": r.get("wall_s", 0.0), "status": r.get("status_counts", {})}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docdb", type=Path, required=True, help="runs/matters-bench")
    ap.add_argument("--arms", type=Path, required=True, help="runs/matters-bench-arms")
    ap.add_argument("--matters-src", type=Path, required=True, help="testMatter/matters")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    matters = sorted(p.name for p in args.arms.iterdir() if p.is_dir())
    per_type: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    per_matter: dict[str, dict[str, dict]] = defaultdict(dict)
    totals: dict[str, list] = defaultdict(lambda: [0, 0, 0.0, 0.0])
    wrong: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for m in matters:
        qtypes = question_types(args.matters_src, m)
        for arm in ARM_ORDER:
            score = (args.docdb / m / "score") if arm == "docdb" else (args.arms / m / arm / "score")
            if not (score / "comparison.csv").exists():
                continue
            v = verdicts(score)
            nums = (report_numbers(args.docdb / m / "out" / "run_report.json") if arm == "docdb"
                    else report_numbers(args.arms / m / arm / "run_report.json"))
            correct = sum(v.values())
            per_matter[m][arm] = {"correct": correct, "total": len(v), **nums}
            t = totals[arm]
            t[0] += correct
            t[1] += len(v)
            t[2] += nums.get("spend", 0.0)
            t[3] = max(t[3], nums.get("wall_s", 0.0))
            for fid, ok in v.items():
                qt = qtypes.get(fid, "unknown")
                per_type[qt][arm][1] += 1
                per_type[qt][arm][0] += int(ok)
                if not ok:
                    wrong[arm][m].append(fid)

    arms = [a for a in ARM_ORDER if totals[a][1]]
    lines = ["| Arm | Correct | Accuracy | Spend | Wall (max matter) |", "|---|---|---|---|---|"]
    for a in arms:
        c, n, s, w = totals[a]
        lines.append(f"| {LABELS[a]} | {c} / {n} | {c / n:.1%} | ${s:.2f} | {w / 60:.1f} min |")
    lines += ["", "| Matter | " + " | ".join(LABELS[a] for a in arms) + " |",
              "|---|" + "---|" * len(arms)]
    for m in matters:
        cells = []
        for a in arms:
            r = per_matter[m].get(a)
            cells.append(f"{r['correct']} / {r['total']} (${r.get('spend', 0):.2f})" if r else "–")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "| Question type | " + " | ".join(LABELS[a] for a in arms) + " |",
              "|---|" + "---|" * len(arms)]
    for qt in sorted(per_type, key=lambda k: -per_type[k][arms[0]][1]):
        cells = [f"{per_type[qt][a][0]} / {per_type[qt][a][1]}" for a in arms]
        lines.append(f"| `{qt}` | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    print(md)
    args.out.write_text(json.dumps({
        "totals": {a: {"correct": totals[a][0], "total": totals[a][1],
                       "spend_usd": round(totals[a][2], 4)} for a in arms},
        "per_matter": per_matter,
        "per_type": {qt: {a: per_type[qt][a] for a in arms} for qt in per_type},
        "misses": wrong, "markdown": md}, indent=2))


if __name__ == "__main__":
    main()
