"""One-command four-arm comparison on a matter directory.

    python -m rnsr.eval.arms_bench --matter testMatter/matters/<name> --out runs/<name>-arms

Given a matter directory laid out as the generators and ``testMatter/matters``
produce (``documents/``, ``questions.csv``, ``golden.json``, ``map.json``,
``answer_key.json``), this

  1. ingests ``documents/`` into ``<out>/work/corpora/corpus.db`` if absent
     (Docling tier by default so tables exist; ``--fast-ingest`` for text only);
  2. runs each arm in ``--arms`` with identical caps, skipping arms already run
     unless ``--force``:
       docdb     ``rnsr answer-csv`` over the artifact (tables, ladder, review)
       stuffing  whole corpus in one prompt per question
       classic   flat string in the REPL
       dci       one file per document with ls/grep/read in the REPL
       dci_raw   as dci, but files are fast text extraction of the original
                 documents (no layout model, no tables, no OCR): isolates the
                 value of ingest from the value of typed tables
  3. scores every arm with ``rnsr regress`` against ``golden.json`` and
     ``map.json`` (``--judge-passes N`` repeats the judge to expose its
     variance; the first pass is the reported one);
  4. mines each arm's trajectories for how the answer was reached (root calls,
     cells, SQL/search/annotation/calculation cells, embedding batches, review
     halts, tiers, verified quotes);
  5. writes ``<out>/summary.json`` and ``<out>/report.md``: per-arm totals,
     per-question-type accuracy, a miss matrix with every arm's answer, and
     the trajectory and evidence statistics.

Everything is a subprocess of the installed CLI so a run is reproducible from
the command line that the report records. Caps come from the environment
(``RNSR_MAX_ROOT_ITERS`` etc.) and are recorded in the summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ARM_LABELS = {"stuffing": "Whole matter in context", "classic": "Classic (flat string in REPL)",
              "dci": "Direct corpus interaction", "dci_raw": "DCI over raw files (no ingest)", "docdb": "DocDB"}
ARM_ORDER = ("stuffing", "classic", "dci", "dci_raw", "docdb")
CAP_VARS = ("RNSR_MAX_ROOT_ITERS", "RNSR_MAX_WALL_S", "RNSR_MAX_SPEND_USD", "RNSR_RUN_SPEND_CEILING_USD",
            "RNSR_ROOT_MODEL", "RNSR_SUB_MODEL", "RNSR_PROVIDER")


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sh(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n")
        fh.flush()
        return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)


def rnsr_bin() -> str:
    return shutil.which("rnsr") or str(Path(sys.executable).with_name("rnsr"))


# ----------------------------------------------------------------------------- steps

def ingest(matter: Path, out: Path, fast: bool) -> Path:
    db = out / "work" / "corpora" / "corpus.db"
    if db.exists():
        return db
    db.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in (matter / "documents").iterdir() if p.is_file() and not p.name.startswith("."))
    cmd = [rnsr_bin(), "ingest", *files, "-o", str(db), "--report", str(out / "ingest.json")]
    if fast:
        cmd.append("--fast-ingest")
    started = time.monotonic()
    rc = sh(cmd, out / "ingest.log")
    if rc != 0 or not db.exists():
        raise SystemExit(f"ingest failed (exit {rc}); see {out / 'ingest.log'}")
    (out / "ingest.timing.json").write_text(json.dumps({"wall_s": round(time.monotonic() - started, 1)}))
    return db


def run_arm(arm: str, matter: Path, out: Path, batch_size: int, concurrency: int, stuffing_concurrency: int) -> None:
    arm_dir = out / arm
    if arm == "docdb":
        cmd = [rnsr_bin(), "answer-csv", "--corpus", str(matter / "documents"), "--questions", str(matter / "questions.csv"),
               "--question-col", "ground_truth_question", "--output", str(arm_dir / "out"), "--work-dir", str(out / "work"),
               "--batch-size", str(batch_size), "--concurrency", str(concurrency), "--max-error-rate", "1"]
    else:
        cmd = [sys.executable, "-u", "-m", "rnsr.eval.matter_arms", "--matter-dir", str(out),
               "--arm", "dci" if arm == "dci_raw" else arm,
               "--out", str(arm_dir), "--questions", str(matter / "questions.csv"), "--id-col", "field_id",
               "--batch-size", str(batch_size),
               "--concurrency", str(stuffing_concurrency if arm == "stuffing" else concurrency)]
        if arm == "dci_raw":
            cmd += ["--source", "raw", "--documents", str(matter / "documents")]
    rc = sh(cmd, out / f"{arm}.log")
    if rc != 0:
        raise SystemExit(f"arm {arm} failed (exit {rc}); see {out / f'{arm}.log'}")


def _answers_dir(arm: str, out: Path) -> Path:
    return out / arm / ("out" if arm == "docdb" else "")


def score_arm(arm: str, matter: Path, out: Path, judge_passes: int) -> None:
    src = _answers_dir(arm, out)
    score = out / arm / "score"
    score.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "answers_chunk1.csv", score / "answers_chunk1.csv")
    # answer-csv keys status rows q000..; re-key to field ids so tiers join.
    qids = [r["field_id"] for r in read_csv(matter / "questions.csv")]
    rows = read_csv(src / "answers_status.csv")
    for r in rows:
        if r.get("query_id", "").startswith("q") and r["query_id"][1:].isdigit():
            r["query_id"] = qids[int(r["row"])]
    with open(score / "answers_status.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    for i in range(judge_passes):
        dest = score if i == 0 else out / arm / f"score_pass{i + 1}"
        dest.mkdir(exist_ok=True)
        if i:
            shutil.copy(score / "answers_chunk1.csv", dest / "answers_chunk1.csv")
            shutil.copy(score / "answers_status.csv", dest / "answers_status.csv")
        sh([rnsr_bin(), "regress", "--answers", str(dest / "answers_chunk1.csv"), "--golden", str(matter / "golden.json"),
            "--map", str(matter / "map.json"), "--out", str(dest)], dest / "regress.log")


def trajectory_stats(arm: str, out: Path) -> dict:
    tdir = (out / "work" / "trajectories") if arm == "docdb" else (out / arm / "trajectories")
    s = Counter()
    cost_root = cost_sub = 0.0
    if not tdir.exists():
        return {}
    for f in tdir.glob("*.jsonl*"):
        if f.suffix == ".enc":
            continue
        with open(f, encoding="utf-8") as fh:
            lines = fh.readlines()
        for line in lines:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = r.get("kind")
            s[k] += 1
            if k == "root_call_completed":
                cost_root += r.get("cost_usd", 0) or 0
            if k == "sub_batch":
                s["sub_prompts"] += r.get("n", 0) or 0
                cost_sub += r.get("cost_usd", 0) or 0
            if k == "search_rung":
                s[f"rung_{r.get('rung')}"] += 1
            if "code" in r:
                c = r.get("code") or ""
                s["cells"] += 1
                s["cells_sql"] += bool(re.search(r"db\.execute|\bSELECT\b", c, re.I))
                s["cells_search"] += bool(re.search(r"(?<![\w.])search\(", c))        # the ladder, not re.search
                s["cells_annotate"] += bool(re.search(r"semantic_(annotate|classify)\(", c))
                s["cells_calc"] += bool(re.search(r"\b(source_number|source_text_number|calculate|calculate_financial|FINAL_CALCS?)\(", c))
                s["cells_llm"] += bool(re.search(r"\bllm_(query|map)\(", c))
    return {"root_calls": s["root_call_completed"], "root_cost_usd": round(cost_root, 4),
            "sub_prompts": s["sub_prompts"], "sub_cost_usd": round(cost_sub, 4),
            "embed_batches": s["embed_batch"], "cells": s["cells"], "cells_sql": s["cells_sql"],
            "cells_search": s["cells_search"], "cells_annotate": s["cells_annotate"], "cells_calc": s["cells_calc"],
            "cells_llm": s["cells_llm"], "claim_reviews": s["claim_review"], "review_halts": s["claim_review_halted"],
            "pushbacks": s["final_observation_review"], "rungs": {k[5:]: v for k, v in s.items() if k.startswith("rung_")}}


def evidence_stats(arm: str, out: Path) -> dict:
    path = out / arm / "score" / "answers_status.csv"
    if not path.exists():      # hand-run arms before the orchestrator existed
        path = _answers_dir(arm, out) / "answers_status.csv"
    rows = read_csv(path)
    tiers = Counter(r.get("tier") or "none" for r in rows)
    quoted = sum(1 for r in rows if (r.get("quotes_verified") or "0/0").split("/")[0] not in ("", "0"))
    return {"tiers": dict(tiers), "answers_with_verified_quotes": quoted, "n": len(rows)}


def run_numbers(arm: str, out: Path) -> dict:
    p = _answers_dir(arm, out) / "run_report.json"
    if not p.exists():
        return {}
    r = json.loads(p.read_text())
    prov = r.get("provider") or {}
    return {"spend_usd": round(prov.get("spend_usd", 0.0), 4), "requests": prov.get("requests", 0),
            "wall_s": r.get("wall_s", 0.0), "status_counts": r.get("status_counts", {})}


def verdicts(path: Path) -> dict[str, dict]:
    return {r["field_id"]: r for r in read_csv(path)}


# ----------------------------------------------------------------------------- report

def build_report(matter: Path, out: Path, arms: list[str], judge_passes: int, caps: dict) -> dict:
    key = {q["id"]: q for q in json.loads((matter / "answer_key.json").read_text())}
    V = {a: verdicts(out / a / "score" / "comparison.csv") for a in arms}
    passes = {a: [sum(1 for r in verdicts(out / a / ("score" if i == 0 else f"score_pass{i + 1}") / "comparison.csv").values()
                      if r["verdict"] == "OK") for i in range(judge_passes)] for a in arms}
    n = len(key)
    totals = {a: {"correct": passes[a][0], "total": n, "judge_passes": passes[a],
                  "judge_range": [min(passes[a]), max(passes[a])], **run_numbers(a, out),
                  "trajectory": trajectory_stats(a, out), "evidence": evidence_stats(a, out)} for a in arms}
    by_type = defaultdict(lambda: {a: [0, 0] for a in arms})
    for qid, q in key.items():
        for a in arms:
            by_type[q["question_type"]][a][1] += 1
            by_type[q["question_type"]][a][0] += V[a][qid]["verdict"] == "OK"
    misses = []
    for qid, q in key.items():
        m = {a: V[a][qid]["verdict"] != "OK" for a in arms}
        if any(m.values()):
            misses.append({"id": qid, "type": q["question_type"], "gold": q["answer"],
                           "arms": {a: {"miss": m[a], "answer": V[a][qid]["answer"], "judge": V[a][qid].get("judge_status", "")}
                                    for a in arms}})
    # ---- markdown
    L = [f"# Four-arm comparison — {matter.name}", "",
         f"Corpus: `{matter}`. Output: `{out}`. Caps: " + ", ".join(f"{k}={v}" for k, v in caps.items() if v) + ".", ""]
    try:
        ing = json.loads((out / "ingest.json").read_text())
        docs_n = ing.get("documents") if isinstance(ing.get("documents"), int) else len(ing.get("documents") or [])
        tables_n = ing.get("tables") if isinstance(ing.get("tables"), int) else len(ing.get("tables") or [])
        L += [f"Ingest: {docs_n} docs, {ing.get('n_chunks', '?')} chunks, {tables_n} tables, "
              f"validation {ing.get('validation_pass_rate', 0):.0%}, scans transcribed {ing.get('scanned_pages_transcribed', 0)}.", ""]
    except Exception:
        pass
    L += ["| Arm | Correct | Accuracy | Judge range | Spend | Requests | Wall |", "|---|---|---|---|---|---|---|"]
    for a in arms:
        t = totals[a]
        rng = f"{t['judge_range'][0]}–{t['judge_range'][1]}" if judge_passes > 1 else "–"
        L.append(f"| {ARM_LABELS[a]} | {t['correct']} / {n} | {t['correct'] / n:.1%} | {rng} | ${t.get('spend_usd', 0):.2f} | "
                 f"{t.get('requests', 0)} | {t.get('wall_s', 0) / 60:.1f} min |")
    L += ["", "| Question type | " + " | ".join(ARM_LABELS[a] for a in arms) + " |", "|---|" + "---|" * len(arms)]
    for qt, d in sorted(by_type.items(), key=lambda kv: -kv[1][arms[0]][1]):
        L.append(f"| `{qt}` ({d[arms[0]][1]}) | " + " | ".join(str(d[a][0]) for a in arms) + " |")
    L += ["", "## How each arm worked", "", "| Arm | Root calls | Cells | SQL | search() | annotate | calc | llm_query | Embed batches | Reviews | Halts | Tiers (H/M/L) | Verified quotes |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        tj, ev = totals[a]["trajectory"], totals[a]["evidence"]
        tiers = ev["tiers"]
        L.append(f"| {ARM_LABELS[a]} | {tj.get('root_calls', 0)} | {tj.get('cells', 0)} | {tj.get('cells_sql', 0)} | {tj.get('cells_search', 0)} | "
                 f"{tj.get('cells_annotate', 0)} | {tj.get('cells_calc', 0)} | {tj.get('cells_llm', 0)} | {tj.get('embed_batches', 0)} | "
                 f"{tj.get('claim_reviews', 0)} | {tj.get('review_halts', 0)} | {tiers.get('high', 0)}/{tiers.get('medium', 0)}/{tiers.get('low', 0)} | "
                 f"{ev['answers_with_verified_quotes']}/{ev['n']} |")
    L += ["", "## Misses (X = scored wrong)", "", "| id | type | gold | " + " | ".join(arms) + " |", "|---|---|---|" + "---|" * len(arms)]
    for m in misses:
        L.append(f"| `{m['id']}` | {m['type']} | {m['gold'][:40]} | " + " | ".join("X" if m["arms"][a]["miss"] else "" for a in arms) + " |")
    L += ["", "### Missed answers", ""]
    for m in misses:
        L.append(f"- `{m['id']}` gold `{m['gold'][:60]}`")
        for a in arms:
            if m["arms"][a]["miss"]:
                L.append(f"  - {a} [{m['arms'][a]['judge']}]: {m['arms'][a]['answer'][:160]!r}")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    summary = {"matter": str(matter), "arms": arms, "caps": caps, "totals": totals,
               "by_type": {qt: {a: d[a] for a in arms} for qt, d in by_type.items()}, "misses": misses}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


# ----------------------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default="stuffing,classic,dci,docdb")
    ap.add_argument("--fast-ingest", action="store_true", help="text-only ingest (no tables)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--stuffing-concurrency", type=int, default=1)
    ap.add_argument("--judge-passes", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="re-run arms that already have scores")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    arms = [a for a in ARM_ORDER if a in {x.strip() for x in args.arms.split(",")}]
    caps = {k: os.environ.get(k, "") for k in CAP_VARS}
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        db = ingest(args.matter, args.out, args.fast_ingest)
        print(f"corpus: {db}")
        for arm in arms:
            if (args.out / arm / "score" / "comparison.csv").exists() and not args.force:
                print(f"{arm}: already scored, skipping (--force to rerun)")
                continue
            started = time.monotonic()
            print(f"{arm}: running …", flush=True)
            run_arm(arm, args.matter, args.out, args.batch_size, args.concurrency, args.stuffing_concurrency)
            score_arm(arm, args.matter, args.out, args.judge_passes)
            print(f"{arm}: done in {time.monotonic() - started:.0f}s")
    summary = build_report(args.matter, args.out, arms, args.judge_passes, caps)
    for a in arms:
        t = summary["totals"][a]
        print(f"{ARM_LABELS[a]:32} {t['correct']}/{t['total']}  ${t.get('spend_usd', 0):.2f}  {t.get('wall_s', 0) / 60:.1f} min")
    print(f"wrote {args.out / 'report.md'}")
    _ = statistics  # reserved for future variance reporting


if __name__ == "__main__":
    main()
