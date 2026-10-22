"""OOLONG diagnostics: strata and label-level accuracy.

    python -m rnsr.eval.oolong_report --run-dir runs/public-arms --systems stuffing,rlm-classic,docdb \
        --corpus-cache runs/public-arms/corpora-oolong --rejudged

Two things the headline accuracy hides. First, OOLONG-synth trec_coarse mixes
five task types (relative frequency is 58% of rows) and seven context lengths
from 1k to 65k tokens; a score on fifty mixed rows says nothing about which
of those an arm can do. This report stratifies every arm by ``task`` and by
``context_len``. Second, for DocDB the labels it assigned are saved in the
corpus (an annotation column on the lines table, certified by a
``classification_aggregate`` proof), and the dataset ships the gold label of
every line in ``context_window_text_with_labels``. Joining the two gives
label accuracy, a per-class confusion matrix, and a cross-tab of label
correctness against answer correctness, which is what separates a labelling
error from a counting error. The other arms keep no labels; their rows carry
only the answer, and the report says so rather than guessing from prose.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

_LINE = re.compile(r"^Date: .*? \|\| User: (?P<user>\d+) \|\| Instance: (?P<instance>.*?)(?: \|\| Label: (?P<label>.+))?$")


def parse_gold_lines(labelled_text: str) -> list[dict]:
    """``context_window_text_with_labels`` -> [{user, instance, label}] in order."""
    out = []
    for line in labelled_text.splitlines():
        m = _LINE.match(line.strip())
        if m and m.group("label"):
            out.append({"user": m.group("user"), "instance": m.group("instance").strip(),
                        "label": m.group("label").strip()})
    return out


def instance_of(text: str) -> str | None:
    m = _LINE.match(text.strip())
    return m.group("instance").strip() if m else None


def text_corpus_path(context: str, cache_dir: Path) -> Path:
    """Mirror of rnsr.eval.harness._text_corpus_for naming."""
    h = sha256(b"rnsr-text-corpus-v2\0" + context.encode()).hexdigest()[:16]
    return cache_dir / f"corpus_text_{h}.db"


def final_classification(trajectory: Path) -> dict | None:
    """The table/column/annotation_id of the last certified classification."""
    if not trajectory.exists():
        return None
    last = None
    with open(trajectory, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "final":
                ver = r.get("verification") or {}
                if ver.get("check") == "classification_aggregate" and isinstance(ver.get("classification"), dict):
                    last = ver["classification"]
    return last


def docdb_labels(corpus_db: Path, table: str, column: str, where: str | None) -> dict[str, str] | None:
    """{instance_text: predicted_label} from the saved annotation column, or
    None when the column is gone or the table has no text column."""
    if not corpus_db.exists():
        return None
    conn = sqlite3.connect(f"file:{corpus_db}?mode=ro", uri=True)
    try:
        cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')]
        if column not in cols or "text" not in cols:
            return None
        sql = f'SELECT text, "{column}" FROM "{table}"'
        if where:
            sql += f" WHERE {where}"
        out = {}
        for text, label in conn.execute(sql):
            inst = instance_of(text or "")
            if inst is not None and label is not None:
                out[inst] = str(label)
        return out
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def label_diagnostics(items_by_qid: dict, results: dict[str, dict], run_dir: Path, cache_dir: Path) -> dict:
    """Per-item label accuracy for DocDB rows plus pooled confusion and the
    labels-vs-answer cross-tab."""
    per_item, confusion = {}, defaultdict(Counter)
    cross = Counter()
    for qid, r in results.items():
        item = items_by_qid.get(qid)
        if item is None:
            continue
        gold_lines = parse_gold_lines(item["labelled"])
        traj = run_dir / "trajectories" / f"{qid}.jsonl"
        if r.get("trajectory_path"):
            traj = Path(r["trajectory_path"])
        cls = final_classification(traj)
        if not cls:
            per_item[qid] = {"labels": "none", "answer_correct": bool(r["correct"])}
            cross[("no_labels", bool(r["correct"]))] += 1
            continue
        pred = docdb_labels(text_corpus_path(item["context"], cache_dir), cls.get("table", ""), cls.get("column", ""), cls.get("where"))
        if pred is None:
            per_item[qid] = {"labels": "unavailable", "answer_correct": bool(r["correct"])}
            cross[("labels_unavailable", bool(r["correct"]))] += 1
            continue
        gold_vocab = {g["label"] for g in gold_lines}
        if not (set(cls.get("allowed_labels") or []) & gold_vocab):
            # e.g. "which user asked most questions": the controller classified
            # user ids, not answer types; nothing to compare with gold labels.
            per_item[qid] = {"labels": "not_a_label_classification", "answer_correct": bool(r["correct"])}
            cross[("not_label_classification", bool(r["correct"]))] += 1
            continue
        vocab = gold_vocab
        matched = [(g["label"], pred.get(g["instance"])) for g in gold_lines]
        covered = [(g, p if p in vocab else "<invalid>") for g, p in matched if p is not None]
        right = sum(1 for g, p in covered if g == p)
        for g, p in covered:
            confusion[g][p] += 1
        gold_counts = Counter(g for g, _ in covered)
        pred_counts = Counter(p for _, p in covered)
        counts_exact = bool(covered) and gold_counts == pred_counts
        order = lambda c: [k for k, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]  # noqa: E731
        order_kept = bool(covered) and order(gold_counts) == order({k: pred_counts.get(k, 0) for k in gold_counts})
        bucket = "counts_exact" if counts_exact else ("ranking_kept" if order_kept else "ranking_changed")
        per_item[qid] = {"labels": "read", "n_gold": len(gold_lines), "n_labelled": len(covered),
                         "label_accuracy": right / len(covered) if covered else None,
                         "counts_exact": counts_exact, "ranking_kept": order_kept,
                         "answer_correct": bool(r["correct"]),
                         "task": item["task"], "context_len": item["context_len"]}
        cross[(bucket, bool(r["correct"]))] += 1
    read = [v for v in per_item.values() if v.get("labels") == "read"]
    pooled_n = sum(sum(c.values()) for c in confusion.values())
    pooled_right = sum(confusion[g][g] for g in confusion)
    return {"per_item": per_item,
            "items_with_labels": len(read),
            "pooled_label_accuracy": pooled_right / pooled_n if pooled_n else None,
            "pooled_labelled_lines": pooled_n,
            "confusion": {g: dict(c) for g, c in sorted(confusion.items())},
            "cross_tab": {f"{k[0]}|answer_{'right' if k[1] else 'wrong'}": v for k, v in sorted(cross.items())},
            "label_accuracy_by_len": _by(read, "context_len"),
            "label_accuracy_by_task": _by(read, "task")}


def _by(read: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for v in read:
        if v.get("label_accuracy") is not None:
            groups[v[key]].append(v["label_accuracy"])
    return {str(k): round(sum(vs) / len(vs), 3) for k, vs in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def load_items(dataset_id: str) -> dict[str, dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="validation")
    out = {}
    for r in ds:
        if r["dataset"] != "trec_coarse":
            continue
        out[f"oolong-{r['id']}"] = {"task": str(r.get("task")).replace("TASK_TYPE.", ""),
                                    "context_len": int(r["context_len"]),
                                    "context": r["context_window_text"],
                                    "labelled": r["context_window_text_with_labels"]}
    return out


def _load(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {json.loads(line)["qid"]: json.loads(line) for line in path.read_text().splitlines() if line.strip()}


def build(run_dir: Path, systems: list[str], items: dict[str, dict], *, cache_dir: Path,
          rejudged: bool = False, benchmark: str = "oolong") -> dict:
    name = "results.rejudged.jsonl" if rejudged else "results.jsonl"
    results = {s: _load(run_dir / f"{benchmark}-{s}" / name) for s in systems}
    common = set.intersection(*(set(r) for r in results.values() if r)) if any(results.values()) else set()
    common = {q for q in common if q in items}
    strata = {"task": defaultdict(lambda: defaultdict(lambda: [0, 0])),
              "context_len": defaultdict(lambda: defaultdict(lambda: [0, 0]))}
    for q in common:
        for s in systems:
            ok = bool(results[s][q]["correct"])
            for key in ("task", "context_len"):
                cell = strata[key][items[q][key]][s]
                cell[0] += ok
                cell[1] += 1
    labels = {}
    for s in systems:
        if s == "docdb" or s.startswith("docdb"):
            labels[s] = label_diagnostics(items, {q: results[s][q] for q in common}, run_dir / f"{benchmark}-{s}", cache_dir)
    L = [f"# {benchmark}: strata and label diagnostics", "",
         f"Run dir `{run_dir}`. {len(common)} questions answered by every system." + (" Re-judged scores." if rejudged else ""), ""]
    for key, title in (("task", "By task type"), ("context_len", "By context length (tokens)")):
        L += [f"## {title}", "", "| Stratum | n | " + " | ".join(systems) + " |", "|---|---|" + "---|" * len(systems)]
        for stratum in sorted(strata[key], key=lambda k: (isinstance(k, str), k)):
            cells = strata[key][stratum]
            n = next(iter(cells.values()))[1] if cells else 0
            L.append(f"| `{stratum}` | {n} | " + " | ".join(f"{cells[s][0]} / {cells[s][1]}" for s in systems) + " |")
        L.append("")
    for s, d in labels.items():
        L += [f"## Label-level accuracy ({s})", "",
              f"Items whose saved labels could be read: {d['items_with_labels']} of {len(common)}. "
              f"Pooled label accuracy over {d['pooled_labelled_lines']} lines: "
              + (f"{d['pooled_label_accuracy']:.1%}." if d["pooled_label_accuracy"] is not None else "n/a."), "",
              "Labels vs answer. `counts_exact`: the per-class counts from the saved labels equal the gold counts, "
              "so any wrong answer is a counting/comparison defect. `ranking_kept`: counts differ but the order of "
              "classes is unchanged (relative-frequency and most/least questions survive; exact counts do not). "
              "`ranking_changed`: labelling errors changed the order.", "",
              "| Labels | Answer right | Answer wrong |", "|---|---|---|"]
        kinds = sorted({k.split("|")[0] for k in d["cross_tab"]})
        for k in kinds:
            L.append(f"| {k} | {d['cross_tab'].get(k + '|answer_right', 0)} | {d['cross_tab'].get(k + '|answer_wrong', 0)} |")
        L += ["", "Label accuracy by context length: " + json.dumps(d["label_accuracy_by_len"]),
              "Label accuracy by task: " + json.dumps(d["label_accuracy_by_task"]), "",
              "Confusion (rows gold, columns predicted):", ""]
        cats = sorted(set(d["confusion"]) | {p for c in d["confusion"].values() for p in c})
        L += ["| gold \\ pred | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
        for g in cats:
            L.append(f"| {g} | " + " | ".join(str(d["confusion"].get(g, {}).get(p, 0)) for p in cats) + " |")
        L.append("")
    others = [s for s in systems if s not in labels]
    if others:
        L += [f"The arms {', '.join(others)} keep no per-line labels; only their answers are scored.", ""]
    (run_dir / f"{benchmark}-strata.md").write_text("\n".join(L), encoding="utf-8")
    summary = {"n": len(common), "systems": systems,
               "strata": {k: {str(st): dict(v) for st, v in d.items()} for k, d in strata.items()},
               "labels": {s: {k: v for k, v in d.items() if k != "per_item"} for s, d in labels.items()},
               "labels_per_item": {s: d["per_item"] for s, d in labels.items()}}
    (run_dir / f"{benchmark}-strata.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--systems", default="stuffing,rlm-classic,docdb")
    ap.add_argument("--corpus-cache", type=Path, required=True)
    ap.add_argument("--dataset-id", default="oolongbench/oolong-synth")
    ap.add_argument("--benchmark", default="oolong")
    ap.add_argument("--rejudged", action="store_true")
    args = ap.parse_args(argv)
    items = load_items(args.dataset_id)
    s = build(args.run_dir, [x.strip() for x in args.systems.split(",") if x.strip()], items,
              cache_dir=args.corpus_cache, rejudged=args.rejudged, benchmark=args.benchmark)
    print(json.dumps({k: v for k, v in s.items() if k not in ("labels_per_item",)}, indent=1, default=str)[:4000])
    print(f"wrote {args.run_dir / (args.benchmark + '-strata.md')}")


if __name__ == "__main__":
    main()
