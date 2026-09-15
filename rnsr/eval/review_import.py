"""Import reviewer corrections into corpus-local golden and playbook diffs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rnsr.eval.audit import _CORRECT, _WRONG


def import_review(
    review_csv: str | Path,
    *,
    corpus_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    questions: dict[str, str] | None = None,
) -> dict:
    """Write golden items and a proposed playbook diff from a filled review.csv.

    Corrections (reviewer_mark in the miss set, or a ``corrected`` column)
    become ``golden/from_review.json``. Notes that look like conventions
    become ``playbook.diff.json``.
    """
    review_csv = Path(review_csv)
    dest = Path(out_dir or corpus_dir or review_csv.parent)
    dest.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    proposed_rules: list[str] = []
    n_ok = n_miss = n_unmarked = 0
    with open(review_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            qid = (row.get("qid") or row.get("query_id") or "").strip()
            mark = (row.get("reviewer_mark") or "").strip().lower()
            note = (row.get("note") or "").strip()
            corrected = (row.get("corrected") or row.get("gold") or "").strip()
            answer = (row.get("answer") or "").strip()
            if mark in _CORRECT:
                n_ok += 1
                continue
            if mark in _WRONG or corrected:
                n_miss += 1
                gold = corrected or note or answer
                items.append({
                    "qid": qid,
                    "question": (questions or {}).get(qid, ""),
                    "golden": [gold] if gold else [],
                    "source": "review",
                    "reviewer_mark": row.get("reviewer_mark") or "",
                    "note": note,
                })
                if note and len(note) > 20:
                    proposed_rules.append(note)
            else:
                n_unmarked += 1

    golden_dir = dest / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_path = golden_dir / "from_review.json"
    golden_path.write_text(json.dumps({
        "items": [{"qid": i["qid"], "golden": i["golden"]} for i in items],
        "fields": [{"id": i["qid"], "golden": i["golden"]} for i in items],
    }, indent=2))

    diff_path = dest / "playbook.diff.json"
    diff_path.write_text(json.dumps({
        "extra_rules": proposed_rules,
        "note": "Proposed playbook additions from reviewer notes. "
                "Merge into playbook.json after review.",
    }, indent=2))

    return {
        "n_ok": n_ok,
        "n_miss": n_miss,
        "n_unmarked": n_unmarked,
        "n_golden": len(items),
        "golden": str(golden_path),
        "playbook_diff": str(diff_path),
        "items": items,
    }
