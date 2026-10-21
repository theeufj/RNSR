"""Arm-neutral inputs for the Test Set 6 (999-document) four-arm comparison.

Writes to runs/ts6-arms/:
  questions.csv  ts6_questions.csv with the one tool-specific evidence bullet
                 ("use search() ... and SQL over tables") replaced by a neutral
                 one, so whole-matter, classic, DCI and DocDB arms see the same
                 task text. Everything else (roles, evidence rules, answer
                 format) is unchanged.
  golden.json    ts6_golden.json items plus one generic field_context for the
                 equivalence judge (no per-item hints).
  map.json       standalone items keyed by qid, so `rnsr regress --map` can
                 join positional answers to the golden.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
TS6 = HERE / "Test Set 6 - Initiating Application (999 Documents)"
OUT = Path("runs/ts6-arms")

TOOL_BULLET = ("- The corpus is LARGE. Do not try to read every document: use search() with "
               "distinctive terms (the field's label, a person's name, a document kind like "
               "'letter of intention' or 'certificate') and SQL over tables, then read the "
               "specific documents that matter.")
NEUTRAL_BULLET = ("- The corpus is LARGE. Do not try to read every document: locate the specific "
                  "documents that matter (by the field's label, a person's name, or a document "
                  "kind like 'letter of intention' or 'certificate') and read those.")

FIELD_CONTEXT = (
    "Form-field answer for an FCFCOA Initiating Application over a 999-document family-law "
    "matter. The reference is the value the form field should contain. The candidate may use "
    "a different checkbox notation (ticks versus listing the selected options), a different "
    "order, or add brief correct supporting detail. Treat the candidate as equivalent when "
    "every value or selected option in the reference is present and no contradictory value "
    "is added; a partial list of required values is not equivalent. 'Not applicable' or "
    "'unknown' is correct only when the reference itself is blank, not applicable or unknown.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(TS6 / "ts6_questions.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    replaced = 0
    for r in rows:
        if TOOL_BULLET in r["ground_truth_question"]:
            r["ground_truth_question"] = r["ground_truth_question"].replace(TOOL_BULLET, NEUTRAL_BULLET)
            replaced += 1
    with open(OUT / "questions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ground_truth_question", "qid"])
        w.writeheader()
        w.writerows(rows)
    golden = json.loads((TS6 / "ts6_golden.json").read_text())
    for item in golden["items"]:
        item["field_context"] = FIELD_CONTEXT
    (OUT / "golden.json").write_text(json.dumps(golden, indent=1, ensure_ascii=False))
    items = [{"kind": "standalone", "field_id": r["qid"], "question": r["ground_truth_question"],
              "needs_value": False} for r in rows]
    (OUT / "map.json").write_text(json.dumps({"items": items}, indent=1, ensure_ascii=False))
    print(f"{len(rows)} questions ({replaced} tool bullets neutralised), "
          f"{len(golden['items'])} golden items -> {OUT}")


if __name__ == "__main__":
    main()
