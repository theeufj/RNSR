"""Build the Test Set 6 question CSV + golden from the verification key.

Reads verification_key.json (extracted from the vendor's xlsx) and emits:
  - ts6_questions.csv   (answer-csv input: ground_truth_question, qid)
  - ts6_golden.json     (qid -> question metadata + correct answer)

Only the question text and matter roles reach the model. Correct answers,
trap notes, and source-document hints stay in the golden file — they are
grading material, not prompt material.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rnsr.eval.regression import infer_expect

HERE = Path(__file__).parent
TS6 = HERE / "Test Set 6 - Initiating Application (999 Documents)"
KEY = TS6 / "verification_key.json"
OUT_CSV = TS6 / "ts6_questions.csv"
OUT_GOLDEN = TS6 / "ts6_golden.json"

# Names and roles only — the same context the vendor's own capture ships
# with every question. Codes, addresses, dates, and the court are answers
# to questions and must not appear here.
ROLES = """\
MATTER ROLES (authoritative - use these to keep the parties straight):
- The form being completed is: FCFCOA Initiating Application (paper form 0625V1).
- Matter: Whitfield & Nguyen (file ref PR/2026/0447).
- Applicant 1 is Amara Jane Whitfield (the client).
- Respondent 1 is Bao Tri Nguyen.
- The children are Harper Mai Nguyen, Elliot Van Nguyen and Thea Grace Nguyen.
- Acting for the applicant: lawyer Priya Raman of Raman & Cole Family Lawyers.
- The respondent's solicitors are Delaney Nash."""

EVIDENCE_RULE = """\
EVIDENCE RULE:
- Answer only from the matter documents in this corpus (999 files across \
seven folders: court/pre-action, correspondence, financial, children, \
property, file notes, miscellaneous).
- The corpus is LARGE. Do not try to read every document: use search() \
with distinctive terms (the field's label, a person's name, a document \
kind like 'letter of intention' or 'certificate') and SQL over tables, \
then read the specific documents that matter.
- Authority matters: court forms, solicitor letters, signed certificates \
and agreements outrank emails and file notes. Where versions or drafts \
conflict, the latest authoritative document governs.
- Internal chronologies, summaries and timelines are SECONDARY evidence \
compiled after the fact and may contain transcription errors. When a \
date or fact in a chronology conflicts with primary documents \
(contemporaneous letters, agreed-facts correspondence, certificates), \
the primary documents govern — especially where both sides' solicitors \
have stated or accepted the same fact in correspondence.
- When several candidate dates exist around the same event, use the date \
the authoritative documents identify as the operative one, not the date \
of an adjacent event.
- Blank form scaffolding is NOT evidence: unticked checkbox labels and \
printed option lists tell you nothing about this matter.
- Never guess from a person's name (gender, ethnicity, anything).
- Search before concluding either way: a negative claimed without a \
targeted search for this question's own subject is a wrong answer."""

ANSWER_FORMAT = """\
ANSWER FORMAT: reply with the exact value(s) the form field should \
contain, and nothing else. For checkbox or multi-select items, list the \
selected option(s), comma-separated. For items with numbered sub-answers \
(e.g. orders sought), give the numbered list. Identify people by their \
FULL NAMES (unless the field itself takes a role value). Where a \
question asks for the parties or persons in a document or case, list \
EVERY one it names, with their roles. Where the field names multiple \
sub-values (e.g. 'home and mobile'), provide each one the documents \
establish — a partial answer is wrong. If the form field is legitimately \
not applicable or must be left blank for this matter, reply exactly: Not \
applicable. If the documents do not establish the answer, reply exactly: \
unknown."""


# Structure of the printed form items whose key question text is too terse
# to answer ("Mark box as applicable" without the boxes). These are the
# public 0625V1 form's own option labels and detail requirements — form
# scaffolding, never matter facts or selections.
ITEM_SPECS = {
    "11": ("Options (mark ALL that apply to Applicant 1): Present in "
           "Australia; Ordinarily resident in Australia; An Australian "
           "citizen; Domiciled in Australia."),
    "21": ("Options (mark ALL that apply to Respondent 1): Present in "
           "Australia; Ordinarily resident in Australia; An Australian "
           "citizen; Domiciled in Australia."),
    "12": ("Options (mark ALL that apply to Applicant 1): Party to a "
           "marriage; Party to a de facto relationship that has broken "
           "down; Parent; Grandparent; Other."),
    "22": ("Options (mark ALL that apply to Respondent 1): Party to a "
           "marriage; Party to a de facto relationship that has broken "
           "down; Parent; Grandparent; Other."),
    "15–24": ("This is the whole Respondent 2 column of Part B. If the "
              "matter has no second respondent, the entire column is left "
              "blank — in that case reply exactly: Not applicable."),
    "50a": ("A bare Yes/No is NOT a valid answer to this item. Answer in "
            "exactly this shape: 'Yes — N: (1) <kind of order/agreement> "
            "<date>; (2) ...' listing EVERY existing order, agreement, "
            "parenting plan or undertaking the documents establish, or "
            "'No' if there are none."),
    "57": ("The form requires: the name of the lawyer who signed the "
           "declaration, the date it was signed, and the date the "
           "Marriage, Families and Separation brochure was given to the "
           "client."),
    "58": ("The form requires: who signed the Statement of Truth and on "
           "what date, for each applicant. If there is no Applicant 2, "
           "say so."),
    "49d": ("List EVERY party named in that case with their role — "
            "including any institutional applicant (e.g. police), any "
            "affected persons, and any protected CHILDREN by name — not "
            "just the two parties to this matter."),
    "8": ("The field asks for BOTH numbers: answer in the shape "
          "'Home <number>; Mobile <number>' with every number the "
          "documents establish."),
    "18": ("The field asks for BOTH numbers: answer in the shape "
           "'Home <number>; Mobile <number>' with every number the "
           "documents establish."),
}

# Item numbers repeat within a part (Part A's service block is all item
# 'A'), so field-specific structure is keyed by part + question text.
QUESTION_SPECS = {
    "Part A|Email": (
        "Part A's service block asks for the lawyer's ADDRESS FOR SERVICE "
        "contact details as given in the firm's notice of address for "
        "service — the firm's service email, not an individual lawyer's "
        "direct email."),
    "Part A|Filed on behalf of": (
        "This field takes a ROLE, not a person's name: the correct value "
        "is 'The applicant', 'The respondent', or similar."),
}


def main() -> None:
    key = json.loads(KEY.read_text())
    items = []
    for i, row in enumerate(key["paper_form"]):
        part = str(row.get("Part / step") or "").strip()
        item_no = str(row.get("Item no.") or "").strip()
        question = str(row.get("Question") or "").strip()
        golden = str(row.get("Correct answer") or "").strip()
        if not question:
            continue
        qid = f"pf{i:03d}"
        header = f"FORM QUESTION ({part}, item {item_no}): {question}"
        spec = QUESTION_SPECS.get(f"{part}|{question}") or ITEM_SPECS.get(item_no)
        if spec:
            header += f"\n\nFORM STRUCTURE: {spec}"
        text = "\n\n".join([ROLES, header, EVIDENCE_RULE, ANSWER_FORMAT])
        note = str(row.get("Note / trap") or "").strip()
        items.append({
            "qid": qid, "part": part, "item": item_no,
            "question": question, "golden": golden,
            "expect": infer_expect(golden, note),
            "note": note,
            "n_sources": row.get("No. of source docs"),
            "prompt": text,
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ground_truth_question", "qid"])
        for it in items:
            w.writerow([it["prompt"], it["qid"]])
    OUT_GOLDEN.write_text(json.dumps(
        {"items": [{k: v for k, v in it.items() if k != "prompt"}
                   for it in items]},
        indent=1, ensure_ascii=False))
    print(f"wrote {OUT_CSV.name}: {len(items)} questions")
    print(f"wrote {OUT_GOLDEN.name}")


if __name__ == "__main__":
    main()
