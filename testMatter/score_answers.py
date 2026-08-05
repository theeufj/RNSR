"""Fan grouped answers out to form fields, then score against the golden set.

The enriched run answers 28 questions; the golden set is 49 fields. Each
group answer names one winning option, which fans out mechanically: the
winner takes the value its own field format asks for, every sibling takes
"No". That is where the consistency guarantee comes from - contradictory
siblings are impossible by construction, rather than something the model
has to remember across independent queries.

Usage:
    python testMatter/score_answers.py --answers runs/<run>/answers_chunk1.csv
        [--baseline runs/matter-eval/comparison.csv] [--out runs/<run>]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
NOT_FOUND = "Not found in matter corpus"

_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+)", re.I)
_VALUE_RE = re.compile(r"VALUE\s*:\s*(.*)", re.I)
_OPT_SUFFIX_RE = re.compile(r"[-:]\s*(yes|no)$")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower()).rstrip(".")


def is_negative(ans: str) -> bool:
    a = norm(ans)
    return a in ("", "no", "unknown", "n/a") or a.startswith(norm(NOT_FOUND))


def agree(gold: list[str], ans: str) -> bool:
    """Field-level agreement with the vendor's golden value.

    The vendor is not internally consistent about how a chosen radio option
    is recorded - sometimes "yes", sometimes the full option text - so a
    yes/no answer and the corresponding option text are treated as equal.
    """
    if not gold:                       # vendor left the field unanswered
        return is_negative(ans)
    g, a = norm(gold[0]), norm(ans)
    if g == a:
        return True
    if g == "no" and is_negative(a):
        return True
    mg, ma = _OPT_SUFFIX_RE.search(g), _OPT_SUFFIX_RE.search(a)
    if mg and a == mg.group(1):
        return True
    if ma and g == ma.group(1):
        return True
    shorter, longer = sorted((g, a), key=len)
    return len(shorter) >= 4 and shorter in longer


def _candidate(text: str) -> str | None:
    """The answer token, whether or not the run used the ANSWER: prefix.

    The harness returns the final answer on its own, so a compliant run
    tends to emit the bare option id rather than the requested label.
    """
    m = _ANSWER_RE.search(text or "")
    raw = m.group(1) if m else (text or "")
    lines = raw.strip().splitlines()
    if not lines:
        return None
    # Tolerate markdown wrappers and trailing commentary on the line.
    tokens = lines[0].strip().strip("`*\"' \t").split()
    return tokens[0].strip("`*\"'.,;") if tokens else None


def parse_value_group_answer(
    text: str, members: list[dict]
) -> tuple[str | None, str | None, str]:
    """(winner_id, value, note) for a group whose answer is a value.

    'Date of marriage' and friends pair a date field with a "not
    applicable" option, so the question asks for the date itself, the
    keyword not_applicable, or unknown.
    """
    value_member = next(m for m in members if m["needs_value"])
    alts = [m for m in members if not m["needs_value"]]
    raw = (text or "").strip()
    vm = _VALUE_RE.search(raw)
    if vm and vm.group(1).strip():
        raw = vm.group(1).strip()
    first = _candidate(raw) or ""
    n = norm(first)
    if n.replace(" ", "_").startswith("not_applicable"):
        return (alts[0]["id"] if alts else None), None, ""
    if n in ("", "unknown") or is_negative(first):
        return None, None, ""
    if first in [m["id"] for m in members]:
        return first, None, "named an option id but supplied no value"
    # Keep the whole first line: dates are single tokens, values may not be.
    value = raw.splitlines()[0].strip().strip("`*\"' ")
    value = re.sub(r"^(?:answer|value)\s*:\s*", "", value, flags=re.I).strip()
    return value_member["id"], value, ""


def parse_group_answer(text: str, member_ids: list[str]) -> tuple[str | None, str | None, str]:
    """(winner_id | None for unknown, value, note) from a group answer."""
    note = ""
    choice = _candidate(text)
    if choice not in member_ids:
        if choice and norm(choice) == "unknown":
            choice = None
        else:
            # Lenient recovery: any member id mentioned anywhere wins.
            mentioned = [mid for mid in member_ids if mid in (text or "")]
            if len(mentioned) == 1:
                note = f"recovered choice from prose (ANSWER line said {choice!r})"
                choice = mentioned[0]
            elif "unknown" in norm(text):
                note = f"read as unknown (ANSWER line said {choice!r})"
                choice = None
            else:
                note = f"unparseable answer, treated as unknown (got {choice!r})"
                choice = None

    vm = _VALUE_RE.search(text or "")
    value = vm.group(1).strip().splitlines()[0].strip() if vm else None
    if value is not None and norm(value) in ("n/a", "na", "none", "unknown", ""):
        value = None
    return choice, value, note


def field_value(member: dict, *, won: bool, value: str | None) -> str:
    """The value this field takes given the group's single choice."""
    if not won:
        return NOT_FOUND if member["needs_value"] else "No"
    if member["needs_value"]:
        return value or NOT_FOUND
    if member["wants_yes"]:
        return "yes"
    return member["option_label"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, type=Path)
    ap.add_argument("--map", type=Path, default=HERE / "questions_enriched_map.json")
    ap.add_argument("--golden", type=Path, default=HERE / "golden.json")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="earlier comparison.csv, for a per-field delta")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out or args.answers.parent

    spec = json.loads(args.map.read_text())
    items = spec["items"]
    golden = {f["id"]: f for f in json.loads(args.golden.read_text())["fields"]}

    with open(args.answers, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != len(items):
        raise SystemExit(f"answers ({len(rows)}) != questions ({len(items)})")

    answers: dict[str, str] = {}
    grouped_fields: set[str] = set()
    notes: list[str] = []
    for item, row in zip(items, rows, strict=True):
        text = row["model_answer"]
        if item["kind"] == "standalone":
            fid = item["field_id"]
            answers[fid] = NOT_FOUND if is_negative(text) and item["needs_value"] else text.strip()
            continue
        if item.get("mode") == "value":
            winner, value, note = parse_value_group_answer(text, item["members"])
        else:
            winner, value, note = parse_group_answer(
                text, [m["id"] for m in item["members"]])
        if note:
            notes.append(f"{item['group']}: {note}")
        for m in item["members"]:
            grouped_fields.add(m["id"])
            answers[m["id"]] = field_value(m, won=(m["id"] == winner), value=value)

    baseline: dict[str, str] = {}
    if args.baseline and args.baseline.exists():
        with open(args.baseline, newline="", encoding="utf-8") as fh:
            baseline = {r["field_id"]: r["verdict"] for r in csv.DictReader(fh)}

    out_rows, n_ok = [], 0
    for fid, f in golden.items():
        ans = answers.get(fid, "")
        ok = agree(f["golden"], ans)
        n_ok += ok
        verdict = "OK" if ok else "DIFF"
        before = baseline.get(fid, "")
        delta = ""
        if before:
            delta = {("DIFF", "OK"): "FIXED", ("OK", "DIFF"): "REGRESSED"}.get(
                (before, verdict), "")
        out_rows.append({
            "field_id": fid,
            "class": "group" if fid in grouped_fields else "standalone",
            # A field whose golden value is blank or a bare "No" is satisfied by
            # answering nothing, so raw agreement over all 49 flatters any
            # cautious run; substantive fields are where the signal is.
            "golden_kind": "negative" if (not f["golden"]
                                          or norm(f["golden"][0]) == "no")
                           else "substantive",
            "golden": "; ".join(f["golden"]) or "(blank)",
            "rnsr_answer": ans,
            "verdict": verdict,
            "baseline_verdict": before,
            "delta": delta,
        })

    (out_dir).mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    with open(out_dir / "field_answers.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["field_id", "model_answer"])
        w.writerows([[r["field_id"], r["rnsr_answer"]] for r in out_rows])

    total = len(out_rows)
    print(f"agreement: {n_ok}/{total}")
    for cls in ("group", "standalone"):
        sub = [r for r in out_rows if r["class"] == cls]
        print(f"  {cls:11s} {sum(r['verdict'] == 'OK' for r in sub)}/{len(sub)}")
    for kind in ("substantive", "negative"):
        sub = [r for r in out_rows if r["golden_kind"] == kind]
        print(f"  {kind:11s} {sum(r['verdict'] == 'OK' for r in sub)}/{len(sub)}"
              + ("   <- real signal: golden holds a value" if kind == "substantive"
                 else "   (satisfied by answering No/nothing)"))
    if baseline:
        fixed = [r["field_id"] for r in out_rows if r["delta"] == "FIXED"]
        regressed = [r["field_id"] for r in out_rows if r["delta"] == "REGRESSED"]
        base_ok = sum(v == "OK" for v in baseline.values())
        print(f"\nbaseline: {base_ok}/{len(baseline)} -> now {n_ok}/{total}")
        print(f"  fixed ({len(fixed)}): {', '.join(fixed) or '-'}")
        print(f"  regressed ({len(regressed)}): {', '.join(regressed) or '-'}")
    if notes:
        print("\nparse notes:")
        for n in notes:
            print(f"  {n}")
    print("\nremaining disagreements:")
    for r in out_rows:
        if r["verdict"] == "DIFF":
            print(f"  [{r['class']}] {r['field_id']}\n"
                  f"    golden: {r['golden']!r}\n"
                  f"    rnsr:   {r['rnsr_answer'][:200]!r}")
    print(f"\nwrote {out_dir / 'comparison.csv'} and {out_dir / 'field_answers.csv'}")


if __name__ == "__main__":
    main()
