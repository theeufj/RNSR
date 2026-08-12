"""Build role-aware, group-collapsed questions from the golden field set.

Two rewrites over testMatter/golden.json:

1. Role enrichment — every question carries the matter's role map, a
   subject line naming the entity it is about, and party-disambiguation
   anchors for the fields where two entities of the same kind appear in
   the documents (two law firms, two parties, two children).
2. Group collapse — fields whose notes declare a MUTUALLY EXCLUSIVE
   GROUP are asked once, as a single-choice question over the group's
   options. One coherent judgement replaces N independent yes/no queries
   that could contradict each other; score_answers.py fans the choice
   back out to the individual fields.

Outputs testMatter/questions_enriched.csv (the answer-csv input) and
testMatter/questions_enriched_map.json (row -> item spec for fan-out).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
GOLDEN = HERE / "golden.json"
OUT_CSV = HERE / "questions_enriched.csv"
OUT_MAP = HERE / "questions_enriched_map.json"

_SUBJECT_RE = re.compile(r"^For\s+(?P<role>[^\u2014]+?)\s+\u2014\s+(?P<name>[^.]+)\.\s*(?P<label>.*)$")
_GROUP_RE = re.compile(
    r'MUTUALLY EXCLUSIVE GROUP "(?P<name>[^"]+)"\s*\((?P<question>[^)]*)\)')
_SIBLINGS_RE = re.compile(r"this field and \[(?P<sibs>.*?)\] are alternative", re.S)
_FIELD_TYPE_RE = re.compile(r"FIELD TYPE:\s*(?P<type>checkbox|radio button|date)", re.I)
_WANTS_YES_RE = re.compile(
    r"""return\s+(?:exactly\s+)?['"]?yes|['"]yes['"]\s+or\s+['"]no['"]""", re.I)

EVIDENCE_RULE = """\
EVIDENCE RULE:
- Answer only from the matter documents in this corpus.
- You may rely on what the documents establish directly (for example, a \
documented Australian residential address establishes that a person is \
present in and ordinarily resident in Australia, and a documented \
Australian place of birth establishes Australian citizenship for form \
purposes).
- Blank form scaffolding is NOT evidence. Unticked checkbox labels, printed \
lists of options with nothing selected, and empty template cells tell you \
nothing about this matter - ignore them.
- NEVER guess from a person's name. In particular, never infer gender from a \
first name: a gender counts as established only where a document records it \
(for example "Gender: Male").
- Search before concluding either way: a No or unknown claimed without a \
search targeted at this question's own subject is a wrong answer.
- Values often sit mid-row in long table text. When a search hit relates \
to this question, scan the hit's FULL text in code (string find or regex \
on the field's label) before concluding the value is absent - never judge \
from a truncated print.
- Person and firm details (addresses, emails, phone numbers, dates and \
places of birth, genders, lawyer codes) are recorded in the client intake \
form, the filled court forms, and the marriage certificate. For such \
fields, READ THOSE DOCUMENTS' FULL TEXT directly (iterate doc.items() and \
scan each document's text in code) rather than relying on search \
snippets; attribute each detail to the correct person by reading the \
surrounding section headings.
- If the documents do not establish the answer, say so instead of guessing."""


def load_roles(context: str) -> dict[str, str]:
    """Parse the 'Roles: k = v; ...' line into a normalized map."""
    roles: dict[str, str] = {}
    m = re.search(r"Roles:\s*(?P<body>.*)", context)
    if not m:
        return roles
    for part in m.group("body").rstrip(".").split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        roles[re.sub(r"\s+", "_", key.strip().lower())] = value.strip()
    return roles


def roles_block(roles: dict[str, str], form: str) -> str:
    lines = ["MATTER ROLES (authoritative - use these to keep the parties straight):",
             f"- The form being completed is: {form}"]
    applicant = roles.get("applicant_1")
    respondent = roles.get("respondent_1")
    if applicant:
        lines.append(
            f"- Applicant 1 is {applicant}, represented by lawyer "
            f"{roles.get('applicant_lawyer', '?')} of the law firm "
            f"{roles.get('applicant_law_firm', '?')}.")
    if respondent:
        lines.append(
            f"- Respondent 1 is {respondent}, represented by lawyer "
            f"{roles.get('respondent_lawyer', '?')} of the law firm "
            f"{roles.get('respondent_law_firm', '?')}.")
    children = [(k, v) for k, v in sorted(roles.items()) if k.startswith("child_")]
    for key, name in children:
        lines.append(f"- {key.replace('_', ' ').title()} is {name}.")
    return "\n".join(lines)


def parse_field(field: dict) -> dict:
    """Extract subject, option label, field type, and group info from notes."""
    notes = field["notes"]
    first_para = notes.split("\n\n")[0].strip()
    role = name = None
    label = first_para
    m = _SUBJECT_RE.match(first_para)
    if m:
        role = m.group("role").strip()
        name = m.group("name").strip()
        label = m.group("label").strip()

    ftype_m = _FIELD_TYPE_RE.search(notes)
    ftype = ftype_m.group("type").lower() if ftype_m else None

    group = siblings = group_question = None
    gm = _GROUP_RE.search(notes)
    if gm:
        group = gm.group("name").strip()
        group_question = gm.group("question").strip()
        sm = _SIBLINGS_RE.search(notes)
        siblings = ([s.strip().strip('"') for s in sm.group("sibs").split(";")]
                    if sm else [])

    return {
        "id": field["id"],
        "title": field["title"],
        "notes": notes,
        "role": role,
        "subject": name,
        "option_label": label,
        "field_type": ftype,
        "needs_value": ftype in (None, "date"),
        "wants_yes": bool(_WANTS_YES_RE.search(field["title"])),
        "group": group,
        "group_question": group_question,
        "siblings": siblings,
    }


def subject_line(f: dict, roles: dict[str, str]) -> str:
    """Name the entity the question is about, plus a same-kind negative anchor."""
    if not f["subject"]:
        return "This question is about the application as a whole, not about one party."
    role, subject = f["role"], f["subject"]
    line = f"THIS QUESTION IS ABOUT: {role} - {subject}."

    applicant, respondent = roles.get("applicant_1"), roles.get("respondent_1")
    if role.lower().startswith("applicant 1") and respondent:
        line += (f"\nDo NOT use details belonging to Respondent 1 ({respondent}); "
                 f"only {subject}'s own details answer this.")
    elif role.lower().startswith("respondent 1") and applicant:
        line += (f"\nDo NOT use details belonging to Applicant 1 ({applicant}); "
                 f"only {subject}'s own details answer this.")
    elif role.lower().startswith("child"):
        others = [v for k, v in sorted(roles.items())
                  if k.startswith("child_") and v != subject]
        if others:
            line += (f"\nDo NOT use details belonging to the other child(ren) "
                     f"({', '.join(others)}); only {subject}'s own details answer this.")
    return line


def firm_anchor(f: dict, roles: dict[str, str]) -> str | None:
    """Hard disambiguation for the applicant-side firm/lawyer fields.

    Both firms appear in the intake form, ~two sections apart; the earlier
    run answered with the respondent firm's address.
    """
    role = (f["role"] or "").lower()
    if not role.startswith("applicant - law"):
        return None
    return (
        "PARTY DISAMBIGUATION - READ CAREFULLY:\n"
        f"This question is about {roles.get('applicant_law_firm')} "
        f"(the APPLICANT's law firm, lawyer {roles.get('applicant_lawyer')}).\n"
        f"The documents ALSO contain details for {roles.get('respondent_law_firm')} "
        f"(the RESPONDENT's law firm, lawyer {roles.get('respondent_lawyer')}) - "
        "those details are NOT the answer.\n"
        "Where a document shows more than one firm, address, or lawyer, use only "
        f"the one belonging to {roles.get('applicant_law_firm')} / "
        f"{roles.get('applicant_lawyer')}, and ignore anything attached to "
        f"{roles.get('respondent_law_firm')} / {roles.get('respondent_lawyer')}.")


def residence_anchor(f: dict) -> str | None:
    """Point presence/residence questions at the recorded address.

    The run retrieved the address and answered "present in Australia"
    correctly, yet answered "ordinarily resident" No from the same
    evidence. Citizenship is deliberately excluded: an address says
    nothing about it.
    """
    t = (f["title"] or "").lower()
    if "citizen" in t or not re.search(r"ordinarily resident|present in australia", t):
        return None
    return (
        "HOW TO READ THE DOCUMENTS FOR THIS QUESTION:\n"
        "The documents record a residential or last-known address for each "
        "party. A recorded Australian address - INCLUDING one labelled "
        "'last known address' - establishes for form purposes that the "
        "party is both present in Australia and ordinarily resident in "
        "Australia, unless a document affirmatively places them elsewhere. "
        "Do not read 'last known' as doubt: answer Yes from that address. "
        "The form's own printed checkbox list (\"I am present in Australia "
        "/ I am ordinarily resident in Australia / I am an Australian "
        "citizen\") is blank scaffolding and is not evidence either way.")


def parentage_anchor(f: dict, roles: dict[str, str]) -> str | None:
    """Resolve the form's numbered parent slots to the matter's parties.

    The documents establish who a child's parents are (each child is
    recorded living with "Client (mother)", i.e. the applicant) but never
    number them - the numbering is form convention, not evidence. On this
    form the applicant's own details go in the Parent 2 slot and the other
    parent's in Parent 1. Without this convention the question is a
    coin-flip between the two parties (an earlier run picked the wrong
    one); with it, the slot is determined once the documents establish who
    the parents actually are.
    """
    t = f"{f['title'] or ''} {f['option_label'] or ''}"
    if not re.search(r"\bparent\s*[12]\b", t, re.I):
        return None
    applicant, respondent = roles.get("applicant_1"), roles.get("respondent_1")
    return (
        "FORM CONVENTION - NUMBERED PARENT SLOTS:\n"
        "The form numbers each child's parents; the documents do not. On this "
        f"form, Parent 2 is the APPLICANT parent ({applicant}) and Parent 1 is "
        f"the other parent ({respondent}) - applicable only where the "
        "documents establish that person is in fact the child's parent (for "
        "example, a child recorded as living with 'Client (mother)' "
        "establishes the client is that child's parent). Answer with the "
        "documented person's details for the numbered slot; do not answer "
        "unknown merely because the documents never use the words 'Parent 1' "
        "or 'Parent 2'.")


# Item-55 family (de facto jurisdiction) on the Initiating Application.
# These are conventions of the form, not facts of any one matter: how the
# vendor's form expects the de facto items to be completed by an applicant
# whose relationship became a marriage.
_ITEM_55_CONVENTIONS = {
    "55c": (
        "FORM CONVENTION - ITEM 55c:\n"
        "'Child of the de facto relationship' on this form means any child "
        "of the two parties' relationship, whether or not the parties "
        "subsequently married. If the documents record children of the "
        "parties, the correct option is Yes."),
    "55e": (
        "FORM CONVENTION - ITEM 55e:\n"
        "A registered relationship is a formal act that would appear in the "
        "matter file if it existed. Where no document mentions a "
        "registration under a state or territory law, the correct option "
        "is No - this is the one item where absence of any record selects "
        "the No option rather than unknown."),
    "55b": (
        "FORM CONVENTION - ITEM 55b:\n"
        "Where the parties married, this form leaves the de facto period "
        "item unanswered. Answer unknown here even if cohabitation dates "
        "are documented - do not compute the period."),
}


def item55_anchor(group_name: str) -> str | None:
    for prefix, convention in _ITEM_55_CONVENTIONS.items():
        if group_name.lower().startswith(prefix):
            return convention
    return None


def relationship_anchor(f: dict) -> str | None:
    """Multi-select 'relationship to application' fields: current status.

    The intake documents pre-marriage cohabitation, which tempted a run
    into also listing 'Party to a de facto relationship' for a married
    party; the form wants the relationship as it stands.
    """
    if "relationship to this initiating application" not in (f["title"] or "").lower():
        return None
    return (
        "FORM CONVENTION - RELATIONSHIP TO APPLICATION:\n"
        "State the party's CURRENT relationship to the application. A de "
        "facto relationship that later became a marriage counts only as "
        "'Party to a marriage' - do not also list 'Party to a de facto "
        "relationship' for parties who married. Include 'Parent' where the "
        "documents record children of the parties.")


def gender_anchor(group_name: str) -> str | None:
    """Hold the no-name-inference line on gender groups specifically.

    A batched run picked genders for both children from their first names
    despite the evidence rule; the children's genders appear in no
    document, so the only supportable answer for them is unknown."""
    if "gender" not in group_name.lower():
        return None
    return (
        "REMINDER FOR THIS GENDER QUESTION:\n"
        "A person's first name is NOT evidence of their gender. Choose an "
        "option ONLY if a document explicitly records this person's gender "
        "(for example 'Gender: Male' on an intake form). If no document "
        "records it, the correct answer is unknown - even when the name "
        "strongly suggests a gender.\n"
        "Before answering unknown, actually check: the client intake form "
        "records gender for the adult parties ('Gender:' fields in its "
        "personal-details and other-party sections) - read that document's "
        "full text for this person first. Children's genders are typically "
        "not recorded anywhere; unknown is the expected answer for a person "
        "whose gender no document states.")


def standalone_question(f: dict, roles: dict[str, str], form: str) -> str:
    parts = [roles_block(roles, form), subject_line(f, roles)]
    for anchor in (firm_anchor(f, roles), residence_anchor(f),
                   parentage_anchor(f, roles), relationship_anchor(f)):
        if anchor:
            parts.append(anchor)
    parts.append(f"FORM QUESTION: {f['title']}")
    parts.append(f"FIELD DETAILS: {f['notes']}")
    parts.append(EVIDENCE_RULE)
    if f["field_type"] in ("checkbox", "radio button"):
        parts.append("ANSWER FORMAT: reply with exactly one word - Yes or No. "
                     "If the documents do not establish it, reply No.")
    elif f["field_type"] == "date":
        parts.append("ANSWER FORMAT: reply with the date in DD/MM/YYYY format and "
                     "nothing else. If the documents do not establish it, reply "
                     "with exactly: unknown")
    else:
        parts.append("ANSWER FORMAT: reply with the requested value only, with no "
                     "explanation. If the documents do not establish it, reply "
                     "with exactly: unknown")
    return "\n\n".join(parts)


def value_group_question(members: list[dict], roles: dict[str, str], form: str) -> str:
    """Question for a group whose real answer is a value, not an option.

    'Date of marriage' pairs a date field with a 'not applicable' option.
    Asking for an option id loses the date (the run chose the option and
    supplied no value), so ask for the value directly and keep the
    alternative as a keyword escape.
    """
    value_member = next(m for m in members if m["needs_value"])
    alts = [m for m in members if not m["needs_value"]]
    lead = members[0]
    question = (lead["group_question"] or "").replace(
        "{{person}}", lead["subject"] or "this party")
    fmt = ("the date in DD/MM/YYYY format" if value_member["field_type"] == "date"
           else "the value itself")
    parts = [roles_block(roles, form)]
    if lead["subject"]:
        parts.append(subject_line(lead, roles))
    parts.append(f"FORM QUESTION: {question}")
    parts.append(
        "Answer in exactly ONE of these three ways, with no other text:\n"
        f"  - {fmt} - if the documents establish it\n"
        "  - the single word not_applicable - if instead the correct answer to "
        f"this form question is \"{alts[0]['option_label']}\"\n"
        "  - the single word unknown - if the documents establish neither")
    parts.append(EVIDENCE_RULE)
    return "\n\n".join(parts)


def group_question(members: list[dict], roles: dict[str, str], form: str) -> str:
    lead = members[0]
    question = (lead["group_question"] or "").replace(
        "{{person}}", lead["subject"] or "this party")
    parts = [roles_block(roles, form), subject_line(lead, roles)]
    for anchor in (firm_anchor(lead, roles), item55_anchor(lead["group"] or ""),
                   gender_anchor(lead["group"] or "")):
        if anchor:
            parts.append(anchor)
    parts.append(
        f"SINGLE-CHOICE FORM QUESTION: {question}\n\n"
        "This is ONE form question whose options are mutually exclusive: exactly "
        "one option may hold a value. Decide the question once, then name the "
        "single option that is correct - do not treat the options as separate "
        "questions and do not select more than one.")

    lines = ["OPTIONS:"]
    for i, m in enumerate(members, 1):
        extra = ""
        if m["needs_value"]:
            fmt = ("a date in DD/MM/YYYY format" if m["field_type"] == "date"
                   else "the value itself")
            extra = f"  (choosing this option also requires {fmt})"
        lines.append(f"  [{i}] option_id: {m['id']}\n"
                     f"      form option: {m['option_label']}{extra}")
    parts.append("\n".join(lines))

    parts.append(EVIDENCE_RULE)
    parts.append(
        "If the documents do not establish which option is correct, choose "
        "unknown - that is a valid and expected answer for questions this "
        "matter's documents never address.")
    parts.append(
        "ANSWER FORMAT - your final answer must be exactly these two lines:\n"
        "ANSWER: <one option_id from the list above, or the word unknown>\n"
        "VALUE: <the value for the chosen option if it requires one, in the "
        "required format; otherwise write n/a>")
    return "\n\n".join(parts)


def main() -> None:
    data = json.loads(GOLDEN.read_text())
    context = data["context"]
    roles = load_roles(context)
    form_m = re.search(r"Form:\s*([^\n.]+)", context)
    form = form_m.group(1).strip() if form_m else "this form"

    fields = [parse_field(f) for f in data["fields"]]
    by_id = {f["id"]: f for f in fields}

    # Cluster by declared group name, preserving first-seen order.
    groups: dict[str, list[dict]] = {}
    standalone: list[dict] = []
    for f in fields:
        if f["group"]:
            groups.setdefault(f["group"], []).append(f)
        else:
            standalone.append(f)

    # Sanity: every sibling label must resolve to a field in the same group.
    label_to_id = {f["option_label"]: f["id"] for f in fields}
    for name, members in groups.items():
        member_ids = {m["id"] for m in members}
        for m in members:
            for sib in m["siblings"]:
                sid = label_to_id.get(sib)
                if sid is None:
                    raise SystemExit(f"group {name!r}: unresolved sibling {sib!r}")
                if sid not in member_ids:
                    raise SystemExit(
                        f"group {name!r}: sibling {sib!r} ({sid}) outside the group")

    items: list[dict] = []
    for name, members in groups.items():
        # Value-bearing options last so the choice list reads options-then-value.
        members = sorted(members, key=lambda m: m["needs_value"])
        value_mode = any(m["needs_value"] for m in members)
        items.append({
            "kind": "group",
            "item_id": f"group::{name}",
            "group": name,
            "mode": "value" if value_mode else "options",
            "members": [
                {"id": m["id"], "option_label": m["option_label"],
                 "needs_value": m["needs_value"], "wants_yes": m["wants_yes"],
                 "field_type": m["field_type"]}
                for m in members
            ],
            "question": (value_group_question(members, roles, form) if value_mode
                         else group_question(members, roles, form)),
        })
    for f in standalone:
        items.append({
            "kind": "standalone",
            "item_id": f["id"],
            "field_id": f["id"],
            "needs_value": f["needs_value"],
            "field_type": f["field_type"],
            "question": standalone_question(f, roles, form),
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ground_truth_question", "item_id"])
        for it in items:
            w.writerow([it["question"], it["item_id"]])

    OUT_MAP.write_text(json.dumps(
        {"context": context, "roles": roles, "items": items},
        indent=2, ensure_ascii=False))

    n_grouped_fields = sum(len(it["members"]) for it in items if it["kind"] == "group")
    print(f"wrote {OUT_CSV.name}: {len(items)} questions "
          f"({sum(1 for i in items if i['kind'] == 'group')} groups covering "
          f"{n_grouped_fields} fields + "
          f"{sum(1 for i in items if i['kind'] == 'standalone')} standalone)")
    print(f"wrote {OUT_MAP.name}")
    assert n_grouped_fields + sum(1 for i in items if i["kind"] == "standalone") == len(by_id)


if __name__ == "__main__":
    main()
