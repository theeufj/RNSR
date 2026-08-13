"""Render form fields into questions the loop answers well.

Every block here earned its place against a measured miss:

  role map            two parties, two firms and three children share a
                      corpus; without an authoritative map, details get
                      attributed to the wrong person
  subject line        names the entity, and names the entity NOT to use
  group collapse      mutually exclusive options asked separately were each
                      answered "yes"; asked once, exactly one can win
  evidence rules      blank form scaffolding is not evidence; a name is not
                      evidence of gender; a negative needs its own search
  answer format       a bare "Yes" where the form wants a list is a miss, so
                      multi-part fields state their shape
"""

from __future__ import annotations

from rnsr.forms.spec import FormField, FormSpec, QuestionItem

EVIDENCE_RULE = """\
EVIDENCE RULE:
- Answer only from the matter documents in this corpus.{corpus_note}
- You may rely on what the documents establish directly, including facts \
that follow from a documented one (a recorded Australian residential \
address establishes presence and ordinary residence; a documented \
Australian place of birth establishes citizenship for form purposes).
- Blank form scaffolding is NOT evidence. Unticked checkbox labels, printed \
lists of options with nothing selected, and empty template cells say \
nothing about this matter - ignore them.
- NEVER guess from a person's name. In particular, never infer gender from \
a first name: a gender counts as established only where a document records \
it (for example "Gender: Male").
- Authority matters: court forms, solicitor letters, signed certificates \
and executed agreements outrank emails and file notes. Internal \
chronologies and summaries are SECONDARY - compiled after the fact and \
sometimes wrong. Where they conflict with contemporaneous primary \
documents, the primary documents govern.
- Search before concluding either way: a No or unknown claimed without a \
search targeted at this question's own subject is a wrong answer.
- Values often sit mid-row in long table text. When a hit relates to this \
question, scan its FULL text in code (string find or regex on the field's \
label) before concluding the value is absent - never judge from a \
truncated print.
- If the documents do not establish the answer, say so instead of guessing."""

_LARGE_CORPUS_NOTE = """ The corpus is LARGE, so do not try to read every \
document: search with distinctive terms (the field's label, a person's \
name, a document kind) and read the specific documents that matter."""


def _roles_block(spec: FormSpec) -> str:
    roles = spec.roles
    lines = ["MATTER ROLES (authoritative - use these to keep the parties "
             "straight):",
             f"- The form being completed is: {spec.form}"]
    if applicant := roles.get("applicant_1"):
        lines.append(
            f"- Applicant 1 is {applicant}"
            + (f", represented by lawyer {roles['applicant_lawyer']}"
               if roles.get("applicant_lawyer") else "")
            + (f" of the law firm {roles['applicant_law_firm']}"
               if roles.get("applicant_law_firm") else "") + ".")
    if respondent := roles.get("respondent_1"):
        lines.append(
            f"- Respondent 1 is {respondent}"
            + (f", represented by lawyer {roles['respondent_lawyer']}"
               if roles.get("respondent_lawyer") else "")
            + (f" of the law firm {roles['respondent_law_firm']}"
               if roles.get("respondent_law_firm") else "") + ".")
    for key, name in sorted(roles.items()):
        if key.startswith("child_"):
            lines.append(f"- {key.replace('_', ' ').title()} is {name}.")
    return "\n".join(lines)


def _subject_line(f: FormField, spec: FormSpec) -> str:
    if not f.subject:
        return ("This question is about the application as a whole, not about "
                "one party.")
    roles, role, subject = spec.roles, f.role or "", f.subject
    line = f"THIS QUESTION IS ABOUT: {role} - {subject}."
    lowered = role.lower()
    if lowered.startswith("applicant 1") and roles.get("respondent_1"):
        line += (f"\nDo NOT use details belonging to Respondent 1 "
                 f"({roles['respondent_1']}); only {subject}'s own details "
                 "answer this.")
    elif lowered.startswith("respondent 1") and roles.get("applicant_1"):
        line += (f"\nDo NOT use details belonging to Applicant 1 "
                 f"({roles['applicant_1']}); only {subject}'s own details "
                 "answer this.")
    elif lowered.startswith("child"):
        others = [v for k, v in sorted(roles.items())
                  if k.startswith("child_") and v != subject]
        if others:
            line += (f"\nDo NOT use details belonging to the other child(ren) "
                     f"({', '.join(others)}); only {subject}'s own details "
                     "answer this.")
    return line


def _evidence_rule(spec: FormSpec) -> str:
    note = f" {spec.corpus_note}" if spec.corpus_note else ""
    if spec.corpus_note:
        note += _LARGE_CORPUS_NOTE
    return EVIDENCE_RULE.format(corpus_note=note)


def _answer_format(f: FormField) -> str:
    if f.field_type in ("checkbox", "radio button"):
        return ("ANSWER FORMAT: reply with exactly one word - Yes or No. If "
                "the documents do not establish it, reply No.")
    if f.field_type == "date":
        return ("ANSWER FORMAT: reply with the date in DD/MM/YYYY format and "
                "nothing else. If the documents do not establish it, reply "
                "with exactly: unknown")
    return ("ANSWER FORMAT: reply with the requested value only, with no "
            "explanation. Identify people by their full names unless the "
            "field itself takes a role. Where the field names several "
            "sub-values, give each one the documents establish - a partial "
            "answer is wrong. If the documents do not establish it, reply "
            "with exactly: unknown")


def render_field_question(f: FormField, spec: FormSpec) -> str:
    parts = [_roles_block(spec), _subject_line(f, spec)]
    parts += spec.conventions_for(f)
    parts.append(f"FORM QUESTION: {f.title}")
    if f.notes:
        parts.append(f"FIELD DETAILS: {f.notes}")
    parts.append(_evidence_rule(spec))
    parts.append(_answer_format(f))
    return "\n\n".join(parts)


def render_group_question(members: list[FormField], spec: FormSpec) -> str:
    """One single-choice question over a mutually exclusive group."""
    lead = members[0]
    question = (lead.group_question or "").replace(
        "{{person}}", lead.subject or "this party")
    parts = [_roles_block(spec), _subject_line(lead, spec)]
    parts += spec.conventions_for(lead)
    parts.append(
        f"SINGLE-CHOICE FORM QUESTION: {question}\n\n"
        "This is ONE form question whose options are mutually exclusive: "
        "exactly one option may hold a value. Decide the question once, then "
        "name the single option that is correct - do not treat the options as "
        "separate questions and do not select more than one.")

    lines = ["OPTIONS:"]
    for i, m in enumerate(members, 1):
        extra = ""
        if m.needs_value:
            shape = ("a date in DD/MM/YYYY format" if m.field_type == "date"
                     else "the value itself")
            extra = f"  (choosing this option also requires {shape})"
        lines.append(f"  [{i}] option_id: {m.id}\n"
                     f"      form option: {m.option_label}{extra}")
    parts.append("\n".join(lines))
    parts.append(_evidence_rule(spec))
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


def render_value_group_question(members: list[FormField], spec: FormSpec) -> str:
    """A group whose real answer is a value with a 'not applicable' escape.

    'Date of marriage' pairs a date field with a not-applicable option:
    asking for an option id loses the date (measured — the run chose the
    option and supplied nothing), so the question asks for the value and
    keeps the alternative as a keyword.
    """
    value_member = next(m for m in members if m.needs_value)
    alts = [m for m in members if not m.needs_value]
    lead = members[0]
    question = (lead.group_question or "").replace(
        "{{person}}", lead.subject or "this party")
    shape = ("the date in DD/MM/YYYY format"
             if value_member.field_type == "date" else "the value itself")
    parts = [_roles_block(spec)]
    if lead.subject:
        parts.append(_subject_line(lead, spec))
    parts += spec.conventions_for(lead)
    parts.append(f"FORM QUESTION: {question}")
    alt_label = alts[0].option_label if alts else "not applicable"
    parts.append(
        "Answer in exactly ONE of these three ways, with no other text:\n"
        f"  - {shape} - if the documents establish it\n"
        "  - the single word not_applicable - if instead the correct answer "
        f"to this form question is \"{alt_label}\"\n"
        "  - the single word unknown - if the documents establish neither")
    parts.append(_evidence_rule(spec))
    return "\n\n".join(parts)


def _member_dict(m: FormField) -> dict:
    return {"id": m.id, "option_label": m.option_label,
            "needs_value": m.needs_value, "wants_yes": m.wants_yes,
            "field_type": m.field_type}


def build_questions(spec: FormSpec) -> list[QuestionItem]:
    """Collapse groups, render every question, and return them in form order.

    Raises ValueError when the spec is structurally inconsistent: a silently
    mis-grouped field produces answers that look fine and are not.
    """
    problems = spec.validate()
    if problems:
        raise ValueError("form spec is inconsistent: " + "; ".join(problems))

    items: list[QuestionItem] = []
    for name, members in spec.groups().items():
        # value-bearing options last, so the list reads options-then-value
        ordered = sorted(members, key=lambda m: m.needs_value)
        value_mode = any(m.needs_value for m in ordered)
        items.append(QuestionItem(
            item_id=f"group::{name}",
            question=(render_value_group_question(ordered, spec) if value_mode
                      else render_group_question(ordered, spec)),
            kind="group",
            members=[_member_dict(m) for m in ordered],
            mode="value" if value_mode else "options",
            group=name,
        ))
    for f in spec.standalone():
        items.append(QuestionItem(
            item_id=f.id,
            question=render_field_question(f, spec),
            kind="standalone",
            field_id=f.id,
            needs_value=f.needs_value,
            field_type=f.field_type,
        ))
    return items
