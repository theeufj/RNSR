"""Fan a collapsed group answer back out to the form's individual fields.

Group collapse is why the mutually exclusive fields stopped contradicting
each other, and this is the half that makes it safe: the winning option
takes the value its own field format asks for, every sibling takes "No".
Contradictory siblings become impossible by construction rather than
something the model has to keep straight across questions.
"""

from __future__ import annotations

import re

NOT_FOUND = "Not found in matter corpus"

_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+)", re.I)
_VALUE_RE = re.compile(r"VALUE\s*:\s*(.*)", re.I)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".")


def is_negative(answer: str) -> bool:
    a = _norm(answer)
    return (a in ("", "no", "unknown", "n/a", "none", "not applicable")
            or a.startswith(_norm(NOT_FOUND)))


def _first_token(text: str) -> str | None:
    """The answer token, whether or not the run used the ANSWER: prefix."""
    m = _ANSWER_RE.search(text or "")
    raw = m.group(1) if m else (text or "")
    lines = raw.strip().splitlines()
    if not lines:
        return None
    tokens = lines[0].strip().strip("`*\"' \t").split()
    return tokens[0].strip("`*\"'.,;") if tokens else None


def parse_group_answer(text: str, member_ids: list[str]) -> tuple[str | None, str | None, str]:
    """(winner_id | None, value, note) for an options-mode group.

    Lenient by design: models name the right option inside a sentence often
    enough that a strict parse would throw away correct answers.
    """
    note = ""
    choice = _first_token(text)
    if choice not in member_ids:
        if choice and _norm(choice) == "unknown":
            choice = None
        else:
            mentioned = [mid for mid in member_ids if mid in (text or "")]
            if len(mentioned) == 1:
                note = f"recovered choice from prose (ANSWER line said {choice!r})"
                choice = mentioned[0]
            elif "unknown" in _norm(text):
                note = f"read as unknown (ANSWER line said {choice!r})"
                choice = None
            else:
                note = f"unparseable answer, treated as unknown (got {choice!r})"
                choice = None
    vm = _VALUE_RE.search(text or "")
    value = vm.group(1).strip().splitlines()[0].strip() if vm else None
    if value is not None and _norm(value) in ("n/a", "na", "none", "unknown", ""):
        value = None
    return choice, value, note


def parse_value_group_answer(text: str, members: list[dict]) -> tuple[str | None, str | None, str]:
    """(winner_id, value, note) for a value-mode group."""
    value_member = next(m for m in members if m["needs_value"])
    alts = [m for m in members if not m["needs_value"]]
    raw = (text or "").strip()
    vm = _VALUE_RE.search(raw)
    if vm and vm.group(1).strip():
        raw = vm.group(1).strip()
    first = _first_token(raw) or ""
    n = _norm(first)
    if n.replace(" ", "_").startswith("not_applicable"):
        return (alts[0]["id"] if alts else None), None, ""
    if n in ("", "unknown") or is_negative(first):
        return None, None, ""
    if first in [m["id"] for m in members]:
        return first, None, "named an option id but supplied no value"
    value = raw.splitlines()[0].strip().strip("`*\"' ")
    value = re.sub(r"^(?:answer|value)\s*:\s*", "", value, flags=re.I).strip()
    return value_member["id"], value, ""


def field_value(member: dict, *, won: bool, value: str | None) -> str:
    """The value one field takes, given the group's single choice."""
    if not won:
        return NOT_FOUND if member["needs_value"] else "No"
    if member["needs_value"]:
        return value or NOT_FOUND
    if member["wants_yes"]:
        return "yes"
    return member["option_label"]


def fan_out(items: list, answers: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map per-item answers to per-field values.

    items: QuestionItem list (or their dict form, as written to the map file).
    Returns (field_id -> value, parse notes).
    """
    if len(items) != len(answers):
        raise ValueError(f"{len(answers)} answers for {len(items)} questions")
    out: dict[str, str] = {}
    notes: list[str] = []
    for item, text in zip(items, answers, strict=True):
        spec = item if isinstance(item, dict) else item.__dict__
        if spec["kind"] == "standalone":
            fid = spec["field_id"]
            out[fid] = (NOT_FOUND if is_negative(text) and spec["needs_value"]
                        else (text or "").strip())
            continue
        members = spec["members"]
        if spec.get("mode") == "value":
            winner, value, note = parse_value_group_answer(text, members)
        else:
            winner, value, note = parse_group_answer(
                text, [m["id"] for m in members])
        if note:
            notes.append(f"{spec.get('group')}: {note}")
        for m in members:
            out[m["id"]] = field_value(m, won=(m["id"] == winner), value=value)
    return out, notes
