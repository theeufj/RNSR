"""Declarative form specification.

A FormSpec is what a vendor's field export becomes once the implicit
structure in its prose has been made explicit: which fields are alternative
answers to one question, which field wants a value rather than a tick, and
which conventions of this particular form the model has to be told because
no document states them.

The vendor conventions parsed here are the ones observed in both matter test
sets: a 'MUTUALLY EXCLUSIVE GROUP "name" (question)' clause with a sibling
list, a 'FIELD TYPE: checkbox|radio button|date' clause, and a 'For <role> —
<name>.' subject prefix.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_SUBJECT_RE = re.compile(
    r"^For\s+(?P<role>[^\u2014]+?)\s+\u2014\s+(?P<name>[^.]+)\.\s*(?P<label>.*)$")
_GROUP_RE = re.compile(
    r'MUTUALLY EXCLUSIVE GROUP "(?P<name>[^"]+)"\s*\((?P<question>[^)]*)\)')
_SIBLINGS_RE = re.compile(r"this field and \[(?P<sibs>.*?)\] are alternative", re.S)
_FIELD_TYPE_RE = re.compile(
    r"FIELD TYPE:\s*(?P<type>checkbox|radio button|date)", re.I)
_WANTS_YES_RE = re.compile(
    r"""return\s+(?:exactly\s+)?['"]?yes|['"]yes['"]\s+or\s+['"]no['"]""", re.I)


@dataclass
class Convention:
    """A rule of the form itself, attached to the fields it governs.

    Conventions carry no matter facts — only how this form expects a field
    to be completed (which slot is the applicant parent, what "child of the
    relationship" covers). Keeping them as data is what stops a per-form
    script from being the only way to reach the accuracy numbers.
    """

    text: str
    fields: tuple[str, ...] = ()          # exact field ids
    groups: tuple[str, ...] = ()          # group-name prefixes (case-insensitive)
    title_pattern: str = ""               # regex against the field title
    group_pattern: str = ""               # regex against the group name
    role_pattern: str = ""                # regex against the field's role

    def applies_to(self, f: FormField) -> bool:
        if f.id in self.fields:
            return True
        if f.group and any(f.group.lower().startswith(g.lower())
                           for g in self.groups):
            return True
        for pattern, value in ((self.title_pattern, f.title),
                               (self.group_pattern, f.group),
                               (self.role_pattern, f.role)):
            if pattern and value and re.search(pattern, value, re.I):
                return True
        return False


@dataclass
class FormField:
    id: str
    title: str = ""
    notes: str = ""
    role: str | None = None               # 'Respondent 1', 'Child 2', ...
    subject: str | None = None            # the person the field is about
    option_label: str = ""                # this option's label within its group
    field_type: str | None = None         # checkbox | radio button | date | None
    group: str | None = None
    group_question: str | None = None
    siblings: tuple[str, ...] = ()
    golden: list[str] = field(default_factory=list)

    @property
    def needs_value(self) -> bool:
        """True when the field takes a typed value rather than a tick."""
        return self.field_type in (None, "date")

    @property
    def wants_yes(self) -> bool:
        return bool(_WANTS_YES_RE.search(self.title or ""))


@dataclass
class FormSpec:
    form: str = "this form"
    roles: dict[str, str] = field(default_factory=dict)
    fields: list[FormField] = field(default_factory=list)
    conventions: list[Convention] = field(default_factory=list)
    corpus_note: str = ""                 # e.g. "999 files across seven folders"

    def conventions_for(self, f: FormField) -> list[str]:
        return [c.text for c in self.conventions if c.applies_to(f)]

    def groups(self) -> dict[str, list[FormField]]:
        """Group name -> members, in first-seen order."""
        out: dict[str, list[FormField]] = {}
        for f in self.fields:
            if f.group:
                out.setdefault(f.group, []).append(f)
        return out

    def standalone(self) -> list[FormField]:
        return [f for f in self.fields if not f.group]

    def validate(self) -> list[str]:
        """Structural problems that would silently degrade answers.

        A sibling label that resolves outside its own group means the vendor
        export disagrees with itself, and the group collapse would then ask
        about options that cannot be selected.
        """
        problems = []
        by_label = {f.option_label: f.id for f in self.fields}
        for name, members in self.groups().items():
            ids = {m.id for m in members}
            for m in members:
                for sib in m.siblings:
                    sid = by_label.get(sib)
                    if sid is None:
                        problems.append(
                            f"group {name!r}: sibling {sib!r} matches no field")
                    elif sid not in ids:
                        problems.append(
                            f"group {name!r}: sibling {sib!r} ({sid}) sits "
                            "outside the group")
        seen: set[str] = set()
        for f in self.fields:
            if f.id in seen:
                problems.append(f"duplicate field id {f.id!r}")
            seen.add(f.id)
        return problems


@dataclass
class QuestionItem:
    """One question put to the model, and the fields its answer fills."""

    item_id: str
    question: str
    kind: str                             # 'group' | 'standalone'
    members: list[dict] = field(default_factory=list)
    mode: str = "options"                 # group: 'options' | 'value'
    group: str | None = None
    field_id: str | None = None
    needs_value: bool = False
    field_type: str | None = None


def parse_field(raw: dict) -> FormField:
    """Build a FormField from a vendor field export entry."""
    notes = raw.get("notes") or ""
    first_para = notes.split("\n\n")[0].strip()
    role = name = None
    label = first_para
    m = _SUBJECT_RE.match(first_para)
    if m:
        role = m.group("role").strip()
        name = m.group("name").strip()
        label = m.group("label").strip()

    ftype_m = _FIELD_TYPE_RE.search(notes)
    group_m = _GROUP_RE.search(notes)
    siblings: tuple[str, ...] = ()
    if group_m:
        sib_m = _SIBLINGS_RE.search(notes)
        if sib_m:
            siblings = tuple(s.strip().strip('"')
                             for s in sib_m.group("sibs").split(";"))
    return FormField(
        id=raw["id"],
        title=raw.get("title", ""),
        notes=notes,
        role=role,
        subject=name,
        option_label=label,
        field_type=ftype_m.group("type").lower() if ftype_m else None,
        group=group_m.group("name").strip() if group_m else None,
        group_question=group_m.group("question").strip() if group_m else None,
        siblings=siblings,
        golden=list(raw.get("golden") or []),
    )


def parse_roles(context: str) -> dict[str, str]:
    """Parse a 'Roles: key = value; ...' context line into a normalized map."""
    roles: dict[str, str] = {}
    m = re.search(r"Roles:\s*(?P<body>.*)", context or "")
    if not m:
        return roles
    for part in m.group("body").rstrip(".").split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        roles[re.sub(r"\s+", "_", key.strip().lower())] = value.strip()
    return roles


def load_spec(path: str | Path) -> FormSpec:
    """Load a spec from JSON.

    Accepts the vendor's own shape ({"context": ..., "fields": [...]}) so an
    export can be used directly, with optional "form", "corpus_note" and
    "conventions" keys layered on top.

    An "extends" key names another spec (relative to this file) to inherit
    fields and context from, so a form's conventions can live beside the
    vendor export instead of being edited into it — the export is evidence,
    and prompt material does not belong in it.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if parent := data.get("extends"):
        base = json.loads((path.parent / parent).read_text(encoding="utf-8"))
        base.update({k: v for k, v in data.items() if k != "extends"})
        data = base
    context = data.get("context", "")
    form_m = re.search(r"Form:\s*([^\n.]+)", context)
    return FormSpec(
        form=data.get("form") or (form_m.group(1).strip() if form_m
                                  else "this form"),
        roles=data.get("roles") or parse_roles(context),
        fields=[parse_field(f) for f in data.get("fields", [])],
        conventions=[Convention(text=c["text"],
                                fields=tuple(c.get("fields", ())),
                                groups=tuple(c.get("groups", ())),
                                title_pattern=c.get("title_pattern", ""),
                                group_pattern=c.get("group_pattern", ""),
                                role_pattern=c.get("role_pattern", ""))
                     for c in data.get("conventions", [])],
        corpus_note=data.get("corpus_note", ""),
    )
