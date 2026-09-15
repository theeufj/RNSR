"""Corpus playbook: domain conventions as data, not prompt rules.

A playbook.json next to a corpus.db (or under the source directory) names
entity aliases, authority order, unit/date conventions, and which prompt
addons to attach. The financial-analysis block that used to live in the
base system prompt is the ``financial`` addon.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ADDONS = ("financial",)


@dataclass
class Playbook:
    entity_aliases: dict[str, list[str]] = field(default_factory=dict)
    authority_order: list[str] = field(default_factory=list)
    date_format: str = ""
    unit_conventions: dict = field(default_factory=dict)
    addons: list[str] = field(default_factory=lambda: list(DEFAULT_ADDONS))
    extra_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entity_aliases": self.entity_aliases,
            "authority_order": self.authority_order,
            "date_format": self.date_format,
            "unit_conventions": self.unit_conventions,
            "addons": self.addons,
            "extra_rules": self.extra_rules,
        }

    def prompt_block(self) -> str:
        lines: list[str] = []
        if self.entity_aliases:
            lines.append("Entity aliases (treat these as the same referent):")
            for canon, aliases in self.entity_aliases.items():
                lines.append(f"- {canon}: {', '.join(aliases)}")
        if self.authority_order:
            lines.append(
                "Authority order (earlier outranks later): "
                + " > ".join(self.authority_order)
            )
        if self.date_format:
            lines.append(f"Date format for answers: {self.date_format}.")
        if self.unit_conventions:
            lines.append("Unit conventions: " + json.dumps(self.unit_conventions))
        lines.extend(self.extra_rules)
        return "\n".join(lines)


def load_playbook(path: str | Path | None) -> Playbook | None:
    """Load a playbook.json. None when the path is missing."""
    if path is None:
        return None
    p = Path(path)
    if p.is_dir():
        p = p / "playbook.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return Playbook(
        entity_aliases={k: list(v) for k, v in (data.get("entity_aliases") or {}).items()},
        authority_order=list(data.get("authority_order") or []),
        date_format=data.get("date_format") or "",
        unit_conventions=dict(data.get("unit_conventions") or {}),
        addons=list(data.get("addons", list(DEFAULT_ADDONS))),
        extra_rules=list(data.get("extra_rules") or []),
    )


def discover_playbook(*candidates: str | Path | None) -> Playbook | None:
    """First playbook.json found among candidate files or directories."""
    for raw in candidates:
        if raw is None:
            continue
        found = load_playbook(raw)
        if found is not None:
            return found
    return None
