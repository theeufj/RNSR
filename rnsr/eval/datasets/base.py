"""Shared evaluation item model (§8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalItem:
    qid: str
    question: str
    gold: str                          # canonical gold answer (string form)
    task_class: str = "default"        # accuracy is reported per class
    context: str | None = None         # flat-text benchmarks (OOLONG, LongBench)
    sources: list[Path] = field(default_factory=list)  # document benchmarks
    meta: dict = field(default_factory=dict)
