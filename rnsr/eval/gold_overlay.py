"""Gold curation overlay: accepted equivalents for benchmark golds whose stated
form disagrees with the dataset's own justification.

The original gold is never edited. Each entry names the qid, the accepted
equivalents, and the reason with the dataset column it rests on, so a reader
can audit every relaxation. Scoring reports overlay-accepted rows separately
(``scored_by="gold_overlay"``) and every table shows accuracy with and
without them. The overlay is deliberately narrow: a gold that is wrong by its
own definition (a different line item, a contested reading of the question)
is not an overlay case; it stays a miss and is discussed as one.

File format::

    {"version": "financebench-2026-10", "entries": {
        "<qid>": {"accepted": ["$8,738 million", "8738"],
                  "magnitudes": [["0.6", "0.62"], ["14.7", "14.76", "14.8"]],
                  "reason": "...", "source": "justification"}}}

``accepted`` strings are compared with the strict scorer and the presentation
normaliser. ``magnitudes`` is a list of groups; every group must be matched,
sign-insensitively, by some number in the prediction (for golds whose only
disagreement with the filing is sign convention).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rnsr.eval.metrics import normalised_match, score_answer

DEFAULT_OVERLAY = Path(__file__).parent / "datasets" / "financebench_overlay.json"
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _magnitudes(text: str) -> set[str]:
    out = set()
    for m in _NUMBER.findall(text or ""):
        digits = m.replace(",", "")
        out.add(digits)
        if "." in digits:
            out.add(digits.rstrip("0").rstrip("."))
    return out


@dataclass
class GoldOverlay:
    version: str
    entries: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_OVERLAY) -> GoldOverlay:
        data = json.loads(Path(path).read_text())
        entries = data.get("entries") or {}
        for qid, entry in entries.items():
            if not entry.get("reason") or not entry.get("source"):
                raise ValueError(f"overlay entry {qid} needs a reason and a source column")
            if not (entry.get("accepted") or entry.get("magnitudes")):
                raise ValueError(f"overlay entry {qid} accepts nothing")
        return cls(version=str(data.get("version", "unversioned")), entries=entries)

    def accepts(self, qid: str, predicted: object) -> bool:
        entry = self.entries.get(qid)
        if not entry or predicted is None:
            return False
        text = str(predicted)
        have = _magnitudes(text)
        for alt in entry.get("accepted") or []:
            if score_answer(text, alt) or normalised_match(text, alt):
                return True
            # The equivalent stated inside a sentence ("... was $8,738 million ...").
            # Per-qid specificity makes a magnitude match safe here where it would
            # not be in the general scorer.
            m = _NUMBER.search(alt)
            if m and _norm(m.group(0)) in have:
                return True
        groups = entry.get("magnitudes") or []
        return bool(groups) and all(any(_norm(g) in have for g in group) for group in groups)


def _norm(s: str) -> str:
    s = s.replace(",", "")
    return s.rstrip("0").rstrip(".") if "." in s else s
