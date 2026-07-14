"""verify(): exact quote checking by code, not model (spec §6).

Final answers must include supporting quotes; each is string-matched
(after normalization) against the retained source text, returning exact
char offsets. A check no LLM can hand-wave.
"""

from __future__ import annotations

import re
import unicodedata

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―"), "-")
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "´": "'", "`": "'"})


def _normalize_char(ch: str) -> str:
    ch = unicodedata.normalize("NFKC", ch)
    ch = ch.translate(_DASHES).translate(_QUOTES)
    return ch


class _NormalizedDoc:
    """Normalized text with a map back to original offsets."""

    def __init__(self, text: str):
        chars: list[str] = []
        self.offsets: list[int] = []
        for i, ch in enumerate(text):
            norm = _normalize_char(ch)
            if norm.isspace():
                if chars and chars[-1] == " ":
                    continue
                norm = " "
            for out in norm:
                chars.append(out)
                self.offsets.append(i)
        self.text = "".join(chars).lower()

    def find(self, needle: str) -> tuple[int, int] | None:
        i = self.text.find(needle)
        if i < 0:
            return None
        j = i + len(needle) - 1
        return self.offsets[i], self.offsets[j] + 1


def _normalize_needle(quote: str) -> str:
    out = "".join(_normalize_char(c) for c in quote)
    return re.sub(r"\s+", " ", out).strip().lower()


class Verifier:
    _CACHE_CAP = 64   # normalized docs are ~2x source size; bound the memory

    def __init__(self, doc):
        self._doc = doc  # any Mapping[str, str], incl. LazyDoc
        self._cache: dict[str, _NormalizedDoc] = {}

    def _norm_doc(self, doc_id: str) -> _NormalizedDoc:
        if doc_id not in self._cache:
            if len(self._cache) >= self._CACHE_CAP:
                self._cache.pop(next(iter(self._cache)))
            self._cache[doc_id] = _NormalizedDoc(self._doc[doc_id])
        return self._cache[doc_id]

    def verify(self, answer: str, quotes: list[str]) -> dict:
        """-> {"passed": bool, "quotes": [{quote, matched, doc_id, char_start,
        char_end}...]}. Passes only if every quote matches somewhere."""
        if isinstance(quotes, str):
            quotes = [quotes]
        results = []
        for quote in quotes:
            needle = _normalize_needle(str(quote))
            hit = None
            if needle:
                for doc_id in self._doc:
                    span = self._norm_doc(doc_id).find(needle)
                    if span:
                        hit = {"doc_id": doc_id, "char_start": span[0],
                               "char_end": span[1]}
                        break
            results.append({"quote": str(quote), "matched": hit is not None,
                            **(hit or {})})
        return {
            "passed": bool(results) and all(r["matched"] for r in results),
            "answer": str(answer),
            "quotes": results,
        }
