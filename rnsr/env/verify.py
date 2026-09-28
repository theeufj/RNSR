"""verify(): exact quote checking by code, not model (spec §6).

Final answers must include supporting quotes; each is string-matched
(after normalization) against the retained source text, returning exact
char offsets. A check no LLM can hand-wave.
"""

from __future__ import annotations

import re
import sqlite3
import time
import unicodedata

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―"), "-")
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "´": "'", "`": "'"})


def _normalize_char(ch: str) -> str:
    ch = unicodedata.normalize("NFKC", ch)
    ch = ch.translate(_DASHES).translate(_QUOTES)
    return ch


class _NormalizedDoc:
    """Normalized text with a map back to original offsets."""

    def __init__(self, text: str, check=None):
        chars: list[str] = []
        self.offsets: list[int] = []
        for i, ch in enumerate(text):
            if check is not None and i % 16384 == 0:
                check()
            norm = _normalize_char(ch)
            if norm.isspace():
                if chars and chars[-1] == " ":
                    continue
                norm = " "
            for out in norm.casefold():
                chars.append(out)
                self.offsets.append(i)
        self.text = "".join(chars)

    def find(self, needle: str, start: int = 0) -> tuple[int, int] | None:
        i = self.text.find(needle, start)
        if i < 0:
            return None
        j = i + len(needle) - 1
        return self.offsets[i], self.offsets[j] + 1


def _normalize_needle(quote: str, check=None) -> str:
    chars = []
    for i, char in enumerate(quote):
        if check is not None and i % 16384 == 0:
            check()
        chars.append(_normalize_char(char))
    out = "".join(chars)
    return re.sub(r"\s+", " ", out).strip().casefold()


class Verifier:
    _CACHE_CAP = 64   # normalized docs are ~2x source size; bound the memory

    def __init__(self, doc):
        self._doc = doc  # any Mapping[str, str], incl. LazyDoc
        self._cache: dict[str, _NormalizedDoc] = {}
        # A disk-backed session index bounds RAM while retaining normalization
        # work beyond the small offset cache. No corpus artifact writes.
        self._index = sqlite3.connect("")
        self._index.execute("CREATE TABLE normalized (doc_id TEXT PRIMARY KEY, text TEXT)")
        self._indexed = False
        self._deadline: float | None = None

    def set_deadline(self, deadline: float | None) -> None:
        self._deadline = deadline
        self._index.set_progress_handler(
            (lambda: int(time.monotonic() >= deadline)) if deadline is not None else None,
            1000)

    def _check_deadline(self) -> None:
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise TimeoutError("source verification exceeded cell wall-clock deadline")

    def close(self) -> None:
        self._index.close()

    def _ensure_index(self) -> None:
        if not self._indexed:
            try:
                for doc_id in self._doc:
                    self._check_deadline()
                    self._index.execute("INSERT INTO normalized VALUES (?,?)", (
                        doc_id, _normalize_needle(self._doc[doc_id], self._check_deadline)))
                self._index.commit()
            except BaseException:
                self._index.rollback()
                raise
            self._indexed = True

    def _norm_doc(self, doc_id: str) -> _NormalizedDoc:
        if doc_id not in self._cache:
            if len(self._cache) >= self._CACHE_CAP:
                self._cache.pop(next(iter(self._cache)))
            self._cache[doc_id] = _NormalizedDoc(self._doc[doc_id], self._check_deadline)
        return self._cache[doc_id]

    def verify(self, answer: str, quotes: list[str], *, doc_id: str | None = None) -> dict:
        """Check lexical occurrence, optionally restricted to one source document.

        Context and bounded alternative matches make source ambiguity visible;
        ``passed`` alone does not establish that a quote supports the answer's
        entity, date, property, or polarity.
        """
        if isinstance(quotes, str):
            quotes = [quotes]
        results = []
        for quote in quotes:
            self._check_deadline()
            needle = _normalize_needle(str(quote), self._check_deadline)
            matches = []
            if needle:
                self._ensure_index()
                sql = 'SELECT doc_id FROM normalized WHERE instr(text, ?) > 0'
                params = [needle]
                if doc_id is not None:
                    sql += ' AND doc_id=?'
                    params.append(doc_id)
                rows = self._index.execute(sql + ' ORDER BY doc_id LIMIT 9', params)
                for row in rows:
                    source_id = row[0]
                    normalized = self._norm_doc(source_id)
                    offset = normalized.text.find(needle)
                    while offset >= 0 and len(matches) < 9:
                        start = normalized.offsets[offset]
                        end = normalized.offsets[offset + len(needle) - 1] + 1
                        hit = {'doc_id': source_id, 'char_start': start, 'char_end': end}
                        context = getattr(self._doc, 'context', None)
                        if context is not None:
                            hit['source_context'] = context(source_id, char_start=start, char_end=end)
                        matches.append(hit)
                        offset = normalized.text.find(needle, offset + 1)
                    if len(matches) >= 9:
                        break
            self._check_deadline()
            results.append({'quote': str(quote), 'matched': bool(matches),
                            **(matches[0] if matches else {}),
                            'matches': matches[:8], 'matches_truncated': len(matches) > 8})
        return {
            'passed': bool(results) and all(r['matched'] for r in results),
            'answer': str(answer),
            'quotes': results,
            'check': 'lexical_source_match',
        }
