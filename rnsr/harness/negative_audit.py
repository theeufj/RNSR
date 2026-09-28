"""Mechanical negative-answer challenge, isolated from loop orchestration."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from rnsr.answer_semantics import coerce_batch, is_negative
from rnsr.harness.trajectory import TrajectoryWriter
from rnsr.obs import get_logger, log, metrics

# Discovery terms, never equivalences used to determine an answer. A source
# may record sex while a form asks gender; the model must check that distinction.
_FIELD_DISCOVERY = {"gender": ("gender", "sex", "male", "female"),
                    "sex": ("sex", "gender", "male", "female")}
_LOG = get_logger("harness.negative_audit")


def audit_negatives(final: dict, questions: list[tuple[str, str]],
                     corpus_db: str, trajectory: TrajectoryWriter) -> str | None:
    """Mechanical audit of negative batch answers against the corpus.

    For unquoted negative answers, try up to eight distinctive phrase
    probes and four single-field discovery probes. Hits are candidates
    for inspection, not proof of a contradiction. The loop permits one
    audit challenge, so a genuine negative costs at most one iteration.
    """
    import itertools
    import sqlite3

    from rnsr.db import fts
    from rnsr.env.search import STOP_WORDS, TOKEN

    answers = coerce_batch(final.get("value")) or {}
    verification = final.get("verification") or {}
    negatives = [
        (qid, text) for qid, text in questions
        if qid in answers and is_negative(str(answers[qid]))
        and not ((verification.get(qid) or {}).get("passed")
                 and (verification.get(qid) or {}).get("quotes"))
    ]
    if not negatives:
        return None
    # Batched questions share heavy boilerplate (role maps, evidence
    # rules), so probe each question's DISTINCTIVE adjacent word pairs
    # — its field labels ("date of birth", "lawyer's code") — as FTS
    # phrases. A phrase hit means the corpus literally contains the
    # question's own wording, which is strong evidence a negative is
    # premature; loose single-term co-occurrence flagged legitimate
    # negatives and churned loops (seen live).
    def bigrams(text: str) -> list[tuple[str, str]]:
        toks = [t.lower() for t in TOKEN.findall(text)]
        return [(a, b) for a, b in itertools.pairwise(toks)
                if a not in STOP_WORDS and b not in STOP_WORDS]

    all_bigrams = {qid: bigrams(text) for qid, text in questions}
    counts = {qid: Counter(t.lower() for t in TOKEN.findall(text)
                           if not t[0].isdigit() and t.lower() not in STOP_WORDS)
              for qid, text in questions}

    def field_probes(qid: str) -> list[str]:
        # Count occurrences, rather than dropping every word shared by batch
        # boilerplate. "Do not guess gender" must not hide a Gender field.
        others = [c for other, c in counts.items() if other != qid]
        distinctive = [t for t, n in counts[qid].items()
                       if n > max((c[t] for c in others), default=0)]
        aliases = [t for t in distinctive if t in _FIELD_DISCOVERY]
        # At most four extra free probes, after the existing phrase probes.
        candidates = aliases + [t for t in distinctive if t not in aliases and len(t) >= 5]
        return [' OR '.join('"' + word + '"' for word in _FIELD_DISCOVERY.get(t, (t,)))
                for t in candidates[:4]]

    def describe(qid: str, hit: dict, probe: str) -> str:
        # Include the matched region, not an arbitrary 120-character prefix
        # which may contain only a repeated letterhead.
        import re

        from rnsr.env.evidence import SourceContext

        needles = re.findall(r'"([^"]+)"', probe)
        found = [re.search(re.escape(word), hit['text'], re.IGNORECASE) for word in needles]
        offset = min((m.start() for m in found if m is not None), default=0)
        start = hit['char_start'] + offset
        context = SourceContext(conn)(hit['doc_id'], char_start=start,
                                      char_end=min(start + 1, hit['char_end']), max_chars=1200)
        sample = ' '.join(context['text'].split())
        headings = ' > '.join(context['heading_paths'])
        return (f"{qid} (doc {hit['doc_id']}, page {context['page']}, "
                f"source {context['source_path']}, section {headings!r}: {sample!r})")
    flagged: list[str] = []
    try:
        conn = sqlite3.connect(Path(corpus_db).resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        for qid, _text in negatives:
            others = [set(bg) for o, bg in all_bigrams.items() if o != qid]
            distinctive = [
                bg for bg in dict.fromkeys(all_bigrams[qid])
                if sum(bg in s for s in others) <= len(others) // 2
            ] if others else list(dict.fromkeys(all_bigrams[qid]))
            probes = [f'"{a} {b}"' for a, b in distinctive[:8]] + field_probes(qid)
            for probe in probes:
                hits = fts.match(conn, probe, k=1)
                if hits:
                    flagged.append(describe(qid, hits[0], probe))
                    break
    finally:
        conn.close()
    if not flagged:
        return None
    flagged_ids = [f.split(" ", 1)[0] for f in flagged]
    trajectory.event("negative_audit", flagged=flagged_ids)
    log(_LOG, logging.INFO, "negative_audit.flagged", flagged=flagged_ids)
    metrics().incr("negative_audit_flags", len(flagged_ids))
    listing = "; ".join(flagged[:6])
    return (
        "These questions were answered negatively, but source passages contain "
        f"related wording: {listing}. These are retrieval candidates, not established "
        "contradictions. Read them against each question's exact person, period, "
        "and requested property before resubmitting. Sex/gender discovery terms "
        "are not interchangeable facts; never infer an attribute from a name or "
        "honorific. Change an answer only if the source establishes that answer "
        "for the requested subject; otherwise resubmit the negative unchanged."
    )
