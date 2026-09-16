"""Mechanical negative-answer challenge, isolated from loop orchestration."""
from __future__ import annotations

import logging
from pathlib import Path

from rnsr.answer_semantics import coerce_batch, is_negative
from rnsr.harness.trajectory import TrajectoryWriter
from rnsr.obs import get_logger, log, metrics

_LOG = get_logger("harness.negative_audit")


def audit_negatives(final: dict, questions: list[tuple[str, str]],
                     corpus_db: str, trajectory: TrajectoryWriter) -> str | None:
    """Mechanical audit of negative batch answers against the corpus.

    For each question answered No/unknown/NOT_FOUND without verified
    quotes, run one free FTS probe (AND of its most distinctive terms).
    Hits mean the corpus contains text where the question's terms
    co-occur — the model must read those passages before its negative
    stands. One shot per loop; a resubmission is accepted, so a
    genuine negative costs at most one extra iteration.
    """
    import itertools
    import sqlite3

    from rnsr.db import fts
    from rnsr.env.search import STOP_WORDS, TOKEN

    answers = coerce_batch(final.get("value")) or {}
    verification = final.get("verification") or {}
    negatives = [
        (qid, text) for qid, text in questions
        if is_negative(str(answers.get(qid, "")))
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
            for a, b in distinctive[:8]:
                hits = fts.match(conn, f'"{a} {b}"', k=1)
                if hits:
                    sample = " ".join(hits[0]["text"][:120].split())
                    flagged.append(
                        f"{qid} (doc {hits[0]['doc_id']}: \"{sample}\")")
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
        "these questions were answered negatively, but the corpus "
        f"contains text matching their terms: {listing}. Read those "
        "passages (and search around them) before resubmitting — "
        "change any answer they establish, or resubmit unchanged if "
        "the negative truly stands."
    )
