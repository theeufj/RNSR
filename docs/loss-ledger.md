# Loss ledger — Job-Grade Accuracy Program, Phase 0

Measured 2026-09-15. Live LLM answering of golden matter / TS6 was **not**
re-run here (provider cost). Causes below come from (1) a real ingest of
the new `office` benchmark, (2) the existing OOLONG miss review, and
(3) the answering-loop / ingest code audits. `rnsr autopsy` is the
mechanical classifier; this file is the sequencing decision.

## Office-gen ingest (seed 7, no transcriber)

10 source files → 1 merged table, 1 untranscribed scan, 0 parse failures.

| exposure | question class | observed at ingest | predicted miss cause |
|---|---|---|---|
| `sheet_identity` | sheet-specific, aggregation | 4 same-header Excel sheets collapsed into `t_budget_001` (4 rows, `title=None`) | **ingest** |
| `scanned_page` | lookup (receipt) | `receipt_scan` page 1 flagged, left untranscribed | **ingest** |
| `attachment` | cross-doc (widget price) | `.eml` body ingested; attached `invoice_widget.pdf` never becomes a child doc | **ingest** |
| `docx` | lookup (Q3 revenue) | `memo_q3_review` parsed; text present | reasoning / retrieval if the loop misses it |
| `supersession` | supersession | v1 and v2 both ingested as peers; no date / `duplicate_of` | **reasoning** (version ranking) unless retrieval lands on the draft |
| absent | absent | memo contains the negative | format / reasoning |

Confirmed structural ingest losses: **sheet merge, attachment drop, scan gap**.
These are Phase 2 items. They will dominate accuracy on any real mixed
office dump before retrieval or the root model get a chance.

## Golden matter (from prior measured runs + code audit)

Mitchell & Mitchell is 49/49 on the current golden when cells-path
semantics are exact (engine-poc Stage 1). Residual risk is not retrieval
at this scale:

| likely cause | evidence |
|---|---|
| reasoning | form conventions (roles, mutually exclusive groups) are prompt-injected; a stale sibling label or a missed subject swap is a reasoning miss |
| format | fan-out of group answers (`yes` vs option text) |
| ingest | scanned exhibits if `--no-transcribe`; otherwise health-gated |
| retrieval | not the limiter at ~hundreds of docs with the cells + FTS ladder |

No live autopsy JSON is checked in (trajectories quote the matter). The
regression workflow now writes one after every golden run (report-only).

## TS6 (~999 initiating-application files)

Same shape as golden matter, larger. The known failure modes from the
ingest audit scale linearly:

- office page identity = 1, so any multi-sheet exhibit merges
- adding one file re-ingests the whole set (Phase 3)
- no document date / supersedes, so amended forms compete as peers

Until Phase 2 lands, TS6 misses that depend on Excel exhibits or email
attachments will classify as **ingest**, not retrieval.

## OOLONG (existing miss review, 18 misses / 50)

| cause | n | notes |
|---|---|---|
| reasoning | ~12 | stale inherited labels, off-by-one counts, comparison flips |
| format | 2 | tie-set gold lists every tied label; predicted one of them |
| retrieval | 0 | context is stuffed / ingested as one doc |
| ingest | 0 | flat text |

This is a **reasoning + format** benchmark. It does not justify Phase 6
retrieval work. The autopsy classifier's `format` rule is written against
these two tie-set items.

## Sequencing decision (phases 2–6)

Phase 0 and Phase 1 stay first (measure, then put a trust signal on every
answer). The ledger **does not reorder** 2–6:

1. **Phase 2 ingest fidelity** — largest measured structural loss on a
   general corpus (sheet merge, attachments, scans, no metadata).
2. **Phase 3 corpus lifecycle** — TS6 / live matters cannot grow in place.
3. **Phase 4 domain-neutral job spec** — accuracy on legal forms is already
   high; the gap is reuse outside family law.
4. **Phase 5 review loop** — compounds only after 1–4 make review cheap.
5. **Phase 6 retrieval** — **conditional, deferred**. Office-gen and
   OOLONG show no retrieval-bound miss class at current scale. Turn this
   on only if a later autopsy ledger shows material `retrieval` share
   (rule of thumb: ≥20% of misses, or gold-doc recall < 0.9).
