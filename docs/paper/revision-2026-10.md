# Paper revision notes — "RNSR DocDB and Programmable Context Management for Language Models"

> Superseded on 22 October 2026 by a full rewrite, `rnsr-docdb-2026-10.html`
> (second version: layer-by-layer evaluation, aggregate certificate, reviewer
> contract). These notes record the 21 October incremental revisions to the
> 30 September manuscript and are kept for the audit trail.

Prepared 21 October 2026 against the 30 September 2026 manuscript (PDF produced
from LibreOffice Writer). The manuscript pins itself to commit `d622678`. The
working tree now carries ~1,800 lines of changes and a 20 October benchmark on
three synthetic legal matters. These notes give replacement or insertion text,
section by section, so the manuscript can be re-pinned to the next commit.

The working tree described here was committed as `ded0c42`
(`ded0c42957ee9f9374938cf157fed5bd050f03b9`).

Conventions: text in blockquotes is ready to paste; everything else is a note
to the editor.

---

## Front matter

The current PDF has no author line. arXiv requires it in both metadata and the
PDF.

> Joshua Theeuf
> 21 October 2026 (revised; first version 30 September 2026)

---

## Abstract

Replace the sentence beginning "We formalize this separation and examine its
consequences…" with:

> We formalize this separation and examine its consequences through an
> implementation audit, a small matched engineering case study, and a later
> 286-question run over three synthetic legal matters under a newer model
> generation. The case study exposes a particularly instructive failure:
> correct labels and correct arithmetic produced incorrect answers when a
> repair operation changed the certified population. Two separate
> trajectories, one month apart and on different model generations, spent
> their budgets re-submitting drafts to an evidence reviewer rather than
> converging; the first exhausted the model context despite truncated cell
> output. These findings motivate scope-preserving evidence contracts,
> explicit review-loop termination, transcript management, and evaluation
> that distinguishes ingestion, retrieval, interpretation, computation, and
> operational completion.

Rationale: the October run is the paper's second recorded instance of the
review-loop pathology and is on `claude-sonnet-5-5`/`claude-haiku-5-5`, which
makes the point that the failure is architectural rather than a property of
one model. The abstract still makes no superiority claim.

---

## 1 Introduction — final paragraph

Replace:

> The implementation examined here is repository revision d622678, dated
> September 2026 [14].

with:

> The implementation examined here is repository revision `ded0c42`, dated
> October 2026 [14]. The engineering runs in Section 6.1–6.3 were recorded
> against the earlier revision d622678; Section 6.4 reports a later run
> against `ded0c42`. Where the two revisions differ in a way that matters to
> a claim, the text says which revision it describes.

The rest of the paragraph ("Mechanisms described as implemented refer to that
revision…") stands.

---

## 2 Context Management as a Data Access Problem

The constants in the paragraph beginning "The current implementation truncates
captured cell output to 4,000 characters…" were re-checked against the working
tree and are unchanged (`_OBSERVATION_LIMIT = 4000` in `harness/loop.py`;
20,000-character manifest clip, 100-document and 100-table inventory limits in
`harness/prompts/base.py`). No edit needed.

Append to that paragraph:

> The revision examined here additionally renders the transcript as an
> append-only sequence of stable blocks with a provider-side prefix-cache
> breakpoint on the system prompt and on the last completed turn. For
> providers that support prefix caching this changes what a growing
> transcript costs; it does not change what the model must read, and it does
> not bound transcript length. The 1M-token context windows of the models used
> in Section 6.4 likewise move the symptom of unbounded growth from a hard
> context-limit failure to spend, not away.

---

## 3 The DocDB Architecture — "Execution and isolation"

Replace the final two sentences of the section ("In the examined configuration,
aggregate query iteration, subcall, wall-time, and spend caps are disabled.
Local execution restrictions, provider limits, and permanent provider errors
still apply, and removing aggregate caps does not ensure convergence.") with:

> In the default research configuration, aggregate query iteration, subcall,
> wall-time, and spend caps are disabled; a separate production preset sets
> finite defaults for each. Local execution restrictions, provider limits, and
> permanent provider errors still apply, and caps bound spend rather than
> ensure convergence. The examined revision adds one convergence control
> specific to the evidence reviewer described in Section 4: when a rejected
> review re-states an objection the reviewer has already adjudicated against
> unchanged evidence, or when the number of rejected reviews reaches a
> configured limit, the loop finalizes with the contested fields marked
> unverified instead of re-submitting the draft. Both thresholds can be set to
> zero to restore the unbounded loop used in Sections 6.1–6.3.

---

## 4 Evidence Contracts and Conditional Correctness

### "Four distinct questions" — tier sentence

Replace:

> High, medium, and low evidence tiers are deterministic rules over recorded
> signals, not calibrated probabilities of correctness.

with:

> High, medium, and low evidence tiers are deterministic rules over recorded
> signals: quotation verification counts, review pushbacks, consensus
> agreement, negative-answer audit state, ingestion health, the validation
> status of source tables bound to the answer, and the outcome of claim
> review. They are not calibrated probabilities of correctness. The table
> binding is deliberately conservative: a quotation on a page that also holds
> an unchecked table lowers the tier even when the quotation does not depend
> on the table. In Section 6.4 this produces high-tier rates between 14% and
> 50% across otherwise similar matters, which is a calibration cost paid for a
> rule that can be audited.

### "Population identity must survive repair" — final paragraph

The scope bug is still present in `ded0c42` (`annotate.py` selects the newest
`annotation_log` row for a table/column; `classification.py` aggregates over
it). Replace the final sentence:

> This contract is a design proposal, not an implemented guarantee of the
> audited revision.

with:

> This contract is a design proposal. It is not implemented in `ded0c42`:
> the finalizer still revalidates the newest contract recorded for a table and
> annotation column, so the failure in Section 6.2 remains reproducible.

---

## 6 Engineering Evidence and Failure Analysis

### Section numbering

Give the existing subsections numbers so the new one can be referenced:

- 6.1 Study scope and protocol
- 6.2 Correct labels with the wrong population
- 6.3 External storage does not control transcript growth
- 6.4 *(new)* A second review-loop pathology under a newer model generation
- 6.5 Operational and evaluation failures *(was unnumbered)*

### 6.1 — add after "Model aliases were recorded rather than immutable provider snapshots."

> Costs in Table 2 are list-price estimates at the September 2026 rates for
> those aliases. The scorer used for Table 2 accepted exact normalized text,
> numeric agreement within one percent, or text containment, with a model
> judge on disagreement; the scorer used in Section 6.4 is stricter (Section
> 6.4). Table 2 is therefore not re-scored and the two sections are not
> directly comparable.

### 6.3 — append a paragraph

> The circuit breaker described in Section 3 is a response to this trajectory
> and to the one in Section 6.4. It bounds the number of review rounds, not
> the transcript. Checkpointing typed execution state and a compact evidence
> ledger, and defining progress in terms of newly covered evidence, remain
> unimplemented proposals.

### 6.4 — new subsection (insert in full)

> **6.4 A second review-loop pathology under a newer model generation**
>
> On 20 October 2026, revision `ded0c42` was run over three synthetic legal
> matters authored for benchmarking: an employment dispute (40 documents, 111
> questions), a family-law property matter (37 documents, 85 questions), and a
> partner-visa matter (43 documents, 90 questions). Documents include Word and
> PDF files, plain-text correspondence, and photographed or scanned pages that
> required vision-model transcription. Each matter carries a question set with
> gold answers, a per-question type, and a list of the source documents an
> answer should rest on. Unanswerable items (false premises, facts absent from
> the corpus) are included and scored as correct only when the system declines
> to answer. The matters were not used for tuning, but they were written to
> exercise known failure modes (superseded facts, distractors, absent
> premises), so they are diagnostic coverage rather than an independent
> held-out test.
>
> The configuration was root model `claude-sonnet-5-5`, sub-model
> `claude-haiku-5-5` for batched sub-calls, claim review, and judging, batches
> of eight questions per exploration, per-query caps of 16 root iterations,
> 300 seconds, and $1 (scaled for batch size), and claim review enabled.
> Scoring used the revision's stricter deterministic scorer: exact normalized
> text or a scalar whose currency, scale, and unit dimension agree, with no
> text-containment rule. Disagreements were sent to a model judge that must
> return a complete YES or NO; an unresolved verdict is a miss.
>
> **The first attempt failed operationally.** Every matter reached a $60 run
> spend ceiling with most questions unanswered: 120 final answers for $181.29
> across the three matters. Four causes were identified, all specific to the
> model generation. The models reject non-default sampling temperature; they
> think adaptively by default and thinking tokens count against the output
> limit, so terse deterministic calls (an eight-token YES/NO judge, a
> 64-token recovery call) returned empty text; the claim reviewer's 2,048-token
> output cap truncated the JSON verdict for batches of eight fields, so 375 of
> 440 reviews failed to parse, the unresolved review sent the draft back to
> the controller, and the cycle repeated until the budget was exhausted; and
> a scan transcription parser rejected a page whose text blocks were returned
> inside the table array. Individual batches reached 145 root iterations and
> 10 million input tokens.
>
> This is the same failure shape as Section 6.3 on a different model
> generation with a different proximate trigger. In September the reviewer
> kept rejecting; in October the reviewer's reply could not be read. In both
> cases the loop treated an unresolved review as an instruction to try again
> with the same evidence, and nothing other than a spend cap stopped it. The
> circuit breaker in Section 3 was added after this run and was not active
> during it.
>
> **After the four fixes, the clean rerun completed.** Table 3 reports it.
> Accuracy counts every question; substantive accuracy excludes unanswerable
> items; judge-scored is the number of correct verdicts that came from the
> model judge rather than exact match. Spend is a list-price estimate and
> under-counts sub-model calls whose prompts exceeded 100,000 tokens, which
> are priced at a higher tier.
>
> *Table 3. October 2026 synthetic legal matters, clean rerun. Three
> family-law questions ended with the budget exhausted; they were unanswerable
> items, so the not-found placeholder scored as correct, but the system did not
> reach that answer on evidence.*
>
> | Matter | Questions | Correct | Substantive | Unanswerable handled | Judge-scored | High tier (correct / n) | Spend |
> |---|---|---|---|---|---|---|---|
> | Employment | 111 | 109 (98.2%) | 92 / 94 | 7 / 7 | 76 | 56 / 56 | $7.86 |
> | Family | 85 | 83 (97.6%) | 70 / 72 | 6 / 6 | 54 | 12 / 12 | $11.61 |
> | Immigration | 90 | 84 (93.3%) | 72 / 76 | 6 / 6 | 63 | 20 / 22 | $14.87 |
> | All | 286 | 276 (96.5%) | 234 / 242 | 19 / 19 | 193 | 88 / 90 | $34.33 |
>
> Three observations qualify these numbers. First, 193 of 276 correct
> verdicts were decided by the model judge, because the stricter scorer sends
> any answer that adds context to the gold value to adjudication. Those are
> model decisions, not human ones, and the judge protocol changed twice during
> scoring (once to add per-question-type context, once to raise the judge's
> reply cap after truncated replies produced four spurious misses); no system
> answer changed between passes. Second, the author's reading of the ten
> misses, which is not folded into any number above, is that six state the
> gold fact and add further context the judge treated as conflicting, one
> appends a superseding corrigendum to a correct quotation, two decline to
> answer where the gold is "No" and the corpus contains no such document, and
> one contains a substantive error. Third, the two high-tier errors are both
> in the immigration matter and both are answers that state the requested
> value and then volunteer a conflicting one; the tier rules do not inspect
> answer text and cannot see that.
>
> Source citation was checked separately: for answerable questions, the
> system's verified evidence cited at least one of the key documents for 259
> of 267 answers, with no answer lacking citations.
>
> Table 3 is a single pass of the configured system on fictional matters. It
> is not a comparison with any alternative, and the matters' authors knew the
> system's failure modes. It establishes that the revision completes a
> realistic multi-document workload at a known cost, and it supplies the
> paper's second record of a review loop that did not terminate on its own.

### 6.5 (formerly "Operational and evaluation failures") — no change to body.

Add one sentence at the end:

> The October attempt in Section 6.4 belongs in the same category: an
> operational failure whose 120 answers should not be aggregated with the 286
> of the rerun.

---

## 7 Related Work

No change required. If a citation for prompt caching is wanted for the Section
2 insertion, add:

> [16] Anthropic. 2026. Prompt caching. Claude Developer Platform
> documentation. https://platform.claude.com/docs/en/build-with-claude/prompt-caching.
> Accessed 21 October 2026.

---

## 8 An Evaluation Agenda — "Measure the chain of evidence"

Append:

> Two further measurements follow from Section 6.4. A review-loop termination
> policy needs its own evaluation: the rate at which a circuit breaker halts a
> loop that would have converged, against the budget it saves when the loop
> would not. And when a model judge decides most correct verdicts, the judge
> is part of the system under test; report the judge-scored share, retain
> judge explanations, and adjudicate a sample of judge decisions
> independently before treating an accuracy figure as a property of the
> answering system alone.

---

## 9 Limitations and Responsible Use

Replace the sentence "The present transcript policy permits context exhaustion;
richer workers and hierarchical routing remain proposed extensions." with:

> The present transcript policy still permits unbounded growth; the review
> circuit breaker bounds one cause of repetition, not transcript size. Richer
> workers and hierarchical routing remain proposed extensions.

Append to the final paragraph:

> The October run adds its own limits: a stricter scorer that routes most
> prose answers to a model judge, cost estimates that under-count tiered
> sub-model pricing, and matters whose authors knew the system. Provider
> behaviour also changed under the paper's feet: sampling, thinking, and
> output-limit semantics differed between the model generations used in
> Sections 6.1–6.3 and 6.4, and each difference produced an operational
> failure before it produced a measurement.

---

## 10 Conclusion

Replace "external storage can still coexist with an overflowing prompt" with:

> external storage can still coexist with an overflowing prompt, and an
> evidence reviewer can turn a stall into a spend.

---

## Artifact and Reproducibility Statement — replace in full

> The implementation audit targets commit `ded0c42957ee9f9374938cf157fed5bd050f03b9` (ded0c42) in the RNSR repository
> [14]. Relevant modules include `db/schema.py`, `ingest/writer.py`,
> `env/search.py`, `env/annotate.py`, `env/classification.py`,
> `env/calculations.py`, `env/evidence.py`, `harness/claim_review.py`,
> `harness/evidence.py`, and `harness/loop.py` under `rnsr/`. Sections
> 6.1–6.3 and 6.5 use preserved September 28 and September 29 run manifests,
> configuration snapshots, source hashes, outcome records, and diagnostics
> recorded against revision d622678 [15]; neither run was resumed for this
> manuscript. Section 6.4 uses the three synthetic matters under
> `testMatter/matters/` in the repository, each with its documents, question
> set, gold answers, and source key, together with the run directories
> described in `docs/benchmarks/matters-2026-10.md`. The matters are fictional
> and contain no client material. The commands that produced Table 3 are
> recorded in that document; re-running them requires provider credentials
> and incurs provider spend, and model aliases are not immutable snapshots.
>
> The September diagnostic bundles [15] remain local research artifacts and
> are not independently archived. Public code availability alone does not make
> those runs independently reproducible. No model inference was performed for
> the purpose of preparing this manuscript; all runs reported were engineering
> runs recorded before writing.

Editorial prerequisites for that statement to be true:

1. Done: the working tree is committed as `ded0c42` and the hash is
   substituted throughout these notes.
2. Done: `testMatter/matters/*.zip` is ignored; `.cache/`, `.DS_Store` and
   `__pycache__/` were already ignored.
3. Add a `LICENSE` file to the repository. The artifact statement points
   readers at public code with no stated terms.
4. Optionally archive the September bundles (synthetic 400-row source, gold
   labels, five trajectories, and a script regenerating Table 2 offline) on
   Zenodo or as a tagged release, and cite the DOI in [15].

---

## References

Update [14]:

> [14] RNSR. 2026. Source implementation at commit `ded0c42957ee9f9374938cf157fed5bd050f03b9`. Versioned
> repository, https://github.com/theeufj/RNSR. Examined 21 October 2026.

[15] unchanged. Add [16] only if the Section 2 caching sentence is kept.

---

## Checklist of claims re-verified against the working tree (21 October 2026)

| Claim | Status |
|---|---|
| 4,000-char observation truncation | unchanged |
| 20,000-char manifest clip; 100-doc / 100-table inventory limits | unchanged |
| Lazy document cache of 32 | unchanged |
| Search ladder stops at first nonempty rung; rung 5 returns an estimate | unchanged |
| Strict classification: all votes valid and strict majority | unchanged |
| Finalizer uses newest `annotation_log` contract (scope bug) | unchanged, still present |
| Aggregate caps default to disabled | unchanged; production preset added |
| Blocking health state prevents answering unless degraded allowed | unchanged |
| Scrubbed child environment | unchanged |
| Sandbox embedding cache private and temporary | unchanged |
| Reviewer not given gold | unchanged |
| Review loop unbounded except by budget | **changed**: stall / round circuit breaker, defaults 1 / 6 |
| Transcript rendered flat each iteration | **changed**: stable blocks with cache breakpoints; same text |
| Trust tiers: signals listed in §4 | **changed**: adds bound-table status and claim-review status |
| Scorer: exact / numeric-1% / containment + judge | **changed**: exact / dimension-aware scalar + judge; no containment |
| Model aliases Sonnet 4.5 / Haiku 4.5 | §6.4 uses Sonnet 5.5 / Haiku 5.5; IDs and prices verified against provider docs 2026-10-21 |
