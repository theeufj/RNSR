# Trust tiers

Every answer carries an `AnswerEvidence` record and a derived **tier**
in `{high, medium, low}`. The tier is a pure function of mechanical
signals — quote verification, negative-answer audit, completeness
pushbacks, recovery, consensus, corpus health. No model is consulted.

Implementation: [`rnsr/harness/evidence.py`](../rnsr/harness/evidence.py).

## Signals

| field | meaning |
|---|---|
| `source` | `final` (quoted FINAL / FINAL_BATCH) · `final_var` (SQL / computed) · `recovered` (namespace salvage) |
| `quotes_total` / `quotes_verified` | quotes submitted vs quotes that matched retained text |
| `third_strike` | historical trajectories only: a failed-quote attempt was accepted; current execution rejects repeated invalid quotes |
| `zero_quotes` | an exempt yes/no/absence answer submitted no quotes; value-bearing answers require quotes |
| `negative_audit` | `none` · `probed` (FTS check, clean) · `flagged` (hits found) · `survived` (flagged, then resubmitted unchanged) |
| `pushbacks` | completeness / batch-gap rejections before accept |
| `status` | loop status (`final` / `recovered` / `budget_exhausted` / `error`) |
| `agreement` / `resolved_by` | consensus vote, when `--consensus > 1` |
| `health_grade` | corpus health at answer time |
| `budget_warned` | the harness told the model to converge |
| `cited_table_statuses` | parent-derived statuses for source-bound tables and canonical cited-page candidates |
| `cited_tables` | source-bound tables, plus conservative table candidates on canonical cited pages |
| `claim_review_status` | `supported` · `disabled` · `not_applicable` · `unverified` · `unknown` |
| `claim_review_required` | whether the answer belongs to a DocDB support-review contract |

## Rules (low first)

An answer is **low** if any of:

- `source == recovered` or `status` is not `final`
- `third_strike`
- `health_grade == blocked`
- `negative_audit == flagged`
- `resolved_by == unresolved`
- required claim review is explicitly `unverified`

A review loop that stalls — the root resubmits claims the independent
adjudication already rejected, with no new evidence, or exceeds
`claim_review_max_rounds` rejected reviews — ends the query with the final
marked `partial` instead of burning the remaining budget. Each contested
field is `unverified` (low); fields the reviewer supported keep `supported`
and are capped at medium by the pushback that preceded them. Setting both
caps to 0 restores the unbounded loop.

Otherwise **medium** if any of:

- `zero_quotes` on a `final` source
- some quotes failed verification
- `negative_audit == survived`
- any completeness pushback
- `health_grade == degraded`
- `budget_warned`
- consensus `tiebreak` / `split`, or `agreement < 1`
- a cited table is `untrusted` or `unchecked`
- table quality is unknown
- support review was disabled, or required review has no supported decision

Otherwise **high**.

In docdb mode, all value-bearing final paths (`FINAL`, `FINAL_VAR`, and
each `FINAL_BATCH` field) require source quotes. The parent re-verifies
them against the read-only corpus. An invalid quote never becomes accepted
because it has been attempted three times. Historical evidence flags remain
readable so older trajectories retain their original conservative tier.

## How callers should use it

- Treat **high** as passing the recorded rule-based checks, not a probability
  of correctness. Decide automatic acceptance only after measuring high-tier
  accuracy and review recall on held-out matter/form data for the deployment.
- Review **medium** and **low**. `--abstain-below medium` writes
  `NEEDS REVIEW` for medium and low-tier rows; `--abstain-below high`
  does the same for all tiers. The threshold is inclusive in the CLI, SDK,
  and HTTP service, and a missing tier is treated conservatively. SDK
  results retain the original value separately as `raw_answer`.
- Gate a run with `rnsr regress --min-high-tier-accuracy`: accuracy
  computed only over high-tier answers must clear the bar, so a run that
  hides misses by marking everything low cannot pass.

Calibration targets (Phase 1 exit): on golden matter + office-gen,
≥90% of misses land in low+medium (**review recall**) and ≥60% of
answers are high (**auto-accept rate**).

The parent attaches table identities and validation statuses from its read-only
source snapshot. Calculation/classification proofs identify exact source tables;
ordinary quotes conservatively include tables on every overlapping canonical
page. That page-level binding may downgrade a prose quote on a page with an
unchecked table. It does not assert that the quotation identifies an exact row.
Per-field table and review signals survive batch and consensus selection.
Protected trajectory modes omit/hash source names and votes while retaining
operational review signals.

These calibration targets are requirements, not an assertion that current
held-out runs meet them. Tuned development fixtures and offline tests cannot
establish calibration. See [evaluation](evaluation.md).
