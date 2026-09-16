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
| `cited_table_statuses` | statuses of tables named in quotes / SQL |

## Rules (low first)

An answer is **low** if any of:

- `source == recovered` or `status` is not `final`
- `third_strike`
- `health_grade == blocked`
- `negative_audit == flagged`
- `resolved_by == unresolved`

Otherwise **medium** if any of:

- `zero_quotes` on a `final` source
- some quotes failed verification
- `negative_audit == survived`
- any completeness pushback
- `health_grade == degraded`
- `budget_warned`
- consensus `tiebreak` / `split`, or `agreement < 1`
- a cited table is `untrusted` or `unchecked`

Otherwise **high**.

In docdb mode, all value-bearing final paths (`FINAL`, `FINAL_VAR`, and
each `FINAL_BATCH` field) require source quotes. The parent re-verifies
them against the read-only corpus. An invalid quote never becomes accepted
because it has been attempted three times. Historical evidence flags remain
readable so older trajectories retain their original conservative tier.

## How callers should use it

- Auto-accept **high**.
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
