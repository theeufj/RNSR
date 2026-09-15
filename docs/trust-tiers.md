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
| `third_strike` | the 3rd failed-quote attempt was accepted so the loop could end |
| `zero_quotes` | a `final` source submitted no quotes (FINAL_BATCH with an empty quotes dict) |
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

- `zero_quotes` on a `final` source (including quote-less FINAL_BATCH)
- some quotes failed verification
- `negative_audit == survived`
- any completeness pushback
- `health_grade == degraded`
- `budget_warned`
- consensus `tiebreak` / `split`, or `agreement < 1`
- a cited table is `untrusted` or `unchecked`

Otherwise **high**.

`final_var` with no quotes is high when nothing else trips — a SQL total
does not owe a verbatim quote. A quote-less `FINAL_BATCH` is not
`final_var`; it is `final` + `zero_quotes` and therefore at most medium.

## How callers should use it

- Auto-accept **high**.
- Review **medium** and **low**. `--abstain-below medium` writes
  `NEEDS REVIEW` in the answers CSV for low-tier rows; `--abstain-below high`
  does the same for medium and low.
- Gate a run with `rnsr regress --min-high-tier-accuracy`: accuracy
  computed only over high-tier answers must clear the bar, so a run that
  hides misses by marking everything low cannot pass.

Calibration targets (Phase 1 exit): on golden matter + office-gen,
≥90% of misses land in low+medium (**review recall**) and ≥60% of
answers are high (**auto-accept rate**).
