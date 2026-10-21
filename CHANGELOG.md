# Changelog

## Unreleased

- Baseline arms for the matter benchmark (`rnsr.eval.matter_arms`): whole
  matter in context, classic flat-string REPL, and a direct-corpus-interaction
  `dci` sandbox mode (one text file per document with `ls`/`grep`/`read`
  helpers, after arXiv:2605.05242). `matter_arms_report` compares arms per
  matter and question type. On the October matters (11–15k tokens each) all
  arms are within noise of DocDB on accuracy at 1/12–1/30 of the cost, and
  on the 999-document Test Set 6 matter (~100k tokens) all four arms remain
  within noise; see `docs/benchmarks/matters-2026-10.md`.
- Benchmark scoring: `map.json` items for free-text questions must carry
  `needs_value: false`; `fan_out` rewrites negative answers ("No", "Not
  applicable") to the not-found phrase for value fields, which mis-scored
  explicit "No" answers against "No" golds in the October runs (corrected as
  scoring pass 4). `testMatter/ts6_arms_prep.py` builds arm-neutral inputs.
- `rnsr.eval.datasets.matter_dense`: long, dense synthetic matter generator
  (375 multi-line invoices, credit notes, remittances, rate variation, stale
  demand figure, ~600k tokens) with 38 exact aggregation/set/join golds. On
  it, whole-corpus prompting answers 4/15 aggregations; classic, DCI and
  DocDB answer 15/15, the two REPL baselines by parsing the raw text; DocDB's
  claim review halts on every batch because it cannot certify an aggregate.
- `rnsr.eval.arms_bench`: one-command four-arm comparison on any matter
  directory (ingest, arms, scoring with optional repeated judge passes,
  trajectory and evidence statistics, report.md/summary.json).
  `matter_dense --heterogeneous` writes the same facts with five invoice
  layouts across PDF, DOCX, EML and scanned PNG so a single parser cannot
  cover the set.
- Root prompts are sent to Anthropic as cache-friendly blocks: the system
  prompt and the completed transcript carry `cache_control` breakpoints, so a
  growing conversation re-reads its prefix instead of re-billing it every
  iteration (`PromptSegments`; other providers receive the identical flat
  text). Cache reads/writes are priced per model (`CACHE_MULTIPLIERS`) and
  reported in `Usage`, ledger snapshots and the run-level governor.
- The claim-review loop has a circuit breaker. A resubmission whose open
  fields were all already adjudicated on unchanged evidence
  (`claim_review_max_stalls`, default 1), or more than
  `claim_review_max_rounds` rejected reviews (default 6), finalises the
  answer with those fields `unverified` (low tier) and a
  `claim_review_halted` trajectory event, rather than running to the spend
  cap and falling back to solo runs. Zero on both restores the old loop.
- Root replies without a complete code fence are salvaged when they still
  parse as statements: a lost opening fence, an XML tool-call wrapper, stray
  tags inside code, or a line of prose on either side. Prose never runs.
- Per-call usage and cost on `root_call_completed` and `sub_batch` events;
  `rnsr ingest --report` and `rnsr regress` record provider spend
  (`provider` / `judge_provider`) so a run's all-in cost is visible.
- Benchmark scoring accepts exact text or compatible scalar values; unit,
  polarity and prose ambiguity require contextual adjudication. Truly absent
  items alone form the false-positive denominator.
- Evaluation manifests bind resume to dataset/source hashes, models, settings,
  code, dependency versions and scorer identity. Changed or legacy evaluations
  require a fresh directory. Offline audits preserve historical predictions.
- Document and attachment ingestion outcomes retain completeness across resume
  and replacement. Optional format-v3 metadata preserves artifact compatibility;
  old aggregate gaps remain conservative until rebuild.
- Parent-derived table quality, support-review state and retrieval rungs reach
  per-field trust evidence and survive consensus. Disabled support review caps
  DocDB answers at medium; trust tiers require held-out calibration.
- SDK and authenticated HTTP results expose citations, evidence, health,
  support-review decisions and opaque audit references. HTTP audit timelines
  omit document content and filesystem locations.
- A production preset adds bounded execution and redacted trajectory retention
  across entry points while preserving unlimited research defaults.
- README separates historical measurements from current validation; generic
  matter/office CI coverage supplements the tuned form regression.

## 1.0.0a7

Job-grade accuracy program.

- **Autopsy + office-gen.** Misses classified by cause; `rnsr eval --benchmark office`;
  loss ledger and report-only CI autopsy.
- **Per-answer trust tiers.** `AnswerEvidence` on results, status CSV, and the
  service. Hidden-failure paths (zero-quote FINAL_BATCH, 3rd-strike, recovery)
  are tier-capped. Batch third-strike applies only to failed fields.
  Unchecked tables no longer degrade corpus health; small-n validation
  failures no longer block a large dump. Calibration metrics and `--abstain-below`.
- **Office ingest fidelity.** Sheet/slide as page + caption; eml/msg/html/zip/image
  dispatch; document metadata + content sha256; exact/near-dup detection;
  footnote/scale coercion; `net` is not a total row.
- **Corpus lifecycle.** `rnsr ingest --append` / `sdk.append`, `--replace`,
  per-file answer-csv cache.
- **Domain-neutral TaskSpec + playbook.** Legal text lives in `legal_base.json`;
  financial analysis is a playbook addon; `answers.xlsx` and `report.md`.
- **Review loop.** `rnsr review-import` writes corpus-local golden and playbook diffs.
- **Retrieval.** Per-question retrieval-recall; rung-4 default-on above a corpus
  size. Tantivy/usearch/TOC stay behind the replay gate.

## 1.0.0a6

Trust-hardening program.

- **Corpus health gate.** Ingest persists `manifest.health`. Answering
  refuses a `blocked` corpus unless `--allow-degraded`. `rnsr health`
  prints the report. Unchecked tables no longer count as trusted.
- **Default transcription.** Scans are transcribed when a vision-capable
  key exists (`transcribe_scans=auto`); ingest prints a cost estimate
  first. `--no-transcribe` opts out.
- **Aggregate-aware tables.** `_row_kind` (`data|total|subtotal|footnote|section`)
  is stamped on every extracted row. Prompt: aggregate with
  `WHERE _row_kind = 'data'`. Messy-table generator + labelled-set scorer
  (`rnsr eval-tables`).
- **Failure-mode eval.** `EvalItem.expect` (`value|absent`); needle-gen
  plants `absent` and `superseded`. `rnsr regress --max-false-positive-rate`.
- **Search-semantics contract.** `rnsr replay` plus a CI replay step.
  See `docs/search-contract.md`.
- **Containment.** Sandbox denies `sqlite3.connect` / `ATTACH`, DNS,
  raw writes to the artifact, `__stdout__` protocol injection, and
  name-spoofed `FINAL`. Guard roots are immutable tuples.
- **Offline provider fixtures.** `scripts/record_llm_fixtures.py` and
  `tests/test_llm_offline.py`. `rnsr doctor --check-models` in weekly CI.
- **Artifact versioning.** `PRAGMA user_version` + `manifest.format_version`.
  `CorpusDB` raises `ArtifactVersionError` on mismatch. `rnsr migrate`.
  Policy: `docs/versioning.md`.
- **Field-trial kit.** `rnsr audit-export` and `rnsr regress --from-review`.

## 1.0.0a5

SDK surface, spend governor, Stage 1 cells path, Test Set 6 ingest.
