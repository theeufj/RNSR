# Changelog

## Unreleased

- Review loop bounded and certificate-aware. `claim_review_certificate_policy`
  defaults to `skip`: a field whose proof is an aggregate certificate with a
  parent-executed selection is supported without a model review
  (`reason_code=certificate_parent_verified`); a certified classification
  count gets one scope-only round in which the reviewer may contradict only
  by naming a clause, row or missing category. `claim_review_max_rounds`
  defaults to 3 (review, adjudication, one resubmission; was 6); contested
  fields then surface as low tier instead of spending the root budget.
  `require_certificate_for_derived` (default on): a final whose headline
  quantity appears in no verified quotation and carries no certificate is
  sent back once with instructions to bind it or quote the cell it came
  from (`certificate_nudge` trajectory event; zero and bare years exempt).
- `aggregate('count', ...)` accepts any original column of a registered
  table (a count is over rows; an invoice number identifies one); the other
  operations still need a numeric column. Seen live: a controller trying to
  certify an invoice count was refused four ways and fell back to hand
  arithmetic. The rubric now says which relations can be rows.
- Claim review keeps large calculation proofs reviewable: beyond 12 operand
  sources the proof carries one summary (count, documents, tables, value
  range and sum) plus the first 12 in full, and id lists on calculation
  records are trimmed to a count and a sample. Previously an uncapped
  derivation (454 records, 317k characters) was dropped as `prompt_limit`
  and the field could never be judged.
- Classification rubric labels once (`semantic_classify(votes=1)`) and
  certifies the count over the saved labels; three votes only on disputed
  rows. OOLONG DocDB: $16.77 -> $2.90, 6,915 -> 895 sub-calls, 11 -> 0
  halts, 34 -> 36/50.
- Scoring: deterministic presentation normaliser
  (`rnsr.eval.metrics.normalised_match`, `scored_by="normalised"`) between
  the strict scorer and the judge: percent/ratio, currency, scale words and
  unit suffixes agree at the gold's stated precision; rounding
  disagreements stay misses. `rnsr.eval.rejudge` defaults to three passes,
  reports `normalised` and `judge_split`, and `--sample N` writes a seeded
  adjudication sample of judge-decided rows (`--adjudicated FILE` reports
  judge-vs-human agreement). `public_arms_report` shows sub-calls, nudges
  and normalised-scored counts.
- Public benchmarks under the five-arm protocol (`docs/benchmarks/public-2026-10.md`):
  FinanceBench 40-question seeded subset over 18 real filings, CUAD-long (10)
  and OOLONG (50), with `rnsr eval` systems `stuffing`, `dci`, `dci-raw`, a
  shared `--corpus-cache`, `--stratify`, `rnsr.eval.migrate_cache` to carry
  historical corpora forward, and `rnsr.eval.public_arms_report`. Arms tie
  within noise on every benchmark; DocDB alone has verified citations on
  every answer and replayable derivations on 12/20 FinanceBench numerics at
  3-4x the REPL arms' cost; the classification-vote path on OOLONG costs
  8-17x and buys nothing. July's FinanceBench predictions rescored under the
  current scorer: 29/40, against 24-25/40 today; the gap is answer
  presentation on the certificate path and judge protocol. Judge v3
  (`rnsr.eval.metrics`): rubric states method/source text is not a
  contradiction and formatting is not substance; the judge explains before a
  final-line verdict (a bare-token judge snapped to NO on long rubrics).
  `rnsr.eval.rejudge` re-scores saved answers offline with repeated passes.
  Under judge v3 on the same 40 questions: July DocDB 30, this week before the
  presentation fix 30, after it 35; classic 35, DCI 34, DCI-raw 33, prompt 30.
- Paper rewritten (second version, `docs/paper/rnsr-docdb-2026-10.html`):
  a layer-by-layer evaluation of ingestion, REPL, typed relations and the
  evidence layer across five arms and four generated corpora; aggregate
  certificate and reviewer contract; Tables 1–5; v1 notes retained in
  `docs/paper/revision-2026-10.md`.
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
  cover the set. A `dci_raw` arm reads fast text extraction of the original
  files (no layout model, no tables, no OCR) to separate the value of ingest
  from the value of typed tables: on the heterogeneous matter it returns
  seven confident wrong aggregates (34 scans unreadable, never flagged) where
  the same agent over ingested text is 37/38 and DocDB 36/38.
- Aggregate certificate (`aggregate()`), certificates in `FINAL_BATCH`
  (`calcs=`), parent-executed `selection_sql`, template/decimals rendering;
  calc limits are settings, unlimited by default. On the dense matters DocDB
  is the only arm with replayable derivations (8–13 of 21 computed answers,
  37–59% key-source coverage) at parity accuracy (36–37/38); the claim
  reviewer accepts ~40% of certificates and every batch still halts, so
  trust tiers stay low. With `claim_review_certificate_policy=scope_only`
  (reviewer may support or contradict a parent-executed certificate by naming
  a scope fault, never withhold) and `embed_auto_max_chunks=1000` (no
  automatic corpus embedding), DocDB reaches 38/38 on the heterogeneous dense
  matter with 223 provider requests and 7 minutes (from 10,554 and 18), at
  3–4x the REPL baselines' cost. See `docs/benchmarks/matters-2026-10.md`.
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
