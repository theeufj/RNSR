# Changelog

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
