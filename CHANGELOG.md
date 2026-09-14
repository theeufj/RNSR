# Changelog

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
