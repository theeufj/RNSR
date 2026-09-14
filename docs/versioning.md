# Versioning

## SDK (`rnsr.sdk`)

`rnsr` follows SemVer for the public names in `rnsr.sdk.__all__` (and the
same names re-exported from the `rnsr` package root):

```
BatchAnswer, answer, answer_batch, answer_batch_sync, answer_sync,
build_questions, corpus_env, fan_out, ingest, make_runner, open_corpus,
score_answers, score_answers_sync
```

- **Patch** (`1.0.0aN` while pre-release): bug fixes and additive optional
  fields that existing callers can ignore.
- **Minor**: new public names, new optional arguments with defaults.
- **Major**: rename, remove, or change the meaning of a public name.

Private modules (`rnsr.env`, `rnsr.ingest`, `rnsr.harness`) may change
without a major bump.

## Artifact format

Every `corpus.db` created by this package stamps:

- `PRAGMA user_version` — integer, currently `1`
- `manifest.format_version` — the same integer as JSON

`CorpusDB` refuses to open an artifact whose version does not match, and
raises `ArtifactVersionError` with a pointer at `rnsr migrate`.

Compatibility rules:

1. A new rnsr **must** open every artifact it wrote at the same
   `format_version`.
2. Adding nullable columns or new tables is a **format bump** only when
   old readers would mis-parse existing tables. Additive annotation
   columns are already the supported extension path and do not bump.
3. A breaking layout change increments `ARTIFACT_FORMAT_VERSION` and
   lands a converter in `rnsr.db.migrate`.
4. `rnsr migrate` is the supported upgrade path. There is no automatic
   rewrite on open.

The package version (`rnsr.__version__`) is recorded separately under
`manifest.versions.rnsr`. A newer package may still read an older
artifact of the same format version.
