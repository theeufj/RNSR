# RNSR — DocDB-RLM

RNSR answers questions across document collections using retained source text,
typed SQLite tables and a sandboxed model loop. It is designed for work such as
reconciling invoice totals, following amended agreements and answering related
fields across a matter.

Ingest once into `corpus.db`. Use the model to interpret evidence, use code for
arithmetic, and return answers with source verification and review signals.

The [design specification](docdb-rlm-design-spec.md) describes the architecture;
current behavior and limitations are documented below and in the linked guides.

## Quickstart

Requires Python 3.11+ and an OS sandbox: `sandbox-exec` on macOS or `bubblewrap`
with user namespaces on Linux. Execution fails closed when containment is
unavailable.

```bash
pip install -e ".[ingest]"
# Configure ANTHROPIC_API_KEY, OPENAI_API_KEY or GOOGLE_API_KEY in .env.
rnsr ingest report.pdf exhibits.docx ledger.xlsx -o corpus.db --report ingest.json
rnsr query corpus.db "What is the total outstanding invoice balance?"
```

The ingestion stack supports PDFs, Office files, CSV, text and email containers.
Scans can use configured vision transcription. Unread scans and parse failures
are reflected in corpus health. Query-only installations need `pip install -e .`
and a prebuilt corpus artifact.

## Choose the workflow

A single document that fits the model context and needs straightforward reading
may be cheaper to answer directly. RNSR is intended for repeated questions over
collections, computed answers over sets, changing document versions and
inspectable source evidence. Interpretation and source selection still require
validation; exact arithmetic alone does not prove the right values were selected.

For a form or question list, related fields share exploration:

```bash
rnsr build-questions --spec form.json --out questions.csv --map questions.map.json
rnsr answer-csv --corpus matter_dir/ --questions questions.csv --output out/ \
  --batch-size 8 --concurrency 4 --consensus 2
rnsr regress --answers out/answers_chunk1.csv --golden golden.json \
  --map questions.map.json --min-accuracy 0.95 --max-false-positive-rate 0.15
```

Outputs include answers, per-field status/evidence and a run report. Checkpoints
resume successful answers for the same source revision. Folder caches **retain
archived sources**: new files are appended and changed files replaced, but
removing a file from the input folder does not delete it from an existing corpus.
Use a fresh work directory to build a snapshot excluding removed sources.

## Python and HTTP

```python
from rnsr import sdk
from rnsr.config import Settings

sdk.ingest(["report.pdf", "ledger.xlsx"], "corpus.db")
result = sdk.answer_sync("Who signed the agreement?", "corpus.db")
print(result.answer, result.status, result.evidence)

answers = sdk.answer_batch_sync(questions, "corpus.db", batch_size=8,
                                settings=Settings.production())
```

The SDK and CLI use the same answering core. Worker recipes are in
[docs/examples](docs/examples/). Install `.[service]` for `rnsr serve`; the service
requires a bearer token and an allowed corpus root. Authorized job results
include evidence, citations, health, support-review state and audit references.
See [operations](docs/operations.md) for deployment, retention and tenancy limits.

## Evidence and trust

Source text and table rows are frozen after ingestion. Final quotes are checked
again by the parent process. Source-bound calculation and classification proofs
are independently recomputed; semantic support review checks whether the
provided evidence addresses the requested claim.

Evidence includes quote counts, cited tables and their validation status,
retrieval rungs, support-review state, recovery and consensus signals. Ordinary
quotes conservatively inherit table quality from their canonical cited pages;
calculation/classification proofs bind their exact tables. Disabling support
review is visible and prevents a high tier in DocDB mode.

A trust tier is a rule-derived review signal, not a calibrated probability or a
legal certification. See [answer validation](docs/answer-validation.md) and
[trust tiers](docs/trust-tiers.md).

Ingestion outcomes persist per document and attachment across resume and
replacement. Existing format-v3 artifacts remain readable. Older artifacts
without per-document outcomes retain their historical aggregate gaps
conservatively when extended; rebuild them to recover complete attribution.
Migration from older formats remains explicit: `rnsr migrate corpus.db`.

## Evaluation status

Historical results and sample sizes are in
[benchmark history](docs/benchmark-history.md). They used an older scorer and
must not be read as verified accuracy of the current implementation. The
[offline scoring audit](docs/benchmarks/scoring-audit.json) preserves aggregate
agreement/adjudication counts without changing original predictions.

New evaluations accept exact normalized text or unambiguous scalar numeric
agreement automatically; other answers need contextual adjudication. Run
manifests bind resume to source/question hashes, models, settings, code and
scorer version. See [evaluation](docs/evaluation.md) for frozen-rule, held-out
and calibration protocols. Live held-out results must be reported separately
from offline implementation tests and tuned form fixtures.

## Deployment settings

Research defaults allow unlimited overall iterations, time and spending, with
full trajectories retained. Individual calls and code cells still time out.
For a bounded deployment starting point:

```bash
export RNSR_PROFILE=production
rnsr doctor
```

The [production example](deploy/.env.production.example) documents the preset and
its overrides. Replace or remove existing research `RNSR_*` overrides when
selecting it: explicit environment values still win, and a zero cap disables
that limit. Generated code has no provider credentials or network access;
provider calls are brokered by the parent. Spending caps stop future requests
once recorded usage reaches the ceiling; already running requests can add cost.

## Development

```bash
pip install -e ".[eval,dev,service,secure]"
pytest                     # offline by default; live tests opt in
ruff check .
```

Useful commands: `rnsr health corpus.db`, `rnsr doctor`, `rnsr trajectory`,
`rnsr eval`, `rnsr gate`, and `rnsr replay`.

The search ladder, index parity requirements and staged storage work are in
[search contract](docs/search-contract.md) and
[storage plan](docs/engine-poc-plan.md). Cross-document schema mapping remains
an explicit proposal-and-join operation. Scale changes should preserve source
provenance and pass replay checks before replacing the SQLite paths.
