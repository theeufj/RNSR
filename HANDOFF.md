# Handoff: running DocDB-RLM against the matter corpus

This is the runbook for testing DocDB-RLM against the fable-replicate
matter set (the 18,000+ file corpus). One command ingests the corpus into
a single SQLite artifact and answers the questions CSV in the exact format
fable-replicate's grader expects. Your corpus directory is **never written
to** — all state lives under `--work-dir`.

## 1. Setup (once, ~5 minutes)

Requirements: macOS (Apple Silicon is what this was tuned on) or Linux,
Python 3.11+, ~2× the corpus size in free disk for the artifact.

```bash
git clone git@github.com:theeufj/RNSR.git
cd RNSR
python3 -m venv .venv
.venv/bin/pip install -e ".[ingest]"
```

Create `.env` in the repo root (gitignored — never commit it):

```
ANTHROPIC_API_KEY=sk-ant-...
RNSR_PROVIDER=anthropic
```

That's the only required key. (`GOOGLE_API_KEY` / `RNSR_EMBED_MODEL` are
only needed for the optional vector-search rung; the matter runs did not
need it.)

Sanity check:

```bash
.venv/bin/rnsr --help        # should list: ingest, query, gate, eval, answer-csv, ablate
```

## 2. Pilot first (~$2–10, ~15 minutes)

Do **not** start with the full question set. Run 10 questions end to end
to confirm the corpus parses, answers come back, and the cost per question
is what you expect:

```bash
head -n 11 /path/to/questions.csv > /tmp/pilot.csv   # header + 10 questions

.venv/bin/rnsr answer-csv \
  --corpus /path/to/matter-corpus \
  --questions /tmp/pilot.csv \
  --output out/pilot \
  --work-dir runs/matter \
  --fast-ingest
```

The first run pays the one-time ingest (see timings below); the questions
themselves then run 3-at-a-time. Check `out/pilot/answers_chunk1.csv`
looks sane, and check spend in the console summary before scaling up.

## 3. The full run

```bash
.venv/bin/rnsr answer-csv \
  --corpus /path/to/matter-corpus \
  --questions /path/to/questions.csv \
  --output out/full \
  --work-dir runs/matter \
  --fast-ingest \
  --concurrency 3
```

Output contract (matches fable-replicate's verify step exactly):
`out/full/answers_chunk1.csv` with header `ground_truth_question,model_answer`,
questions verbatim in input order, no empty answers. Questions whose answer
is not in the corpus get the phrase `Not found in matter corpus`.

If your questions CSV uses a different column name, pass
`--question-col <name>`.

Then grade with fable-replicate's grader as usual, pointing it at
`out/full`.

## 4. What it costs and how long it takes

| Stage | Time | API cost |
|---|---|---|
| Ingest, 18k files (`--fast-ingest`) | ~10–40 min depending on page counts (parses in parallel across cores−2 workers; measured 850–4,600 pages/s on M-series) | $0 — no LLM calls |
| Per question | ~1–3 min wall each, 3 concurrent | typically $0.05–0.30; hard budget cap **$2.00/question** |
| 100 questions | a few hours | usually $10–40, worst case $200 |

The $2/question figure is a hard cap enforced by the harness (along with
20 iterations / 600 s per question) — a runaway question is cut off, not
billed open-endedly.

## 5. Resume semantics (crashes are cheap)

Everything checkpoints. **Rerun the identical command and it resumes:**

- **Ingest** checkpoints per document into `corpus_*.db.ingesting` under
  `runs/matter/corpora/`. A crash at file 17,999 resumes at 17,999, not 0.
  Already-ingested files are skipped by a stat-based identity
  (path+size+mtime — if you touch/replace corpus files mid-ingest they
  will be re-ingested as new).
- **Answers** checkpoint to `runs/matter/answers_partial.jsonl` after each
  question. Interrupting a 100-question run at 60 costs nothing; the rerun
  answers only the remaining 40 and re-emits the full CSV.
- The finished corpus artifact is cached — later runs with more questions
  skip ingest entirely.

## 6. Scanned pages

`--fast-ingest` extracts text layers only. Pages with no text layer
(scans) are counted and reported at the end of ingest as
`scanned_pages_untranscribed` — check that number. If it is material,
rerun with `--llm` added: scanned pages are transcribed by the vision
model during ingest (adds roughly $0.005–0.01 per scanned page; the rerun
only processes what's missing).

## 7. Troubleshooting

- **`PARSE FAILED <file>`** lines during ingest: that file is skipped and
  counted in `parse_failed`; the run continues. Common cause in the wild:
  HTML or zero-byte files with a `.pdf` extension.
- **All parses failed** → the run aborts *before* finalizing rather than
  producing an empty corpus. Usually means the `--corpus` path is wrong.
- **Network blips mid-run**: root API calls retry with backoff; a question
  that still fails is recorded and retried on the next resume.
- **Want to inspect an answer?** Each question's full REPL trajectory is
  under `runs/matter/` — every answer is backed by verified verbatim
  quotes or SQL lineage, so misses are auditable, not mysterious.

## 8. Guardrails already in place

- The corpus directory is opened read-only; nothing under `--corpus` is
  ever created, modified, or deleted.
- Ground-truth / rubric files must never be passed to this side — the
  answering process should only ever see the corpus and the questions CSV.
