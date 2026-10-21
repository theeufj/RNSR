# Matter benchmarks — October 2026

Run date: 2026-10-20. Three synthetic legal matters from `testMatter/matters/`, each with its own `qa.json` golden set, were ingested into separate `corpus.db` artifacts and answered with `rnsr answer-csv`, then scored with `rnsr regress` (deterministic scorer plus the sub-LM judge for disagreements). All artifacts live under `runs/matters-bench/<matter>/` (git-ignored).

> These are live, single-pass results for the current implementation on fictional matters authored for benchmarking. They are not independent held-out validation: the matters were not used for tuning, but they were authored to exercise known failure modes (superseded facts, distractors, unanswerable premises). Judge-scored agreements are model decisions, not human adjudication. See `docs/evaluation.md` for the reporting protocol.

## Configuration

| Setting | Value |
|---|---|
| Provider | anthropic |
| Root model (orchestrator) | `claude-sonnet-5-5` |
| Sub model (batched sub-calls, claim review, judge) | `claude-haiku-5-5` |
| Vision model (scan transcription) | `claude-haiku-5-5` |
| Embedding model | `gemini-embedding-2` |
| Ingest | `rnsr ingest` default (LLM-free parse; scans transcribed via VLM) |
| Answering | `answer-csv --batch-size 8 --concurrency 3 --consensus 1` |
| Per-query caps | `RNSR_MAX_ROOT_ITERS=16`, `RNSR_MAX_WALL_S=300`, `RNSR_MAX_SPEND_USD=1` (scaled ×4.5 for a batch of 8; a capped batch falls back to per-question loops) |
| Run spend ceiling | `RNSR_RUN_SPEND_CEILING_USD=30` per matter (not reached) |
| Claim review | enabled (default) |
| Scorer | `rnsr regress` schema v2 + sub-LM judge on string disagreements; generic per-question-type evaluator context supplied via `golden.json` (see Scoring) |

## Headline results

> Pass-3 scoring. See [Scoring pass 4](#scoring-pass-4-2026-10-21-map-needs_value-correction) for the corrected totals (275/286) and the reason; the tables in this and the following three sections are retained as the pass-3 record.

| Matter | Q | Accuracy | Substantive | Unanswerable handled | False-positive rate | Judge-scored | Spend | Wall |
|---|---|---|---|---|---|---|---|---|
| Employment law — Reyes v Harbourline Freight | 111 | 109/111 (98.2%) | 92/94 (97.9%) | 7/7 (100.0%) | 0.0% | 76 | $7.86 | 4.1 min |
| Family law — Okafor & Nguyen | 85 | 83/85 (97.6%) | 70/72 (97.2%) | 6/6 (100.0%) | 0.0% | 54 | $11.61 | 6.0 min |
| Immigration law — Tanaka Partner visa | 90 | 84/90 (93.3%) | 72/76 (94.7%) | 6/6 (100.0%) | 0.0% | 63 | $14.87 | 9.1 min |
| **All matters** | 286 | 276/286 (96.5%) | 234/242 (96.7%) | 19/19 (100.0%) | 0.0% | 193 | $34.33 | 9.1 min (parallel) |

*Accuracy* counts every question. *Substantive* excludes unanswerable items (golden holds a value). *Unanswerable handled* is the share of false-premise / not-in-corpus questions where the system declined rather than guessed; the *false-positive rate* is the complement (confident wrong answers on absent items). *Judge-scored* is the number of correct verdicts that came from the sub-LM equivalence judge rather than exact normalized match.

Caveat: Family law — Okafor & Nguyen: 3 question(s) ended `budget_exhausted` (`trung_criminal_record`, `ada_hecs_debt`, `full_court_appeal`). Those were unanswerable items, so the not-found placeholder scored as correct, but the system did not reach that answer on evidence. Discounting them, unanswerable handling is 16/19 on evidence.

### Scoring passes

The same answers were scored three times while the scoring inputs were corrected; no answer changed between passes.

| Matter | Pass 1: no evaluator context | Pass 3: context + judge cap fix |
|---|---|---|
| Employment law — Reyes v Harbourline Freight | 109/111 | 109/111 |
| Family law — Okafor & Nguyen | 79/85 | 83/85 |
| Immigration law — Tanaka Partner visa | 81/90 | 84/90 |

1. **No evaluator context.** `golden.json` carried only question, id and gold. The judge is instructed to return `unavailable` when context cannot resolve equivalence, and an unresolved verdict is a miss; most pass-1 misses were correct answers with extra detail (e.g. *60% to the Wife (Ada Okafor) and 40% to the Husband (Trung Nguyen) (draft consent orders v2…)* against gold *60% to the Wife, 40% to the Husband*).
2. **Generic per-type context added** (`field_context`: one shared paragraph plus one sentence per `question_type`, e.g. "Yes/No question; a leading Yes or No followed by an explanation is acceptable"). No per-item hints. Four previously valid verdicts then became `unavailable` because the judge's 300-token reply cap truncated longer reasons mid-JSON.
3. **Judge reply cap raised** to 1,024 tokens in `rnsr/eval/regression.py` (a truncated reply can only ever produce a miss, never a pass). These are the numbers reported above.

## Trust tiers and review signals

| Matter | High tier (correct/n) | Medium tier | Low tier | Review recall | Auto-accept rate |
|---|---|---|---|---|---|
| Employment law — Reyes v Harbourline Freight | 56/56 (100.0%) | 53/55 (96.4%) | – | 100.0% | 50.4% |
| Family law — Okafor & Nguyen | 12/12 (100.0%) | 68/70 (97.1%) | 3/3 (100.0%) | 100.0% | 14.1% |
| Immigration law — Tanaka Partner visa | 20/22 (90.9%) | 64/68 (94.1%) | – | 66.7% | 24.4% |

*Review recall* is the share of wrong answers that carried a non-high tier (i.e. would have been routed to review). *Auto-accept rate* is the share of answers at the high tier. A trust tier is a rule-derived review signal, not a calibrated probability (see `docs/trust-tiers.md`).

## Accuracy by question type

| Question type | Employment law | Family law | Immigration law | All |
|---|---|---|---|---|
| `count_list` | 9/10 (90.0%) | 8/8 (100.0%) | 8/9 (88.9%) | 25/27 (92.6%) |
| `existence` | 14/15 (93.3%) | 12/12 (100.0%) | 12/14 (85.7%) | 38/41 (92.7%) |
| `multi_fact` | 13/13 (100.0%) | 10/10 (100.0%) | 11/11 (100.0%) | 34/34 (100.0%) |
| `quote` | 6/6 (100.0%) | 3/3 (100.0%) | 3/4 (75.0%) | 12/13 (92.3%) |
| `relationship` | 12/12 (100.0%) | 6/8 (75.0%) | 9/9 (100.0%) | 27/29 (93.1%) |
| `summary` | 7/7 (100.0%) | 6/6 (100.0%) | 5/6 (83.3%) | 18/19 (94.7%) |
| `unanswerable` | 7/7 (100.0%) | 6/6 (100.0%) | 6/6 (100.0%) | 19/19 (100.0%) |
| `value_lookup` | 41/41 (100.0%) | 32/32 (100.0%) | 30/31 (96.8%) | 103/104 (99.0%) |

## Source citation check

`answer_key.json` lists the source documents each answer should rest on. The check below asks whether at least one document the system cited in its verified evidence is one of those sources (answerable questions only).

| Matter | Cited a key source (all answerable) | Cited a key source (correct answers) | Answers with no citations |
|---|---|---|---|
| Employment law — Reyes v Harbourline Freight | 102/104 (98.1%) | 100/102 (98.0%) | 0 |
| Family law — Okafor & Nguyen | 78/79 (98.7%) | 76/77 (98.7%) | 0 |
| Immigration law — Tanaka Partner visa | 79/84 (94.0%) | 74/78 (94.9%) | 0 |

## Ingestion

| Matter | Files | Docs ingested | Chunks | Tables | Scanned pages transcribed | Untranscribed | Parse failed | Health |
|---|---|---|---|---|---|---|---|---|
| Employment law — Reyes v Harbourline Freight | 40 | 40 | 156 | 19 | 7 | 1 | 0 | ok |
| Family law — Okafor & Nguyen | 37 | 37 | 67 | 18 | 6 | 0 | 0 | ok |
| Immigration law — Tanaka Partner visa | 43 | 43 | 131 | 4 | 8 | 0 | 0 | ok |

### Corpus size

Retained text per matter (`doc_text`, all pages), measured 2026-10-21. Token
counts are characters ÷ 4; the Claude tokenizer will be within ~20% of that.

| Matter | Retained text | ≈ tokens | Largest single document |
|---|---|---|---|
| Employment law | 52,538 chars | ~13k | 2,205 chars |
| Family law | 44,381 chars | ~11k | 1,846 chars |
| Immigration law | 60,054 chars | ~15k | 2,372 chars |

Every matter fits in a single model call with room to spare, and all three
together are ~40k tokens. These matters are therefore *inside* the regime
where the archived regime map (`benchmark-history.md`) found no accuracy
advantage for DocDB over putting the document in context. They test whether
the system completes a multi-document workload on realistic document types
(scans, Word, PDF, email, tables); they do not test the beyond-window regime.
The baseline arms below exist to make that explicit.

One image in the employment matter (the payslip photo) initially failed transcription because the vision model placed trailing text blocks inside the JSON `tables` array and the parser rejected the page. `_parse_transcription` now relocates such blocks; the document was re-ingested with `--replace` and all three corpora reached health grade `ok` before answering.

## Scoring pass 4 (2026-10-21): map `needs_value` correction

The headline, per-type, tier and miss tables above are pass 3. Pass 4 fixes a
scoring defect that affected every arm in this document: `map.json` marked
every question `needs_value: true`, which `fan_out` treats as a typed value
field and so rewrites any negative answer ("No", "Not applicable", "unknown")
to the not-found phrase before scoring. An explicit "No" to a Yes/No question
therefore scored as an abstention and failed against a gold of "No". The two
DocDB misses read above as *Not found in matter corpus* where the gold is *No*
were "No" answers. Pass 4 sets `needs_value: false` for every benchmark
question (pass-3 scores are kept in `score_pass3/`). No answer changed.

| Matter | Pass 3 | Pass 4 | Substantive | Judge-scored | High tier acc. | Auto-accept |
|---|---|---|---|---|---|---|
| Employment law | 109/111 | 108/111 (97.3%) | 91/94 | 77 | 98.2% | 50.5% |
| Family law | 83/85 | 82/85 (96.5%) | 69/72 | 54 | 100.0% | 14.1% |
| Immigration law | 84/90 | 85/90 (94.4%) | 71/76 | 60 | 90.9% | 24.4% |
| **All** | 276/286 | **275/286 (96.2%)** | 231/242 | 191 | – | – |

Pass 4 also exposes judge variance. Re-running the judge over unchanged answers
moved three DocDB items from OK to miss (`mf_two_dates`, `combined_super_balance`,
`s56_response_attachment_count`: each states the gold value and adds detail; the
judge returned `unavailable` or rejected) and one from miss to OK. Every DocDB
miss on pass 4 is a correct value with added context that the judge did not
accept. Treat differences of fewer than about three questions on 286 as judge
noise, on top of the binomial error of about three.

## Baseline arms (2026-10-21)

Three baselines were run on the same 286 questions, the same `corpus.db`
retained text, the same models (`claude-sonnet-5-5` root, `claude-haiku-5-5`
sub/judge), the same per-query caps, and scored with the same `rnsr regress`
protocol and `golden.json` as the DocDB run above. Each arm presents the corpus
differently and nothing else changes:

| Arm | Corpus presentation | Loop | Claim review |
|---|---|---|---|
| Whole matter in context | every document in the prompt; one call per question; corpus as a cached prefix | none | none |
| Classic | one flat string `context` in the REPL (the RLM-classic environment) | same root loop, batch 8 | not applicable |
| Direct corpus interaction | one `.txt` per document under `CORPUS_DIR` with `ls` / `grep` (≈ `rg -n`) / `read` (≈ `sed -n`) helpers (arXiv:2605.05242) | same root loop, batch 8 | not applicable |
| DocDB | typed tables, search ladder, annotations, claim review (the run above) | same root loop, batch 8 | enabled |

Runner: `rnsr/eval/matter_arms.py`; comparison: `rnsr/eval/matter_arms_report.py`;
artifacts under `runs/matters-bench-arms/<matter>/<arm>/` (git-ignored).

| Arm | Correct | Accuracy | Spend (3 matters) | Wall (slowest matter) |
|---|---|---|---|---|
| Whole matter in context | 278 / 286 | 97.2% | $1.15 | 1.3 min |
| Classic (flat string in REPL) | 280 / 286 | 97.9% | $2.78 | 2.4 min |
| Direct corpus interaction | 285 / 286 | 99.7% | $2.79 | 2.5 min |
| DocDB (October run, pass 4) | 275 / 286 | 96.2% | $34.33 | 9.1 min |

| Matter | Whole matter | Classic | DCI | DocDB |
|---|---|---|---|---|
| Employment law | 109 / 111 ($0.45) | 108 / 111 ($0.96) | 111 / 111 ($0.88) | 108 / 111 ($7.86) |
| Family law | 83 / 85 ($0.32) | 83 / 85 ($0.80) | 84 / 85 ($0.81) | 82 / 85 ($11.61) |
| Immigration law | 86 / 90 ($0.38) | 89 / 90 ($1.01) | 90 / 90 ($1.10) | 85 / 90 ($14.87) |

| Question type | Whole matter | Classic | DCI | DocDB |
|---|---|---|---|---|
| `value_lookup` | 102 / 104 | 103 / 104 | 104 / 104 | 103 / 104 |
| `existence` | 38 / 41 | 40 / 41 | 41 / 41 | 40 / 41 |
| `multi_fact` | 34 / 34 | 32 / 34 | 33 / 34 | 32 / 34 |
| `relationship` | 28 / 29 | 28 / 29 | 29 / 29 | 27 / 29 |
| `count_list` | 25 / 27 | 26 / 27 | 27 / 27 | 24 / 27 |
| `summary` | 19 / 19 | 19 / 19 | 19 / 19 | 18 / 19 |
| `unanswerable` | 19 / 19 | 19 / 19 | 19 / 19 | 19 / 19 |
| `quote` | 13 / 13 | 13 / 13 | 13 / 13 | 12 / 13 |

All numbers are scoring pass 4. Reading. The four arms are within ten
questions of each other on 286; binomial error is about three questions and
judge variance about three more, so the ordering is not reliable and no arm is
distinguishable from another on accuracy. They are distinguishable on cost by
12–30× and on wall time by 4–7×. On the dense question types the structure was
meant to serve (`count_list`, `relationship`), DocDB is not ahead: DCI is 27/27
on counts, DocDB 24/27. Every baseline miss is a correct answer with added
context that the judge rejected (e.g. `overtime_hours_total`: gold "208.0
hours", answer "208.0 hours. The underpayment calculation …", rejected). None
is a retrieval failure.

What this does and does not show. At 11–15k tokens per matter, the corpus fits
in one call, and putting it there is the cheapest and about the most accurate
option. DocDB's retrieval layer (tables, ladder, annotations) buys nothing here
and its claim-review loop accounts for most of the 30× cost. This is the result
the archived regime map predicted for within-window corpora. It says nothing
about collections that exceed a context window, about dense aggregation over
hundreds of rows, or about the evidence layer (source-bound calculations,
coordinate-checked quotations, trust tiers), which no baseline arm produces and
which the scorer does not measure. The matched comparison the paper asks for in
its evaluation agenda, at a scale where the arms can diverge, remains to be run.

Reproduce:

```bash
export RNSR_MAX_ROOT_ITERS=16 RNSR_MAX_WALL_S=300 RNSR_MAX_SPEND_USD=1 RNSR_RUN_SPEND_CEILING_USD=20
for arm in stuffing classic dci; do
  python -m rnsr.eval.matter_arms --matter-dir runs/matters-bench/$M --arm $arm \
    --out runs/matters-bench-arms/$M/$arm --batch-size 8 --concurrency 3
  rnsr regress --answers runs/matters-bench-arms/$M/$arm/answers_chunk1.csv \
    --golden runs/matters-bench/$M/golden.json --map runs/matters-bench/$M/map.json \
    --out runs/matters-bench-arms/$M/$arm/score
done
python -m rnsr.eval.matter_arms_report --docdb runs/matters-bench \
  --arms runs/matters-bench-arms --matters-src testMatter/matters \
  --out runs/matters-bench-arms/summary.json
```

## Four arms on the 999-document matter (2026-10-21)

The same four arms on Test Set 6, the vendor-built family-law matter used for
the archived corpus-scale result (`benchmark-history.md`): 1,004 files (467
PDF, 390 email, 65 Word, 30 text, 27 CSV, 22 PNG) across seven folders, and a
90-field verification key for the FCFCOA Initiating Application with planted
traps (a decoy separation date, a chronology with wrong dates, fields that must
be left blank). Documents are short: 999 ingested, 400k characters of retained
text (~100k tokens), largest document 3,180 characters, 30 extracted
transaction tables. Still inside one Sonnet 5.5 call.

Inputs (`testMatter/ts6_arms_prep.py` → `runs/ts6-arms/`): the 90 enriched
form questions as shipped, except that the one tool-specific evidence bullet
("use `search()` … and SQL over tables") is replaced by a tool-neutral one, so
all four arms, including the fresh DocDB run, see identical task text. Roles,
evidence rules and the answer format are unchanged. One generic `field_context`
for the judge; `needs_value: false`. These questions are the development
fixture the archived 90/90 was tuned on, which favours DocDB if anything.

Fresh fast-tier ingest under `ded0c42`+ (999 docs, 22 scans transcribed, health
`ok`, no parse failures). Same caps as above; run ceiling $40.

| Arm | Correct | Accuracy | Spend | Requests | Wall |
|---|---|---|---|---|---|
| Whole matter in context | 78 / 90 | 86.7% | $2.87 | 90 | 1.1 min |
| Classic (flat string in REPL) | 81 / 90 | 90.0% | $1.09 | 109 | 1.8 min |
| Direct corpus interaction | 78 / 90 | 86.7% | $1.03 | 92 | 1.5 min |
| DocDB (claim review on) | 80 / 90 | 88.9% | $1.75 | 198 | 4.2 min |

DocDB spend includes the 22 scan transcriptions; its judge pass returned 79 on
a first run and 80 on a rerun with no answer changed. Trust tiers: 34 high /
55 medium / 1 low; high-tier accuracy 85.3%, auto-accept 37.8%, review recall
50%. One claim-review halt (`review_stalled`).

| Part | Whole matter | Classic | DCI | DocDB |
|---|---|---|---|---|
| Part A (orders sought, lawyer details) | 11 / 13 | 13 / 13 | 12 / 13 | 12 / 13 |
| Part B (parties) | 25 / 25 | 24 / 25 | 23 / 25 | 24 / 25 |
| Part D (children) | 19 / 22 | 22 / 22 | 21 / 22 | 22 / 22 |
| Part F (existing orders, cases) | 6 / 9 | 6 / 9 | 6 / 9 | 6 / 9 |
| Part H (de facto relationship) | 3 / 7 | 2 / 7 | 2 / 7 | 2 / 7 |
| Cover, C, E, G, I–K | 14 / 14 | 14 / 14 | 14 / 14 | 14 / 14 |

Misses. Five are shared by all four arms: four Part H items where the gold packs
the sub-field detail into the Yes/No ("Yes — Victoria", "Yes — three children")
and every arm answered "Yes", and one two-date field (`50c`) where every arm
returned one of the two dates. One more (`49a`, "Yes — one") is missed by three
arms the same way. These are answer-format mismatches shared across arms, not
retrieval. The arm-specific misses are small in number: whole-matter gave a
wrong postcode (3058 for 3000), the applicant's phone numbers for the lawyer's,
"Grace" for "Thea Grace" and a role for a name (four substantive errors, the
only ones in the table that read as lost-in-context); DCI and DocDB each
dropped "Melbourne" from an address and "Registered nurse" from an occupation;
classic missed one language field. There is no DocDB-only miss.

Reading. At 999 documents and ~100k tokens the four arms remain within noise
of one another (binomial error ≈ 3 on 90, judge variance ≈ 1). The planted
traps (decoy separation date, wrong chronology) were handled by every arm.
DocDB is now cheaper than whole-matter prompting, because the baseline pays
~100k input tokens per question and the REPL arms do not, but it is 2–4× slower
and not more accurate. Whole-matter prompting shows the first substantive
errors of the kind long-context evaluation predicts, at a rate (4/90) too low
to establish. Document *count* is not the dimension that separates these
designs; token volume and answer-support density are, and neither corpus here
reaches them: the average document is 400 characters and the form asks for
lookups, not aggregates over the 30 transaction tables. The archived 90/90 on
this matter was a loose-scorer, multi-run result on tuned questions; under the
current scorer, single-pass, the comparable figure is 80/90.

Reproduce:

```bash
python testMatter/ts6_arms_prep.py
export RNSR_MAX_ROOT_ITERS=16 RNSR_MAX_WALL_S=300 RNSR_MAX_SPEND_USD=1 RNSR_RUN_SPEND_CEILING_USD=40
rnsr answer-csv --corpus "testMatter/Test Set 6 - Initiating Application (999 Documents)/Matter Documents" \
  --questions runs/ts6-arms/questions.csv --output runs/ts6-arms/docdb/out \
  --work-dir runs/ts6-arms/work --fast-ingest --batch-size 8 --concurrency 3 --max-error-rate 1
for arm in stuffing classic dci; do
  python -m rnsr.eval.matter_arms --matter-dir runs/ts6-arms --arm $arm --out runs/ts6-arms/$arm \
    --questions runs/ts6-arms/questions.csv --id-col qid --batch-size 8 --concurrency 3
done
# score each arm: rnsr regress --answers <arm>/answers_chunk1.csv --golden runs/ts6-arms/golden.json \
#   --map runs/ts6-arms/map.json --out <arm>/score   (re-key docdb answers_status.csv row -> qid first)
```

## Four arms on a long, dense synthetic matter (2026-10-21)

The experiment the two comparisons above could not reach: a corpus that is
long rather than numerous, with aggregation questions over tables.
`rnsr.eval.datasets.matter_dense` (seed 11 → `testMatter/matters/dense_corvid_meridian`)
writes a 36-month, three-project professional-services engagement as 566
documents: 375 multi-line tax invoices (consultant × role × hours × rate ×
amount, Subtotal, GST, TOTAL DUE rows), 15 credit notes cancelling and
reissuing invoices, 34 monthly remittance advices, 8 pro-forma quotes shaped
like invoices, 5 dispute letters, a demand letter stating a stale outstanding
figure, a SOW rate variation with two later invoices that wrongly keep the old
rate, and 108 multi-page status reports whose "indicative" hours disagree with
the invoices. Retained text 2.23M characters (~560–620k tokens); 3,772 chunks;
510 tables extracted by Docling (all 375 invoice tables, typed `hours`, `rate`,
`amount`), validation 100%. The Subtotal/GST/TOTAL rows were extracted as
ordinary data rows. 38 questions with exact golds computed from the same facts
(15 aggregation, 3 set logic, 3 join, 4 cross-doc, 6 lookup, 4 timeline, 1
existence, 2 unanswerable); two golds were independently re-derived from the
written PDFs before the run.

Caps: 24 root iterations, 600 s, $2 per query (scaled for batch 8), run ceiling
$40. Same models and scorer as above. Whole-matter arm at concurrency 1 (one
~600k-token request per question, cached prefix).

| Arm | Correct | Accuracy | Spend | Requests | Wall |
|---|---|---|---|---|---|
| Whole matter in context | 24 / 38 | 63.2% | $5.14 | 38 | 3.4 min |
| Classic (flat string in REPL) | 37 / 38 | 97.4% | $0.63 | 60 | 1.4 min |
| Direct corpus interaction | 37 / 38 | 97.4% | $0.59 | 63 | 1.4 min |
| DocDB (claim review on) | 34 / 38 | 89.5% | $1.79 | 2,947 | 6.7 min |

| Question type | Whole matter | Classic | DCI | DocDB |
|---|---|---|---|---|
| `aggregation` (15) | 4 | 15 | 15 | 11 |
| `set_logic` (3) | 1 | 3 | 3 | 3 |
| `join` (3) | 3 | 3 | 3 | 3 |
| `cross_doc` (4) | 3 | 4 | 4 | 4 |
| `value_lookup` (6) | 6 | 6 | 6 | 6 |
| `timeline` (4) | 4 | 4 | 4 | 4 |
| `existence` (1) | 1 | 1 | 1 | 1 |
| `unanswerable` (2) | 2 | 1 | 1 | 2 |

Misses. Whole-matter prompting failed 11 of 15 aggregations and 2 of 3 set-logic
items: nine answers begin "I need to list all…" / "I can't produce a reliable
total" and never reach a number, one gives a wrong total ($381,825.00 for the
disputed invoices against $305,841.25), and one correct answer with a trailing
explanation was rejected by the judge. Classic and DCI share one miss,
`abs_delta`, where both answered "0" invoices for the nonexistent project and
the gold expects a decline; "0" is defensible. DocDB's four misses are all
correct values followed by an itemisation ("$1,068,801.25 (15 credit notes:
CN-115 77,976.25, …)") that the judge rejected; substantively DocDB is 38/38.

How the arms got there. Classic and DCI did not retrieve; they built the table
themselves. The classic trajectory for the aggregation batch is fifteen cells:
split the string on document headers, regex every invoice into number, date,
project, line rows and TOTAL DUE, parse credit notes and remittances, reconcile
remittance totals against invoice totals (all matched), then compute every
aggregate in Python and quote a line from a source document per answer. DocDB
used SQL over the invoice tables in 12 of 125 cells, `search()` in 41, and
spent 45 provider batches embedding all 3,772 chunks for search rung 4 (that is
where 2,900 of its 2,947 requests went). All five DocDB batches ended in
`claim_review_halted`: the reviewer could not certify a sum over hundreds of
invoices from quotations, the circuit breaker stopped the loop, and every
answer was published as medium or low tier (0 high / 12 medium / 26 low).
Source-bound calculations (`FINAL_CALC`) were not used.

Reading. This is the first corpus on which the arms separate, and they
separate on the axis the paper predicted: whole-corpus prompting cannot
aggregate over 360 invoices at ~600k tokens (4/15), and every arm with a
Python REPL can (15/15). But the REPL arms *without* tables matched the one
*with* them, at a third of the cost and a fifth of the wall time, because the
controller wrote its own parser over the raw text. That works here because the
corpus is synthetic and every invoice shares one layout. On real matters where
invoice layouts vary by vendor and period, layout-aware extraction at ingest is
the step regex cannot replicate; that is the untested hypothesis on which
DocDB's retrieval layer now rests. Two further findings are independent of the
synthetic caveat: the search ladder's embedding pass is a fixed cost that
bought nothing on a corpus the model could parse directly, and the claim-review
contract has no way to certify an aggregate, so on exactly the workload where
the answers were right and hard, the trust tiers said "do not trust".

Reproduce:

```bash
python -m rnsr.eval.datasets.matter_dense --out testMatter/matters/dense_corvid_meridian --seed 11
M=testMatter/matters/dense_corvid_meridian
rnsr ingest $M/documents/* -o runs/dense-arms/work/corpora/corpus.db --report runs/dense-arms/ingest.json   # ~18 min
export RNSR_MAX_ROOT_ITERS=24 RNSR_MAX_WALL_S=600 RNSR_MAX_SPEND_USD=2 RNSR_RUN_SPEND_CEILING_USD=40
rnsr answer-csv --corpus $M/documents --questions $M/questions.csv --output runs/dense-arms/docdb/out \
  --work-dir runs/dense-arms/work --batch-size 8 --concurrency 3 --max-error-rate 1
for arm in stuffing classic dci; do
  python -m rnsr.eval.matter_arms --matter-dir runs/dense-arms --arm $arm --out runs/dense-arms/$arm \
    --questions $M/questions.csv --batch-size 8 --concurrency $([ $arm = stuffing ] && echo 1 || echo 3)
done
# score each arm with rnsr regress --golden $M/golden.json --map $M/map.json (re-key docdb status row -> field_id)
```

## Run status and cost

| Matter | Status counts | Not-found answers | Provider requests | Spend | Rate-limit hits | Wall |
|---|---|---|---|---|---|---|
| Employment law — Reyes v Harbourline Freight | {"final": 111} | 0 | 176 | $7.86 | 0 | 4.1 min |
| Family law — Okafor & Nguyen | {"final": 82, "budget_exhausted": 3} | 3 | 312 | $11.61 | 0 | 6.0 min |
| Immigration law — Tanaka Partner visa | {"final": 90} | 0 | 315 | $14.87 | 0 | 9.1 min |

Spend is computed from `rnsr/llm/cost.py` list prices. Haiku 5.5 is priced at its ≤100K-token prompt tier, so long-prompt sub-calls may be under-counted.

## First attempt (aborted) and fixes applied

The first run of this benchmark was aborted when every matter hit a $60 spend ceiling with most questions unanswered. Root causes, all specific to the Claude 5.5 models and fixed in this change set:

1. **Sampling parameters rejected.** `claude-sonnet-5-5` / `claude-haiku-5-5` return 400 for any `temperature` other than the default. `AnthropicClient` now sends `temperature` only to the ≤4.6 generations.
2. **Adaptive thinking consumed `max_tokens`.** Both 5.5 models think by default and thinking tokens count against `max_tokens`, so terse calls (the YES/NO judge at 8 tokens, recovery at 64, coverage check at 100) returned empty text, and the strict-JSON claim review at 2,048 tokens was truncated. The client now turns pre-response thinking off (`between_tools` on Sonnet 5.5, `disabled` on Haiku 5.5; each model rejects the other form).
3. **Claim-review truncation looped.** A batch of eight fields with requirement checks needs ~3k output tokens. With the reply truncated, `_parse_reply` raised `JSONDecodeError` (375 of 440 reviews), the unresolved review sent the draft back to the root, and the cycle repeated until the budget ran out: single batches reached 145 root iterations and 10M input tokens. `MAX_OUTPUT_TOKENS` is now 8,192.
4. **Scan transcription parser.** The vision model placed trailing text blocks inside the `tables` array for one payslip photo; the parser now relocates them instead of discarding the page.

| Matter | Attempt-1 status | Attempt-1 spend | Attempt-1 wall |
|---|---|---|---|
| Employment law — Reyes v Harbourline Freight | {"final": 64, "error": 47} | $60.29 | 42 min |
| Family law — Okafor & Nguyen | {"final": 32, "error": 53} | $60.69 | 33 min |
| Immigration law — Tanaka Partner visa | {"final": 24, "error": 66} | $60.31 | 34 min |

Attempt-1 spend totalled $181.29 for 120 final answers; the clean rerun below cost $34.33 for all 286 questions. The `ReviewGate` design never stops a query on review count alone, so a reviewer that fails persistently is bounded only by the budget caps; that remains worth a dedicated guard.

## Misses

Every question the scorer marked wrong, with the golden answer, the system's answer and the judge's status/reason. Where the system abstained on an answerable item the answer reads *Not found in matter corpus*.

### Employment law — Reyes v Harbourline Freight — 2 miss(es) of 111

| id | type | tier | question | golden | answer | judge |
|---|---|---|---|---|---|---|
| `attended_toolbox` | existence | medium | Does Aisha Rahman say Marisol Reyes attended the 8 April 2026 toolbox meeting? | Yes | Yes. Aisha Rahman says she attended the 8 April 2026 toolbox meeting, that Marisol arrived at about 7:04 am, and that she saw her sign the attendance sheet. | rejected: The question asks whether Aisha Rahman says Marisol Reyes attended the meeting. The reference answer is a plain 'Yes'. The candidate's leading 'Yes' matches that, but the candidate then attributes a different claim to Ai |
| `leave_weeks_count` | count_list | medium | How many weeks of annual leave are included in the leave loading claim? | 5 weeks | 5 weeks of annual leave: 2 weeks (9 to 20 December 2024) plus 3 weeks (5 to 23 January 2026). | unavailable: Judge response was not a valid explained verdict. |

### Family law — Okafor & Nguyen — 2 miss(es) of 85

| id | type | tier | question | golden | answer | judge |
|---|---|---|---|---|---|---|
| `lan_nguyen_relationship` | relationship | medium | What is Lan Nguyen's relationship to Zara and Lachlan? | Trung's mother | Lan Nguyen is Trung Nguyen's mother, i.e. the children's paternal grandmother. | unavailable: The reference states Lan Nguyen is 'Trung's mother' and the candidate states she is Trung Nguyen's mother, which matches. However, the question asks about her relationship to Zara and Lachlan, and the candidate's additio |
| `biscuit_owner` | relationship | medium | In the family report, whose dog is Biscuit? | The father's dog | Biscuit (a beagle) is the father's (Trung Nguyen's) dog. | unavailable: The reference answer "The father's dog" identifies the owner only as a role, while the candidate names the father as Trung Nguyen. The field context says a role plus the person's name is acceptable and that role aliases  |

### Immigration law — Tanaka Partner visa — 6 miss(es) of 90

| id | type | tier | question | golden | answer | judge |
|---|---|---|---|---|---|---|
| `sponsor_police_check_provided` | existence | high | Has Eleanor Marsh provided her own police check for the application? | No | Not found in matter corpus | rejected: The reference answer is 'No', a definite answer to the Yes/No existence question. The candidate says 'Not found in matter corpus', which the field context states is correct only when the reference itself says the matter  |
| `form_888_count` | count_list | medium | How many Form 888 declarations were lodged for Haruto Tanaka? | 2 | Two Form 888 declarations (from Daniel Okafor and Mei-Ling Chen) | rejected: The question asks for a count of Form 888 declarations for Haruto Tanaka. The reference answer is 2. The candidate's count is also 2 ("Two"), and it lists two names (Daniel Okafor and Mei-Ling Chen). However, the candida |
| `us_police_certificate_provided` | existence | medium | Has Haruto Tanaka provided a police certificate from the United States? | No | Not found in matter corpus | rejected: The question is a Yes/No existence question and the reference answer is 'No', meaning the matter corpus does not show a US police certificate was provided. The candidate says 'Not found in matter corpus', which the field |
| `summary_art_grounds` | summary | medium | Summarise the grounds of review in Haruto Tanaka's ART application. | That the delegate erred in finding the relationship was not genuine and continuing, failed to give weight to evidence that the parties began staying together from 1 December 2022 and to the explanation for the difference in their statements, and did not properly consider the s56 response, the two Form 888 declarations or the February 2025 travel to Japan. | The ART application (lodged 6 October 2025, case 2025/07731) challenges the 12 September 2025 refusal of the Partner (Temporary) visa subclass 820 on three grounds: (1) the delegate erred in finding the relationship is not genuine and continuing, as the couple have lived together since early 2023 and share a lease, joint account and household bills; (2) the delegate failed to give adequate weight to evidence that they began staying together from 1 December 2022 and to the explanation for the difference in their statements; (3) the delegate did not properly consider the s56 response of 8 January 2025, the two Form 888 declarations, or the February 2025 travel to Japan. | rejected: The candidate covers all three of the reference's grounds (relationship not genuine, failure to weigh the 1 December 2022 cohabitation evidence and statement-difference explanation, failure to consider the s56 response,  |
| `passport_number_in_s56` | value_lookup | high | What passport number did the Department's section 56 request letter record for Haruto Tanaka? | TR9004172 | TR9004172 (the Department's request of 12 December 2024 recorded this; the correct number is TR9004127) | rejected: The candidate gives two conflicting passport numbers: it first states TR9004172, matching the reference, then says the correct number is TR9004127. The candidate ends by asserting a different value as correct, so it cont |
| `quote_refusal_review_rights` | quote | medium | What does the refusal letter say about when Haruto Tanaka must apply for review? | You must apply within 28 days of being notified of this decision, that is by 9 October 2025 | The refusal letter (dated 11 September 2025) says Haruto Tanaka may apply to the Administrative Review Tribunal and must apply within 28 days of being notified of the decision, that is by 9 October 2025 (lodged with the correct fee). A later corrigendum (22 September 2025) corrected the notification date to 12 September 2025, making the last day to apply 10 October 2025. | rejected: The candidate reproduces the reference's quoted substance (apply within 28 days of notification, by 9 October 2025) but then states a superseding corrigendum changed the deadline to 10 October 2025. The field context all |

### Informal reading of the misses

This is the author's reading of the ten misses, not an adjudication, and it is not folded into any number above.

- Six state the reference fact and add further correct context that the judge treated as a conflict or could not resolve: `attended_toolbox`, `leave_weeks_count` (the judge emitted two JSON objects and the reply was discarded), `lan_nguyen_relationship`, `biscuit_owner`, `form_888_count`, `passport_number_in_s56` (gives the requested wrong-in-source number and notes the correct one).
- One quotes the reference text and appends the corrigendum that superseded it (`quote_refusal_review_rights`); whether the appended context is a defect depends on the use.
- Two answer *Not found in matter corpus* where the gold is *No* (`sponsor_police_check_provided`, `us_police_certificate_provided`): the corpus contains no such document, so the abstention is defensible, but the question asked for a Yes/No.
- One contains a characterisation that conflicts with the gold (`summary_art_grounds` says the couple lived together "since early 2023" where the gold's evidence is from 1 December 2022); this is the one miss that reads as a substantive error.

## Reproduce

```bash
# per matter (M = employment_law_reyes_v_harbourline | family_law_okafor_nguyen | immigration_law_tanaka_partner_visa)
rnsr ingest testMatter/matters/$M/documents/* -o runs/matters-bench/$M/work/corpora/corpus.db \
  --report runs/matters-bench/$M/ingest.json
RNSR_MAX_ROOT_ITERS=16 RNSR_MAX_WALL_S=300 RNSR_MAX_SPEND_USD=1 RNSR_RUN_SPEND_CEILING_USD=30 \
rnsr answer-csv --corpus testMatter/matters/$M/documents \
  --questions runs/matters-bench/$M/questions.csv --output runs/matters-bench/$M/out \
  --work-dir runs/matters-bench/$M/work --concurrency 3 --batch-size 8
rnsr regress --answers runs/matters-bench/$M/score/answers_chunk1.csv \
  --golden runs/matters-bench/$M/golden.json --map runs/matters-bench/$M/map.json \
  --out runs/matters-bench/$M/score
```

`questions.csv`, `golden.json` and `map.json` are derived from each matter's `qa.json`: the question text and id become the question row, the golden value is the gold answer (empty, with the explanation as a note, for `unanswerable` items) and the map is a standalone item per question so `regress` can key answers by qa id. `answers_status.csv` is re-keyed from `q###` to the qa id before scoring so trust tiers join correctly.
