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

One image in the employment matter (the payslip photo) initially failed transcription because the vision model placed trailing text blocks inside the JSON `tables` array and the parser rejected the page. `_parse_transcription` now relocates such blocks; the document was re-ingested with `--replace` and all three corpora reached health grade `ok` before answering.

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
