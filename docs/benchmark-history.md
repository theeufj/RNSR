# Historical benchmark notes

Archived from the README on 1 October 2026. These measurements were reported
before the stricter scalar/text scorer and durable ingestion/trust fixes.
They are historical results, not certification of the current implementation.

The original scorer could accept mismatched units, polarity and the first
number in a prose answer. [The scoring audit](benchmarks/scoring-audit.json)
records which historical answers remain deterministic agreements and which
require independent adjudication. An unresolved answer is not automatically
wrong, and deterministic agreement is not corrected benchmark accuracy.

The original go/no-go result was **8/9 DocDB versus 7/9 classic**, not a large
comparison. Form conventions were refined after observed misses; those results
are development-fixture results. Baseline comparisons apply to the particular
implementations and settings tested, not every retrieval system. Claims of
unique capabilities were not independently established.

Current methodology and the held-out protocol are in
[evaluation.md](evaluation.md). Original local predictions are preserved.

## Archived observations


**FinanceBench** (numeric needles in real SEC filings; 114 reachable
questions, 36 of 150 excluded as undownloadable):

| Metric | Result |
|---|---|
| Overall accuracy | **88.6%** |
| Numeric questions | **97.4%** (37/38) |
| Textual questions | 84.2% |
| Cost per question | p50 $0.25 · p95 $0.65 |
| Terminal status | 112 final · 2 recovered · 0 errors |

Published GPT-4-with-retrieval baselines on this benchmark sit near 50%.

**Go/no-go gate** (spec §8: DocDB vs RLM-classic — same loop and budgets,
flat-string environment — on a numeric-needle set with exact golds): **PASS**.
DocDB 89% vs classic 78% accuracy, at lower cost (p50 $0.055 vs $0.075;
p95 $0.106 vs $0.225 — classic re-reads its giant context every turn).

**Legal benchmarks** (30-question samples; scoring as above; ContractNLI
and LegalBench items carry no document, so classic and DocDB environments
are identical for them by construction):

| Benchmark | DocDB | RLM-classic (same questions) |
|---|---|---|
| CUAD clause extraction | **90%** (27/30) | 83% (25/30) |
| — of which absent-clause questions | **20/23** | 18/23 |
| ContractNLI (3-way NLI, clause-scale) | 67% | — (identical env) |
| LegalBench (4-task slice) | 84% | — (identical env) |

On CUAD the A/B split is informative: extraction parity (7/7 both) —
sampled contracts are small (median 23k chars) and fit one context
window — but DocDB is better at *absent* clauses (all four disagreements
were absent-clause questions, 3–1 DocDB), i.e. it resists finding
plausible-but-nonexistent clauses. Classic is cheaper on sub-window
documents ($0.10 vs $0.19/question). On a long-contract cut (230–300k-char
agreements, ~60–75k tokens): **no accuracy advantage** — classic 7/10,
DocDB 6/10 (n too small to rank either way; two misses shared, incl. one
contestable gold), classic slightly cheaper, DocDB 2× faster (p50 114s vs
230s — classic re-reads ~75k tokens every turn). Note CUAD cannot test the beyond-window regime for modern
models: its largest contract (~190k tokens) still fits Claude's window —
beyond-window means multi-document corpora, covered by the needle gate
and FinanceBench. These are QA-protocol numbers on samples, not the
official CUAD span-AUPR metric — not leaderboard-comparable.

**OOLONG** (Phase B harness acceptance; `oolongbench/oolong-synth`
trec_coarse, 50 questions, 1k–65k-token contexts): RLM-classic scores
**60%/58%** across two runs, in the ballpark the RLM paper reports —
closing the reproduction gate. DocDB progressed 56% → 62% → **64%**
through targeted fixes (lines-table for semantic_annotate, per-item
labeling rubric, majority voting); the residual misses are dominated by
per-item label ambiguity that OOLONG's aggregation amplifies (a 97.8%
per-line labeler still miscounts), plus a measured share of contestable
golds. Flat-text costs ~1.5× classic: when the whole context fits in one
window and carries no structure, structure buys nothing (§1.3) — the
harness exposes classic mode as a flag for exactly that regime. Bonus
finding, measured live: annotation columns persist in the artifact, so a
second run over the same corpora answered at **half cost with a median of
1 sub-call per question** — semantic work amortizes across queries (§4.1).

**Real-filing ingestion health**: 87% table-validation pass rate on two 3M
10-Ks (412 pages, 228 detected tables), against the spec's 70% stop threshold.
Scoring: exact string/numeric match first; sub-model equivalence judge only
on string failure. Answers carry code-verified quotes (§6) — supporting
quotes are string-matched against retained source text, with failures fed
back into the loop.

**Matter files** (the multi-document legal regime: a synthetic
commercial-dispute matter as real PDFs — MSA + overriding amendments,
~40 invoices, breach/cure correspondence, a superseded draft with wrong
numbers, bulk file notes; 79 docs, ~213k tokens — beyond one context
window, so prompt-stuffing is disqualified at the door):

Across **three independently generated matters** (36 questions/system;
by question class):

| System | Total | Aggregations (9) | Timelines (6) | Other (21) | Cost/q |
|---|---|---|---|---|---|
| **DocDB** | **34/36 (94%)** | 7/9 | **6/6** | **21/21** | $0.17 |
| RLM-classic | 32/36 (89%) | 7/9 | 4/6 | **21/21** | $0.07 |
| Vector-RAG (gemini-embedding-2) | 27/36 (75%) | **0/9** | 6/6 | **18/18** | $0.005 |
| Graph-RAG (lean reimpl., entity graph + community summaries) | 30/36 (83%) | 3/9 | 6/6 | **21/21** | $0.17 incl. index |
| BM25-RAG / +LLM reranker (1 matter) | 7/12 both | 0/3 | — | — | $0.006/$0.018 |

False-positive rate (confident answer on an absent/not-applicable item) is
now a first-class metric, reported by `rnsr regress` next to accuracy and
gated by `--max-false-positive-rate`. Matter-gen already plants `absent`
and `cross-doc` items; needle-gen plants `absent` and `superseded`. A
weekly regression fails if the rate exceeds 15%.

RAG aced the retrieval-friendly v1 (12/12 at 1/30th the cost — stated
plainly) and collapsed to 7/12 on the realistic v2: its five misses are
structural, not marginal — every invoice aggregation (the set exceeds any
top-k), the amendment-override question (it confidently answered the
*original* MSA terms, never seeing the amendment), and a timeline it
failed to retrieve. The RAG ladder decomposes the failure honestly:
an LLM reranker over a top-60 lexical pool changed nothing (identical
7/12 at 3× cost), but semantic embeddings (gemini-embedding-2) fixed both
retrieval-quality misses — the amendment-override and the timeline —
reaching 9/12. What NO retrieval flavor fixed is the aggregation class:
39 invoices cannot occupy 12 excerpt slots regardless of ranking quality.
That is the architectural boundary of single-shot retrieval:
vector-RAG missed ALL NINE aggregations across three seeds while scoring
perfectly on everything else. GraphRAG — the incumbent answer to exactly
this criticism — raises the ceiling (30/36; its community summaries
sometimes carry the needed totals, 3/9) but every one of its six misses
is still an aggregation: a knowledge graph's summaries contain whatever
the index-time LLM happened to compile, not computation. The retrieval
ladder climbs — BM25 7 → rerank 7 → vector 9 → graph 10 per matter —
and every rung stops at the same wall, the one SQL walks through. Honest ledger for the leaders too: DocDB's
two misses are SQL slips (a unit error; a double-count from summing
invoice tables including their TOTAL rows — a fair real-world trap), and
classic twice answered a timeline with "28 days from the notice" instead
of computing the date. Classic's single miss (v1) was a silent arithmetic
slip while aggregating by reading — the failure class SQL makes a
non-event. DocDB was perfect across both versions with every answer
carrying verified quotes.


**Batched answering** (many related questions over one corpus — the
form-fill regime): `answer-csv` groups consecutive questions into shared
RLM loops (`--batch-size`, default 8) submitting via `FINAL_BATCH`; one
exploration of the corpus serves the whole group, and any question a
batch fails to answer is retried in its own loop automatically. Measured
on a real family-law matter (49 form fields, 11 documents, scored against
golden answers):

| Mode | Correct | Wall time | LLM spend |
|---|---|---|---|
| **Batched (8/loop)** | **39/49** | **7m 12s** | **$5.46** |
| One loop per question | 30/49 | 26m 55s | $23.11 |

False-positive rate (confident answer on an absent/not-applicable field) is
reported beside accuracy by `rnsr regress` and gated by
`--max-false-positive-rate` (weekly CI fails above 15%).

3.7× faster and 4.2× cheaper — and *more accurate*, for a structural
reason: forms carry mutually-exclusive field groups (radio buttons,
checkbox families), and a solo loop seeing only its own field happily
answers "yes" to every sibling option. A batched loop sees the whole
group in one context and picks one. When positive query caps are configured,
they scale sub-linearly with batch size (each extra question adds half a
single question's caps). The default zero caps remain unlimited for batches.

With enriched questions (`rnsr build-questions`: group collapse, role
maps, evidence rules, per-field answer shapes) the same matter reaches
**49/49**, holding across repeated fresh runs at ~2.6–2.9 min and ~$1.50
per full form fill. Two mechanical guards keep it there, neither of them
a request to the model: a **negative-answer audit** re-checks every
No/unknown/NOT_FOUND against the corpus by FTS phrase probe before
FINAL_BATCH is accepted (lazy loops declared documented facts missing),
and **quotes are mandatory** for value-bearing answers, verified against
source text (a run invented an address that appears in no document).
Per-field accuracy across nine fresh runs is 98.8%, with roughly two runs
in three perfect end-to-end — `--consensus 2` is the lever when every run
must be.

**Corpus scale** (the same regime, 90× the documents): a vendor
999-document family-law matter — 467 PDFs, 390 emails, 64 Word files, 30
text, 26 CSV across seven folders — with a 90-question verification key
for the FCFCOA Initiating Application, built with deliberate traps (decoy
separation date, a chronology document whose dates are wrong, questions
that must be left blank):

| Metric | Result |
|---|---|
| Questions correct | **90/90** |
| False-positive rate (absent / leave-blank items) | gated by `--max-false-positive-rate` (CI 15%) |
| Ingest (977 readable files, fast text tier) | **1.3 s**, zero parse failures |
| Wall time per 90-question run | 7–15 min |
| LLM spend per run | $5–9 |

Retrieval did not degrade with corpus size: the search ladder found
needles like a s 60J advice note buried in one FDR intake record among
999 files. What grew was exploration per question, not the cost of
having more documents. The instructive trap is the chronology: on the
first run the model trusted its tidy (wrong) summary table; once told
that internal chronologies are secondary evidence, it preferred the
letter where the *opposing* solicitor accepted the true date in writing
— which is what a careful human does.

**Fix-and-confirm cycle** (post three-seed autopsy): the two DocDB SQL
slips were converted into prompt disciplines (trust document units/headers;
sum line items OR total rows, never both; reconcile computed aggregates
against document-stated figures; compute calendar dates). Retested on the
originating questions: all six misses across systems converted. Confirmed
on a fresh unseen matter (seed 8): **DocDB 12/12**; classic 11/12 — its
remaining miss answered $0.00 for a $2.4M invoice total, the silent
grep-aggregation failure again. The disciplines generalize for the
architecture that computes; they cannot rescue the one that greps.

**The honest regime map**, consistent across every controlled A/B:

| Regime | Verdict |
|---|---|
| Multi-document / beyond-window corpora (matter files) | DocDB 24/24; classic strong but slips silently on aggregation; RAG structurally fails aggregation/override/absence |
| Single documents fitting one context window (incl. 75k-token contracts) | **No accuracy advantage.** Classic somewhat cheaper; DocDB ~2× faster at depth |
| Absent-needle questions ("no such clause") | DocDB's verification discipline resists confabulation |
| Many questions per document | DocDB amortizes: ingest + annotations pay once |
| Any regime | Only DocDB returns code-verified quotes with char offsets |

## When to use it (legal decision guide)

**Just load the document into model context** when all of these hold: a
single document under ~150k tokens; one-off questions; lookup/reading
answers rather than computation; no independently verifiable citation
required. We measured accuracy parity there, and stuffing is cheaper.

**Use DocDB** when any of these hold — each measured in the results above:

| Trigger | Measured basis |
|---|---|
| Multi-document matter / exceeds the context window | Stuffing disqualified at ~200k tokens; a modest matter file is already there |
| Answers computed over sets (invoice totals, counts, chronologies) | 0/3 for every RAG flavor; silent arithmetic slip for the flat-string loop; SQL makes it exact |
| Absence must be provable ("no such clause/guarantee") | Retrieval can't tell not-found from not-there; DocDB enumerates |
| Superseded versions in the file (drafts, amendments) | RAG confidently returned pre-amendment terms; DocDB date-orders the file |
| Working sessions: many questions per matter | Ingest + annotations amortize — second pass at half cost, median 1 model call; batched loops answer 8 questions per exploration (3.7× faster, 4.2× cheaper, more accurate on form-field groups) |
| Citations that survive scrutiny | Code-verified quotes with character offsets — unique to DocDB |

One line: **read a document → context window; interrogate a matter → DocDB.**

