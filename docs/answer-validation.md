# Answer validation

DocDB answers now receive semantic support review before acceptance by default
(`RNSR_CLAIM_REVIEW_ENABLED=true`). Exact quotes prove that text exists; review
checks whether it supports the requested person, period, metric, relationship,
and category. Review uses the task and retained evidence, never evaluation gold.

An unsupported draft returns to the root with a specific gap to resolve. An
unchanged disputed candidate can receive independent adjudication using the root
model. Adjudication must resolve the first reviewer's specific objection using
real quotations; a second approval alone cannot clear it. Question-derived checks
distinguish price restrictions from royalty terms, unlimited rights from fee
waivers, and directional financial answers from requests for precise amounts.
A failed reviewer call does not certify support. Changed evidence can be
reviewed again; the gate does not add an overall iteration, time, or spend cap.
Review is itself model judgment, so its false approvals and false rejections
must be measured. The explicit configuration switch remains available for
controlled comparisons and applications that provide their own review.

## Classified counts

Use `semantic_classify`, rather than general-purpose `semantic_annotate`, for
classify-then-count, frequency comparisons, and extrema:

```python
labels = ["category A", "category B", "category C"]  # the complete task vocabulary
result = semantic_classify(
    table, "category", "Apply the task's full category definitions ...",
    allowed_labels=labels, where="kind = 'instance'", expected_count=verified_count,
    votes=3, model="sub",
)
counts = classification_counts(table, "category")
print(counts)
# Read the output before submitting another cell:
FINAL_CLASSIFICATION(table, "category", "compare", ["category A", "category B"])
```

The predicate selects original source fields, excluding metadata/footer rows.
The expected count must be established from the source and reconciled against
any declared dataset size. Never reduce a multiclass task to the two categories
being compared. `count` takes one label; `compare` takes two; `least` and `most`
take no labels and include every tie, including zero-count categories.

Incomplete votes, ties without a majority, invalid labels, mismatched scope, or
changed labels cannot certify a count. Reuse verifies the active annotation
version, actual configured model identity, source selection, label vocabulary,
batch context, and saved values. A failed replacement clears old selected
labels instead of leaving them looking current. Coverage certification proves
the declared scope and arithmetic; it does not prove semantic label accuracy.
The default acceptance gate requires the parent classification proof for
classification-and-aggregate tasks. A quoted literal count cannot replace it.
Internal verification work does not have to be narrated when the caller requests
only the final count.

## Financial calculations

Bind original values with `source_number(table,rowid,column,unit_span=...,
period_span=...)`. When the value appears only in prose, `source_text_number`
accepts its exact document offsets. It parses a single original numeric span;
it cannot accept a replacement literal or infer scaling.

`calculate` supports sum, mean, subtraction, multiplication, division, and
percentage conversion over parent-owned source/result IDs. `calculate_financial`
adds named, explicit conventions:

- Working capital: total or operating current assets less the corresponding liabilities.
- Quick ratio: liquid assets including short-term investments, or an explicitly labelled cash/receivables basis.
- Gross margin: gross profit over revenue, expressed as a percentage.
- Inventory turnover: cost of sales over year-end or average inventory.
- Effective tax rate: tax expense over pretax income, expressed as a percentage.

These operations require source-bound units and periods. Choosing the appropriate
definition and interpreting period/header relationships remain part of evidence
review. If the question/source does not settle a material convention difference,
state and calculate the alternatives. Never choose a formula to match a reference.
Negative results stay negative unless a caller explicitly supplies a different
metric convention. Directional questions may be answered from explicit comparative
prose without inventing an unavailable numeric ratio.

An immutable caller-supplied metric contract is checked before provider calls.
An invalid source selector, missing formula, or zero denominator raises a clear
error instead of sending the model through an unrepairable loop. Questions and
definitions that exceed the reviewer's input capacity likewise fail before calls.

`FINAL_CALC(result_id)` submits one raw result. `FINAL_CALCS` renders several:

```python
FINAL_CALCS(
    "Total working capital: {total}; operating working capital: {operating}.",
    {"total": total_result_id, "operating": operating_result_id},
    decimals=2,
)
```

Templates accept named result placeholders, not numeric literals, attribute
lookups, or arbitrary formatting. Rounding occurs in the parent. Raw results,
input identities, source contexts, and the operation graph remain in the proof.
The semantic gate can reject a correct calculation of the wrong metric or period.

## Table checks and evaluation

Arithmetic validation respects section boundaries and supported subtotal
hierarchies. Ambiguous structures are recorded as unchecked; they are not
silently considered valid. Genuine wrong totals continue to fail. This repairs
false table failures without disabling the corpus health gate.

Field regression judging receives the actual question and explicit output
requirements, not only a field ID. Boolean, role, full-name, and detailed answers
have different contracts. Every judge outcome, including rejection and provider
failure, is retained for review. Existing stored benchmark results are not
rewritten by these changes.

Validate changes on known failures, previously correct controls, and held-out
documents or contexts. Report classification confusion and exact count accuracy
separately, and record reviewer errors, answer accuracy, cost, and latency. Offline
tests demonstrate the checks; they do not establish a live benchmark accuracy gain.
