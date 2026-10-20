# Evaluation and release evidence

The stricter scorer accepts exact normalized text or an unambiguous scalar
numeric expression. Scalar comparison retains currency, scale and percentage
semantics. Text containment, a matching first number inside prose, and opposite
polarity are not automatic passes. Ambiguous answers need a contextual judge;
the judge must agree with every requested component, and its entire reply must
be `YES` or `NO`. Provider failures never certify correctness.

## Historical scoring audit

```bash
python -m rnsr.eval.rescore \
  --results runs/old-evaluation/results.jsonl \
  --out runs/scoring-audit/old-evaluation
```

This reads saved predictions without provider calls or modifying historical
files. It writes deterministic agreement and unresolved-adjudication counts.
An unresolved old positive may be a perfectly valid paraphrase; it is not
automatically a false pass. The audit does not establish corrected accuracy.
The versioned [aggregate audit](benchmarks/scoring-audit.json) contains counts,
source result hashes and scorer identity, without private predictions or golds.

## Run identity and resume

Every new `run_eval` directory has a sanitized `run_manifest.json`, binding
the complete dataset (questions, golds, context, metadata and source hashes),
resolved provider/model roles, settings, system, scorer, package content and
evaluation options. Secrets and source paths are omitted or represented by
hashes. Limit changes, source changes or code/model changes require a new run
directory. A legacy directory without a manifest cannot silently resume.
Original results remain available for diagnosis and offline audit.

The local results and trajectories can contain document content. The sanitized
manifest is not permission to publish those files. Public reports should include
the denominator, excluded/unavailable items, scoring method, cost and latency,
review decisions, false-positive rate and calibration signals.

## Frozen-rule and held-out protocol

1. Freeze the generic form policy, prompt code and scorer before evaluating.
   The manifest package-content hash identifies that policy. Keep form-specific
   conventions derived from observed misses in the development-fixture lane.
2. Evaluate untouched matters/forms separately. Generated matter seed `901`
   and office seed `902` are reserved for the added generic regression lane;
   neither uses the tuned Mitchell form conventions. They provide new synthetic
   coverage, not proof of performance on independent real matters.
3. Do not tune rules against those results and continue calling the same samples
   untouched. If failures inform a change, move those samples into development
   and reserve a fresh independently prepared set before making new claims.
4. Report overall/per-class accuracy, truly absent-item false-positive rate,
   high-tier error rate, review recall, auto-accept rate, reviewer errors,
   runtime and spend. A small sample needs its count alongside the percentage.
5. Report at least one previously correct control set as well as known failure
   cases. Offline tests verify mechanisms; live evaluation measures model behavior.

For a local live run with the frozen generic configuration:

```bash
RNSR_PROFILE=production rnsr eval --benchmark matter --seed 901 \
  --system docdb --concurrency 2 --run-dir runs/held-out-matter-901
RNSR_PROFILE=production rnsr eval --benchmark office --seed 902 \
  --system docdb --concurrency 2 --run-dir runs/held-out-office-902
```

These commands use configured provider credentials and incur provider spend.
Replace/remove existing research environment overrides first; profile defaults
do not replace explicit zero caps from `.env` or the process environment.
Use fresh directories when the manifest identity changes. Do not add custom
conventions after observing a run while reporting it as independent validation.

The weekly workflow retains Mitchell as a development/regression control and
adds generated generic matter/office coverage under finite caps. It skips live
work when no provider key is configured; that skip is not a passing live result.
No fresh live accuracy or trust-calibration claim accompanies these implementation
changes. Historical measurements remain explicitly historical.
