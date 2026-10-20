# Deployment and operations

For a bounded starting configuration, set `RNSR_PROFILE=production` or use
`Settings.production()`. Individual settings override the preset. Research
settings remain unlimited by default. The supplied
[production environment example](../deploy/.env.production.example) sets finite
query and process spending caps, request pacing, redacted trajectories and
30-day retention. Replace/remove research overrides in an existing `.env` before
selecting the preset; explicit zero caps still disable their limits. Tune these
values to the workload.

Retention runs before queries and service jobs; an idle deployment needs a
scheduled cleanup if expiry must happen at an exact age. Retention applies to
trajectory files, not corpus artifacts or every output file. Encryption remains
an explicit key-management choice. In-flight requests may finish after a
recorded spending ceiling is reached.

The authorized HTTP job response now carries evidence, corpus health, source
citations, support-review decisions and opaque audit references. Audit retrieval
uses the same bearer authorization as job retrieval and returns an operational
timeline without document text, generated code or filesystem paths. Source
citations expose document IDs, verified spans and quotes, with path metadata
removed. Each service instance still has one authorization scope.

## Operational controls


OS containment and HTTP authentication are required. Trajectory protection
and aggregate spending limits remain deployment choices.

For Docker, use the supplied [container seccomp policy](../deploy/README.md)
to permit bubblewrap's namespace setup while retaining syscall filtering.

| Concern | Control | Why it exists |
|---|---|---|
| Untrusted documents | OS sandbox plus `RNSR_SANDBOX_FS_GUARD=true` | Generated code sees read-only runtime/corpus data and its private scratch directory |
| Kernel-level isolation | macOS `sandbox-exec`; Linux `bubblewrap` with user namespaces | Unsupported or restricted hosts refuse to execute generated code; containers must permit the nested namespaces |
| Service authorization | `RNSR_SERVICE_TOKEN`, `RNSR_SERVICE_CORPUS_ROOT` | Every route except liveness requires a bearer token; corpus paths cannot escape the configured root |
| Privileged text at rest | `RNSR_TRAJECTORY_CONTENT`, `RNSR_TRAJECTORY_KEY`, `RNSR_TRAJECTORY_RETENTION_DAYS` | Trajectories quote client documents verbatim |
| Run governance | `RNSR_RUN_SPEND_CEILING_USD`, `RNSR_MAX_IN_FLIGHT_REQUESTS`, `RNSR_MAX_REQUESTS_PER_MINUTE` | An optional spend ceiling applies across the job; concurrency and request pacing do not bound total spend |
| Silent failure | `--max-error-rate` (default: any error fails), `answers_status.csv`, `run_report.json` | A provider outage must not read as "the corpus does not say" |
| Run-to-run variance | `--consensus 2` | Two independent passes rarely make the same mistake, so a split is a signal |
| Model rot | `rnsr doctor --check-models` | Checks configured model availability; local pricing must also cover the selected models |
| Accuracy drift | `.github/workflows/regression.yml` + `rnsr regress --min-accuracy` | Provider behaviour changes outside this repository |

Known limits, stated plainly: jobs in `rnsr serve` live in one process's
memory (single node by design — a multi-node deployment needs a real
queue). Each service instance is one authorization scope: its token holders
can read all jobs, so use distinct instances, tokens and corpus roots per
tenant/matter. Completed jobs are evicted at the configured capacity;
submissions receive HTTP 503 when all slots are active. Lexical quote verification proves occurrence. The default semantic support
review additionally checks the requested entity, period, metric and relationship;
that model judgment can still be wrong, so trust tiers require calibration; and the form conventions in
`testMatter/mitchell_form_spec.json` were derived from observed misses on
that matter, so a new form starts from the generic rules and earns its own
conventions.

The governor enforces request concurrency and RPM across threads and event
loops. Every explicit retry consumes a sub-call attempt; hidden SDK retries
are disabled. Spend ceilings stop new requests after recorded usage reaches
the limit; already in-flight requests can still add cost.

Artifacts in older formats require `rnsr migrate /path/to/corpus.db` before use with
format v3. Migration is explicit and transactional; it validates metadata and
references before committing. Back up important artifacts before migrating.
