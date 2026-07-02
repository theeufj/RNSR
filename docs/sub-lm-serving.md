# Sub-LM serving economics (spec Appendix A — deployment guidance, no code)

The system's cost centers are `semantic_annotate` and rung-5 sweeps:
embarrassingly parallel batch calls over ~200k-char contexts to a cheap
sub-LM. These are the source of the long-tailed p95 costs the RLM paper
reports; watch `sub_batch` events in trajectories and the p95 columns in
eval summaries to know when volume justifies self-hosting.

## When to consider self-hosting

Stay on API providers until *all* of the following hold:

1. Sustained sub-LM volume (not root-LM — root traffic is interactive and
   latency-sensitive; leave it on an API).
2. p95 cost per query is dominated by sub-calls (check the ladder-rung
   histogram: heavy rung-5 usage is the signal).
3. Batch density, not per-sequence latency, is the objective.

## PolarQuant configuration (applied literally, per the paper)

If self-hosting the sub-LM, quantize the transformer's internal KV cache
with PolarQuant (arXiv:2502.02617). This "KV" is the attention cache, not
the corpus store — same paper, different layer of the stack.

The trade, from the paper's own numbers:

| Dimension            | Effect                                            |
|----------------------|---------------------------------------------------|
| KV-cache compression | 4.2×+ → ~4× longer contexts or ~4× more concurrent sequences per GPU |
| Generation speed     | ~14% slower per sequence (43.7s vs 38.4s exact, 16k-prefill/1k-generate) |
| Quality              | within ~0.3 LongBench points of exact (45.45 vs 45.71) |

For our batch sub-LM workload, throughput-per-dollar dominates and 4×
batch density wins decisively; the latency penalty only matters for
interactive traffic, which stays on API providers anyway.

**Use the offline codebook variant.** Prefill is 3.4s vs 11.6s for online
per-prompt clustering — the online variant's per-prompt clustering cost is
unacceptable at our call volumes. The quality cost of offline vs online is
~0.7 LongBench points, acceptable for sub-LM duty.

## The design theme this closes

Precondition so the data distribution becomes predictable, then exploit
predictability with a fixed cheap scheme. The same principle appears in
rung-4 vector compression (random rotation → analytically known angle
distribution → no per-block constants), in checksum-validated table
extraction (document redundancy → free ground truth), and here in serving.
Spend structure once; compute cheaply forever after.
