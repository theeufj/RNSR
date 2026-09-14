# Search-ladder contract

Rung-0 hit semantics are part of the agent contract, not an implementation
detail. A faster `_rung0_cells` that returned a slightly different row set
dropped golden-matter accuracy (49/49 → 44–46/49) twice before exact
parity was restored.

The gate is `rnsr replay` / `rnsr.eval.replay.replay`:

1. Collect `search_rung` events with `rung == 0` from trajectories, or a
   committed `queries.json`.
2. Run each query on the cells path and the legacy scan path.
3. Compare both to each other and to a frozen `baseline.json` row set.
4. Exit 2 on any set or order diff.

CI runs `tests/test_replay.py` on every push. Stage 2 provider swaps
(Tantivy behind rung 2, usearch behind rung 4) must pass the same replay
plus the golden-matter gate before merge.

To refresh the baseline after a *deliberate* semantics change:

```bash
rnsr replay --db tests/fixtures/replay/corpus.db \
  --queries tests/fixtures/replay/queries.json \
  --write-baseline tests/fixtures/replay/baseline.json
```
