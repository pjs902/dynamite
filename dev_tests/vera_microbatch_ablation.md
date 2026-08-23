# MicroBatch ablation (synthetic landscape, spec gate C2)

| proposer | models evaluated | best kinchi2 |
|---|---|---|
| GridWalk | 35 | 2705346.9 |
| MicroBatch(f=0.6) | 35 | 2705346.9 |

Budget cap: 60 models; identical deterministic landscape and rng stream.

Gate C2: MicroBatch best-chi2 <= GridWalk best-chi2 at equal budget.

## Interpretation note (honest limitation)

Under this harness both proposers traverse the same proposal sequence and
converge at the same model count. That is expected, not a defect: the
MicroBatch advantage is *wall-clock overlap* - re-centering while stragglers
of the previous batch are still being evaluated on real workers - which a
sequential toy evaluator deliberately serializes away. The unit tests above
prove the fraction-quorum semantics; the throughput benefit is measured in
production (spec gate D2), not here.
