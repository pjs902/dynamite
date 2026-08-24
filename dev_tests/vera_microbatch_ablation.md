# MicroBatch ablation (synthetic landscape, spec gate C2)

| proposer | models evaluated | best kinchi2 |
|---|---|---|
| GridWalk | 35 | 2705346.9 |
| MicroBatch(f=0.6) | 24 | 2705346.9 |

Budget cap: 60 models; identical deterministic landscape and rng stream.

Gate C2: MicroBatch best-chi2 <= GridWalk best-chi2 at equal budget.

## Scope of this result

MicroBatch reaches the same best chi2 from fewer models: re-centring
on a solved fraction rather than a whole batch turns the walk over
sooner. Verified across seed offsets 0-5, though the harness noise
term is 1e-9, so those are one deterministic trajectory rather than
six independent samples -- this is one landscape, not a distribution
of them. The wall-clock overlap on real workers (stragglers still
running while new work goes out) is additional to this and is not
measured here; that is spec gate D2, in production.

Earlier revisions of this file recorded a 35/35 tie with a note
explaining why no difference was expected. That tie was an artifact:
the fractional quorum was inert until 6071b79, so both arms ran the
same code path.
